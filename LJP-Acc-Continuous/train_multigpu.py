import os
import argparse
import pickle
import time
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from datetime import datetime
import torch.cuda
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import AdamW

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import BERTPrompt4LJP
from prepro_data import *
from utils import evaluate
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23346'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def load_tokenizer(model_name, args):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    conti_tokens1 = []
    for i in range(args.num_conti1):
        conti_tokens1.append('[P' + str(i + 1) + ']')
    conti_tokens2 = []
    for i in range(args.num_conti2):
        conti_tokens2.append('[Q' + str(i + 1) + ']')
    conti_tokens3 = []
    for i in range(args.num_conti3):
        conti_tokens3.append('[M' + str(i + 1) + ']')

    new_tokens = ['[NSEP]']
    tokenizer.add_tokens(new_tokens)

    conti_tokens = conti_tokens1 + conti_tokens2 + conti_tokens3
    tokenizer.add_tokens(conti_tokens)

    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    return tokenizer, conti_tokens1, conti_tokens2, conti_tokens3


def load_model(model_path, tokenizer, args):
    # tokenizer = BertTokenizer.from_pretrained(model_path)

    # new_vocab_size = len(tokenizer)
    # args.vocab_size = new_vocab_size

    answer = ['否', '是']  # "No" and "Yes" in Chinese
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    # def __init__(self, model_name, answer_ids, args):
    net = BERTPrompt4LJP(model_path, answer_ids, args)
    return net


def train(model, optimizer, data_loader, rank, world_size, epoch, sampler):
    model.train()
    # The metrics and loss are moved to the designated GPU
    mean_loss = torch.zeros(2).to(rank)
    acc_cnt = torch.zeros(2).to(rank)
    acc_cnt_pos = torch.zeros(2).to(rank)
    data_loader = tqdm(data_loader)
    if sampler:
        sampler.set_epoch(epoch)
    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs, batch_imp = data

        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs, rank)

        # Backpropagation and gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss[0] += loss.item()
        mean_loss[1] += 1

        predict = torch.argmax(scores.detach(), dim=1)
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)

        positive_idx = torch.where(batch_labs == 1)[0]
        num_correct_pos = (predict[positive_idx] == batch_labs[positive_idx]).sum()
        acc_cnt_pos[0] += num_correct_pos
        acc_cnt_pos[1] += positive_idx.size(0)

    # Distributed reduction operation
    # The operation sums up the variables across all processes in the distributed system to synchronize them
    # This is a synchronous operation
    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt_pos, op=dist.ReduceOp.SUM)

    loss_epoch = mean_loss[0] / mean_loss[1]
    acc = acc_cnt[0] / acc_cnt[1]
    acc_pos = acc_cnt_pos[0] / acc_cnt_pos[1]
    pos_ratio = acc_cnt_pos[1] / acc_cnt[1]

    return loss_epoch.item(), acc.item(), acc_pos.item(), pos_ratio.item()

@torch.no_grad()
def eval(model, rank, world_size, data_loader):
    model.eval()
    data_loader = tqdm(data_loader)

    labels = []
    imp_ids = []
    predicts = []
    acc_cnt = torch.zeros(2).to(rank)
    acc_cnt_pos = torch.zeros(2).to(rank)

    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs, batch_imp = data
        imp_ids = imp_ids + batch_imp
        labels = labels + batch_labs.cpu().numpy().tolist()
        # Move data to the GPU
        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs)
        # scores.detach() creates a new tensor sharing data with scores but without gradient information
        # It is a copy of the original tensor without the gradient
        predict = torch.argmax(scores.detach(), dim=1)
        # print("predict = {}".format(predict))
        predicts = predicts + predict.cpu().numpy().tolist()
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)

        positive_idx = torch.where(batch_labs == 1)[0]
        num_correct_pos = (predict[positive_idx] == batch_labs[positive_idx]).sum()
        acc_cnt_pos[0] += num_correct_pos
        acc_cnt_pos[1] += positive_idx.size(0)

    dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt_pos, op=dist.ReduceOp.SUM)

    acc = acc_cnt[0] / acc_cnt[1]
    acc_pos = acc_cnt_pos[0] / acc_cnt_pos[1]
    pos_ratio = acc_cnt_pos[1] / acc_cnt[1]

    # Concatenate multiple tensors in the val_scores list along the specified dimension. dim=0 means along the first dimension (rows).
    # torch.IntTensor converts to tensor
    val_impids = torch.IntTensor(imp_ids).to(rank)
    val_labels = torch.IntTensor(labels).to(rank)
    val_predicts = torch.IntTensor(predicts).to(rank)

    # Create a new tensor with the same shape as val_scores but all zeros
    val_predicts_list = [torch.zeros_like(val_predicts).to(rank) for _ in range(world_size)]
    val_impids_list = [torch.zeros_like(val_impids).to(rank) for _ in range(world_size)]
    val_labels_list = [torch.zeros_like(val_labels).to(rank) for _ in range(world_size)]

    dist.all_gather(val_impids_list, val_impids)
    dist.all_gather(val_predicts_list, val_predicts)
    dist.all_gather(val_labels_list, val_labels)

    return val_predicts_list, acc.item(), acc_pos.item(), pos_ratio.item(), val_impids_list, val_labels_list


def fsdp_main(rank, world_size, args):
    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    init_seed(rank + 1)
    if rank == 0:
        if args.log:
            sys.stdout = Logger(args.log_file, sys.stdout)
    setup(rank, world_size)

    print('| distributed init rank {}'.format(rank))
    # Wait for all processes to synchronize
    dist.barrier()

    if rank == 0:
        print(args)

    # Load model
    # net, tokenizer = load_model(args.model_path, args)

    # Load tokenizer
    tokenizer, conti_tokens1, conti_tokens2, conti_tokens3 = load_tokenizer(args.model_path, args)
    conti_tokens = [conti_tokens1, conti_tokens2, conti_tokens3]

    # Load model
    net = load_model(args.model_path, tokenizer, args)

    # Load data
    train_dataset = MyDataset(os.path.join(args.data_path,"train_cs.json"), tokenizer, conti_tokens = conti_tokens, status='train', args = args)
    if rank == 0:
        print('Vocabulary size of tokenizer after adding new tokens : %d' % args.vocab_size)
        print('num train: %d' % (len(train_dataset)))

    # Distributed data sampler
    train_sampler = DistributedSampler(train_dataset,
                                       rank=rank,
                                       num_replicas=world_size,
                                       shuffle=True)
    # val_sampler = DistributedSampler(val_dataset,
    #                                  rank=rank,
    #                                  num_replicas=world_size)
    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler,
                    'shuffle': False, 'pin_memory': True, 'collate_fn': train_dataset.collate_fn}
    # val_kwargs = {'batch_size': args.test_batch_size, 'sampler': val_sampler,
    #               'shuffle': False, 'pin_memory': True, 'collate_fn': val_dataset.collate_fn}

    nw = 8
    cuda_kwargs = {'num_workers': nw, 'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    # val_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    # val_loader = DataLoader(val_dataset, **val_kwargs)
    # Move the model to GPU
    net = net.to(rank)
    # And use DDP for distributed training setup.
    net = DDP(net, device_ids=[rank])

    # AdamW
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm
    # Initialize distributed training across multiple processes
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
    t1 = time.time()
    run_time = (t1 - t0) / 3600
    print('Running time: %0.4f' % run_time)
    # fsdp_main(args)  # Start the FSDP training function
