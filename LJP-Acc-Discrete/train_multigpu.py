
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
    os.environ['MASTER_PORT'] = '23347'
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


def load_model(model_path, args):
    tokenizer = BertTokenizer.from_pretrained(model_path)

    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    answer = ['否', '是']
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    # def __init__(self, model_name, answer_ids, args):
    net = BERTPrompt4LJP(model_path, answer_ids, args)
    return net, tokenizer


def train(model, optimizer, data_loader, rank, world_size, epoch, sampler):
    model.train()
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

        loss, scores = model(batch_enc, batch_attn, batch_labs)

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
       
        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs)

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

    val_impids = torch.IntTensor(imp_ids).to(rank)
    val_labels = torch.IntTensor(labels).to(rank)
    val_predicts = torch.IntTensor(predicts).to(rank)

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
    dist.barrier()

    if rank == 0:
        print(args)

    # load model
    net, tokenizer = load_model(args.model_path, args)

    # load data
    train_dataset = MyDataset(os.path.join(args.data_path,"laic_train.json"), tokenizer, status='train', args = args)
    if rank == 0:
        print('Vocabulary size of tokenizer after adding new tokens : %d' % args.vocab_size)
        print('num train: %d' % (len(train_dataset)))

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
    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    # AdamW
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wd},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)

    metrics = ['Acc', 'MP', 'MR', 'F1']
    best_val_result = {"acc":0, "mp":0, "mr":0, "f1":0}
    best_val_epoch = {"acc":0, "mp":0, "mr":0, "f1":0}
    for m in metrics:
        best_val_result[m] = 0.0
        best_val_epoch[m] = 0

    for epoch in range(args.epochs):
        st_tra = time.time()
        if rank == 0:
            print('--------------------------------------------------------------------')
            print('start training: ', datetime.now())
            print('Epoch: ', epoch)
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        loss, acc_tra, acc_pos_tra, pos_ratio_tra = \
            train(net, optimizer, train_loader, rank, world_size, epoch, train_sampler)
        # scheduler.step()

        end_tra = time.time()
        train_spend = (end_tra - st_tra) / 3600
        if rank == 0:
            print("Train Loss: %0.4f" % loss)
            print("Train ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
                  (acc_tra, acc_pos_tra, pos_ratio_tra, train_spend))
            if args.model_save:
                file = args.save_dir + '/Epoch-' + str(epoch) + '.pt'
                print('save file', file)
                torch.save(net.module.state_dict(), file)
        dist.barrier()
    cleanup()


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='DATA', type=str, help='Path')
    parser.add_argument('--data_file', default='data_train_bert.json', type=str, help='Path')
    parser.add_argument('--model_path', default='../../models/bert-base-chinese', type=str, help='Path')
    parser.add_argument('--neg_num', default=5, type=int, help='negtive sample nums')
    parser.add_argument('--data_ratio', default=100, type=int, help='train data ratio')
    parser.add_argument('--epochs', default=5, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size')
    parser.add_argument('--test_batch_size', default=200, type=int, help='test batch_size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')

    parser.add_argument('--device', default='cuda', help='device id')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')

    parser.add_argument('--model_save', default=False, type=bool, help='save model file')
    parser.add_argument('--model_save_path', default=False, type=str, help='save model file path')
    parser.add_argument('--log', default=False, type=bool, help='whether write log file')

    args = parser.parse_args()

    if args.model_save:
        # save_dir = './model_save/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_dir = args.model_save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        args.save_dir = save_dir
    if args.log:
        args.log_file = os.path.join(save_dir, "log.txt")

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
    t1 = time.time()
    run_time = (t1 - t0) / 3600
    print('Running time: %0.4f' % run_time)
    # fsdp_main(args)
