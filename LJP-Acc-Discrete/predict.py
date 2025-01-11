import os
import argparse
import pickle
import shutil
import time
import sys
from random import random

import numpy as np
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

import utils
from model import BERTPrompt4LJP
from prepro_data import *
from utils import evaluate


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '23342'
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


def load_model(model_path, args):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    answer = ['否', '是']
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    net = BERTPrompt4LJP(model_path, answer_ids, args)
    return net, tokenizer


def eval(model, rank, world_size, data_loader):
    model.eval()
    data_loader = tqdm(data_loader)

    labels = []
    imp_ids = []
    predicts = []
    crimeids = []
    judge_scores = []
    acc_cnt = torch.zeros(2).to(rank)
    acc_cnt_pos = torch.zeros(2).to(rank)

    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs, batch_imp, batch_crimeids = data
        imp_ids = imp_ids + batch_imp
        crimeids = crimeids + batch_crimeids
        labels = labels + batch_labs.cpu().numpy().tolist()

      
        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs)
    
        # predict = torch.argmax(scores.detach(), dim=1)
        max_score, predict = torch.max(scores.detach(), dim=1)
        judge_scores += max_score.cpu().numpy().tolist()
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


    test_impids = torch.IntTensor(imp_ids).to(rank)
    test_labels = torch.IntTensor(labels).to(rank)
    test_predicts = torch.IntTensor(predicts).to(rank)
    test_crimeids = torch.IntTensor(crimeids).to(rank)
    test_scores = torch.FloatTensor(judge_scores).to(rank)

    test_predicts_list = [torch.zeros_like(test_predicts).to(rank) for _ in range(world_size)]
    test_impids_list = [torch.zeros_like(test_impids).to(rank) for _ in range(world_size)]
    test_labels_list = [torch.zeros_like(test_labels).to(rank) for _ in range(world_size)]
    test_crimeids_list = [torch.zeros_like(test_crimeids).to(rank) for _ in range(world_size)]
    test_scoress_list = [torch.zeros_like(test_scores).to(rank) for _ in range(world_size)]

    dist.all_gather(test_impids_list, test_impids)
    dist.all_gather(test_predicts_list, test_predicts)
    dist.all_gather(test_labels_list, test_labels)
    dist.all_gather(test_crimeids_list, test_crimeids)
    dist.all_gather(test_scoress_list, test_scores)

    return test_predicts_list, acc.item(), acc_pos.item(), pos_ratio.item(), test_impids_list, test_labels_list, test_crimeids_list, test_scoress_list


def ddp_main(rank, world_size, args):
    args.rank = rank
    args.world_size = world_size
    init_seed(rank + 1)

    setup(rank, world_size)

    print('| distributed init rank {}'.format(rank))
    dist.barrier()

    # load model
    net, tokenizer = load_model(args.model_path, args)
    # net, tokenizer = load_model("../../models/bert-base-chinese", args)

    # load data
    test_dataset = MyDataset(os.path.join(args.data_path, args.data_name), tokenizer,
                             status="test", args = args)

    if rank == 0:
        print(args)
        print('Vocabulary size of tokenizer after adding new tokens : %d' % args.vocab_size)
        print('num test: %d' % len(test_dataset))

    test_sampler = DistributedSampler(test_dataset,
                                      rank=rank,
                                      num_replicas=world_size)
    nw = 2
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': test_sampler,
                   'shuffle': False, 'pin_memory': False,
                   'num_workers': nw, 'collate_fn': test_dataset.collate_fn}

    test_loader = DataLoader(test_dataset, **test_kwargs)

    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    net.module.load_state_dict(torch.load(args.model_file, map_location=map_location))

    with torch.no_grad():
        st_test = time.time()
        test_predicts, acc_test, acc_pos_test, pos_ratio_test, test_impids, test_labels, test_crimeids, test_scores = \
            eval(net, rank, world_size, test_loader)
        impressions = {}  # {1: {'score': [], 'lab': []}}

        for i in range(world_size):
            predicts, imp_id, labs, crimeids, scores = test_predicts[i], test_impids[i], test_labels[i], test_crimeids[i], test_scores[i]
            assert predicts.size() == imp_id.size() == labs.size() == crimeids.size() == scores.size()
            predicts = predicts.cpu().numpy().tolist()
            imp_id = imp_id.cpu().numpy().tolist()
            labs = labs.cpu().numpy().tolist()
            crimeids = crimeids.cpu().numpy().tolist()
     
            scores_cpu = scores.cpu()
     
            numpy_array = scores_cpu.numpy()

            scores_list = numpy_array.tolist()
            for j in range(len(predicts)):
                pre, imp, lab, index, score = predicts[j], imp_id[j], labs[j], crimeids[j], scores_list[j]
                if imp not in impressions:
                    size = utils.get_crimes_list_len()
                    impressions[imp] = {'pre': [0] * size, 'lab': [0] * size, 'score': [0] * size}
                    impressions[imp]['pre'][index] = pre
                    impressions[imp]['lab'][index] = lab
                    impressions[imp]['score'][index] = score
                else:
                    impressions[imp]['pre'][index] = pre
                    impressions[imp]['lab'][index] = lab
                    impressions[imp]['score'][index] = score
        process_datas = []
        file = os.path.join(args.data_path, args.data_name)
        with open(file, 'r') as json_file:
            lines = json_file.readlines()
        impid = 0
        for line in lines:
            json_data = json.loads(line)
            fact = json_data["fact_cut"]
            accuid = json_data["accu"]
            crime = utils.get_crime_by_id(accuid)
            data = {}
            data["id"] = impid
            data["fact"] = fact
            data["true_crime"] = crime
            process_datas.append(data)
            impid += 1

   
        predicts, truths, all_scores = [], [], []
        for imp in impressions:
            pres, labs, scores = impressions[imp]['pre'], impressions[imp]['lab'], impressions[imp]['score']
            assert len(pres) == len(labs) == len(scores) == utils.get_crimes_list_len()
            predicts.append(pres)
            truths.append(labs)
            # all_scores.append(scores)
            assert int(imp) == process_datas[int(imp)]["id"]
            process_datas[int(imp)]["bert_pred_crimes"] = utils.one_hot_to_list(pres)
            indices = [index for index, value in enumerate(pres) if value == 1]
            process_datas[int(imp)]["yes_scores"] = [value for index, value in enumerate(scores) if index in indices]
            assert len(process_datas[int(imp)]["bert_pred_crimes"]) == len(process_datas[int(imp)]["yes_scores"])


        if rank == 0:
            if not os.path.exists(args.truth_path):
                with open(args.truth_path, 'w') as truths_file:
                    for truth_list in truths:
                        line = str(truth_list) + '\n'
                        truths_file.write(line)
            else:
                with open(args.truth_path, 'a') as truths_file:
                    for truth_list in truths:
                        line = str(truth_list) + '\n'
                        truths_file.write(line)
            if not os.path.exists(args.predict_path):
                with open(args.predict_path, 'w') as predicts_file:
                    for predict_list in predicts:
                        line = str(predict_list) + '\n'
                        predicts_file.write(line)
            else:
                with open(args.predict_path, 'a') as predicts_file:
                    for predict_list in predicts:
                        line = str(predict_list) + '\n'
                        predicts_file.write(line)
            # output_file_path = "DATA/CAIL2018/finetune_temp.json"
            utils.evaluate(predicts, truths)
            # output_file_path = args.result_path + "/test_cs_" + str(args.index) +".json"
            
            output_file_path = args.result_path + "/laic_result.json"
            
            # output_file_path = 'RESULT/temp_result.json'
            # with open(output_file_path, 'w', encoding='utf-8') as file:
            #     for sample in process_datas:
            #         file.write(json.dumps(sample, ensure_ascii=False) + '\n')
            json_datas = json.dumps(process_datas, ensure_ascii=False, indent=4)
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json_datas)

    cleanup()


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='DATA/CAIL2018', type=str, help='Path')
    parser.add_argument('--model_path', default='../../models/bert-base-chinese', type=str, help='Path')
    parser.add_argument('--test_batch_size', default=15, type=int, help='test batch_size')
    parser.add_argument('--max_tokens', default=500, type=int, help='max number of tokens')
    parser.add_argument('--result_path', default='resut file path', type=str, help='Path')

    parser.add_argument('--device', default='cuda', help='device id')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')

    parser.add_argument('--model_file', default='', type=str, help='model file')
    parser.add_argument('--log', default=False, type=bool, help='whether write log file')
    # parser.add_argument('--log', default=True, type=bool, help='whether write log file')

    args = parser.parse_args()


    result_path = args.result_path

    if os.path.exists(result_path):
      
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    predict_path = result_path + "/predicts.txt"
    truth_path = result_path +   "/truths.txt"
    args.predict_path = predict_path
    args.truth_path = truth_path

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    WORLD_SIZE = torch.cuda.device_count()

    args.data_name = "laic_test.json"
    args.index = 7
    WORLD_SIZE = torch.cuda.device_count()
    print(WORLD_SIZE)
    mp.spawn(ddp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)

    t1 = time.time()
    run_time = (t1 - t0) / 3600
    print('testing time: %0.4f' % run_time)