import json
from torch.utils.data import Dataset
import torch
import random
import numpy as np

import utils

class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, status, args):
 
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.status = status
        self.crimes = utils.get_crimes_list()
        if status == 'train':
            self.neg_num = args.neg_num
            self.data_ratio = args.data_ratio
        if status == 'train':
            self.data = self.prepro_train()
        elif status == 'test':
            self.data = self.prepro_test()
        else:
            self.data = self.prepro_dev()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def prepro_train(self):
        processed_data = []
        with open(self.data_path, 'r') as json_file:
            lines = json_file.readlines()
        selected_samples = utils.select_samples(lines, self.data_ratio)
        template = "根据以下事实描述，被告人是否构成<crime>罪：[MASK]。<fact>"
        impid = 0
        for line in selected_samples:
            json_data = json.loads(line)
            fact = json_data["fact_cut"]
            fact = fact.replace(" ", "")
            accuid = json_data["accu"]
            accu = utils.get_crime_by_id(accuid)

            # 正样本
            prompt = template.replace("<crime>", accu).replace("<fact>", fact)
            prompt_json = {"sentence": prompt,
                           "target": 1,  # Label is 1 or 0
                           "impid": impid
                           }
            processed_data.append(prompt_json)

            # 负样本
            n = self.neg_num
            crimes = utils.get_crimes_list()
            temp_crimes = random.sample([x for x in self.crimes if x != accu], n)
            for crime in temp_crimes:
                prompt = template.replace("<crime>", crime).replace("<fact>", fact)
                prompt_json = {"sentence": prompt,
                               "target": 0,  # Label is 1 or 0
                               "impid": impid
                               }
                processed_data.append(prompt_json)
            impid += 1

        return processed_data

    def prepro_dev(self):
        # Initialize an empty list to store processed data
        processed_data = []
        # Read the JSON file
        with open(self.data_path, 'r') as json_file:
            lines = json_file.readlines()
        # Define the prompt template
        template = "根据以下事实描述，被告人是否构成<crime>罪：[MASK]。<fact>"
        # Iterate through each line in the JSON file
        impid = 0;
        for line in lines:
            impid += 1
            # Load JSON data
            json_data = json.loads(line)
            fact = json_data["fact"]
            crimes_ = json_data["meta"]["accusation"]
            # Positive samples
            for crime in crimes_:
                prompt = template.replace("<crime>", crime).replace("<fact>", fact)
                prompt_json = {"sentence": prompt,
                               "target": 1 , # Label is 1 or 0
                               "impid":impid
                               }
                processed_data.append(prompt_json)
            # Negative samples
            abundance_crimes = [crime for crime in self.crimes if crime not in crimes_]
            neg_len = 10 - len(crimes_)
            negative_crimes = random.sample(abundance_crimes, neg_len)
            for crime in negative_crimes:
                prompt = template.replace("<crime>", crime).replace("<fact>", fact)
                prompt_json = {"sentence": prompt,
                               "target": 0 , # Label is 1 or 0
                               "impid": impid
                               }
                processed_data.append(prompt_json)
        return processed_data

    def prepro_test(self):
        processed_data = []
        with open(self.data_path, 'r') as json_file:
            lines = json_file.readlines()
        # Define the prompt template
        template = "根据以下事实描述，被告人是否构成<crime>罪：[MASK]。<fact>"
        # Iterate through each line in the JSON file
        impid = 0
        for line in lines:
            # Load JSON data
            json_data = json.loads(line)
            fact = json_data["fact_cut"]
            fact = fact.replace(" ", "")
            accuid = json_data["accu"]
            accu = utils.get_crime_by_id(accuid)
            prompt = template.replace("<crime>", accu).replace("<fact>", fact)
            # 正样本
            prompt_json = {"sentence": prompt,
                           "target": 1,  # Label is 1 or 0
                           "impid": impid,
                           "crimeid": accuid
                           }
            processed_data.append(prompt_json)
            # 负样本
            for index, crime in enumerate(self.crimes):
                prompt = template.replace("<crime>", crime).replace("<fact>", fact)
                if crime != accu:
                    prompt_json = {"sentence": prompt,
                                   "target": 0,  # Label is 1 or 0
                                   "impid": impid,
                                   "crimeid": index
                                   }
                    processed_data.append(prompt_json)
            impid += 1
        return processed_data


    def collate_fn(self, batch):
        sentences = [x['sentence'] for x in batch]
        target = [x['target'] for x in batch]
        imp_id = [x['impid'] for x in batch]
        if self.status == 'test':
            # Process crime IDs
            crimeids = [x['crimeid'] for x in batch]

        encode_dict = self.tokenizer.batch_encode_plus(
            sentences,
            # add_special_tokens=True,
            padding='max_length',
            max_length=512,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        target = torch.LongTensor(target)
        if self.status == 'test':
            return batch_enc, batch_attn, target, imp_id, crimeids
        return batch_enc, batch_attn, target, imp_id
