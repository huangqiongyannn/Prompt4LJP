import json
from torch.utils.data import Dataset
import torch
import random
import numpy as np

import utils


class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer, conti_tokens, status, args):
        # Load the tokenizer
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.status = status
        self.crimes = utils.get_crimes_list()
        # self.neg_num = args.neg_num
        # self.data_ratio = args.data_ratio
        self.args = args
        self.conti_tokens = conti_tokens
        if status == 'train':
            self.data = self.prepro_train()
        elif status == 'test':
            self.data = self.prepro_test()
        else:
            self.data = self.prepro_dev()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a sample from the dataset
        return self.data[idx]

    def prepro_train(self):
        # Initialize an empty list to store the processed data
        processed_data = []
        # Read the JSON file
        with open(self.data_path, 'r') as json_file:
            lines = json_file.readlines()
        data_ratio = self.args.data_ratio
        selected_samples = utils.select_samples(lines, data_ratio)
        # Define the template
        template1 = ''.join(self.conti_tokens[0]) + "<crime>"
        template2 = ''.join(self.conti_tokens[1]) + "[MASK]"
        template3 = ''.join(self.conti_tokens[2]) + "<fact>"
        template = template1  + template2  + template3
        # Iterate through each line in the JSON file
        impid = 0
        for line in selected_samples:
            # Load the JSON data
            json_data = json.loads(line)
            fact = json_data["fact_cut"]
            # fact = json_data["fact"]
            fact = fact.replace(" ","")
            # accuid = json_data["accu"]
            accuid = json_data["accu"]
            accu = utils.get_crime_by_id(accuid)

            # Positive sample
            prompt = template.replace("<crime>", accu).replace("<fact>", fact)
            prompt_json = {"sentence": prompt,
                           "target": 1,  # Label is 1 or 0
                           "impid": impid
                           }
            processed_data.append(prompt_json)

            # if impid == 0:
            #     print(prompt)

            # Negative sample
            n = self.args.neg_num
            # n = 8
            # few_crimes = utils.get_few_crime_list()
            
            # if accu in few_crimes:
            #     n = 12
            
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
        # Initialize an empty list to store the processed data
        processed_data = []
        # Read the JSON file
        with open(self.data_path, 'r') as json_file:
            lines = json_file.readlines()
        template1 = ''.join(self.conti_tokens[0]) + "<crime>"
        template2 = ''.join(self.conti_tokens[1]) + "[MASK]"
        template3 = ''.join(self.conti_tokens[2]) + "<fact>"
        template = template1 + template2  + template3
        # Iterate through each line in the JSON file
        impid = 0;
        for line in lines:
            impid += 1
            # Load the JSON data
            json_data = json.loads(line)
            fact = json_data["fact"]
            crimes_ = json_data["meta"]["accusation"]
            # Positive class
            for crime in crimes_:
                prompt = template.replace("<crime>", crime).replace("<fact>", fact)
                prompt_json = {"sentence": prompt,
                               "target": 1 , # Label is 1 or 0
                               "impid":impid
                               }
                processed_data.append(prompt_json)
            # Negative class
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
        template1 = ''.join(self.conti_tokens[0]) + "<crime>"
        template2 = ''.join(self.conti_tokens[1]) + "[MASK]"
        template3 = ''.join(self.conti_tokens[2]) + "<fact>"
        template = template1 + template2  + template3
        # Iterate through each line in the JSON file
        impid = 0
        for line in lines:
            # Load the JSON data
            json_data = json.loads(line)
            fact = json_data["fact_cut"]
            # fact = json_data["fact"]
            fact = fact.replace(" ", "")
            # accuid = json_data["accu"]
            accuid = json_data["accu"]
            accu = utils.get_crime_by_id(accuid)
            prompt = template.replace("<crime>", accu).replace("<fact>", fact)
            # Positive sample
            prompt_json = {"sentence": prompt,
                           "target": 1,  # Label is 1 or 0
                           "impid": impid,
                           "crimeid": accuid
                           }
            processed_data.append(prompt_json)
            # Negative sample
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
            # print("-----------------------------------------Processing crime ID---------------------------------------------------")
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
