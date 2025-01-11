
import json

import utils

all_predicts = []
all_truths = []

def calculate(i, all_predicts, all_truths):
    predicts = []
    truths = []
    false_sample = []
    # data_path = "result/crime-100-10-Epoch-7/test_cs_" + str(i) + ".json"
    data_path = "result/crime-100-10-Epoch-7/test.json"
    # data_path = "result/laic-crime-100-10-Epoch-6/laic_result.json"
    with open(data_path, 'r', encoding='utf-8') as file:
        # datas = json.load(file)
        datas = file.read()
        # datas = file.readlines()
    datas = eval(datas)
    print(len(datas))
    total = 0
    for data in datas:
        total += 1
        true_crime = data["true_crime"]
        # print(true_crimes)
        bert_pred_crimes = data["bert_pred_crimes"]
        scores = data["yes_scores"]

        # if len(scores) == 0:
        #
        #     total += 1
        #     continue
        real_crimes = []
        if len(bert_pred_crimes) == 1:
            real_crimes.append(bert_pred_crimes[0])
        elif len(bert_pred_crimes) > 1:
            j = scores.index(max(scores))
            real_crimes.append(bert_pred_crimes[j])

        true_crimes_list = []
        true_crimes_list.append(true_crime)
        predicts.append(utils.list_to_one_hot(real_crimes))
        truths.append(utils.list_to_one_hot(true_crimes_list))
        if utils.list_to_one_hot(real_crimes) != utils.list_to_one_hot(true_crimes_list):
            false_sample.append(data)
    json_datas = json.dumps(false_sample, ensure_ascii=False, indent=4)
    # data_path = "result/temp/crime_data_test_" + str(i) + "_false.json"
    data_path = "result/temp/crime_data_test_false.json"
    with open(data_path, 'w', encoding='utf-8') as file:
        file.write(json_datas)
    utils.evaluate(predicts, truths)
    all_predicts += predicts
    all_truths += truths
    # print(total)
    return total
all = 0
for i in range(1, 2):
    all += calculate(i, all_predicts, all_truths)
print("-------------------------------------------------------------")
print("results：")
utils.evaluate(all_predicts, all_truths)