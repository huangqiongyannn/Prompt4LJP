import json

import utils

all_predicts = []
all_truths = []

def calculate(i, all_predicts, all_truths):
    predicts = []
    truths = []
    false_sample = []
    data_path = "result/crime-100-10-15-Epoch-7/test_cs_" + str(i) + ".json"
    # data_path = "result/crime-100-10-6-Epoch-7/test.json"
    with open(data_path, 'r', encoding='utf-8') as file:
        # Load the data
        datas = file.read()
    datas = eval(datas)
    print(len(datas))
    total = 0
    for data in datas:
        true_crime = data["true_crime"]
        bert_pred_crimes = data["bert_pred_crimes"]
        scores = data["yes_scores"]

        if len(scores) == 0:
            total += 1
            continue

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
        # Save mismatched results for further review
        if utils.list_to_one_hot(real_crimes) != utils.list_to_one_hot(true_crimes_list):
            false_sample.append(data)

    json_datas = json.dumps(false_sample, ensure_ascii=False, indent=4)
    # Save mismatched samples to a file (commented out)
    # data_path = "result/2024-04-12-15-59-24-Epoch-2/temp/data_test_" + str(i) + "_false.json"
    # with open(data_path, 'w', encoding='utf-8') as file:
    #     file.write(json_datas)

    print("Results for file number {}".format(i))
    utils.evaluate(predicts, truths)
    all_predicts += predicts
    all_truths += truths
    return total

all = 0
for i in range(1, 6):
    all += calculate(i, all_predicts, all_truths)
print("Total test sample size: {}".format(all))
print("-------------------------------------------------------------")
print("Final results")
utils.evaluate(all_predicts, all_truths)
print(all)
