import json

import numpy as np
import random

crime_path = "../../data/laic_accu.txt"
with open(crime_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
crimes = [line.strip() for line in lines]
# print(len(crimes))
def get_crimes_list():
    return crimes

def get_crimes_list_len():
    return len(crimes)

def get_crime_by_id(id):
    return crimes[id]


def list_to_one_hot(temp_crimes):
    # crimes = [line.strip() for line in lines]
    result = []
    for crime in crimes:
        if crime in temp_crimes:
            result.append(1)
        else:
            result.append(0)
    return result

def one_hot_to_list(onehot):
    indices = [index for index, value in enumerate(onehot) if value == 1]
    selected_crimes = [crimes[index] for index in indices]
    return selected_crimes

def select_samples(samples, percentage):
    num = 0
    data = {}
    for line in samples:
        dic = json.loads(line)
        crime = dic["accu"]
        if crime not in data:
            data[crime] = []
        data[crime].append(line)
        num += 1
    # print(num)

    selected_samples = []
    for label, samples in data.items():
        num_samples = len(samples)
        # print(num_samples)
        num_selected = int(num_samples * (percentage / 100))
        selected_samples += random.sample(samples, num_selected)
    # print(len(selected_samples))
    return selected_samples

def macro_precision_recall_f1_accuracy(y_true, y_pred):
    L = y_true.shape[1] 
    M = y_true.shape[0] 
    print("标签总数：", L)

    precision_per_label = np.zeros(L)
    recall_per_label = np.zeros(L)
    f1_per_label = np.zeros(L)

    for j in range(L):
        true_positives = np.sum(np.logical_and(y_true[:, j], y_pred[:, j]))
        false_positives = np.sum(np.logical_and(np.logical_not(y_true[:, j]), y_pred[:, j]))
        false_negatives = np.sum(np.logical_and(y_true[:, j], np.logical_not(y_pred[:, j])))


        precision_per_label[j] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0

        recall_per_label[j] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_per_label[j] = 2 * true_positives / (2 * true_positives + false_positives + false_negatives) if (2 * true_positives + false_positives + false_negatives) != 0 else 0


    macro_precision = np.mean(precision_per_label)
    macro_recall = np.mean(recall_per_label)
    macro_f1 = np.mean(f1_per_label)

    accuracy = np.sum(np.all(y_true == y_pred, axis=1)) / M

    return macro_precision, macro_recall, macro_f1, accuracy, precision_per_label, recall_per_label

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]



def find_crime_category(crime):
    categories = crimes_class.categories
    for category, crime_list in categories.items():
        if crime in crime_list:
            return category
    return None

def select_megative_crimes(crime, n):
    available_crimes = [c for c in crime if c != crime]

    if n > len(available_crimes):
        return available_crimes 

    selected_crimes = random.sample(available_crimes, n)
    return selected_crimes

def evaluate(y_preds, y_trues):
    y_true = np.array(y_trues)
    y_pred = np.array(y_preds)

    macro_precision, macro_recall, macro_f1, accuracy, precision_per_label, recall_per_label  = macro_precision_recall_f1_accuracy(y_true, y_pred)
    print("test: Acc: %0.4f\tMP: %0.4f\tMR: %0.4f\tF1: %0.4f\t" %
          (accuracy, macro_precision, macro_recall, macro_f1))

    return accuracy, macro_precision, macro_recall, macro_f1
