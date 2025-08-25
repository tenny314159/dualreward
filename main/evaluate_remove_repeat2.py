import json
import numpy as np
import pandas as pd
import ast
from collections import Counter


def sort_by_frequency(lst):
    count = Counter(lst)
    unique_list = list(dict.fromkeys(lst))
    return sorted(unique_list, key=lambda x: (-count[x], lst.index(x)))


def evaluate(result, under_ten_count):
    eval = {"P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0, "P@5": 0.0, "R@5": 0.0,
            "F1@5": 0.0, "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0, "MRR@3": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0,
            "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0}

    distractors = result["target"].split(' ')
    generations = []

    answer = result['source'].split(' [SEP] ')[-1]

    predict_str = result['predict']
    predict_list = ast.literal_eval(predict_str)
    predict = predict_list[0]['generated_text'].split(' ')
    generations = predict

    generations = list(dict.fromkeys(generations))

    if answer in generations:
        generations.remove(answer)

    original_length = len(generations)
    generations = generations[:10]

    # 如果原始长度小于10，则增加计数
    if original_length < 10:
        under_ten_count += 1

    distractors = [d.lower() for d in distractors]
    relevants = [int(generation.lower() in distractors) for generation in generations]

    if relevants == []:
        relevants = [0, 0, 0]
        print(generations)
        print(distractors)

    # 计算各项指标
    # P@1
    if relevants[0] == 1:
        eval["P@1"] = 1
    else:
        eval["P@1"] = 0

    # R@1
    eval["R@1"] = relevants[:1].count(1) / len(distractors)

    # F1@1
    try:
        eval["F1@1"] = (2 * eval["P@1"] * eval["R@1"]) / (eval["P@1"] + eval["R@1"])
    except ZeroDivisionError:
        eval["F1@1"] = 0

    # P@3
    eval["P@3"] = relevants[:3].count(1) / 3

    # R@3
    eval["R@3"] = relevants[:3].count(1) / len(distractors)

    # F1@3
    try:
        eval["F1@3"] = (2 * eval["P@3"] * eval["R@3"]) / (eval["P@3"] + eval["R@3"])
    except ZeroDivisionError:
        eval["F1@3"] = 0

    # MRR@3
    try:
        for i in range(3):
            if relevants[i] == 1:
                eval["MRR@3"] = 1 / (i + 1)
                break
    except:
        eval["MRR@3"] = 0

    # NDCG@1
    eval["NDCG@1"] = ndcg_at_k(relevants, 1)

    # NDCG@3
    eval["NDCG@3"] = ndcg_at_k(relevants, 3)

    # NDCG@10
    eval["NDCG@10"] = ndcg_at_k(relevants, 10)

    return eval, under_ten_count


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


train_num = 'train50'
result_path = '/media/disk3/CXL/DG/wang-2023/result/%s_result/%s_checkpoint-92820.csv' % (train_num, train_num)
df = pd.read_csv(result_path)

print("Evaluating...")
avg_eval = {"P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0, "P@5": 0.0, "R@5": 0.0,
            "F1@5": 0.0, "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0, "MRR@3": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0,
            "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0}
under_ten_count = 0

for id, row in df.iterrows():
    eval, under_ten_count = evaluate(row, under_ten_count)
    for k in avg_eval.keys():
        avg_eval[k] += eval[k]

# 计算平均值
for k in avg_eval.keys():
    avg_eval[k] /= len(df)

# 输出不足10个元素的情况出现的次数
print(f"Number of times generations had fewer than 10 elements: {under_ten_count}")

save_evaluation_path = result_path.split('/')[-1].split('.')[0]
with open('./evaluation/%s_evaluation/%s_remove_repeat.json' % (train_num, save_evaluation_path), 'w') as json_file:
    json.dump(avg_eval, json_file, indent=4)