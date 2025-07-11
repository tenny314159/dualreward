import json
import numpy as np
import pandas as pd
import ast

from collections import Counter


def sort_by_frequency(lst):
    # 统计每个元素的出现次数
    count = Counter(lst)
    unique_list = list(dict.fromkeys(lst))
    # 按照出现次数排序，出现次数多的排前面，如果出现次数相同，则按照元素在列表中的第一次出现的顺序
    return sorted(unique_list, key=lambda x: (-count[x], lst.index(x)))


def evaluate(result):
    eval = {"P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0, "P@5": 0.0, "R@5": 0.0,
            "F1@5": 0.0,
            "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0, "MRR@3": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0, "NDCG@3": 0.0,
            "NDCG@5": 0.0, "NDCG@10": 0.0}
    # distractors = [d.lower() for d in result["distractors"]]
    # generations = [d.lower() for d in result["generations"]]
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

    generations = generations[:3]
    # distractor_first = generations[0]
    # generations = generations[1:]
    # generations = sort_by_frequency(generations)[:2]
    #
    # generations = [distractor_first] + generations

    distractors = [d.lower() for d in distractors]
    relevants = [int(generation.lower() in distractors) for generation in generations]

    # relevants = [int(generation in distractors) for generation in generations]
    # print(relevants)

    if relevants == []:
        relevants = [0, 0, 0]
        print(generations)
        print(distractors)

    # P@1
    if relevants[0] == 1:
        eval["P@1"] = 1
    else:
        eval["P@1"] = 0

    # R@1
    eval["R@1"] = relevants[:1].count(1) / len(distractors)

    # F1@1
    try:
        eval["F1@1"] = (2 * eval["P@1"] * eval["R@1"]) / \
                       (eval["P@1"] + eval["R@1"])
    except ZeroDivisionError:
        eval["F1@1"] = 0

    # P@3
    eval["P@3"] = relevants[:3].count(1) / 3

    # R@3
    eval["R@3"] = relevants[:3].count(1) / len(distractors)

    # F1@3
    try:
        eval["F1@3"] = (2 * eval["P@3"] * eval["R@3"]) / \
                       (eval["P@3"] + eval["R@3"])
    except ZeroDivisionError:
        eval["F1@3"] = 0

    # # P@5
    # eval["P@5"] = relevants[:5].count(1) / 5

    # # R@5
    # eval["R@5"] = relevants[:5].count(1) / len(distractors)

    # # F1@5
    # try:
    #     eval["F1@5"] = (2 * eval["P@5"] * eval["R@5"]) / \
    #         (eval["P@5"] + eval["R@5"])
    # except ZeroDivisionError:
    #     eval["F1@5"] = 0

    # P@10
    # eval["P@10"] = relevants[:10].count(1) / 10

    # R@10
    # eval["R@10"] = relevants[:10].count(1) / len(distractors)

    # F1@10
    # try:
    #     eval["F1@10"] = (2 * eval["P@10"] * eval["R@10"]) / \
    #         (eval["P@10"] + eval["R@10"])
    # except ZeroDivisionError:
    #     eval["F1@10"] = 0

    # MRR@3
    try:
        for i in range(3):
            if relevants[i] == 1:
                eval["MRR@3"] = 1 / (i+1)
                break
    except:
        eval["MRR@3"] = 0

    # for i in range(3):
    #     try:
    #         if relevants[i] == 1:
    #             eval["MRR@3"] = 1 / (i + 1)
    #             break
    #     except:
    #         pass

    # # MAP@5
    # rel_num = 0
    # for i in range(5):
    #     if relevants[i] == 1:
    #         rel_num += 1
    #         eval["MAP@5"] += rel_num / (i+1)
    # eval["MAP@5"] = eval["MAP@5"] / len(distractors)

    # NDCG@1
    eval["NDCG@1"] = ndcg_at_k(relevants, 1)

    # NDCG@3
    eval["NDCG@3"] = ndcg_at_k(relevants, 3)

    # NDCG@5
    # eval["NDCG@5"] = ndcg_at_k(relevants, 5)

    # # NDCG@10
    # eval["NDCG@10"] = ndcg_at_k(relevants, 10)

    return eval


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

train_num = 'train58'

# path of the result
result_path = '/media/disk3/CXL/DG/wang-2023/result/%s_result/%s_checkpoint-11600.csv' % (train_num, train_num)
df = pd.read_csv(result_path)

# evaluation = evaluate(df)

print("Evaluating...")
avg_eval = {"P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0, "P@5": 0.0, "R@5": 0.0,
            "F1@5": 0.0,
            "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0, "MRR@3": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0, "NDCG@3": 0.0,
            "NDCG@5": 0.0, "NDCG@10": 0.0}
for id, row in df.iterrows():
    eval = evaluate(row)
    for k in avg_eval.keys():
        avg_eval[k] += eval[k]

# calculate average
for k in avg_eval.keys():
    avg_eval[k] /= len(df)
# print(avg_eval)

save_evaluation_path = result_path.split('/')[-1].split('.')[0]

with open('./evaluation/%s_evaluation/%s_remove_repeat.json'% (train_num, save_evaluation_path), 'w') as json_file:
# with open('./evaluation/%s_evaluation/%s_with_test_token.json'% (train_num, save_evaluation_path), 'w') as json_file:
    json.dump(avg_eval, json_file, indent=4)





