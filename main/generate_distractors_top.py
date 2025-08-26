# 用bert生成k个，拼接在原来的三个干扰项后面，并且附带置信度分数。（注意：原来三个干扰项在这里的数据中并没有分数）。这里的不同之处在于生成的干扰项如果跟原来的三个干扰项有重复，则从后面补够。
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from tqdm import tqdm

# 使用训练后的模型进行预测
model_file = '/media/disk3/CXL/DG/cdgp-main/models/CSG/cdgp-csg-bert-cloth-download'
# model_file = '/media/disk3/CXL/DG/cdgp-main/models/CSG/cdgp-csg-bert-cloth-download-dgen-finetune'

tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForMaskedLM.from_pretrained(model_file)

translator = pipeline('fill-mask', tokenizer=tokenizer, model=model, top_k=20)

df = pd.read_csv('/media/disk3/CXL/DG/t5-cdgp/data/sciq-train_source_target.csv')

predict_list = []
score_list = []

# 使用 tqdm 为迭代加上进度条
for id, row in tqdm(df.iterrows(), total=len(df)):
    # 处理原有的 target 列，取前三个元素
    original_targets = str(row['target']).strip().split()[:3]

    predict = translator(row['source'])
    distractors = [d['token_str'] for d in predict]
    scores = [s['score'] for s in predict]

    new_distractors = []
    new_scores = []
    index = 0

    while len(new_distractors) < 7:
        if index >= len(distractors):
            break
        distractor = distractors[index]
        score = scores[index]
        if distractor not in original_targets and distractor not in new_distractors:
            new_distractors.append(distractor)
            new_scores.append(score)
        index += 1

    # 将预测出的干扰项用空格连接成一个字符串
    distractors_str = " ".join(new_distractors)

    # 如果原'target'列存在数据，则在其后添加新预测的干扰项，用空格分隔
    if original_targets:
        updated_target = " ".join(original_targets) + " " + distractors_str
    else:
        updated_target = distractors_str

    predict_list.append(updated_target)
    score_list.append(" ".join(map(str, new_scores)))  # 分数列表转字符串

# 更新DataFrame中的'target'和'score'列
df['target'] = predict_list
df['score'] = score_list

# 保存到新的 CSV 文件中
output_path = '/media/disk3/CXL/DG/t5-cdgp/data/sciq-train_source_target(k=7)_without_repeat.csv'
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")