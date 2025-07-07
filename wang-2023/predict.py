import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # 引入tqdm库

train_num = 'train79'

# 使用训练后的模型进行预测
# model_file = './model_output/%s_model/checkpoint-5395' % train_num
model_file = '/media/disk3/CXL/DG/wang-2023/model_output/%s_model/checkpoint-61880' % train_num
# model_file = '/media/disk3/CXL/DG/wang-2023/model_output/train26_model/checkpoint-77625'

translator = pipeline("text2text-generation", model=model_file, device=0)

df = pd.read_csv('./data/%s_data/CLOTH-F-test.csv' % train_num)
# df = pd.read_csv('./data/%s_data/DGen-test.csv' % train_num)
# df = pd.read_csv('./data/%s_data/multipul_task_split_with_token_CLOTH-F-test.csv' % train_num)

predict_list = []

# 使用 tqdm 为迭代加上进度条
for id, row in tqdm(df.iterrows(), total=len(df)):
    predict = translator(row['source'])
    # print(row['source'], predict)
    predict_list.append(predict)

df['predict'] = predict_list

save_file = model_file.split('/')[-1]
df.to_csv('./result/%s_result/%s_%s.csv'% (train_num, train_num, save_file), index=False)

