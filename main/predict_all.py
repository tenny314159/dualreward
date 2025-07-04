import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # 引入tqdm库
import os

train_num = 'train80'

# 读取数据集
# df = pd.read_csv('./data/%s_data/CLOTH-F-test.csv' % train_num)
df = pd.read_csv('./data/%s_data/DGen-test.csv' % train_num)
# df = pd.read_csv('./data/%s_data/multipul_task_split_with_token_CLOTH-F-test.csv' % train_num)

# 获取指定目录下的所有checkpoint文件夹
model_dir = '/media/disk3/CXL/DG/wang-2023/model_output/%s_model/' % train_num
checkpoint_folders = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('checkpoint-')]

# 遍历每个checkpoint文件夹进行预测
for checkpoint_folder in checkpoint_folders:
    # 使用训练后的模型进行预测
    print(f"当前正在使用的checkpoint: {checkpoint_folder}")
    translator = pipeline("text2text-generation", model=checkpoint_folder, device=0)

    predict_list = []

    # 使用 tqdm 为迭代加上进度条
    for id, row in tqdm(df.iterrows(), total=len(df)):
        predict = translator(row['source'])
        # print(row['source'], predict)
        predict_list.append(predict)

    df['predict'] = predict_list

    save_file = checkpoint_folder.split('/')[-1]
    df.to_csv('./result/%s_result/%s_%s.csv' % (train_num, train_num, save_file), index=False)