import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm

train_num = 'train49'

model_file = '/media/disk3/CXL/DG/wang-2023/model_output/%s_model/checkpoint-179452' % train_num
#model_file = '/media/disk3/CXL/DG/model/t5-large-generation-race-Distractor'

tokenizer = T5Tokenizer.from_pretrained(model_file)
model = T5ForConditionalGeneration.from_pretrained(model_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv('./data/%s_data/race_test_updated_source.csv' % train_num)

predict_list = []

for id, row in tqdm(df.iterrows(), total=len(df)):
    input_ids = tokenizer.encode(row['source'], return_tensors='pt').to(device)

    outputs = model.generate(input_ids, max_new_tokens=512)

    predict = tokenizer.decode(outputs[0], skip_special_tokens=False)
    predict_list.append(predict)

df['predict'] = predict_list

save_file = model_file.split('/')[-1]
df.to_csv('./result/%s_result/%s_%s.csv' % (train_num, train_num, save_file), index=False)
