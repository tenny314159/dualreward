
# import pandas as pd
#
# # 读取 Parquet 文件
# df = pd.read_parquet('RACE/all/validation-00000-of-00001.parquet')
#
# # 将 DataFrame 写入 CSV 文件，指定编码为 UTF-8
# df.to_csv('RACE/all/race-validation.csv', index=False, encoding='utf-8')




# import pandas as pd
# import re
#
# # 读取CSV文件
# df = pd.read_csv('RACE/all/race-test.csv')
#
# # 定义一个函数用于从字符串中解析出选项列表
# def parse_options(options_str):
#     # 使用正则表达式匹配每个选项，支持单引号和双引号
#     options = re.findall(r'\'([^\']+)\'|"([^"]+)"', options_str)
#     # 将匹配到的结果合并为单一的列表项
#     options = [''.join(t) for t in options]
#     return options
#
# # 应用上述函数将字符串转换为列表
# df['options'] = df['options'].apply(parse_options)
#
# # 创建'source'列的辅助函数
# def create_source_column(row):
#     answer_index = {'A':0, 'B':1, 'C':2, 'D':3}.get(row['answer'])
#     if answer_index is None:
#         print(f"警告: 未知的答案 '{row['answer']}' 在行 {row.name}")
#         return ''
#     if answer_index >= len(row['options']):
#         print(f"警告: 索引超出范围 - 行 {row.name} 的答案索引为 {answer_index}, 但'options'列只有 {len(row['options'])} 个选项")
#         return ''
#     return row['question'] + " <sep> " + row['options'][answer_index] + " <sep> " + row['article']
#
# # 创建'target'列的辅助函数
# def create_target_column(row):
#     answer_index = {'A':0, 'B':1, 'C':2, 'D':3}.get(row['answer'])
#     if answer_index is None or answer_index >= len(row['options']):
#         return ''
#     return ' <sep> '.join([option for idx, option in enumerate(row['options']) if idx != answer_index])
#
# # 应用create_source_column函数来创建'source'列
# df['source'] = df.apply(create_source_column, axis=1)
#
# # 应用create_target_column函数来创建'target'列
# df['target'] = df.apply(create_target_column, axis=1)
#
# # 检查生成的'source'列是否有空值，表示处理过程中出现了问题
# problem_rows = df[df['source'].eq('') | df['target'].eq('')].index
# print(f"有问题的行索引: {problem_rows.tolist()}")
#
# # 删除有问题的行
# df.drop(problem_rows, inplace=True)
#
# # 如果需要，可以保存修改后的DataFrame到新的CSV文件中
# df.to_csv('RACE/all/race-test-source.csv', index=False)


# import pandas as pd
#
# # 读取原始CSV文件
# df = pd.read_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors.csv')  # 将 'your_file.csv' 替换为实际的文件名
#
# # 取前20行数据
# new_df = df.head(20)
#
# # 将前20行数据另存为新的CSV文件
# new_df.to_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors_20.csv', index=False)  # 将 'new_file.csv' 替换为你想要的新文件名


# import pandas as pd
#
# # 读取 CSV 文件
# file_path = 'RACE/all/race-train-ten-distractors-score.csv'
# df = pd.read_csv(file_path)
#
# # 假设要处理的列名为 'your_column'，float1 为 1.2345
# float1 = 1.0000
# column_name = 'score'
#
# # 将 float1 保留四位小数
# formatted_float1 = '{:.4f}'.format(float1)
#
# # 对指定列的每个元素前面补上三个格式化后的 float1
# df[column_name] = [f"{formatted_float1},{formatted_float1},{formatted_float1},{str(item)}" for item in df[column_name]]
#
# # 保存修改后的 DataFrame 到新的 CSV 文件
# output_file_path = 'RACE/all/race-train-ten-distractors-score2.csv'
# df.to_csv(output_file_path, index=False)
#
# print(f"处理完成，结果已保存到 {output_file_path}")


# import json
# import pandas as pd
#
#
# def load_json(file_path):
#     """加载包含多个JSON对象的文件"""
#     data_list = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data_list.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON: {e}")
#                 continue
#     return data_list
#
#
# def convert_to_csv(data_list, output_file):
#     # 准备数据帧的列名
#     columns = ['file_id', 'question_id', 'distractor_id', 'question', 'distractor', 'answer_text', 'related_sentences']
#
#     # 重组数据
#     rows = []
#     for data in data_list:
#         related_sentences = ' '.join([' '.join(sent) for sent in data['sent']])  # 将相关的句子合并成字符串
#         row = [
#             data['id']['file_id'],
#             data['id']['question_id'],
#             data['id']['distractor_id'],
#             ' '.join(data['question']),
#             ' '.join(data['distractor']),
#             ' '.join(data['answer_text']),
#             related_sentences
#         ]
#         rows.append(row)
#
#     # 创建 DataFrame 并写入 CSV 文件
#     df = pd.DataFrame(rows, columns=columns)
#     df.to_csv(output_file, index=False, encoding='utf-8')
#     print(f"CSV file '{output_file}' has been created successfully.")
#
#
# if __name__ == "__main__":
#     # JSON文件路径
#     json_file_path = 'Distractor-Generation-RACE-master\data\distractor/race_test_updated.json'  # 替换为你的JSON文件路径
#     # 输出CSV文件路径
#     output_csv_file = 'Distractor-Generation-RACE-master\data\distractor/race_test_updated.csv'  # 替换为你希望保存的CSV文件路径
#
#     # 加载JSON数据
#     data_list = load_json(json_file_path)
#
#     # 转换为CSV文件
#     convert_to_csv(data_list, output_csv_file)


# import json
# import pandas as pd
#
#
# def load_json(file_path):
#     """加载包含多个JSON对象的文件"""
#     data_list = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
#         lines = content.split('\n')  # 分割成行
#         for line in lines:
#             if line.strip():  # 检查是否为空行
#                 try:
#                     data_list.append(json.loads(line))
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding JSON: {e}")
#                     continue
#     return data_list
#
#
# def convert_to_csv(data_list, output_file):
#     # 准备数据帧的列名
#     columns = ['file_id', 'question_id', 'distractor_id', 'question', 'distractor', 'answer_text', 'related_sentences']
#
#     # 重组数据
#     rows = []
#     for data in data_list:
#         related_sentences = ' '.join([' '.join(sent) for sent in data['sent']])  # 将相关的句子合并成字符串
#         row = [
#             data['id'],
#             data['question_id'],
#             data['distractor_id'],
#             ' '.join(data['question']),
#             # ' '.join(data['distractor_list'][0]),  # test数据集需要
#             ' '.join(data['distractor']),
#             ' '.join(data['answer_text']),
#             related_sentences
#         ]
#         rows.append(row)
#
#     # 创建 DataFrame 并写入 CSV 文件
#     df = pd.DataFrame(rows, columns=columns)
#     df.to_csv(output_file, index=False, encoding='utf-8')
#     print(f"CSV file '{output_file}' has been created successfully.")
#
#
# if __name__ == "__main__":
#     # JSON文件路径
#     json_file_path = 'Distractor-Generation-RACE-master/data/distractor/race_dev_original.json'  # 使用正斜杠或原始字符串
#     # 输出CSV文件路径
#     output_csv_file = 'Distractor-Generation-RACE-master/data/distractor/race_dev_original.csv'
#
#     # 加载JSON数据
#     data_list = load_json(json_file_path)
#
#     # 转换为CSV文件
#     convert_to_csv(data_list, output_csv_file)


# import pandas as pd
#
#
# def process_csv(input_file, output_file):
#     # 读取CSV文件
#     df = pd.read_csv(input_file)
#
#     # 检查必要的列是否存在
#     required_columns = ['question', 'answer_text', 'related_sentences']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         raise ValueError(f"缺少必要的列: {', '.join(missing_columns)}")
#
#     # 创建source列
#     df['source'] = df['question'].astype(str) + ' <sep> ' + \
#                    df['answer_text'].astype(str) + ' <sep> ' + \
#                    df['related_sentences'].astype(str)
#
#     # 如果需要，可以只保留某些列
#     columns_to_keep = ['file_id', 'question_id', 'distractor_id', 'question', 'distractor', 'answer_text',
#                        'related_sentences', 'source']
#     df = df[[col for col in columns_to_keep if col in df.columns]]
#
#     # 将结果写入新的CSV文件
#     df.to_csv(output_file, index=False, encoding='utf-8')
#     print(f"Processed CSV file has been saved to '{output_file}'.")
#
#
# if __name__ == "__main__":
#     # 输入和输出文件路径
#     input_csv_file = 'Distractor-Generation-RACE-master\data\distractor/race_test_updated.csv'  # 替换为你的输入CSV文件路径
#     output_csv_file = 'Distractor-Generation-RACE-master\data\distractor/race_test_updated_source.csv'  # 替换为你希望保存的输出CSV文件路径
#
#     # 处理CSV文件
#     process_csv(input_csv_file, output_csv_file)


# import pandas as pd
#
# # 读入CSV文件
# df = pd.read_csv('data/train49_data/race_train_updated_source_target_score.csv')
#
# # 对 target 列做 <sep> 分隔，将字符串转换为列表
# df['target'] = df['target'].str.split('<sep>')
#
# # 保留每个列表的前 10 个元素
# df['target'] = df['target'].apply(lambda x: x[:13] if len(x) > 13 else x)
#
# # 使用 <sep> 将列表中的元素重新拼接成字符串
# df['target'] = df['target'].apply(lambda x: '<sep>'.join(x))
#
# # 将修改后的 DataFrame 保存到新的 CSV 文件
# df.to_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors.csv', index=False)


import pandas as pd

# 读入CSV文件
df = pd.read_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors.csv')

# 对 target 列做 <sep> 分隔，将字符串转换为列表
df['score'] = df['score'].str.split(',')

# 保留每个列表的前 10 个元素
df['score'] = df['score'].apply(lambda x: x[:13] if len(x) > 13 else x)

df['score'] = df['score'].apply(lambda x: ','.join(x))

# 将修改后的 DataFrame 保存到新的 CSV 文件
df.to_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors.csv', index=False)


