# import Levenshtein as lev
#
# str1 = "kitten"
# str2 = "sitting"
#
# distance = lev.distance(str1, str2)
# print(f"Levenshtein distance between '{str1}' and '{str2}': {distance}")


import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/train49_data/race_train_updated_source_target_score_ten_distractors.csv')

# 方法1：统计 target 列中 '<sep>' 的数量 + 1（因为 n 个分隔符对应 n+1 个元素）
df['target_count'] = df['target'].str.count('<sep>') + 1

# 方法2：统计 score 列中 ',' 的数量 + 1（因为 n 个分隔符对应 n+1 个元素）
df['score_count'] = df['score'].str.count(',') + 1

# 打印每个样本 score 和 target 的元素数量
print(df[['score', 'target', 'score_count', 'target_count']])

# 如果你还想查看该列的基本描述性统计信息
score_description = df['score_count'].describe()
target_description = df['target_count'].describe()

print(f"Description of elements in column 'score':\n{score_description}")
print(f"Description of elements in column 'target':\n{target_description}")

# 将修改后的 DataFrame 保存到新的 CSV 文件
#df.to_csv('RACE/all/race-train-ten-distractors-with-counts.csv', index=False)