
import pandas as pd
from transformers import T5Tokenizer

MODEL = '/model/t5-base'

tokenizer = T5Tokenizer.from_pretrained(MODEL)

df = pd.read_csv('data/train59_data/CLOTH-F-train(k=10)-count-bigger-0.csv')

df['token_length'] = df['target'].apply(lambda x: len(tokenizer(x, add_special_tokens=False)['input_ids']))

print(df['token_length'].describe())

coverage_percentile = 0.95
max_input_length = int(df['token_length'].quantile(coverage_percentile))
print(f"To cover {coverage_percentile*100}% of the data, max_input_length should be set to {max_input_length}")

print(f"The maximum token length in source column is {df['token_length'].max()}")