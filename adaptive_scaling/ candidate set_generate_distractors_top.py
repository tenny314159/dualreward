
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from tqdm import tqdm


model_file = '/models/CSG/cdgp-csg-bert-cloth-download'

tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForMaskedLM.from_pretrained(model_file)

translator = pipeline('fill-mask', tokenizer=tokenizer, model=model, top_k=20)

df = pd.read_csv('/data/sciq-train_source_target.csv')

predict_list = []
score_list = []


for id, row in tqdm(df.iterrows(), total=len(df)):
    
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

    
    distractors_str = " ".join(new_distractors)

    if original_targets:
        updated_target = " ".join(original_targets) + " " + distractors_str
    else:
        updated_target = distractors_str

    predict_list.append(updated_target)
    score_list.append(" ".join(map(str, new_scores)))  


df['target'] = predict_list
df['score'] = score_list


output_path = '/data/sciq-train_source_target(k=7)_without_repeat.csv'
df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")