import torch
import logging
import pandas as pd
from datetime import datetime
from datasets import Dataset
from transformers import TrainerCallback, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import gc
import json
from pathlib import Path
import matplotlib.pyplot as plt


class EvaluationLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        eval_results = kwargs['metrics']
        logging.info(f"Evaluation results at step {state.global_step}: {eval_results}")


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_input_length,
        truncation=True,
        padding=True,
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_target_length,
        truncation=True,
        padding=True,
    )
    model_inputs["labels"] = list(labels["input_ids"])

    if "target" in examples:
        model_inputs["original_target"] = examples["target"]
    else:
        raise ValueError("Expected 'target' field not found in examples.")
    return model_inputs


class RewardDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        original_targets = [feature["original_target"] if "original_target" in feature else "" for feature in features]
        batch["original_target"] = original_targets
        return batch


MODEL = '/model/t5-base'
train_num = 'train59'

max_input_length = 128
max_target_length = 20
batch_size = 8
epochs = 3

tokenizer = T5Tokenizer.from_pretrained(MODEL, model_max_length=max_input_length)
model = T5ForConditionalGeneration.from_pretrained(MODEL)

logging.basicConfig(filename='../log.txt',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a+'
                    )

df_train = pd.read_csv('/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-train(k=7)-without-repeat.csv' % train_num)
df_val = pd.read_csv('/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-valid.csv' % train_num)[:1000]

if "target" not in df_train.columns or "target" not in df_val.columns:
    raise ValueError("The dataset does not contain the 'target' column.")

train_ds = Dataset.from_pandas(df_train, split="train")
val_ds = Dataset.from_pandas(df_val, split="val")

tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_val = val_ds.map(preprocess_function, batched=True)

print("Tokenized train dataset columns:", tokenized_train.column_names)
print("Sample original_target values:", tokenized_train['original_target'][:5])

GLOBAL_TOKENIZED_TRAIN = {
    'input_ids': tokenized_train["input_ids"],
    'score': tokenized_train["score"]
}

input_ids_to_index = {tuple(input_ids): index for index, input_ids in enumerate(GLOBAL_TOKENIZED_TRAIN["input_ids"])}

import pickle

pickle.dump(tokenized_train, open(
    '/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-train(k=7)-without-repeat.pkl' % train_num,
    'wb'))
pickle.dump(tokenized_val, open('/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-valid.pkl' % train_num, 'wb'))

ds_train = pickle.load(
    open(
        '/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-train(k=7)-without-repeat.pkl' % train_num,
        'rb'))
ds_valid = pickle.load(open('/media/disk3/CXL/DG/wang-2023/data/%s_data/CLOTH-F-valid.pkl' % train_num, 'rb'))

print("Loaded train dataset columns:", ds_train.column_names)
print("Sample original_target values in loaded train dataset:", ds_train['original_target'][:5])

if "original_target" not in ds_train.column_names or "original_target" not in ds_valid.column_names:
    raise ValueError("The loaded dataset does not contain the 'original_target' column.")

data_collator = RewardDataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir="../model_output/%s_model" % train_num,
    evaluation_strategy="steps",
    eval_steps=100000000,
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=100,
    num_train_epochs=epochs,
    save_steps=12376,
)


def sigmoid(x):
    return torch.sigmoid(x)


match_count = 0
no_match_count = 0
reward_scales = []
global_step_list = []
avg_loss_list = []


def custom_train_step(model, inputs):
    global match_count, no_match_count
    labels = inputs.get("labels")
    original_targets = []
    input_ids = inputs.get("input_ids")
    inputs.pop("original_target")

    for label in labels:
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        original_targets.append(label_text)

    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(labels.size())

    avg_loss = loss.mean()
    avg_loss_list.append(avg_loss.item())

    base_scale = 0.1
    max_scale = 0.2
    alpha = 5
    threshold = 1.0

    reward_scale = base_scale + (max_scale - base_scale) * sigmoid(alpha * (avg_loss - threshold))
    reward_scales.append(reward_scale.item())
    global_step = trainer.state.global_step
    global_step_list.append(global_step)

    rewards = []
    for i in range(len(original_targets)):
        current_input_ids = tuple(input_ids[i].tolist())
        match_index = input_ids_to_index.get(current_input_ids)
        if match_index is None:
            no_match_count += 1
            reward = [0] * max_target_length
        else:
            match_count += 1
            score_str = GLOBAL_TOKENIZED_TRAIN["score"][match_index]
            scores = [float(s) for s in score_str.split()]
            target = original_targets[i].split()
            target_first_three = target[:3]
            target_last = target[3:]

            output_ids = torch.argmax(logits[i], dim=-1)
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).split()

            reward = []
            for word in output_text:
                if word in target_first_three:
                    reward.append(1 * reward_scale)
                elif word in target_last:
                    index = target_last.index(word)
                    reward.append(0.9 * reward_scale * scores[index])
                else:
                    reward.append(0)

            reward = reward + [0] * (max_target_length - len(reward))

        rewards.append(reward)

    rewards = torch.tensor(rewards, dtype=torch.float32, device=loss.device)
    adjusted_loss = loss - rewards
    adjusted_loss = torch.clamp(adjusted_loss, min=0)

    del outputs, logits, rewards
    torch.cuda.empty_cache()
    gc.collect()

    return adjusted_loss.mean()


class RewardSeq2SeqTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        print("First batch in train dataloader:", next(iter(dataloader)))
        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = custom_train_step(model, inputs)
        if return_outputs:
            return loss, {}
        return loss


trainer = RewardSeq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EvaluationLogger],
)

logging.info('logging --------------------------------------- started')

try:
    trainer.train()
finally:
    output_dir = Path("../training_metrics")
    output_dir.mkdir(exist_ok=True)

    metrics = {
        "global_step": global_step_list,
        "reward_scale": reward_scales,
        "avg_loss": avg_loss_list,
        "match_count": match_count,
        "no_match_count": no_match_count
    }

    with open(output_dir / f"training_metrics_{train_num}.json", "w") as f:
        json.dump(metrics, f)

    pd.DataFrame({
        "global_step": global_step_list,
        "reward_scale": reward_scales,
        "avg_loss": avg_loss_list
    }).to_csv(output_dir / f"training_metrics_{train_num}.csv", index=False)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(global_step_list, reward_scales)
    plt.xlabel('Global Step')
    plt.ylabel('Reward Scale')
    plt.title('Reward Scale during Training')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(global_step_list, avg_loss_list)
    plt.xlabel('Global Step')
    plt.ylabel('Average Loss')
    plt.title('Loss during Training')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_plots_{train_num}.png", dpi=300)
    plt.show()