# DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation

This repository contains the official implementation of the paper *"DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation"*.

---

## Repository Descriptions

### adaptive_scaling

####  `create_file.py`
Undertake the responsibility of creating folders. Used to build structured data files for specific experiments, create storage directories for data, prediction results, evaluation, and model parameters.

####  `evaluate.py`
Basic evaluation script. Basic evaluation script that calculates core metrics such as Precision (P), Recall (R), F1, MRR, MAP, and NDCG around the generated distractors to measure their quality and relevance.

####  `evaluate_remove_repeat.py`
Advanced evaluation script. Introducing "de duplication" logic and removing answer operations. Remove duplicate distractors and answer before evaluation to avoid redundant content affecting the results and make the evaluation more accurate.

####  `candidate_set_generate_distractors_top.py`
Uses a pre-trained BERT model to generate 7 non-repetitive distractors. Updates the "target" column with the original top 3 targets plus the new distractors. Extract the first 3 original targets from the "target" column, use the model to predict up to 20 potential distractors from the "source" text, filter out distractors that overlap with original targets to get 7 valid ones, combine these distractors with the original targets to update the "target" column, generate 10 candidate distractors and record the distractors' scores.

####  `max_input_length.py`
Used to analyze the length distribution of the target column. Output the maximum number of tokens required to cover 95% of the data, as well as the maximum token length in the target column, to provide reference for the input length limit of the model. This helps set the maximum input length parameter of the model to balance data coverage and computational efficiency.

####  `predict.py`
Basic distractor generation prediction script. Generates candidate distractors for cloze test questions using a trained model — serving as the primary entry point for distractor generation.

### datasets

#### `CLOTH-F`
CLOTH-F (Filtered) is a refined subset of the CLOTH dataset. The original CLOTH dataset contains questions with numbered blanks (e.g., ”1”). To avoid training data inconsistency and create a more standardized evaluation setting, CLOTH-F removes the questions with numbered blanks, resulting in 5,041/720/739 instances for train/dev/test splits.

#### `DGen`
DGen (MCQ) is a cross-domain, sentence-level cloze dataset spanning science, vocabulary, common sense, and trivia domains. It contains 2,321/258 questions for train/test splits (with dev data created by a 9:1 split from training), where each instance has a sentence-level context with blanks marked as blank.
