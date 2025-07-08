# DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation

This repository contains the official implementation of the paper *"DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation"*.

---

## 🔍 Script Descriptions

### 📁 `create_file.py`
Undertake the responsibility of creating folders. Used to build structured data files for specific experiments, create storage directories for prediction results, evaluation reports, and model parameters, and standardize the project file system.

### 📈 `evaluate.py`
Basic evaluation script. Basic evaluation script that calculates core metrics such as Precision (P), Recall (R), F1, MRR, MAP, and NDCG around the generated distractors to measure their quality and relevance.

### 🧹 `evaluate_remove_repeat.py`
Advanced evaluation script, introducing "de duplication" logic and removing answer operations. Remove duplicate interference items and answers before evaluation to avoid redundant content affecting the results and make the evaluation more accurate.

### 🔄 `evaluate_remove_repeat_all.py`
The enhanced version of the "de duplication" evaluation script can be applied in a wider range of scenarios (such as multiple experimental results) to facilitate the evaluation of experimental results.

### 🔄 `evaluate_remove_repeat2.py`
Another type of 'de duplication' evaluation variant. In the original method, the frequency of situations where the output is less than 10 distractors was counted, providing diverse options for evaluation and adapting to different needs.

### 🔢 `max_input_length.py`
Used to analyze the length distribution of the target column. Output the maximum number of tokens required to cover 95% of the data, as well as the maximum token length in the target column, to provide reference for the input length limit of the model. This helps set the maximum input length parameter of the model to balance data coverage and computational efficiency.

### 🚀 `predict.py`
Basic distractor generation prediction script. Generates candidate distractors for cloze test questions using a trained model and specific algorithm — serving as the primary entry point for distractor generation.

### 📦 `predict_all.py`
The prediction script for generating interference items can be applied to a wider range of scenarios (such as multiple experimental results) to generate interference items using cloze test questions at once, improving batch processing efficiency.



# DualReward：A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation

本仓库包含论文《DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation》的实现。

---

## 🔍 脚本说明

### 📁 `create_file.py`
承担文件夹创建职责。用于搭建特定实验的结构化的数据文件、创建预测结果、评估报告、模型参数的存储目录，规范项目文件体系。

### 📈 `evaluate.py`
基础评估脚本，围绕干扰项生成结果计算核心指标，如P，R，F1，MRR，MAP，NDCG等，衡量干扰项质量与匹配度。

### 🧹 `evaluate_remove_repeat.py`
进阶评估脚本，引入“去重复”逻辑和去除答案操作。剔除重复干扰项和答案后再评估，避免冗余内容影响结果，让评估更精准。

### 🔄 `evaluate_remove_repeat_all.py`
强化版“去重复”评估脚本，在更广泛场景（如多次实验结果）应用，方便评估实验结果。

### 🔄 `evaluate_remove_repeat2.py`
另一种“去重复”评估变体。在原来方式上统计输出不足10个元素的情况出现的次数，为评估提供多样化选择，适配不同需求。

### 🔢 `max_input_length.py`
用于分析（target列）的长度分布。输出覆盖 95% 数据所需的最长 token 数量，以及 target 列中的最大 token 长度以便为模型输入长度限制提供参考，这有助于设置模型的最大输入长度参数，以兼顾数据覆盖率与计算效率。

### 🚀 `predict.py`
基础干扰项生成预测脚本，依托训练好的模型和特定算法，为完形填空题目生成候选干扰项，是干扰项产出的基础入口。

### 📦 `predict_all.py`
面向干扰项生成的预测脚本，可一次性在更广泛场景（如多次实验结果）应用完形填空题目生成干扰项，提升批量处理效率。

