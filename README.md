# DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation

This repository contains the official implementation of the paper *"DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation"*.

---

## 🔍 Script Descriptions

### 📁 `create_file.py`
Responsible for file creation tasks. It can be used to set up structured data files, create directories for storing predictions or evaluation reports, and initialize log files to standardize the project's file system.

### 📈 `evaluate.py`
Basic evaluation script. It calculates core metrics like accuracy, precision, recall, etc., to assess the quality and relevance of generated distractors.

### 🧹 `evaluate_remove_repeat.py`
Advanced evaluation script with a "de-duplication" mechanism. It filters out repeated distractors before evaluation to avoid redundancy and improve result accuracy.

### 🔄 `evaluate_remove_repeat_all.py`
Enhanced version of the de-duplication evaluation script. Applies deduplication rules across broader scenarios (e.g., full dataset or entire evaluation pipeline) to ensure consistency and enhance result reliability.

### 🔄 `evaluate_remove_repeat2.py`
Alternative variation of the de-duplication evaluation. May differ in deduplication algorithms or metric calculation methods, offering more flexible evaluation options for different needs.

### 🔢 `max_input_length.py`
Determines the maximum input length for models or data processing steps. Analyzes text paragraphs and questions to find the longest sequence length, providing reference for configuring models with input length constraints (e.g., certain neural networks).

### 🚀 `predict.py`
Basic distractor prediction script. Generates candidate distractors for cloze test questions using a trained model or specific algorithm — serving as the primary entry point for distractor generation.

### 📦 `predict_all.py`
Batch prediction script for large-scale datasets. Can generate distractors for all cloze test questions in a dataset (train, validation, test sets, etc.) at once, improving processing efficiency.



# DualReward：A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation

本仓库包含论文《DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation》的实现。

---

## 🔍 脚本说明

### 📁 `create_file.py`
承担文件创建职责。可用于搭建特定结构的数据文件、创建预测结果或评估报告的存储目录，也可初始化日志文件，规范项目文件体系。

### 📈 `evaluate.py`
基础评估脚本，围绕干扰项生成结果计算核心指标，如准确率、精确率、召回率等，衡量干扰项质量与匹配度。

### 🧹 `evaluate_remove_repeat.py`
进阶评估脚本，引入“去重复”逻辑。剔除重复干扰项后再评估，避免冗余内容影响结果，让评估更精准。

### 🔄 `evaluate_remove_repeat_all.py`
强化版“去重复”评估脚本，在更广泛场景（如全数据集、全评估流程）应用去重规则，保障评估一致性，提升结果可靠性。

### 🔄 `evaluate_remove_repeat2.py`
另一种“去重复”评估变体。可能在去重算法、指标计算方式上有差异，为评估提供多样化选择，适配不同需求。

### 🔢 `max_input_length.py`
用于测定模型或数据处理环节的最大输入长度。分析文本段落、题目等内容，确定最长序列长度，为有输入长度限制的模型（如部分神经网络）配置参数提供依据。

### 🚀 `predict.py`
基础干扰项预测脚本，依托训练好的模型或特定算法，为完形填空题目生成候选干扰项，是干扰项产出的基础入口。

### 📦 `predict_all.py`
面向大规模数据的预测脚本，可一次性对整个数据集（训练集、验证集、测试集等）的完形填空题目生成干扰项，提升批量处理效率。

