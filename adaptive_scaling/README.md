## Folder Descriptions

### adaptive_scaling

#### `data`
Including the specific data file of an experiment

#### `train`
This code fine-tunes a T5-base model for a distractors generation task with a custom reward-based loss. 
Specifically, it loads datasets, preprocesses text data via tokenization, uses a custom data collator to retain original targets, 
defines a training loop that adjusts loss with adaptive scaling rewards, and finally generates plots to visualize training progress.
