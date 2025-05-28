from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer
import transformers

import torch
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(transformers.__version__)

dataset = stage_3.load('processed_data/stg_3')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=index2tag, label2id=tag2index)

args = TrainingArguments(
    output_dir='distilbert-ner-best-quality',
    evaluation_strategy='steps',
    eval_steps=250,  
    save_strategy='steps',
    save_steps=250,
    logging_dir='runs_quality',
    logging_steps=100,
    learning_rate=1e-5, 
    num_train_epochs=8, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2, 
    weight_decay=0.01,
    save_total_limit=2,
    warmup_ratio=0.1, 
    lr_scheduler_type='cosine', 
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard"
)


trainer = Trainer(model = model,
                  args = args,
                  train_dataset = dataset['train'],
                  eval_dataset = dataset['validation'],
                  data_collator = data_collator,
                  compute_metrics = compute_metrics,
                  tokenizer = tokenizer
                 )

trainer.train()