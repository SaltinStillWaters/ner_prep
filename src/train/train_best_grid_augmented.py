from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer

import transformers
import torch

from transformers import set_seed

import optuna
from optuna.samplers import GridSampler
import os
set_seed(42)

print('='*30)
print('DEVICE DATA:')
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print('bf16 is available:', torch.cuda.is_bf16_supported())
print(transformers.__version__)
print('='*30)

dataset = stage_3.load('combined_processed')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2tag,
        label2id=tag2index,
    )
    
# Use the best hyperparameters
best_args = TrainingArguments(
    output_dir='post_grid_models/best_grid_augmented',
    learning_rate=1.608840634195285e-05,
    weight_decay=0.024306886701424227,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    save_total_limit=2,
    eval_strategy='epoch',
    save_strategy='epoch',
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="post_grid_runs/best_grid_augmented",
    logging_steps=500,
    logging_strategy="steps",
    report_to="tensorboard",
)

best_trainer = Trainer(
    model_init=model_init,
    args=best_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

best_trainer.train()
best_trainer.save_model("final_models/best_grid_augmented")