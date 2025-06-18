from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer

import transformers
import torch

from transformers import set_seed
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

dataset = stage_3.load('old_data/combined_processed')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2tag,
        label2id=tag2index,
    )
    
# Use the best hyperparameters
args = TrainingArguments(
    output_dir="new_models/2/new_bayesian_run",       # 👈 new save location
    logging_dir="new_logs/2",                         # 👈 optional: new logs location
    run_name="new_run",                             # 👈 optional: new run name
    evaluation_strategy="epoch",                       # based on your previous config
    save_strategy="epoch",
    logging_strategy="steps",
    save_steps=500,
    logging_steps=500,
    num_train_epochs=5.0,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.04,
    logging_nan_inf_filter=True,
    seed=42,
    report_to=["tensorboard"],
    fp16=False,
    bf16=True,
    disable_tqdm=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    optim="adamw_torch",
)

best_trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

best_trainer.train()
best_trainer.save_model("new_models/2/new_bayesian")