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

dataset = stage_3.load('temp/3')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)


search_space = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "weight_decay": [0.0, 0.01],
    "per_device_train_batch_size": [8, 16],
    "num_train_epochs": [3, 5, 7]
}

sampler = GridSampler(search_space)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 5, 7])
    }
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2tag,
        label2id=tag2index,
    )
    
args = TrainingArguments(
    'finetuned-ner',
    evaluation_strategy='epoch',
    per_device_eval_batch_size=16,
    save_total_limit=2,
    save_strategy='epoch',
    warmup_ratio=0.1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    logging_dir="new_runs",
    logging_strategy="steps",
    report_to="tensorboard",
)

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    study_name="ner-grid-search"
)

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",  # use optuna
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["eval_f1"],
    n_trials=48  # 4 x 2 x 2 x 3 = 48 combinations
)

# Use the best hyperparameters
best_args = TrainingArguments(
    output_dir='new_models/best_grid_search_model',
    learning_rate=best_run.config["learning_rate"],
    weight_decay=best_run.config["weight_decay"],
    per_device_train_batch_size=best_run.config["per_device_train_batch_size"],
    num_train_epochs=best_run.config["num_train_epochs"],
    
    save_total_limit=2,
    save_strategy='epoch',
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="new_runs/grid_search",
    logging_strategy="epoch",
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
best_trainer.save_model("new_models/best_grid_search_model")