from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer
import transformers

import torch, torchvision
import optuna 
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
print(torch.version.cuda)
print('---->Torch version:', torch.__version__)
print('->>>>', 'torch and torch vision must be the same')
print('---->Torch Vision version', torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(transformers.__version__)

dataset = stage_3.load('processed_data/stg_3')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)

# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=index2tag, label2id=tag2index)

training_args = TrainingArguments(
    output_dir="ner-optuna",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="runs",
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
)


def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2tag,
        label2id=tag2index
    )

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }
    
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    model_init=model_init,  # important for tuning
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    hp_space=hp_space,
    backend="optuna",
    n_trials=20  # Adjust as needed
)

best_args = TrainingArguments(
    output_dir="ner-best-optuna",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="runs",
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
    learning_rate=best_trial.hyperparameters["learning_rate"],
    num_train_epochs=best_trial.hyperparameters["num_train_epochs"],
    per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
    weight_decay=best_trial.hyperparameters["weight_decay"],
)

model = model_init()

trainer = Trainer(
    args=best_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("ner-optuna-saved")
