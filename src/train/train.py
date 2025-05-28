import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_ENABLE_AIOHTTP_SERVER"] = "0"

os.environ["RAY_LOG_TO_STDERR"] = "1"   
from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer
import ray
ray.init(include_dashboard=False)
ray.init(dashboard_host=None)
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import transformers
import torch

print('='*30)
print('DEVICE DATA:')
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print('bf16 is available:', torch.cuda.is_bf16_supported())
print(transformers.__version__)
print('='*30)

dataset = stage_3.load('processed_data/stg_3')
batch = data_collator([dataset['train'][x] for x in range(2)])
print(batch)

scheduler = ASHAScheduler(
    max_t=7,             # max epochs (should match your max num_train_epochs)
    grace_period=1,      # min epochs to run before considering stopping
    reduction_factor=2,  # halving factor to decide how many trials to stop
)

def hp_space(trial):
    if trial is None:
      # Return default hyperparameters when trial is None (used internally)
      return {
          "learning_rate": 2e-5,
          "weight_decay": 0.0,
          "per_device_train_batch_size": 16,
          "num_train_epochs": 3
      }
      
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", [3, 5, 7])
    }
    
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2tag,
        label2id=tag2index,
    )
    
args = TrainingArguments('finetuned-ner',
                          eval_strategy = 'epoch',
                          per_device_eval_batch_size=16,
                          
                          save_total_limit=2,
                          save_strategy = 'epoch',
                          
                          fp16 = not torch.cuda.is_bf16_supported(),
                          bf16 = torch.cuda.is_bf16_supported(),
        
                          load_best_model_at_end=True,
                          metric_for_best_model="f1",
                          greater_is_better=True,
                          
                          logging_steps=100,
                          logging_dir = "new_runs",
                          logging_strategy = "epoch",
                          report_to="tensorboard",
                          run_name='v1'
                        )

trainer = Trainer(model_init = model_init,
                  args = args,
                  train_dataset = dataset['train'],
                  eval_dataset = dataset['validation'],
                  data_collator = data_collator,
                  compute_metrics = compute_metrics,
                  tokenizer = tokenizer
                 )

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    direction="maximize",  # optimize for F1
    backend="ray",         # default if ray is installed
    n_trials=40,            # total combinations
    compute_objective=lambda metrics: metrics["eval_f1"],
    local_dir="runs/grid_search",
    scheduler=scheduler
)

best_args = TrainingArguments(
    output_dir='new_models/best_grid_search_model',
    learning_rate=best_run.config["learning_rate"],
    weight_decay=best_run.config["weight_decay"],
    per_device_train_batch_size=best_run.config["per_device_train_batch_size"],
    num_train_epochs=best_run.config["num_train_epochs"],
    
    save_total_limit=2,
    save_strategy = 'epoch',
    
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    logging_dir = "new_runs/grid_search",
    logging_strategy = "epoch",
    report_to="tensorboard",
    run_name='best_grid_search_model'
)

best_trainer = Trainer(
    model_init=model_init,
    args=best_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

best_trainer.train()
best_trainer.save_model("new_models/best_grid_search_model")