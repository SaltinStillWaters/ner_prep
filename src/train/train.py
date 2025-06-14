# train.py
import torch
import random
import numpy as np
import torchvision

from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(torch.version.cuda)
    print('Torch version:', torch.__version__)
    print('TorchVision version:', torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    

    dataset = stage_3.load('reannotated_processed/3')
    
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=index2tag,
            label2id=tag2index
        )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 10e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.5),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        }
    
    training_args = TrainingArguments(
        num_train_epochs = 10,
        output_dir="super_out_3/trials/",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="super_out_3/trials/logs",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        fp16=True,
    )

    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        model_init=model_init,  # important for tuning
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        backend="optuna",
        n_trials=200  # Adjust as needed
    )

    best_args = TrainingArguments(
        output_dir="super_out_3/best/",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="super_out_3/best/logs",
        logging_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        learning_rate=best_trial.hyperparameters["learning_rate"],
        num_train_epochs=10,
        per_device_train_batch_size=best_trial.hyperparameters["per_device_train_batch_size"],
        weight_decay=best_trial.hyperparameters["weight_decay"],
        warmup_ratio=best_trial.hyperparameters["warmup_ratio"],
        fp16=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=index2tag,
            label2id=tag2index
        )
    
    best_trainer = Trainer(
        args=best_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        import json
        with open('super_out_3/results.txt', 'w', encoding='utf-8') as f:
            f.write("best args:\n")
            f.write(json.dumps(best_args.to_dict(), indent=4))
        print(f"best args:\n{best_args}")
    except:
        pass
    
    best_trainer.train()
    best_trainer.save_model("super_out_3/best_saved/")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but avoids bugs on frozen apps or weird IDEs
    main()
