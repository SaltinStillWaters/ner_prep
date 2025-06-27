import torch
import random
import numpy as np
import torchvision
import os
import shutil
import sys
import time
import traceback

from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from src.misc.globals import *
from src.pre_process import stage_3, pre_proc_func
from src.misc.metrics import *
from pathlib import Path

base_out_path = 'out/best_model_training/' # must be suffixed with '/'

def main(dataset_path, hparams, dataset_name):
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
    

    eval_dataset = stage_3.load(dataset_path)

    best_args = TrainingArguments(
        output_dir=f"{base_out_path}/{dataset_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{base_out_path}/logs/{dataset_name}",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        
        per_device_eval_batch_size=8,

        num_train_epochs=hparams[0],
        per_device_train_batch_size=hparams[1],
        learning_rate=hparams[2],
        weight_decay=hparams[3],
        warmup_ratio=hparams[4],
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
        train_dataset=eval_dataset['train'],
        eval_dataset=eval_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_span,
    )
    
    best_trainer.train()
    best_trainer.save_model(f'{base_out_path}/{dataset_name}/best/')

    compute_confusion_matrix(best_trainer, eval_dataset['validation'], labels, f'{base_out_path}/{dataset_name}/summary/matrix_eval.png')
    
    
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but avoids bugs on frozen apps or weird IDEs
    
    datasets = [
        'processed/orig',
        'processed/removed/3',
        'processed/base_dataset/3',
    ]
    
    hparams = [
        [10, 8, 1.1651559481113404e-05, 0.132954636707164, 0.0002026766944821],
        [10, 8, 1.3419955052352191e-05, 0.2797867317383964, 0.0386854827420473],
        [10, 8, 2.960741071471581e-05, 0.421287573508828, 0.1310828907212573],
    ]
    
    for x in range(len(datasets)):
        main(datasets[x], hparams[x], datasets[x].split('/')[1])