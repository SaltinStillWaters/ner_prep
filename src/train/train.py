import torch
import random
import numpy as np
import torchvision
import os
import shutil
import sys
import time

from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from src.misc.globals import *
from src.pre_process import stage_3, pre_proc_func
from src.misc.metrics import *
from pathlib import Path

base_out_path = 'out/super_out_7_aug' # must be suffixed with '/'
eval_dataset_path = 'processed/base_dataset/3'
base_model_path = 'best_saved/' # change to path to baseline model

def main(aug_dataset, dataset_name, eval_to_use):
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
    

    eval_dataset = stage_3.load(eval_dataset_path)

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
        
        per_device_eval_batch_size=eval_to_use,
        
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        weight_decay=0.421288,
        warmup_ratio=0.131083,
        fp16=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
            base_model_path,
            id2label=index2tag,
            label2id=tag2index
        )
    
    best_trainer = Trainer(
        args=best_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=aug_dataset['train'],
        eval_dataset=eval_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_span,
    )

    a = time.time()
    best_trainer.train()
    b = time.time()
    
    try:
        compute_confusion_matrix(best_trainer, eval_dataset['validation'], labels, f'{base_out_path}/{dataset_name}/summary/matrix_eval.png')
    except:
        pass
    
    return b - a

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but avoids bugs on frozen apps or weird IDEs
    
    datasets_dir = 'data_aug/'
    processed_aug_path = 'processed_aug/'
    
    eval_sizes_to_test = [8, 16, 32, 64]
    ctr = 0
    best_eval = eval_sizes_to_test[0]
    best_time = sys.float_info.max
    
    for filename in os.listdir(datasets_dir):
        dataset_name = Path(filename).stem
        file_path = os.path.join(datasets_dir, filename)

        print(f'===STARTING TRAINING OF {dataset_name}')
        print(file_path)
        try:
            pre_proc_func.run(file_path, processed_aug_path)
        except FileExistsError:
            # pass here because no need to pre-process
            pass
        
        aug_dataset = stage_3.load(f'{processed_aug_path}{dataset_name}/3')
        
        if ctr < len(eval_sizes_to_test):
            eval_to_use = eval_sizes_to_test[ctr]
            ctr += 1
        else:
            eval_to_use = best_eval
            
        train_time = main(aug_dataset, dataset_name, eval_to_use)
        
        if ctr <= len(eval_sizes_to_test):
            with open(f'{base_out_path}/train_time.txt', 'a', encoding='utf-8') as f:
                f.write(f'{eval_to_use}:\t{train_time}\n')
            
        if train_time < best_time:
            best_eval = eval_to_use
            best_time = train_time
        
        checkpoint_dir = Path(f"{base_out_path}/{dataset_name}")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)