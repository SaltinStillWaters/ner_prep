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



scheduler = ASHAScheduler(
    max_t=7,             # max epochs (should match your max num_train_epochs)
    grace_period=1,      # min epochs to run before considering stopping
    reduction_factor=2,  # halving factor to decide how many trials to stop
)

args = TrainingArguments(
    output_dir='distilbert-ner-high-acc',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='runs',
    logging_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=2,
    weight_decay=0.01,
    warmup_steps=500,
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