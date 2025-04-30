from src.misc.globals import *
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import TrainingArguments, AutoModelForTokenClassification, Trainer

import torch
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# dataset = stage_3.load('stg_3')
# batch = data_collator([dataset['train'][x] for x in range(2)])
# print(batch)

# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=index2tag, label2id=tag2index)

# args = TrainingArguments('distilbert-finetuned-ner',
#                           evaluation_strategy = 'epoch',
#                           save_strategy = 'epoch',
#                           learning_rate = 2e-5,
#                           num_train_epochs = 3,
#                           weight_decay = 0.01
#                         )

# trainer = Trainer(model = model,
#                   args = args,
#                   train_dataset = dataset['train'],
#                   eval_dataset = dataset['validation'],
#                   data_collator = data_collator,
#                   compute_metrics = compute_metrics,
#                   tokenizer = tokenizer
#                  )

