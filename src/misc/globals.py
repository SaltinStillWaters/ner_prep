from transformers import AutoTokenizer, DataCollatorForTokenClassification
from src.pre_process.stage_3 import load

model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

labels = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']

index2tag = {x:ent for x, ent in enumerate(labels)}
tag2index = {ent:x for x, ent in enumerate(labels)}