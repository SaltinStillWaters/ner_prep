import stage_3
import seqeval
import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer

stg_3 = stage_3.load_from_disk('stg_3')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([stg_3['train'][x] for x in range(2)])
metric = evaluate.load('seqeval')

print(stg_3['train'][0]['labels'])

ents = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    predictions = np.argmax(logits, axis=1)
    true_labels = [[[ents[l]] for l in label if l != 100] for label in labels]
    
    true_predictions = [[ents[l] for p, l in zip(prediction, label) if l != -100] 
                        for prediction, label in zip(predictions, labels)]
    
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    
    return all_metrics

