import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from src.pre_process import stage_3
from src.misc.metrics import *
from src.pre_process.stage_2 import align_labels
import json

# Define label mappings
labels = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 
          'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']
index2tag = {x: ent for x, ent in enumerate(labels)}
tag2index = {ent: x for x, ent in enumerate(labels)}


dir_path = 'model_out'

models = [
    f'model_out/23k_best'
]

# for data in ['test', 'validation', 'train']:
for model_name in models:
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load and tokenize test set
    dataset = stage_3.load('old_data/combined_processed')
    test_dataset = dataset['test']
    print(len(test_dataset))
    # Set up Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    results = trainer.evaluate(eval_dataset=test_dataset)
    print('======================================================================')
    print("Evaluation Results on", model_name)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

# Unique entity classes you care about
# entity_classes = ['command', 'command_attribute',