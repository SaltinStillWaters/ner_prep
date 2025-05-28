import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from src.pre_process import stage_3
from src.misc.metrics import *
from src.pre_process.stage_2 import align_labels

# Define label mappings
labels = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 
          'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']
index2tag = {x: ent for x, ent in enumerate(labels)}
tag2index = {ent: x for x, ent in enumerate(labels)}


dir_path = 'model_out'
# Model paths
models = [
    # f'{dir_path}/7k_1/checkpoint-606',
    # f'{dir_path}/23k_best',
    # f'{dir_path}/distilbert-ner-best-quality/checkpoint-4072',
    # f'{dir_path}/distilbert-ner-high-acc/checkpoint-5090',
    f'{dir_path}/optuna/ner-optuna-saved',
    f'{dir_path}/optuna/ner-optuna-lime-saved',
    # 'distilbert-base-uncased',
    
    # f'{dir_path}/23k_best',
    # f'{dir_path}/distilbert-finetuned-ner/checkpoint-3054',
    # f'{dir_path}/distilbert-finetuned-ner/checkpoint-5090',
    # f'{dir_path}/distilbert-ner-best-quality/checkpoint-2500',
    # f'{dir_path}/distilbert-ner-best-quality/checkpoint-4072',
    # f'{dir_path}/distilbert-ner-high-acc/checkpoint-3054',
    # f'{dir_path}/distilbert-ner-high-acc/checkpoint-5090',
    # f'{dir_path}/optuna/ner-optuna-saved',
    # f'{dir_path}/optuna/ner-optuna-lime-saved',
]

# for data in ['test', 'validation', 'train']:
for model_name in models:
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load and tokenize test set
    dataset = stage_3.load('processed_data/stg_3')
    test_dataset = dataset['test']
    print(len(test_dataset))
    # Set up Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    results = trainer.evaluate(eval_dataset=test_dataset)
    print('======================================================================')
    print("Evaluation Results on", model_name)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
