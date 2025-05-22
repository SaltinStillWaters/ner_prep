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


# Model paths
models = [
    "model_out/distilbert-finetuned-ner/checkpoint-3054",
    "model_out/distilbert-finetuned-ner/checkpoint-5090",
    "model_out/distilbert-ner-best-quality/checkpoint-2500",
    "model_out/distilbert-ner-best-quality/checkpoint-4072",
    "model_out/distilbert-ner-high-acc/checkpoint-3054",
    "model_out/distilbert-ner-high-acc/checkpoint-5090",
]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load and tokenize test set
    dataset = stage_3.load('processed_data/stg_3')
    test_dataset = dataset["test"]
    tokenized_test = test_dataset

    # Set up Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    results = trainer.evaluate(eval_dataset=tokenized_test)
    print('======================================================================')
    print("Evaluation Results on", model_name)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
