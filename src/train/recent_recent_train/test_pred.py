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
    f'model'
]

def get_token_mismatches(eval_preds, tokenizer, label_list, dataset):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)

    mismatches = []

    for i in range(len(predictions)):
        pred_seq = predictions[i]
        label_seq = labels[i]
        input_ids = dataset[i]['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        for idx, (pred, true) in enumerate(zip(pred_seq, label_seq)):
            if true == -100:
                continue
            true_label = label_list[true]
            pred_label = label_list[pred]
            if pred_label != true_label:
                mismatches.append({
                    "token": tokens[idx],
                    "true_label": true_label,
                    "predicted_label": pred_label
                })

    return mismatches


# for data in ['test', 'validation', 'train']:
for model_name in models:
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load and tokenize test set
    dataset = stage_3.load('old_data/combined_processed')
    test_dataset = dataset['validation']
    print(len(test_dataset))
    # Set up Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    eval_preds = trainer.predict(test_dataset)
    mismatches = get_token_mismatches(eval_preds, tokenizer, labels, test_dataset)
    
    df = pd.DataFrame(mismatches)
    df.to_csv(f"{model_name}_mismatches.csv", index=False)
    print(f"Saved mismatches to {model_name}_mismatches.csv")
    