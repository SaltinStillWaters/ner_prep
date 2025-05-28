import torch
from src.pre_process import stage_3
from src.misc.metrics import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

checkpoint_path = 'model_out/test_slimmed' #Change this to your desired model
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

dir_path = ''
models = [
    
]
def evaluate_model(model_path, tokenizer, test_dataset, data_collator, compute_metrics):
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Results for {model_path}:")
    print(results)
    
sentences = ['transpose everything to the left side', 'transpose everything to the left hand side', "so transpose all the terms to the left hand side", 'okay now we transpose all the terms to the right side']

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors = 'pt')

    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    pred = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [model.config.id2label[p.item()] for p in pred[0]]

    print('====================================================================================')
    for token, label in zip(tokens, labels):
        print(f'{token:15} -> {label}')