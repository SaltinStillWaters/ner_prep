import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

checkpoint_path = 'distilbert-finetuned-ner/checkpoint-1818' #Change this to your desired model
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

sentence = "graph the function of <equation>"
inputs = tokenizer(sentence, return_tensors = 'pt')

with torch.no_grad():
    output = model(**inputs)
logits = output.logits
pred = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = [model.config.id2label[p.item()] for p in pred[0]]

for token, label in zip(tokens, labels):
    print(f'{token:15} -> {label}')