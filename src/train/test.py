import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

checkpoint_path = 'model_out/test_slimmed' #Change this to your desired model
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

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