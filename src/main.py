
        
def prepare_dataset(jsonl: list[dict[str, any]]):
    for json in jsonl:
        json['text'] = json['text'].lower()
        
def align_labels(text, entities, tokenizer):
    encodings = tokenizer(text, return_offsets_mapping=True, truncation=True)
    labels = []

    entity_ranges = [
        (ent["start_offset"], ent["end_offset"], ent["label"])
        for ent in entities
    ]

    for idx, (start, end) in enumerate(encodings["offset_mapping"]):
        if start == end:
            labels.append("O")
            continue

        label = "O"
        for ent_start, ent_end, ent_label in entity_ranges:
            if start == ent_start:
                label = f"B-{ent_label}"
                break
            elif ent_start < start < ent_end:
                label = f"I-{ent_label}"
                break
        labels.append(label)
    
    return encodings.tokens(), labels

# data = read_jsonl('concat.jsonl')





# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# tokens, labels = align_labels(data[0]['text'], data[0]['entities'], tokenizer)

# for idx, token in enumerate(tokens):
#     print(f'{token}: {labels[idx]}')