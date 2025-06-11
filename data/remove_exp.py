import json

def read_jsonl(in_file: str):
    data = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(out_file: str, data: list[dict]):
    with open(out_file, 'w', encoding='utf-8') as f:
        for x in data:
            f.write(f'{json.dumps(x)}\n')
            
file = read_jsonl('combined_raw.jsonl')

new = []
for line in file:
    ents = []
    for ent in line['entities']:
        if ent['label'] in ['expression', 'equation', 'term']:
            continue
        ents.append(ent)
    temp = {
        'text': line['text'],
        'entities': ents
    }
    new.append(temp)
    
save_jsonl('removed.jsonl', new)