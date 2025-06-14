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
            
            
file = read_jsonl('reannotated.jsonl')

extracted = []
base = []
to_extract = ['illustrate', 'graph', 'plot']

for line in file:
    ents = []
    for ent in line['entities']:
        word = line['text'][ent['start_offset']:ent['end_offset']]
        if word in to_extract:
            extracted.append(line)
            break
    else:
        base.append(line)
    
save_jsonl('extracted.jsonl', extracted)
save_jsonl('base.jsonl', base)