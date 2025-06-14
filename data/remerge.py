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
            
            
base = read_jsonl('reannotated.jsonl')
to_merge = read_jsonl('extracted.jsonl')
merged = []

for line in base:
    for line_to_merge in to_merge:
        if line['id'] == line_to_merge['id']:
            merged.append(line_to_merge)
            break
    else:
        merged.append(line)
    
save_jsonl('merged.jsonl', merged)