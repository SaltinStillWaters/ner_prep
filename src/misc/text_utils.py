import json
import pickle
from pathlib import Path

# Reads a jsonl file and returns an array of dicts
def read_jsonl(in_file: str):
    data = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Saves an array of dicts as a jsonl file
def save_jsonl(out_file: str, data: list[dict]):
    with open(out_file, 'w', encoding='utf-8') as f:
        for x in data:
            f.write(f'{json.dumps(x)}\n')

def read_pkl(in_file: str):
    with open(in_file, 'rb') as f:
        return pickle.load(f)
        
def save_pkl(out_file: str, data):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)        

# Concats all jsonl files in `in_root_dir` and merges them into one
def concat_jsonl_data(in_root_dir: str, out_file: str):
    dir = Path(in_root_dir)
    
    result = []
    for file_path in dir.rglob('*.jsonl'):
        print(len(read_jsonl(file_path)))
        result += read_jsonl(file_path)
    
    save_jsonl(out_file, result)
    
def convert_jsonl_structure(in_dir):
    dir_path = Path(in_dir)
    print('started')
    for jsonl_file in dir.rglob('*.jsonl'):
        orig = read_jsonl(jsonl_file)
        result = []
        for line in orig:
            ents = []
            for (start, end, label) in line['label']:
                temp = {
                    'label': label,
                    'start_offset': start,
                    'end_offset': end
                }
                ents.append(temp)
            
            new_line = {
                'id': line['id'],
                'text': line['text'],
                'entities': ents
            }
            result.append(new_line)
            print('appended', result)
        out_path = jsonl_file.with_stem(jsonl_file.stem + '-new')
        save_jsonl(out_path, result)

convert_jsonl_structure('data/raw_jsonl/rb/')