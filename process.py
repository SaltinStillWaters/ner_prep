import json
import re

def extract_words_from_jsonl(input_file, output_file):
    unique_words = set()
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            text = data.get("text", "")
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words.update(words)
    sorted_words = sorted(unique_words)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        formatted = '["' + '", "'.join(sorted_words) + '"]'
        outfile.write(formatted)
extract_words_from_jsonl(r'./processed_data/stg_1.jsonl', 'grammar1.txt')
