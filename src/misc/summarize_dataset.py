import re

from text_utils import *

# # GET SENTENCE WITH COMMAND
# orig = read_jsonl('processed_data/stg_1.jsonl')

# arr = []
# arr_2 = []

# for line in orig:
#     for ent in line['entities']:
#         if ent['label'] == 'command':
#             arr.append(line)
#             break
#     else:
#         arr_2.append(line)
    
# save_jsonl('with_cmd.jsonl', arr)
# save_jsonl('without_cmd.jsonl', arr_2)


# transforms ent: [[start, end, label], [start, end, label]]
# to ent: [0, 0, 'ent']
def restructure_jsonl(in_file, out_file, in_ent_col='entities', out_ent_col='entities'):
    orig = read_jsonl(in_file)
    arr = []
    for line in orig:
        ents = []
        for ent in line[in_ent_col]:
            temp_ent = [
                ent['start_offset'],
                ent['end_offset'],
                ent['label'],
            ]
            ents.append(temp_ent)
            
        temp_dict = {
            'text': line['text'],
            out_ent_col: ents
        }
        arr.append(temp_dict)
    save_jsonl(out_file, arr)
    
def summarize_dataset(in_file, out_file):
    ctr = {}
    file = read_jsonl(in_file)
    for line in file:
        for ent in line['entities']:
            start = ent['start_offset']
            end = ent['end_offset']
            text = line['text']
            keyword = text[start : end]
            
            if not keyword in ctr:
                ctr[keyword] = 1
            else:
                ctr[keyword] += 1             
    
    ctr = dict(sorted(ctr.items(), key=lambda item: item[1], reverse=True))
    data = str(ctr)
    data = re.sub(r',|{|}', '\n', data)
    save_text(out_file, data)
    
def summarize_dataset_indepth(in_file, out_file):
    ctr = {}
    file = read_jsonl(in_file)
    for line in file:
        for ent in line['entities']:
            label = ent['label']
            start = ent['start_offset']
            end = ent['end_offset']
            text = line['text']
            keyword = text[start : end]
            
            if label not in ctr:
                ctr[label] = {}
                
            if not keyword in ctr[label]:
                ctr[label][keyword] = 1
            else:
                ctr[label][keyword] += 1             
    
    data = ''
    for category, val in ctr.items():
        val = dict(sorted(val.items(), key=lambda item: item[1], reverse=True))
        data += '=' * 20 + '\n'
        data += category + '\n'
        data += str(val)

    data = re.sub(r',|{|}', '\n', data)
    save_text(out_file, data)

summarize_dataset_indepth('processed_data/stg_1.jsonl', '23k_summary.txt')