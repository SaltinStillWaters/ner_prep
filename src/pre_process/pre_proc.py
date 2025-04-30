"""
THIS IS THE DRIVER CODE.
YOU ONLY NEED TO RUN THE FUNCTION HERE
TO START THE PRE-PROCESSING
"""

import stage_1
import stage_2
import stage_3

from text_utils import *
from transformers import AutoTokenizer

## Global variables

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

labels = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']

index2tag = {x:ent for x, ent in enumerate(labels)}
index2tag[-100] = 'IGN'
tag2index = {ent:x for x, ent in enumerate(labels)}

raw_jsonl = read_jsonl('data/combined_raw.jsonl')


## Pipeline

stg_1 = stage_1.pre_process(raw_jsonl)
stage_1.save('stg_1.jsonl', stg_1)

stg_2 = stage_2.pre_process(stg_1, tokenizer, tag2index)
stage_2.save('stg_2.pkl', stg_2)
stage_2.print_token_alignment(stg_2[1], tokenizer)

stg_3 = stage_3.pre_process(stg_2)
stage_3.save('stg_3', stg_3)