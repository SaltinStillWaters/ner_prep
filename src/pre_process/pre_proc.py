"""
THIS IS THE DRIVER CODE.
YOU ONLY NEED TO RUN THE FUNCTION HERE
TO START THE PRE-PROCESSING
"""

from src.pre_process import stage_1
from src.pre_process import stage_2
from src.pre_process import stage_3

from src.misc.text_utils import *
from src.misc.globals import *

## Pipeline
print('Merging raw data...')
concat_jsonl_data('lime_only', 'data/lime_only_combined.jsonl')
raw_jsonl = read_jsonl('data/lime_only_combined.jsonl')
print('Finished merging\n')

print('Started pre-processing stage 1/3...')
stg_1 = stage_1.pre_process(raw_jsonl)
stage_1.save('processed_data/lime_stg_1.jsonl', stg_1)
print('Finished pre-processing stage 1/3\n')

print('Started pre-processing stage 2/3...')
stg_2 = stage_2.pre_process(stg_1, tokenizer, tag2index)
stage_2.save('processed_data/lime_stg_2.pkl', stg_2)
stage_2.print_token_alignment(stg_2[1], tokenizer)
print('Finished pre-processing stage 2/3\n')

print('Started pre-processing stage 3/3...')
stg_3 = stage_3.pre_process(stg_2)
stage_3.save('processed_data/lime_stg_3', stg_3)
print('Finished pre-processing stage 3/3\n')
print('Done Pre-processing!!!')