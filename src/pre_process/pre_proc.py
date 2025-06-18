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

## You only need to change tha variables here:
data_to_merge_path = '' # leave blank if none
data_path = '' #leave blank if $data_to_merge_path is filled

## Don't edit unless you want custom paths
base_out = 'processed/'
merged_data_path = 'data/combined/'


## Pipeline
if data_to_merge_path:
    print('Merging raw data...')
    out_file_name = data_to_merge_path.split('/')[-1]
    merged_path = f'{merged_data_path}{out_file_name}.jsonl'
    concat_jsonl_data(data_to_merge_path, merged_data_path)

data_path = data_path if data_path else merged_path
out_path = f'{base_out}{data_path.split('/')[-1].split('.')[-2]}'
raw_jsonl = read_jsonl(data_path)
print('Finished merging\n')

print('Started pre-processing stage 1/3...')
stg_1 = stage_1.pre_process(raw_jsonl)
stage_1.save(f'{out_path}1.jsonl', stg_1)
print('Finished pre-processing stage 1/3\n')

print('Started pre-processing stage 2/3...')
stg_2 = stage_2.pre_process(stg_1, tokenizer, tag2index)
stage_2.save(f'{out_path}2.pkl', stg_2)
stage_2.print_token_alignment(stg_2[1], tokenizer)
print('Finished pre-processing stage 2/3\n')

print('Started pre-processing stage 3/3...')
stg_3 = stage_3.pre_process(stg_2, tag2index)
stage_3.save(f'{out_path}3', stg_3)
print('Finished pre-processing stage 3/3\n')
print('Done Pre-processing!!!')