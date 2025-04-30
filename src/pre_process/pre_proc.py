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

raw_jsonl = read_jsonl('data/combined_raw.jsonl')

stg_1 = stage_1.pre_process(raw_jsonl)
stage_1.save('stg_1.jsonl', stg_1)

stg_2 = stage_2.pre_process(stg_1, tokenizer, tag2index)
stage_2.save('stg_2.pkl', stg_2)
stage_2.print_token_alignment(stg_2[1], tokenizer)

stg_3 = stage_3.pre_process(stg_2)
stage_3.save('stg_3', stg_3)