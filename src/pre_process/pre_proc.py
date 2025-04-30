"""
THIS IS THE DRIVER CODE.
YOU ONLY NEED TO RUN THE FUNCTION HERE
TO START THE PRE-PROCESSING
"""

import stage_1
import stage_2
import stage_3

from text_utils import *

raw_jsonl = read_jsonl('data/combined_raw.jsonl')

stg_1 = stage_1.pre_process(raw_jsonl)
stage_1.save('stg_1.jsonl', stg_1)

stg_2 = stage_2.pre_process(stg_1)
stage_2.save('stg_2.pkl', stg_2)
stage_2.print_token_alignment(stg_2, tok)

stg_3 = stage_3.pre_process(stg_2)
stage_3.save('stg_3', stg_3)