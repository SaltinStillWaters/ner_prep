"""
THIS FILE SIMPLY BUILDS THE FINAL DatasetDict
"""

from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from src.misc.text_utils import *

__all__ = ['pre_process', 'save', 'load']

def pre_process(stg_2_data):
    train_data, temp_data = train_test_split(stg_2_data, test_size=.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=.5, random_state=42)

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    return dataset_dict

def save(out_file, stg_3_data):
    stg_3_data.save_to_disk(out_file)

def load(in_file):
    return load_from_disk(in_file)