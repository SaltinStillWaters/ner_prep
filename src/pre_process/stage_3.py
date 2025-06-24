"""
THIS FILE SIMPLY BUILDS THE FINAL DatasetDict
"""

from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from src.misc.text_utils import *
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
np.random.seed(42)

__all__ = ['pre_process', 'save', 'load']

def extract_entity_types(sample, index2tag):
    tag_ids = [label for label in sample["labels"] if label >= 0]
    tag_names = [index2tag[i] for i in tag_ids if i in index2tag]
    entity_types = list(set(tag.split("-")[-1] for tag in tag_names if "-" in tag))
    return entity_types

def pre_process(stg_2_data, tag2index):
    # print(type(stg_2_data))           # Should be list
    # print(type(stg_2_data[0]))        # Should be dict
    # print(stg_2_data[0], '\n\n')   
    
    index2tag = {v: k for k, v in tag2index.items()}
    label_matrix = [extract_entity_types(sample, index2tag) for sample in stg_2_data]
    
    print(len(stg_2_data))       # Should match...
    print(len(label_matrix))
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_matrix)

    pure_dict_data = [dict(sample) for sample in stg_2_data]
    X = np.array(pure_dict_data, dtype=object).reshape(-1, 1) # shape (n_samples, 1)
    Y = np.array(Y)

    print(X.shape)  # Should be (n_samples, 1)
    print(Y.shape)

    X_train = X.ravel().tolist()

    dummy_example = {key: "" if isinstance(val, str) else [] for key, val in X_train[0].items()}
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(X_train),
        "validation": Dataset.from_list([dummy_example]),
        "test": Dataset.from_list([dummy_example])
    })
    
    return dataset_dict
    
    # train_data, temp_data = train_test_split(stg_2_data, test_size=.3, random_state=42)
    # val_data, test_data = train_test_split(temp_data, test_size=.5, random_state=42)


def save(out_file, stg_3_data):
    stg_3_data.save_to_disk(out_file)

def load(in_file):
    return load_from_disk(in_file)