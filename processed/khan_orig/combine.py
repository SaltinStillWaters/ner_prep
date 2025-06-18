from datasets import load_dataset, DatasetDict, concatenate_datasets
from src.pre_process import stage_3

# Load two datasetdicts
dataset1 = DatasetDict(stage_3.load('temp_khan/3'))
dataset2 = DatasetDict(stage_3.load('temp/3'))

# Combine their train and validation splits
combined = DatasetDict({
    key: concatenate_datasets([dataset1[key], dataset2[key]])
    for key in dataset1.keys()
})

combined.save_to_disk('temp_combined')