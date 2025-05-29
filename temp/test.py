from src.pre_process import stage_3

dataset_dict = stage_3.load('temp_khan/3')
print(dataset_dict)

print(dataset_dict["train"][0])
print(dataset_dict["validation"][0])
print(dataset_dict["test"][0])

labels = ['O', 'B-command', 'I-command', 'B-equation', 'I-equation', 'B-expression', 'I-expression', 'B-term', 'I-term', 'B-command_attribute', 'I-command_attribute', 'B-method', 'I-method']

index2tag = {x:ent for x, ent in enumerate(labels)}
tag2index = {ent:x for x, ent in enumerate(labels)}

def extract_entity_types_from_dataset(dataset, index2tag):
    all_types = []
    for sample in dataset:
        tag_ids = [label for label in sample["labels"] if label >= 0]
        tag_names = [index2tag[i] for i in tag_ids if i in index2tag]
        entity_types = [tag.split("-")[-1] for tag in tag_names if "-" in tag]
        all_types.extend(entity_types)
    return all_types

from collections import Counter

train_types = extract_entity_types_from_dataset(dataset_dict["train"], index2tag)
val_types = extract_entity_types_from_dataset(dataset_dict["validation"], index2tag)
test_types = extract_entity_types_from_dataset(dataset_dict["test"], index2tag)

print("Train:", Counter(train_types))
print("Val:  ", Counter(val_types))
print("Test: ", Counter(test_types))
