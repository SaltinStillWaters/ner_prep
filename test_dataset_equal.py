from datasets import DatasetDict

def compare_datasetdicts(d1: DatasetDict, d2: DatasetDict) -> bool:
    if d1.keys() != d2.keys():
        print("Different keys:", d1.keys(), "!=", d2.keys())
        return False

    for split in d1.keys():
        ds1 = d1[split]
        ds2 = d2[split]

        if len(ds1) != len(ds2):
            print(f"Split '{split}' has different lengths: {len(ds1)} != {len(ds2)}")
            return False

        for i in range(len(ds1)):
            if ds1[i] != ds2[i]:
                print(f"Mismatch in split '{split}' at index {i}")
                print("ds1:", ds1[i])
                print("ds2:", ds2[i])
                return False

    print("✅ DatasetDicts are identical")
    return True

from src.pre_process.stage_3 import load

d1 = load("processed/base_dataset/3")
d2 = load("processed/khan_orig/3")

compare_datasetdicts(d1, d2)
