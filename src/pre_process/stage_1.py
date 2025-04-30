"""
This is the first stage of pre-processing.
It does the following cleaning:
    - punctuation removal
    - lowercasing
"""

import string
from text_utils import *

__all__ = ['pre_process', 'save']

def pre_process(jsonl_full: list[dict]):
    """
    Preprocesses a list of JSONL lines by validating structure, lowercasing text, 
    and removing specific punctuation characters.

    Args:
        jsonl_full (list[dict]): A list of dictionary entries, each representing a JSONL line 
                                 with at least a 'text' key.

    Returns:
        list[dict]: The modified list with lowercased text and specified punctuation removed.

    Raises:
        Exception: If the input structure does not pass validation via is_valid_jsonl_line.
    """
    if not is_valid_jsonl_line(jsonl_full):
        raise Exception('Unexpected structure of jsonl')
    
    for x, line in enumerate(jsonl_full):
        line = lowercase_text(line)
        line = remove_punctuation(line, {'<', '>'})
        jsonl_full[x] = line
    
    return jsonl_full

def save(out_file, stg_1_data):
    save_jsonl(out_file, stg_1_data)
    
def is_valid_jsonl_line(jsonl_full: list[dict]) -> bool:
    try:
        if not isinstance(jsonl_full, list):
            return False
        
        jsonl_line = jsonl_full[0]
        required_keys = {"id", "text", "entities", "relations", "Comments"}

        if not isinstance(jsonl_line, dict):
            return False
        if not required_keys.issubset(jsonl_line):
            return False
        if not isinstance(jsonl_line["id"], int):
            return False
        if not isinstance(jsonl_line["text"], str):
            return False
        if not isinstance(jsonl_line["entities"], list):
            return False

        for ent in jsonl_line["entities"]:
            if not isinstance(ent, dict):
                return False
            if not {"id", "label", "start_offset", "end_offset"}.issubset(ent):
                return False
            if not isinstance(ent["id"], int):
                return False
            if not isinstance(ent["label"], str):
                return False
            if not isinstance(ent["start_offset"], int):
                return False
            if not isinstance(ent["end_offset"], int):
                return False
    except KeyError:
        return False
    
    return True

def lowercase_text(jsonl_line: dict):
    lowered = jsonl_line.copy()
    lowered['text'] = lowered['text'].lower()
    return lowered
    
def remove_punctuation(jsonl_line: dict, excepted_puncs: set):
    """
    Remove punctuation from the input text and realign entity offsets accordingly.

    Parameters:
        jsonl_line (dict): A dictionary representing one line from a JSONL file.
            It must contain:
                - 'text' (str): The original input string.
                - 'entities' (list): A list of entities, each with 'start_offset' and 'end_offset'.
        excepted_puncs (set): A set of punctuation characters to exclude from removal.

    Returns:
        dict: The updated dictionary with punctuation removed (except for those in `excepted_puncs`)
              and entity offsets adjusted to match the new text.
    """
    original_text = jsonl_line["text"]
    punctuation = set(string.punctuation) - excepted_puncs

    new_text = ""
    offset_map = {}  # Maps original char index → new index
    new_index = 0

    for i, char in enumerate(original_text):
        if char not in punctuation:
            new_text += char
            offset_map[i] = new_index
            new_index += 1

    updated_entities = []
    for ent in jsonl_line["entities"]:
        old_start = ent["start_offset"]
        old_end = ent["end_offset"]

        if not any(i in offset_map for i in range(old_start, old_end)):
            continue

        new_start = offset_map.get(old_start)
        if new_start is None:
            for i in range(old_start + 1, old_end):
                if i in offset_map:
                    new_start = offset_map[i]
                    break
        if new_start is None:
            continue

        valid_indices = [i for i in range(old_start, old_end) if i in offset_map]
        if not valid_indices:
            continue
        new_end = offset_map[valid_indices[-1]] + 1

        updated_entities.append({
            **ent,
            "start_offset": new_start,
            "end_offset": new_end
        })

    return {
        **jsonl_line,
        "text": new_text,
        "entities": updated_entities
    }


def show_allignment(dict):
    """
    Prints entity-text and its entity-type.
    Used to check if they are still alligned.

    Parameters:
        dict (dict): A dictionary with:
            - 'text' (str): The sentence string.
            - 'entities' (list): A list of entities with 'start_offset', 'end_offset', and 'label'.

    Returns:
        None
    """
    text = dict['text']
    
    for ent in dict['entities']:
        print(f'{text[ent["start_offset"]:ent["end_offset"]]}; {ent["label"]}')