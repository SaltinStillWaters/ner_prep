import string
from text_utils import *

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