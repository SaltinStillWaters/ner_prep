from src.misc.text_utils import *

__all__ = ['pre_process', 'save', 'print_token_alignment']

def pre_process(stg_1_data: list[dict], tokenizer, tag2index):
    """
    Processes a list of JSONL lines from stage_1 by tokenizing the text and aligning NER labels.
    
    Each entry in the resulting list will contain:
        - 'id': Index of the line
        - 'tokens': List of tokenized strings
        - 'labels': List of numerical NER tag indices

    The final result is saved to 'processed_stage2.jsonl'.

    Parameters:
        stg_1_data (list): List of dictionaries, each with 'text' and 'entities' fields from stage_1.

    Returns:
        None
    """
    
    result = list()
    for x, line in enumerate(stg_1_data):
        tokens = align_labels(line['text'], line['entities'], tokenizer, tag2index)
        del tokens['offset_mapping']
        result.append(tokens)
        
    return result

def save(out_file,stg_2_data):
    save_pkl(out_file, stg_2_data)
    
def print_token_alignment(tokens, tokenizer):
    input_ids = tokens['input_ids']
    labels = tokens['labels']
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    
    word_ids = tokens.word_ids() if hasattr(tokens, "word_ids") else [None] * len(input_ids)
    
    print(f"{'Idx':<4} {'Input ID':<10} {'Token':<15} {'Word ID':<8} {'Label':<10}")
    print("-" * 50)
    
    for i, (id_, tok, wid, label) in enumerate(zip(input_ids, token_strings, word_ids, labels)):
        print(f"{i:<4} {id_:<10} {tok:<15} {str(wid):<8} {label:<10}")

def align_labels(text, entities, tokenizer, tag2index):
    # Tokenize normally (not split into words), but get offset mapping and word_ids
    encodings = tokenizer(text, return_offsets_mapping=True, truncation=True)
    offset_mapping = encodings["offset_mapping"]
    word_ids = encodings.word_ids()
    
    # Prepare entity lookup by character offset
    entity_map = {}
    for ent in entities:
        for i in range(ent["start_offset"], ent["end_offset"]):
            entity_map[i] = ent["label"]

    labels = []
    previous_word_id = None

    is_begin = False
    for idx, (offset, word_id) in enumerate(zip(offset_mapping, word_ids)):
        start, end = offset
        
        if word_id is None:
            # CLS, SEP, or padding
            labels.append(-100)
        elif word_id != previous_word_id:
            # First subword of a word
            if any(i in entity_map for i in range(start, end)):
                label_char = entity_map.get(start, None)
                if label_char:
                    if is_begin:
                        labels.append(tag2index[f"I-{label_char}"])
                    else:
                        labels.append(tag2index[f"B-{label_char}"])
                        is_begin = True
                else:
                    is_begin = False
                    labels.append(tag2index["O"])
            else:
                is_begin = False
                labels.append(tag2index["O"])
        else:
            is_begin = False
            # Subsequent subword: ignore
            labels.append(0) #set this to -100 if you want to ignore it instead

        previous_word_id = word_id

    encodings['labels'] = labels
    return encodings

# tokenizes text using disitilbert tokenizer and aligns the ner tags
# def align_labels(text, entities, tokenizer):
#     """
#     Tokenizes the input text and aligns NER entity labels with the resulting tokens.

#     Each token is assigned a label using the BIO tagging scheme:
#         - 'B-<label>' for the beginning of an entity
#         - 'I-<label>' for the inside of an entity
#         - 'O' for tokens not part of any entity

#     Parameters:
#         text (str): The input text to tokenize.
#         entities (list): A list of entity dictionaries with 'start_offset', 'end_offset', and 'label'.
#         tokenizer: A Hugging Face tokenizer (e.g., DistilBERT tokenizer).

#     Returns:
#         tuple: A tuple containing:
#             - tokens (list of str): Tokenized text
#             - labels (list of str): Corresponding NER tags for each token
#     """
#     encodings = tokenizer(text, return_offsets_mapping=True, truncation=True)
#     labels = []

#     entity_ranges = [
#         (ent["start_offset"], ent["end_offset"], ent["label"])
#         for ent in entities
#     ]

#     for idx, (start, end) in enumerate(encodings["offset_mapping"]):
#         if start == end:
#             labels.append("O")
#             continue

#         label = "O"
#         for ent_start, ent_end, ent_label in entity_ranges:
#             if start == ent_start:
#                 label = f"B-{ent_label}"
#                 break
#             elif ent_start < start < ent_end:
#                 label = f"I-{ent_label}"
#                 break
#         labels.append(label)
    
#     return encodings.tokens(), labels