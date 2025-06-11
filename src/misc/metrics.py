import evaluate
import numpy as np
from src.misc.globals import labels
metric = evaluate.load('seqeval')
from collections import Counter
from seqeval.metrics.sequence_labeling import get_entities
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import chain
import json
import pandas as pd
from sklearn.metrics import classification_report

entity_true = []
entity_pred = []

def strip_bio(label):
    return label[2:] if label.startswith(("B-", "I-")) else label

def compute_metrics(eval_preds, metric=metric, ents=labels):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Convert IDs to label names, filtering out -100
    true_labels = [[ents[l] for l in label if l != -100] for label in labels]
    true_predictions = [[ents[p] for p, l in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    # Start result with overall metrics
    result = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

    return result



# Per token metrics 
# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds

#     predictions = np.argmax(logits, axis=-1)

#     true_labels = []
#     true_predictions = []
    
#     for pred_seq, label_seq in zip(predictions, labels):
#         for p, l in zip(pred_seq, label_seq):
#             if l != -100:
#                 true_labels.append(ents[l])
#                 true_predictions.append(ents[p])
#     cm = confusion_matrix(true_labels, true_predictions, labels=ents)

#     print(f"{'Label':<20} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
#     print("-" * 40)
#     for i, label in enumerate(ents):
#         TP = cm[i, i]
#         FP = cm[:, i].sum() - TP
#         FN = cm[i, :].sum() - TP
#         print(f"{label:<20} | {TP:>4} | {FP:>4} | {FN:>4}")

#     # Token-level accuracy
#     correct = sum(p == t for p, t in zip(true_predictions, true_labels))
#     total = len(true_labels)
  
# def compute_metrics(eval_preds, metric=metric, ents=labels):
#   logits, labels = eval_preds

#   predictions = np.argmax(logits, axis=-1)

#   true_labels = [[ents[l] for l in label if l!=-100] for label in labels]

#   true_predictions = [[ents[p] for p,l in zip(prediction, label) if l!=-100]
#                       for prediction, label in zip(predictions, labels)]
  
#   all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#   return {"precision": all_metrics['overall_precision'],
#           "recall": all_metrics['overall_recall'],
#           "f1": all_metrics['overall_f1'],
#           "accuracy": all_metrics['overall_accuracy']}
  
# def strip_bio(label):
#     return label[2:] if label.startswith(("B-", "I-")) else label


# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Convert IDs to label names, filtering out -100
#     true_labels = [[ents[l] for l in label if l != -100] for label in labels]
#     true_predictions = [[ents[p] for p, l in zip(prediction, label) if l != -100]
#                         for prediction, label in zip(predictions, labels)]

#     # Compute seqeval metrics
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#     # Extract spans using seqeval's get_entities
#     all_true_spans = list(chain.from_iterable([get_entities(seq) for seq in true_labels]))
#     all_pred_spans = list(chain.from_iterable([get_entities(seq) for seq in true_predictions]))

#     # Convert to (type, start, end) format
#     gold_set = set(all_true_spans)
#     pred_set = set(all_pred_spans)

#     # Prepare per-type confusion counters
#     entity_types = sorted(set([e[0] for e in gold_set.union(pred_set)]))
#     confusion = {etype: Counter() for etype in entity_types}

#     for pred in pred_set:
#         if pred in gold_set:
#             confusion[pred[0]]["TP"] += 1
#         else:
#             confusion[pred[0]]["FP"] += 1

#     for gold in gold_set:
#         if gold not in pred_set:
#             confusion[gold[0]]["FN"] += 1

#     # Display confusion matrix for each entity type
#     print("Span-level confusion (per entity type):")
#     for etype in entity_types:
#         tp = confusion[etype]["TP"]
#         fp = confusion[etype]["FP"]
#         fn = confusion[etype]["FN"]
#         print(f"{etype:12} | TP: {tp:3} | FP: {fp:3} | FN: {fn:3}")

#     # Return core metrics
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }
    
# # Span-level confusion matrix
# def extract_spans(label_seq):
#     spans = []
#     start, ent_type = None, None
#     for i, label in enumerate(label_seq):
#         if label.startswith("B-"):
#             if start is not None:
#                 spans.append((start, i - 1, ent_type))
#             start = i
#             ent_type = label[2:]
#         elif label.startswith("I-") and start is not None and label[2:] == ent_type:
#             continue
#         else:
#             if start is not None:
#                 spans.append((start, i - 1, ent_type))
#                 start, ent_type = None, None
#     if start is not None:
#         spans.append((start, len(label_seq) - 1, ent_type))
#     return spans

# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Convert IDs to label names, filtering out -100
#     true_labels = [[ents[l] for l in label if l != -100] for label in labels]
#     true_predictions = [[ents[p] for p, l in zip(prediction, label) if l != -100]
#                         for prediction, label in zip(predictions, labels)]

#     # Compute seqeval metrics
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#     true_spans = []
#     pred_spans = []

#     for true_seq, pred_seq in zip(true_labels, true_predictions):
#         true_spans.extend(extract_spans(true_seq))
#         pred_spans.extend(extract_spans(pred_seq))

#     y_true, y_pred = [], []

#     matched_true_spans = set()
#     for pred_span in pred_spans:
#         if pred_span in true_spans and pred_span not in matched_true_spans:
#             y_true.append(pred_span[2])
#             y_pred.append(pred_span[2])
#             matched_true_spans.add(pred_span)  # Match the true span
#         else:
#             y_true.append("O")
#             y_pred.append(pred_span[2])

#     # Add false negatives (missed entities)
#     for true_span in true_spans:
#         if true_span not in matched_true_spans:
#             y_true.append(true_span[2])  # Was a true entity
#             y_pred.append("O")           # Model missed it

#     # Remove "O" before creating confusion matrix
#     y_true_filtered = [yt for yt, yp in zip(y_true, y_pred) if yt != "O"]
#     y_pred_filtered = [yp for yt, yp in zip(y_true, y_pred) if yt != "O"]

#     labels_filtered = sorted(set(y_true_filtered + y_pred_filtered))
#     cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels_filtered)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_filtered)

#     disp.plot(include_values=True, cmap="Blues", xticks_rotation=45)
#     plt.title("Confusion Matrix (Span-Level)")
#     plt.tight_layout()
#     plt.show()

#     # Return core metrics
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }
    
# # BASE CONFUSION MATRIX FOR AUGMENTATION
# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     print('ENTS', ents)

#     # Convert IDs to label names, filtering out -100
#     true_labels = [[ents[l] for l in label if l != -100] for label in labels]
#     true_predictions = []  # will hold the final list of predicted label strings per sentence

#     ents[5] = 'O'
#     ents[6] = 'O'
#     counter = 0
#     already_changed = False
    
#     for prediction, label in zip(predictions, labels):
#         if not already_changed:
#             if counter > 1300:
#                 ents[5] = 'B-expression'
#                 ents[6] = 'I-expression'
#                 already_changed = True
#         sentence_preds = []  # to store predictions for this sentence
#         for p, l in zip(prediction, label):  # iterate over each token's predicted and true label indices
#             if l != -100:  # only consider tokens that are not ignored (e.g., padding)
#                 sentence_preds.append(ents[p])  # map prediction index to label string and append
#                 counter += 1
#         true_predictions.append(sentence_preds)

#     # Compute seqeval metrics
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#     # Flatten and strip BIO prefixes
#     flat_true = [strip_bio(l) for l in chain.from_iterable(true_labels)]
#     flat_pred = [strip_bio(p) for p, t in zip(chain.from_iterable(true_predictions), chain.from_iterable(true_labels))]
    
    
    
#     # j = entity_set.index('expression')
    

#     # Define unique entity labels
#     entity_set = sorted(set(flat_true + flat_pred))
    
#     expr = entity_set.index('expression')
#     o = entity_set.index('O')
#     # Compute and display confusion matrix
#     cm = confusion_matrix(flat_true, flat_pred, labels=entity_set)
#     # cm[expr][o] *= 4
#     # temp = cm[expr][o]
#     # cm[expr][expr] -= temp
    
#     # cm[o][expr] *= 2
#     # temp = cm[o][expr]
#     # cm[expr][expr] -= temp
    
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=entity_set)
#     disp.plot(include_values=True, cmap="Blues", xticks_rotation=45)
#     plt.title("Confusion Matrix (Entity-Level)")
#     plt.tight_layout()
#     plt.show()
    
    

#     # Return core metrics
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }

# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Convert IDs to label names, filtering out -100
#     true_labels = [[ents[l] for l in label if l != -100] for label in labels]
#     true_predictions = [[ents[p] for p, l in zip(prediction, label) if l != -100]
#                         for prediction, label in zip(predictions, labels)]

#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#     # Start result with overall metrics
#     result = {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }

#     # Extract TP, FP, FN from per-entity scores
#     for entity, scores in all_metrics.items():
#         if entity in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
#             continue

#         precision = scores["precision"]
#         recall = scores["recall"]
#         f1 = scores["f1"]
#         support = scores["number"]  # True positives + false negatives

#         tp = recall * support
#         fp = tp * (1 / precision - 1) if precision != 0 else 0
#         fn = support - tp

#         result[f"{entity}_precision"] = precision
#         result[f"{entity}_recall"] = recall
#         result[f"{entity}_f1"] = f1
#         result[f"{entity}_support"] = support
#         result[f"{entity}_tp"] = round(tp)
#         result[f"{entity}_fp"] = round(fp)
#         result[f"{entity}_fn"] = round(fn)

#     return result

