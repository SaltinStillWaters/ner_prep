import evaluate
import numpy as np
from src.misc.globals import labels
metric = evaluate.load('seqeval')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import chain

def strip_bio(label):
    return label[2:] if label.startswith(("B-", "I-")) else label

def compute_metrics(eval_preds, metric=metric, ents=labels):
  logits, labels = eval_preds

  predictions = np.argmax(logits, axis=-1)

  true_labels = [[ents[l] for l in label if l!=-100] for label in labels]

  true_predictions = [[ents[p] for p,l in zip(prediction, label) if l!=-100]
                      for prediction, label in zip(predictions, labels)]

  all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

  return {"precision": all_metrics['overall_precision'],
          "recall": all_metrics['overall_recall'],
          "f1": all_metrics['overall_f1'],
          "accuracy": all_metrics['overall_accuracy']}
  
# def compute_metrics(eval_preds, metric=metric, ents=labels):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Convert IDs to label names, filtering out -100
#     true_labels = [[ents[l] for l in label if l != -100] for label in labels]
#     true_predictions = [[ents[p] for p, l in zip(prediction, label) if l != -100]
#                         for prediction, label in zip(predictions, labels)]

#     # Compute seqeval metrics
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

#     # Flatten and strip BIO prefixes
#     flat_true = [strip_bio(l) for l in chain.from_iterable(true_labels) if strip_bio(l) != "O"]
#     flat_pred = [strip_bio(p) for p, t in zip(chain.from_iterable(true_predictions), chain.from_iterable(true_labels))
#                  if strip_bio(t) != "O"]

#     # Define unique entity labels
#     entity_set = sorted(set(flat_true + flat_pred))
    
#     # Compute and display confusion matrix
#     cm = confusion_matrix(flat_true, flat_pred, labels=entity_set)
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

