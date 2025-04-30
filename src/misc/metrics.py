import evaluate
import numpy as np

metric = evaluate.load('seqeval')

def compute_metrics(eval_preds, metric, ents):
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