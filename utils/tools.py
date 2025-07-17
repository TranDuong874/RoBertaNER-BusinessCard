import numpy as np
from seqeval.metrics import classification_report
from transformers.trainer_callback import TrainerCallback
import pandas as pd

def label_to_id(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[label] = i
    return dic
    
def id_to_label(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[i] = label
    return dic

# Custom callback for CSV logging
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.metrics = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Log evaluation metrics
        metrics["epoch"] = state.epoch
        self.metrics.append(metrics)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_path, index=False)

def make_compute_metrics(label_list):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        pred_labels = [[label_list[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
        report = classification_report(true_labels, pred_labels, output_dict=True)
        return {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        }
    return compute_metrics