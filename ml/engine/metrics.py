import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

def compute_topk_accuracy(outputs, targets, topk=(1, 3)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        if maxk > outputs.size(1):
            maxk = outputs.size(1)
            topk = [k for k in topk if k <= maxk]

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() / batch_size)
        return res

def compute_classification_metrics(y_true, y_pred):
    """Computes overall classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted metrics
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1
    }

def compute_per_class_report(y_true, y_pred, class_names):
    """Computes a per-class report as a DataFrame."""
    # Ensure all classes are represented even if absent in y_true/y_pred
    labels = list(range(len(class_names)))
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Support
    support = np.bincount(y_true, minlength=len(class_names))
    
    df = pd.DataFrame({
        'class_name': class_names,
        'precision': p,
        'recall': r,
        'f1': f1,
        'support': support
    })
    
    return df

def compute_confusion_matrices(y_true, y_pred, num_classes):
    """Computes raw and row-normalised confusion matrices."""
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Row-normalised
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Replace NaNs with 0
    
    return cm, cm_norm

def compute_most_confused_pairs(cm, class_names):
    """
    Finds the most confused pairs.
    Returns: list of dicts (true_class, predicted_class, count, share) sorted by count.
    """
    pairs = []
    num_classes = len(class_names)
    
    # Calculate row sums for sharing
    row_sums = cm.sum(axis=1)
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                share = float(cm[i, j] / row_sums[i]) if row_sums[i] > 0 else 0.0
                pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(cm[i, j]),
                    'share': share
                })
                
    # Sort by count descending
    pairs.sort(key=lambda x: x['count'], reverse=True)
    return pairs
