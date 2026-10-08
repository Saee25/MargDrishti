import json
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from ml.engine.trainer import load_checkpoint
from ml.engine.metrics import (
    compute_classification_metrics,
    compute_per_class_report,
    compute_confusion_matrices,
    compute_most_confused_pairs,
    compute_topk_accuracy
)
from ml.data.loaders import build_loaders

def plot_curves(history_csv, out_dir, best_epoch):
    """Plots training and validation curves from history.csv."""
    if not Path(history_csv).exists():
        return
        
    df = pd.read_csv(history_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation')
    if best_epoch > 0:
        ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(df['epoch'], df['train_acc'], label='Train')
    ax2.plot(df['epoch'], df['val_acc'], label='Validation')
    if best_epoch > 0:
        ax2.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=200)
    plt.close(fig)

def plot_confusion_matrix(cm_norm, class_names, out_path):
    """Plots a large row-normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(24, 24))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Purples')
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Row-Normalised Confusion Matrix')
    
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_per_class_f1(df_report, out_path):
    """Plots horizontal bars for per-class F1."""
    df_sorted = df_report.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(df_sorted['class_name'], df_sorted['f1'], color='#9479B8') # purple-400
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Class F1 Score')
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_image_grid(results, class_names, out_path, title, max_images=36):
    """Plots a grid of images (correct or misclassified)."""
    n = min(len(results), max_images)
    if n == 0:
        return
        
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*3))
    
    # Ensure axes is a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
        
    for i, item in enumerate(results[:n]):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        
        img = Image.open(item['path'])
        ax.imshow(img)
        ax.axis('off')
        
        color = 'green' if item['correct'] else 'red'
        title_text = f"T: {item['true_name']}\nP: {item['pred_name']}\nConf: {item['confidence']:.2f}"
        ax.set_title(title_text, fontsize=8, color=color)
        
    # Hide empty subplots
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
        
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def evaluate(checkpoint_path, split, out_dir, config_overrides=None):
    """
    Evaluates the model on the specified split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    
    from ml.config import Config
    config = Config(checkpoint["config"])
    
    if config_overrides:
        from ml.config import set_nested_value
        for override in config_overrides:
            if "=" in override:
                key_path, value_str = override.split("=", 1)
                keys = key_path.split(".")
                
                if value_str.lower() == "true": value = True
                elif value_str.lower() == "false": value = False
                else:
                    try: value = int(value_str)
                    except ValueError:
                        try: value = float(value_str)
                        except ValueError: value = value_str
                set_nested_value(config, keys, value)
    
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if it's a smoke run from config or out_dir
    is_smoke = "_smoke_" in str(out_dir)
    train_loader, val_loader, test_loader, _ = build_loaders(config, smoke=is_smoke)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[split]
    
    model.eval()
    
    all_preds = []
    all_targets = []
    all_paths = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f"Evaluating on {split}")
    # We need paths, so we have to retrieve them from dataset
    # Standard DataLoader returns (images, targets)
    # The dataset has .samples containing (path, label_idx)
    # But shuffling is off for val/test if we configure it right, 
    # Actually, Dataset returns (img, target), we need paths.
    # We will reconstruct paths by using the dataset's sample list directly if it's not shuffled.
    # But let's just modify the dataset or get paths directly.
    # Our dataset (ml/data/dataset.py) doesn't return path by default.
    # Since we need paths, let's access loader.dataset.samples if available.
    
    is_shuffled = False
    
    with torch.no_grad():
        for batch_idx, (images, targets, batch_paths) in enumerate(pbar):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.append(probs.cpu())
            all_paths.extend(batch_paths)
            
    all_probs = torch.cat(all_probs, dim=0)
    
    metrics = compute_classification_metrics(all_targets, all_preds)
    
    # Top-k accuracy (k=1 and k=3)
    # all_probs is [N, C], targets is [N]
    targets_tensor = torch.tensor(all_targets)
    topk_accs = compute_topk_accuracy(all_probs, targets_tensor, topk=(1, 3))
    metrics['top1_accuracy'] = topk_accs[0]
    metrics['top3_accuracy'] = topk_accs[1]
    
    # Save metrics JSON
    metrics_path = out_dir / f"{split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Class report
    df_report = compute_per_class_report(all_targets, all_preds, class_names)
    df_report.to_csv(out_dir / "classification_report.csv", index=False)
    
    # Confusion matrix
    cm, cm_norm = compute_confusion_matrices(all_targets, all_preds, num_classes)
    np.save(out_dir / "confusion_matrix.npy", cm)
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(out_dir / "confusion_matrix.csv")
    
    # Most confused pairs
    confused_pairs = compute_most_confused_pairs(cm, class_names)
    df_pairs = pd.DataFrame(confused_pairs)
    df_pairs.to_csv(out_dir / "most_confused_pairs.csv", index=False)
    
    predictions = []
    correct_results = []
    wrong_results = []
    
    for i in range(len(all_targets)):
        true_idx = all_targets[i]
        pred_idx = all_preds[i]
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        correct = (true_idx == pred_idx)
        
        path = all_paths[i]
        
        p = all_probs[i]
        conf = p[pred_idx].item()
        top3_vals, top3_idx = p.topk(3)
        top3_indices = top3_idx.tolist()
        top3_probs = top3_vals.tolist()
        
        res = {
            'path': path,
            'true_idx': true_idx,
            'true_name': true_name,
            'pred_idx': pred_idx,
            'pred_name': pred_name,
            'confidence': conf,
            'top3_indices': top3_indices,
            'top3_probs': top3_probs,
            'correct': correct
        }
        predictions.append(res)
        
        if correct:
            correct_results.append(res)
        else:
            wrong_results.append(res)
            
    df_preds = pd.DataFrame(predictions)
    # Flatten list columns for CSV saving
    df_preds['top3_indices'] = df_preds['top3_indices'].apply(json.dumps)
    df_preds['top3_probs'] = df_preds['top3_probs'].apply(json.dumps)
    df_preds.to_csv(out_dir / "predictions.csv", index=False)
    
    # Plots
    if split == 'val' or split == 'test':
        history_csv = Path(checkpoint_path).parent / "history.csv"
        best_epoch = checkpoint.get("epoch", 0) # approximation, from checkpoint best
        if history_csv.exists():
            plot_curves(history_csv, out_dir, best_epoch)
            
    plot_confusion_matrix(cm_norm, class_names, out_dir / "confusion_matrix.png")
    plot_per_class_f1(df_report, out_dir / "per_class_f1.png")
    
    # Sort wrong by confidence descending (most confident mistakes)
    wrong_results.sort(key=lambda x: x['confidence'], reverse=True)
    # Sort correct by confidence descending
    correct_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    plot_image_grid(wrong_results, class_names, out_dir / "misclassified_grid.png", "Misclassified", max_images=36)
    plot_image_grid(correct_results, class_names, out_dir / "correct_grid.png", "Correctly Classified", max_images=24)

    return metrics, confused_pairs, df_report
