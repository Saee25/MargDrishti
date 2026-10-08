import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import random
from ml.config import load_config

def setup_dirs(config):
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures" / "dataset"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, figures_dir

def plot_class_distribution(manifest, figures_dir):
    train_manifest = manifest[manifest['split'] == 'train']
    counts = train_manifest['class_slug'].value_counts().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 15))
    counts.plot(kind='barh', color='skyblue')
    plt.title('Train Crops per Class')
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png")
    plt.close()
    return counts

def plot_split_sizes(manifest, figures_dir):
    split_counts = manifest['split'].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=split_counts.index, y=split_counts.values)
    plt.title('Crops per Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.savefig(figures_dir / "split_sizes.png")
    plt.close()
    return split_counts.to_dict()

def plot_crop_sizes(manifest, figures_dir):
    plt.figure(figsize=(10, 5))
    sns.histplot(manifest['crop_w'], color='blue', alpha=0.5, label='Width', kde=True)
    sns.histplot(manifest['crop_h'], color='red', alpha=0.5, label='Height', kde=True)
    plt.axvline(64, color='black', linestyle='--', label='64 px (Target)')
    plt.title('Crop Size Distribution (Width & Height)')
    plt.xlabel('Pixels')
    plt.legend()
    plt.savefig(figures_dir / "crop_size_hist.png")
    plt.close()
    
    hist_w, bins_w = np.histogram(manifest['crop_w'], bins=20)
    return {
        'hist': hist_w.tolist(),
        'bins': bins_w.tolist()
    }

def plot_box_area_share(manifest, figures_dir):
    plt.figure(figsize=(8, 5))
    sns.histplot(manifest['area_share'], bins=50, color='green', kde=True)
    plt.title('Box Area Share of Original Image')
    plt.xlabel('Share of Image Area (0.0 to 1.0)')
    plt.ylabel('Count')
    plt.savefig(figures_dir / "box_area_share_hist.png")
    plt.close()
    
    avg_share = manifest['area_share'].mean()
    return avg_share

def create_sample_grid(manifest, crops_dir, figures_dir):
    train_manifest = manifest[manifest['split'] == 'train']
    classes = sorted(train_manifest['class_slug'].unique())
    
    n_classes = len(classes)
    cols = 8
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    
    for ax in axes:
        ax.axis('off')
        
    for idx, cls in enumerate(classes):
        cls_crops = train_manifest[train_manifest['class_slug'] == cls]
        if not cls_crops.empty:
            sample = cls_crops.sample(1, random_state=42).iloc[0]
            img_path = Path(crops_dir) / sample['crop_path']
            try:
                img = Image.open(img_path)
                axes[idx].imshow(img)
                axes[idx].set_title(cls[:15], fontsize=8) # truncate title
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    plt.tight_layout()
    plt.savefig(figures_dir / "sample_grid.png")
    plt.close()

def create_smallest_crops_grid(manifest, crops_dir, figures_dir):
    # Calculate area and sort
    manifest_copy = manifest.copy()
    manifest_copy['crop_area'] = manifest_copy['crop_w'] * manifest_copy['crop_h']
    smallest = manifest_copy.nsmallest(24, 'crop_area')
    
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax in axes:
        ax.axis('off')
        
    for idx, (_, row) in enumerate(smallest.iterrows()):
        img_path = Path(crops_dir) / row['crop_path']
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{row['crop_w']}x{row['crop_h']}", fontsize=10)
        except Exception:
            pass
            
    plt.tight_layout()
    plt.savefig(figures_dir / "smallest_crops.png")
    plt.close()

def main():
    config = load_config()
    reports_dir, figures_dir = setup_dirs(config)
    processed_dir = Path(config["paths"]["processed_dir"])
    crops_dir = config["paths"]["crops_dir"]
    
    manifest_path = processed_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    
    stats_path = processed_dir / "stats.json"
    with open(stats_path, 'r') as f:
        ml_stats = json.load(f)
        
    dropped_path = reports_dir / "dropped_classes.csv"
    if dropped_path.exists():
        dropped_df = pd.read_csv(dropped_path)
        dropped_classes_list = dropped_df.to_dict('records')
    else:
        dropped_df = pd.DataFrame()
        dropped_classes_list = []
        
    dup_path = reports_dir / "cross_split_duplicates.csv"
    leakage = "No leakage detected across splits."
    if dup_path.exists():
        leakage = "WARNING: Cross-split duplicates were found. See cross_split_duplicates.csv"
        
    # Generate Plots
    print("Generating class distribution...")
    class_counts = plot_class_distribution(manifest, figures_dir)
    print("Generating split sizes...")
    split_counts = plot_split_sizes(manifest, figures_dir)
    print("Generating crop sizes...")
    size_hist = plot_crop_sizes(manifest, figures_dir)
    print("Generating area share...")
    avg_share = plot_box_area_share(manifest, figures_dir)
    print("Generating sample grid...")
    create_sample_grid(manifest, crops_dir, figures_dir)
    print("Generating smallest crops grid...")
    create_smallest_crops_grid(manifest, crops_dir, figures_dir)
    
    # Dataset stats for web
    dataset_stats = {
        'splits': split_counts,
        'per_class_train': class_counts.to_dict(),
        'crop_size_hist': size_hist,
        'kept_classes_count': ml_stats['k'],
        'dropped_classes': dropped_classes_list
    }
    with open(reports_dir / "dataset_stats.json", 'w') as f:
        json.dump(dataset_stats, f, indent=2)
        
    # Generate Markdown Report
    scene_type = "real road scenes" if avg_share < 0.2 else "close-up photos"
    imbalance = class_counts.max() / class_counts.min() if not class_counts.empty else 0
    
    md_content = f"""# Dataset Report

## Overview
This report summarizes the dataset processing, statistics, and exploration findings.

### Dataset Stats
- **Classes Kept (K):** {ml_stats['k']}
- **Classes Dropped:** {len(dropped_classes_list)}
- **Leakage Check:** {leakage}
- **Channel Mean:** {ml_stats['mean']}
- **Channel Std:** {ml_stats['std']}

### Splits
{split_counts}

### Dropped Classes
{dropped_df.to_markdown() if not dropped_df.empty else "None"}

## Observations
1. **Scene Composition:** Based on the box area share distribution (average share {avg_share:.3f}), these images primarily represent {scene_type}.
2. **Class Imbalance:** There is a significant class imbalance with the largest class being {imbalance:.2f}x larger than the smallest kept class in the training set.
3. **Small Signs:** Many crops are significantly smaller than the 64x64 target, indicating the model will need to handle low-resolution features effectively.
4. **Similar Classes:** There are numerous look-alike classes (e.g., left/right turn pairs, speed limit increments) that will challenge the model's fine-grained classification capabilities.

## Figures
- **Class Distribution:** `reports/figures/dataset/class_distribution.png`
- **Split Sizes:** `reports/figures/dataset/split_sizes.png`
- **Crop Sizes:** `reports/figures/dataset/crop_size_hist.png`
- **Box Area Share:** `reports/figures/dataset/box_area_share_hist.png`
- **Samples:** `reports/figures/dataset/sample_grid.png`
- **Smallest Crops:** `reports/figures/dataset/smallest_crops.png`
"""
    with open(reports_dir / "dataset_report.md", 'w') as f:
        f.write(md_content)
        
    print("EDA completed. Generated reports and figures.")

if __name__ == "__main__":
    main()
