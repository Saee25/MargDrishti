import time
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
import yaml
import sys
import csv
import json
import random

from ml.data.loaders import build_loaders
from ml.data.transforms import build_transforms

def load_config():
    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)

def denormalize(tensor, mean, std):
    # tensor: (C, H, W)
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return torch.clamp(t, 0, 1)

def main():
    cfg = load_config()
    
    # 1. Augmentation preview
    manifest_path = Path(cfg["paths"]["processed_dir"]) / "manifest.csv"
    crops_dir = Path(cfg["paths"]["crops_dir"])
    manifest = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                row["class_index"] = int(row["class_index"])
                row["crop_path"] = str(crops_dir / row["crop_path"])
                manifest.append(row)
                
    rng = random.Random(42)
    sample_crops = rng.sample(manifest, 6)
    
    img_size = cfg["model"]["img_size"]
    
    with open(Path(cfg["paths"]["processed_dir"]) / "stats.json", "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    
    eval_transform = build_transforms(img_size, mean, std, train=False, augment=False)
    train_transform = build_transforms(img_size, mean, std, train=True, augment=True)
    
    fig, axes = plt.subplots(6, 8, figsize=(12, 9))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    from PIL import Image
    for i, item in enumerate(sample_crops):
        path = item["crop_path"]
        img = Image.open(path).convert("RGB")
        
        t_eval = eval_transform(img)
        img_show = denormalize(t_eval, mean, std).permute(1, 2, 0)
        
        axes[i, 0].imshow(img_show)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original")
            
        for j in range(1, 8):
            t_aug = train_transform(img)
            img_aug_show = denormalize(t_aug, mean, std).permute(1, 2, 0)
            axes[i, j].imshow(img_aug_show)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"Aug {j}")
                
    out_dir = Path("reports/figures/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "augmentation_preview.png", dpi=200, bbox_inches="tight")
    plt.close()

    def test_loaders(size, cache, msg_prefix):
        c = copy.deepcopy(cfg)
        c["model"]["img_size"] = size
        c["data"]["cache_in_ram"] = cache
        c["train"]["augment"] = True
        
        train_loader, val_loader, test_loader, info = build_loaders(c)
        
        start = time.perf_counter()
        val_min, val_max = float('inf'), float('-inf')
        batch_shape = None
        
        for batch_idx, (x, y, paths) in enumerate(train_loader):
            if batch_idx == 0:
                batch_shape = x.shape
                val_min = x.min().item()
                val_max = x.max().item()
                if not cache:
                    grid = torchvision.utils.make_grid(x[:16], nrow=4, normalize=True, value_range=(val_min, val_max))
                    plt.figure(figsize=(6, 6))
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.axis("off")
                    plt.savefig(out_dir / f"batch_preview_{size}.png", dpi=200, bbox_inches="tight")
                    plt.close()
            else:
                pass
            
        elapsed = time.perf_counter() - start
        
        print(f"{msg_prefix}: Size {size}, Cache {cache}, Epoch Time {elapsed:.2f}s")
        if not cache:
            print(f"Batch shape: {batch_shape}, Range: [{val_min:.3f}, {val_max:.3f}]")
            print(f"Loader lengths: Train {len(train_loader)}, Val {len(val_loader)}, Test {len(test_loader)}")
        return info, len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)

    print("Testing 64px without cache...")
    info_64, train_n, val_n, test_n = test_loaders(64, False, "Test")
    print("Testing 64px with cache...")
    test_loaders(64, True, "Test")
    
    print("Testing 224px without cache...")
    info_224, _, _, _ = test_loaders(224, False, "Test")
    print("Testing 224px with cache...")
    test_loaders(224, True, "Test")
    
    w = info_64.class_weights
    names = info_64.class_names
    
    sorted_idx = torch.argsort(w)
    print("\nLowest class weights:")
    for i in range(5):
        idx = sorted_idx[i].item()
        print(f"  {names[idx]}: {w[idx]:.3f}")
        
    print("\nHighest class weights:")
    for i in range(1, 6):
        idx = sorted_idx[-i].item()
        print(f"  {names[idx]}: {w[idx]:.3f}")
        
    print("\n===== NOTE THIS FOR PPT =====")
    print(f"Train samples: {train_n}")
    print(f"Val samples: {val_n}")
    print(f"Test samples: {test_n}")
    print(f"Number of classes (K): {info_64.K}")
    print(f"Input sizes evaluated: 64, 224")
    print(f"Normalisation mean: {[round(m, 3) for m in info_64.mean]}")
    print(f"Normalisation std: {[round(s, 3) for s in info_64.std]}")
    print(f"Smallest class weight: {w[sorted_idx[0]].item():.3f}")
    print(f"Largest class weight: {w[sorted_idx[-1]].item():.3f}")
    print("===== END NOTE =====")

if __name__ == '__main__':
    main()
