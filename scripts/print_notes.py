import yaml
import torch
import copy
from ml.data.loaders import build_loaders

def main():
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["model"]["img_size"] = 64
    cfg["data"]["cache_in_ram"] = False
    cfg["train"]["augment"] = False
    
    train_loader, val_loader, test_loader, info_64 = build_loaders(cfg)
    
    train_n = len(train_loader.dataset)
    val_n = len(val_loader.dataset)
    test_n = len(test_loader.dataset)
    
    w = info_64.class_weights
    sorted_idx = torch.argsort(w)
    
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
