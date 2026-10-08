import json
import csv
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from ml.data.transforms import build_transforms
from ml.data.dataset import SignCropDataset

@dataclass
class DataInfo:
    class_names: list[str]
    K: int
    train_counts: dict[int, int]
    class_weights: torch.Tensor
    mean: list[float]
    std: list[float]
    img_size: int

def build_loaders(cfg: dict, smoke: bool = False) -> tuple[DataLoader, DataLoader, DataLoader, DataInfo]:
    """
    Build train, valid and test dataloaders.
    
    Args:
        cfg: The configuration dictionary.
        smoke: If True, limit the number of crops per class for fast testing.
        
    Returns:
        train_loader, val_loader, test_loader, data_info
    """
    processed_dir = Path(cfg["paths"]["processed_dir"])
    
    # Load class map
    class_map_path = processed_dir / "class_map.json"
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map_list = json.load(f)
        
    class_names = [None] * len(class_map_list)
    class_map = {}
    for item in class_map_list:
        cid = item["index"]
        cname = item["slug"]
        class_names[cid] = cname
        class_map[cname] = cid
        
    # Load stats
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
        
    K = stats["k"]
    train_counts = {}
    for cname, count in stats["class_counts_train"].items():
        cid = class_map[cname]
        train_counts[cid] = count
        
    # Calculate class weights
    # w_c = N / (K * n_c)
    # This weights each class inversely proportional to its frequency.
    # N is total samples, K is number of classes, n_c is samples in class c.
    # We clip to [0.5, 5.0] to prevent extreme weights, then scale so the mean is 1.0.
    N = sum(train_counts.values())
    w = []
    for c in range(K):
        n_c = train_counts.get(c, 0)
        if n_c > 0:
            weight = N / (K * n_c)
        else:
            weight = 1.0
        w.append(weight)
        
    w_tensor = torch.tensor(w, dtype=torch.float32)
    w_tensor = torch.clamp(w_tensor, 0.5, 5.0)
    w_tensor = w_tensor / w_tensor.mean()
    
    # Normalisation preset
    norm_preset = cfg["data"]["normalisation"]
    if norm_preset == "dataset":
        mean = stats["mean"]
        std = stats["std"]
    elif norm_preset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown normalisation preset: {norm_preset}")
        
    img_size = cfg["model"]["img_size"]
    
    # Transforms
    train_transform = build_transforms(img_size, mean, std, train=True, augment=cfg["train"]["augment"])
    eval_transform = build_transforms(img_size, mean, std, train=False, augment=False)
    
    # Load manifest
    manifest_path = processed_dir / "manifest.csv"
    crops_dir = Path(cfg["paths"]["crops_dir"])
    manifest = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["class_index"] = int(row["class_index"])
            row["crop_path"] = str(crops_dir / row["crop_path"])
            manifest.append(row)
            
    # Limit per class for smoke mode
    limit_per_class = cfg["smoke"]["max_per_class"] if smoke else None
    
    # Datasets
    cache_in_ram = cfg["data"].get("cache_in_ram", False)
    train_ds = SignCropDataset(manifest, "train", train_transform, limit_per_class, cache_in_ram)
    val_ds = SignCropDataset(manifest, "valid", eval_transform, limit_per_class, cache_in_ram)
    test_ds = SignCropDataset(manifest, "test", eval_transform, limit_per_class, cache_in_ram)
    
    num_workers = cfg["data"]["num_workers"]
        
    batch_size = cfg["train"]["batch_size"]
    persistent_workers = num_workers > 0
    pin_memory = torch.cuda.is_available()
    
    # The train loader shuffles with a seeded generator
    train_gen = torch.Generator()
    train_gen.manual_seed(cfg["project"]["seed"])
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, persistent_workers=persistent_workers, 
        pin_memory=pin_memory, generator=train_gen
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, persistent_workers=persistent_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, persistent_workers=persistent_workers, 
        pin_memory=pin_memory
    )
    
    info = DataInfo(
        class_names=class_names,
        K=K,
        train_counts=train_counts,
        class_weights=w_tensor,
        mean=mean,
        std=std,
        img_size=img_size
    )
    
    return train_loader, val_loader, test_loader, info
