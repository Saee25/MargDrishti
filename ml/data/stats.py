import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any

def compute_dataset_stats(manifest_path: str | Path, crops_dir: str | Path, img_size: int = 64) -> Dict[str, Any]:
    """Compute per-channel mean and std of training crops resized to img_size x img_size."""
    manifest = pd.read_csv(manifest_path)
    train_manifest = manifest[manifest['split'] == 'train']
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    crops_dir = Path(crops_dir)
    
    pixel_sum = torch.zeros(3)
    pixel_sq_sum = torch.zeros(3)
    count = 0
    
    class_counts = train_manifest['class_slug'].value_counts().to_dict()
    k = len(class_counts)
    
    for _, row in tqdm(train_manifest.iterrows(), total=len(train_manifest), desc="Computing stats"):
        crop_path = crops_dir / row['crop_path']
        img = Image.open(crop_path).convert('RGB')
        tensor = transform(img) # 3 x H x W
        
        pixel_sum += tensor.sum(dim=[1, 2])
        pixel_sq_sum += (tensor ** 2).sum(dim=[1, 2])
        count += (tensor.shape[1] * tensor.shape[2])
        
    mean = pixel_sum / count
    variance = (pixel_sq_sum / count) - (mean ** 2)
    std = torch.sqrt(variance)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'k': k,
        'class_counts_train': class_counts
    }
