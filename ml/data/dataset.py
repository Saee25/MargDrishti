import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class SignCropDataset(Dataset):
    """
    Dataset for Indian Traffic Sign crops.
    """
    def __init__(self, manifest: list[dict], split: str, transform, limit_per_class: int = None, cache_in_ram: bool = False):
        """
        Args:
            manifest: List of dicts, each with 'split', 'class_id', 'crop_path'.
            split: Which split to keep (e.g. 'train', 'valid', 'test').
            transform: The composed torchvision transform.
            limit_per_class: For smoke mode, keeps the first N samples per class after a seeded shuffle.
            cache_in_ram: If True, decodes every crop once and keeps it in memory (already resized to the target size, as uint8).
        """
        # Filter for the specific split
        self.samples = [item for item in manifest if item['split'] == split]
        
        if limit_per_class is not None:
            # Seeded shuffle to ensure reproducibility
            rng = random.Random(42)
            rng.shuffle(self.samples)
            
            class_counts = {}
            limited_samples = []
            for item in self.samples:
                cid = item['class_index']
                if class_counts.get(cid, 0) < limit_per_class:
                    limited_samples.append(item)
                    class_counts[cid] = class_counts.get(cid, 0) + 1
            self.samples = limited_samples

        self.transform = transform
        self.cache_in_ram = cache_in_ram
        self.ram_cache = {}
        
        self.resize_transform = None
        if self.cache_in_ram and hasattr(transform, 'transforms'):
            self.resize_transform = transform.transforms[0]
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        item = self.samples[idx]
        path = item['crop_path']
        
        if self.cache_in_ram and idx in self.ram_cache:
            img = self.ram_cache[idx]
        else:
            # Load with Pillow and convert to RGB
            img = Image.open(path).convert("RGB")
            
            if self.cache_in_ram:
                if self.resize_transform is not None:
                    img = self.resize_transform(img)
                self.ram_cache[idx] = img
        
        # Apply transforms
        tensor = self.transform(img)
        
        return tensor, int(item['class_index']), path
