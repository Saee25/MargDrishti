import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def build_transforms(img_size: int, mean: list[float], std: list[float], train: bool, augment: bool) -> transforms.Compose:
    """
    Build data transforms for training or evaluation.
    
    Args:
        img_size: Target image size for resizing.
        mean: Normalisation mean.
        std: Normalisation standard deviation.
        train: Whether building for training.
        augment: Whether to apply augmentation.
    
    Returns:
        A composed torchvision transform.
    """
    base_transforms = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
    ]

    if train and augment:
        kernel_size = 3 if img_size <= 64 else 5
        aug_transforms = [
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.02),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 1.5))
            ], p=0.2),
            # No flip of any kind is used.
            # Mirror-image signs such as "turn left" and "turn right", or "keep left" and "keep right",
            # would swap meaning if horizontally flipped.
            # Hue variation stays tiny because red and blue carry strict meaning on signs.
        ]
        
        post_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.10))
        ]
        
        return transforms.Compose(base_transforms + aug_transforms + post_transforms)
    else:
        post_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        return transforms.Compose(base_transforms + post_transforms)
