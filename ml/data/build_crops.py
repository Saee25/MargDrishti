import math
from typing import Tuple
from PIL import Image, ImageOps

def clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clamp a bounding box to image dimensions."""
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    return x1, y1, x2 - x1, y2 - y1

def pad_and_clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int, padding_frac: float) -> Tuple[int, int, int, int]:
    """Expand box by padding_frac on each side and clamp to image dimensions."""
    pad_x = w * padding_frac
    pad_y = h * padding_frac
    new_x = x - pad_x
    new_y = y - pad_y
    new_w = w + 2 * pad_x
    new_h = h + 2 * pad_y
    return clamp_box(new_x, new_y, new_w, new_h, img_w, img_h)

def process_crop(img: Image.Image, box: Tuple[int, int, int, int], max_side: int) -> Image.Image:
    """Crop, convert to RGB, and downscale so the longer side is at most max_side."""
    x, y, w, h = box
    crop = img.crop((x, y, x + w, y + h))
    crop = crop.convert('RGB')
    
    # Downscale if longer side exceeds max_side
    cw, ch = crop.size
    if max(cw, ch) > max_side:
        if cw > ch:
            new_w = max_side
            new_h = int(ch * (max_side / cw))
        else:
            new_h = max_side
            new_w = int(cw * (max_side / ch))
        
        # Ensure at least 1px
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    return crop
