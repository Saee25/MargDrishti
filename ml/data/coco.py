import json
import re
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

def clean_class_name(name: str) -> Tuple[str, str]:
    """
    Given an original category name, return (display_name, slug).
    - display_name: trimmed, single spaces, consistent case (kept as is mostly but trimmed).
    - slug: filesystem-safe, lowercase, alphanumeric and underscores.
    """
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return name, slug

def find_coco_splits(raw_dir: str | Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Find COCO splits in the given raw data directory.
    Returns a dict mapping split name to (json_path, images_dir).
    """
    raw_dir = Path(raw_dir)
    splits = {}
    
    # Check direct subdirectories for coco JSONs
    for json_path in raw_dir.rglob("*.json"):
        if "coco" in json_path.name.lower():
            images_dir = json_path.parent
            split_name = images_dir.name
            splits[split_name] = (json_path, images_dir)
            
    return splits

def load_coco_split(json_path: str | Path, images_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a COCO split and return tidy DataFrames for images, annotations, and categories.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    images_df = pd.DataFrame(data.get('images', []))
    if not images_df.empty:
        images_df = images_df[['id', 'file_name', 'width', 'height']]
        images_df = images_df.rename(columns={'id': 'image_id'})
    else:
        images_df = pd.DataFrame(columns=['image_id', 'file_name', 'width', 'height'])
        
    ann_list = data.get('annotations', [])
    if ann_list:
        anns = []
        for ann in ann_list:
            bbox = ann['bbox']
            # bbox is [x, y, width, height]
            anns.append({
                'annotation_id': ann['id'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox_x': bbox[0],
                'bbox_y': bbox[1],
                'bbox_w': bbox[2],
                'bbox_h': bbox[3],
                'area': ann.get('area', bbox[2] * bbox[3])
            })
        anns_df = pd.DataFrame(anns)
    else:
        anns_df = pd.DataFrame(columns=['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'area'])
        
    cats_list = data.get('categories', [])
    if cats_list:
        cats = []
        for cat in cats_list:
            disp, slug = clean_class_name(cat['name'])
            cats.append({
                'category_id': cat['id'],
                'name': cat['name'],
                'display_name': disp,
                'slug': slug
            })
        cats_df = pd.DataFrame(cats)
    else:
        cats_df = pd.DataFrame(columns=['category_id', 'name', 'display_name', 'slug'])
        
    # Validation
    if not images_df.empty and not anns_df.empty:
        missing_images = set(anns_df['image_id']) - set(images_df['image_id'])
        if missing_images:
            raise ValueError(f"Annotations reference missing image_ids: {missing_images}")
            
    return images_df, anns_df, cats_df
