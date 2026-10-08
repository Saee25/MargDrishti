import argparse
import hashlib
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageOps
import pandas as pd

from ml.config import load_config
from ml.data.coco import find_coco_splits, load_coco_split
from ml.data.build_crops import clamp_box, pad_and_clamp_box, process_crop

def compute_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Build cropped dataset from COCO annotations.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing crops directory.")
    parser.add_argument("--limit", type=int, help="Limit number of images processed per split.")
    parser.add_argument("--out", type=str, help="Output directory override for crops.")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir = Path(args.out) if args.out else Path(config["paths"]["crops_dir"])
    if out_dir.exists():
        if args.force:
            shutil.rmtree(out_dir)
        else:
            print(f"Output directory {out_dir} exists. Use --force to overwrite.")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = find_coco_splits(raw_dir)
    if not splits:
        print(f"No COCO splits found in {raw_dir}")
        return

    # Leakage check logic
    # source image file is in two splits (by relative path or name)
    # MD5 of every source image and count exact duplicates across splits
    img_name_to_splits = defaultdict(set)
    img_md5_to_splits = defaultdict(set)
    img_md5_to_paths = defaultdict(list)
    
    # Store data for processing
    split_data = {}
    for split_name, (json_path, images_dir) in splits.items():
        images_df, anns_df, cats_df = load_coco_split(json_path, images_dir)
        split_data[split_name] = {
            'images': images_df,
            'anns': anns_df,
            'cats': cats_df,
            'images_dir': images_dir
        }
        
    print("Performing leakage checks...")
    for split_name, data in split_data.items():
        images_df = data['images']
        images_dir = data['images_dir']
        
        # Limit if requested for MD5 check? No, limit affects processing later.
        # But we want to only check the images we process? The instruction says:
        # "leakage checks: assert that no source image file is in two splits; compute an MD5 of every source image and count exact duplicates across splits."
        for _, row in images_df.iterrows():
            fname = row['file_name']
            fpath = images_dir / fname
            if not fpath.exists():
                continue
            
            img_name_to_splits[fname].add(split_name)
            md5_hash = compute_md5(fpath)
            img_md5_to_splits[md5_hash].add(split_name)
            img_md5_to_paths[md5_hash].append(str(fpath))

    # Cross-split duplicates
    cross_split_dups = []
    for md5_hash, splits_set in img_md5_to_splits.items():
        if len(splits_set) > 1:
            cross_split_dups.append({
                'md5': md5_hash,
                'splits': ', '.join(sorted(list(splits_set))),
                'paths': ', '.join(img_md5_to_paths[md5_hash])
            })
            
    name_cross_split = [fname for fname, splits_set in img_name_to_splits.items() if len(splits_set) > 1]
    
    if cross_split_dups or name_cross_split:
        print("Leakage found!")
        dup_df = pd.DataFrame(cross_split_dups)
        dup_path = reports_dir / "cross_split_duplicates.csv"
        dup_df.to_csv(dup_path, index=False)
        print(f"Cross-split duplicates written to {dup_path}")
        print("Move nothing, tell the user.")
        return
        
    print("Leakage checks passed. Processing crops...")
    
    manifest_rows = []
    class_counts = defaultdict(lambda: defaultdict(int)) # class_slug -> split -> count
    slug_to_cat = {} # map class slug to original category data
    
    stats_tracking = {
        'source_images': defaultdict(int),
        'annotations_read': 0,
        'crops_written': defaultdict(int),
        'boxes_skipped_small': 0,
    }

    # First pass: collect categories
    raw_classes = set()
    for split_name, data in split_data.items():
        for _, row in data['cats'].iterrows():
            slug = row['slug']
            raw_classes.add(slug)
            if slug not in slug_to_cat:
                slug_to_cat[slug] = {
                    'original_category_id': row['category_id'],
                    'display_name': row['display_name'],
                    'slug': slug
                }
                
    for split_name, data in splits.items():
        images_df = split_data[split_name]['images']
        anns_df = split_data[split_name]['anns']
        images_dir = split_data[split_name]['images_dir']
        
        if args.limit:
            images_df = images_df.head(args.limit)
            valid_image_ids = set(images_df['image_id'])
            anns_df = anns_df[anns_df['image_id'].isin(valid_image_ids)]
            
        stats_tracking['source_images'][split_name] = len(images_df)
        stats_tracking['annotations_read'] += len(anns_df)
        
        img_id_to_fname = dict(zip(images_df['image_id'], images_df['file_name']))
        
        # We need original image dimensions for padding/clamping, let's group annotations by image
        anns_by_img = defaultdict(list)
        for _, row in anns_df.iterrows():
            anns_by_img[row['image_id']].append(row)
            
        for img_id, fname in tqdm(img_id_to_fname.items(), desc=f"Processing {split_name}"):
            fpath = images_dir / fname
            if not fpath.exists():
                continue
                
            try:
                img = Image.open(fpath)
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(f"Error opening {fpath}: {e}")
                continue
                
            img_w, img_h = img.size
            img_area = img_w * img_h
            
            for ann in anns_by_img[img_id]:
                cat_id = ann['category_id']
                cat_row = split_data[split_name]['cats'][split_data[split_name]['cats']['category_id'] == cat_id]
                if cat_row.empty:
                    continue
                cat_slug = cat_row.iloc[0]['slug']
                
                x, y = ann['bbox_x'], ann['bbox_y']
                w, h = ann['bbox_w'], ann['bbox_h']
                
                # clamping the original box
                cx, cy, cw, ch = clamp_box(x, y, w, h, img_w, img_h)
                
                if min(cw, ch) < config["data"]["min_box_side"]:
                    stats_tracking['boxes_skipped_small'] += 1
                    continue
                    
                # pad and clamp
                px, py, pw, ph = pad_and_clamp_box(cx, cy, cw, ch, img_w, img_h, config["data"]["crop_padding"])
                
                # process crop
                crop_img = process_crop(img, (px, py, pw, ph), config["data"]["max_crop_side"])
                
                # save crop
                img_stem = Path(fname).stem
                ann_id = ann['annotation_id']
                crop_fname = f"{img_stem}__{ann_id}.jpg"
                crop_rel_path = f"{split_name}/{cat_slug}/{crop_fname}"
                crop_out_path = out_dir / split_name / cat_slug / crop_fname
                
                crop_out_path.parent.mkdir(parents=True, exist_ok=True)
                crop_img.save(crop_out_path, "JPEG", quality=95)
                
                stats_tracking['crops_written'][split_name] += 1
                class_counts[cat_slug][split_name] += 1
                
                # share of road image
                box_area = pw * ph
                share = box_area / img_area if img_area > 0 else 0
                
                manifest_rows.append({
                    'crop_path': crop_rel_path.replace("\\", "/"),
                    'split': split_name,
                    'class_slug': cat_slug,
                    'source_image': fname,
                    'annotation_id': ann_id,
                    'original_box': f"[{x}, {y}, {w}, {h}]",
                    'crop_w': crop_img.width,
                    'crop_h': crop_img.height,
                    'area_share': share
                })

    # Drop classes
    dropped_classes = []
    kept_classes = []
    
    for slug in sorted(raw_classes): # sorting raw classes too for deterministic behaviour
        counts = class_counts[slug]
        total = sum(counts.values())
        train_count = counts.get('train', 0)
        test_count = counts.get('test', 0)
        
        reason = None
        if total < config["data"]["min_crops_per_class"]:
            reason = f"Total crops {total} < {config['data']['min_crops_per_class']}"
        elif train_count == 0:
            reason = "Zero crops in train"
        elif test_count == 0:
            reason = "Zero crops in test"
            
        if reason:
            dropped_classes.append({
                'class': slug,
                'train': train_count,
                'valid': counts.get('valid', 0),
                'test': test_count,
                'total': total,
                'reason': reason
            })
        else:
            kept_classes.append(slug)
            
    if dropped_classes:
        pd.DataFrame(dropped_classes).to_csv(reports_dir / "dropped_classes.csv", index=False)
        
    # Write class map
    # kept classes in order, each with index, slug, display name and original category id
    # Indices run 0 to K-1 in alphabetical order of slug
    kept_classes = sorted(kept_classes)
    class_map = []
    for idx, slug in enumerate(kept_classes):
        cat_info = slug_to_cat[slug]
        class_map.append({
            'index': idx,
            'slug': slug,
            'display_name': cat_info['display_name'],
            'original_category_id': cat_info['original_category_id']
        })
        
    with open(processed_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
        
    # Filter manifest to only kept classes and add class_index
    slug_to_index = {slug: idx for idx, slug in enumerate(kept_classes)}
    final_manifest = []
    for row in manifest_rows:
        if row['class_slug'] in slug_to_index:
            row['class_index'] = slug_to_index[row['class_slug']]
            final_manifest.append(row)
            
    pd.DataFrame(final_manifest).to_csv(processed_dir / "manifest.csv", index=False)
    
    # Calculate PPT stats
    k = len(kept_classes)
    largest_class = max(kept_classes, key=lambda s: class_counts[s].get('train', 0)) if kept_classes else None
    smallest_class = min(kept_classes, key=lambda s: class_counts[s].get('train', 0)) if kept_classes else None
    
    largest_train = class_counts[largest_class].get('train', 0) if largest_class else 0
    smallest_train = class_counts[smallest_class].get('train', 0) if smallest_class else 0
    imbalance_ratio = largest_train / smallest_train if smallest_train > 0 else 0
    
    print("\n--- NOTE THIS FOR PPT ---")
    print(f"Source images per split: {dict(stats_tracking['source_images'])}")
    print(f"Annotations read: {stats_tracking['annotations_read']}")
    print(f"Crops written per split: {dict(stats_tracking['crops_written'])}")
    print(f"Boxes skipped as too small: {stats_tracking['boxes_skipped_small']}")
    print(f"Classes in raw data: {len(raw_classes)}")
    print(f"Classes kept (K): {k}")
    print(f"Classes dropped: {len(dropped_classes)}")
    if k > 0:
        print(f"Largest kept class (train): {largest_class} ({largest_train} crops)")
        print(f"Smallest kept class (train): {smallest_class} ({smallest_train} crops)")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print("-------------------------\n")

if __name__ == "__main__":
    main()
