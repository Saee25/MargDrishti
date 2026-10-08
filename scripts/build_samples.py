import json
import shutil
import random
from pathlib import Path
import pandas as pd
from ml.config import load_config

def main():
    config = load_config()
    samples_dir = Path(config["paths"]["samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # clear existing
    for f in samples_dir.glob("*"):
        f.unlink()
        
    processed_dir = Path(config["paths"]["processed_dir"])
    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        print("Manifest not found")
        return
        
    manifest = pd.read_csv(manifest_path)
    test_manifest = manifest[manifest['split'] == 'test']
    
    random.seed(42)
    # Pick about 40 crops spread across classes
    classes = test_manifest['class_slug'].unique()
    num_classes = len(classes)
    
    crops_per_class = max(1, 40 // num_classes) if num_classes > 0 else 0
    
    selected_rows = []
    for cls in classes:
        cls_rows = test_manifest[test_manifest['class_slug'] == cls]
        n_sample = min(len(cls_rows), crops_per_class)
        if n_sample > 0:
            selected_rows.extend(cls_rows.sample(n_sample, random_state=42).to_dict('records'))
            
    # If we need more to reach 40
    while len(selected_rows) < 40 and len(selected_rows) < len(test_manifest):
        remaining = test_manifest[~test_manifest['crop_path'].isin([r['crop_path'] for r in selected_rows])]
        if not remaining.empty:
            selected_rows.append(remaining.sample(1, random_state=42).iloc[0].to_dict())
        else:
            break
            
    crops_dir = Path(config["paths"]["crops_dir"])
    samples = []
    
    for idx, row in enumerate(selected_rows):
        src_path = crops_dir / row['crop_path']
        ext = src_path.suffix
        dst_name = f"sample_{idx:03d}{ext}"
        dst_path = samples_dir / dst_name
        
        shutil.copy2(src_path, dst_path)
        samples.append({
            'file_name': dst_name,
            'true_class': row['class_slug'],
            'class_index': row['class_index']
        })
        
    with open(samples_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
        
    print(f"Copied {len(samples)} samples to {samples_dir}")

if __name__ == "__main__":
    main()
