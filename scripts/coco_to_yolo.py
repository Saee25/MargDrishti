import json
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

def convert_coco_to_yolo():
    print("Starting COCO to YOLO conversion...")
    raw_dir = Path("data/raw/Indian Traffic SignBoards.v3-final-correct-label.coco")
    yolo_dir = Path("data/processed/yolo")
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class map
    class_map_path = Path("data/processed/class_map.json")
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
        
    orig_to_new = {c["original_category_id"]: c["index"] for c in class_map}
    class_names = [c["display_name"] for c in class_map]

    splits = ["train", "valid", "test"]
    
    for split in splits:
        print(f"Processing {split} split...")
        split_dir = yolo_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        coco_json = raw_dir / split / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"Warning: {coco_json} not found.")
            continue
            
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
            
        images = {img["id"]: img for img in coco_data["images"]}
        
        # Group annotations by image
        img_to_anns = {img_id: [] for img_id in images}
        for ann in coco_data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
            
        for img_id, img_info in images.items():
            img_anns = img_to_anns[img_id]
            valid_anns = [a for a in img_anns if a["category_id"] in orig_to_new]
            
            if not valid_anns:
                continue # Skip images with no valid annotations
                
            # Copy image
            src_img = raw_dir / split / img_info["file_name"]
            dst_img = split_dir / "images" / img_info["file_name"]
            if not dst_img.exists():
                 shutil.copy2(src_img, dst_img)
            
            # Create label
            label_file = split_dir / "labels" / (Path(img_info["file_name"]).stem + ".txt")
            with open(label_file, 'w') as f:
                for ann in valid_anns:
                    cat_id = orig_to_new[ann["category_id"]]
                    x, y, w, h = ann["bbox"]
                    img_w = img_info["width"]
                    img_h = img_info["height"]
                    
                    # YOLO format: center_x, center_y, width, height (normalized)
                    cx = (x + w/2) / img_w
                    cy = (y + h/2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    
                    f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Write dataset YAML
    yaml_data = {
        "path": str(yolo_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"YOLO dataset created at {yolo_dir}")

def check_labels():
    print("Checking labels...")
    yolo_dir = Path("data/processed/yolo")
    train_images_dir = yolo_dir / "train" / "images"
    train_labels_dir = yolo_dir / "train" / "labels"
    
    if not train_images_dir.exists():
        print("Train images not found.")
        return
        
    images = list(train_images_dir.glob("*.jpg"))
    if not images:
        print("No images found.")
        return
        
    sample = random.sample(images, min(12, len(images)))
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    with open("data/processed/class_map.json", 'r') as f:
        class_map = json.load(f)
    names = {c["index"]: c["display_name"] for c in class_map}
    
    for i, img_path in enumerate(sample):
        ax = axes[i]
        img = Image.open(img_path)
        ax.imshow(img)
        
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cat_id = int(parts[0])
                        cx, cy, nw, nh = map(float, parts[1:])
                        
                        w = nw * img.width
                        h = nh * img.height
                        x = (cx * img.width) - (w / 2)
                        y = (cy * img.height) - (h / 2)
                        
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='purple', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, y-5, names.get(cat_id, str(cat_id)), color='purple', fontsize=8, backgroundcolor='white')
                        
        ax.axis('off')
        
    plt.tight_layout()
    out_dir = Path("reports/figures/detection")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "label_check.png", dpi=200)
    print(f"Label check saved to {out_dir / 'label_check.png'}")

if __name__ == '__main__':
    convert_coco_to_yolo()
    check_labels()
