import argparse
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    project_dir = Path("experiments")
    name = "yolo_nano_smoke" if args.smoke else "yolo_nano"
    
    # Define timing probe start
    start_time = time.time()
    
    model = YOLO("yolo11n.pt")
    
    epochs = 1 if args.smoke else 30
    patience = 8
    
    fraction = 0.1 if args.smoke else 1.0
    
    import yaml
    
    yaml_path = Path("data/processed/yolo/dataset.yaml")
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        
    yaml_data["path"] = str(yaml_path.parent.absolute())
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=epochs,
        patience=patience,
        batch=16,
        imgsz=640,
        seed=42,
        project=str(project_dir.absolute()),
        name=name,
        fliplr=0.0,
        flipud=0.0,
        exist_ok=True,
        fraction=fraction,
    )
    
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Training took {duration:.2f} seconds")
    
    if args.smoke:
        # Estimate full training time based on smoke test
        # Assume 1 epoch takes X seconds. 30 epochs will take roughly 30 * X.
        # But this is smoke test, wait, smoke test just trains 1 epoch on full data or a small fraction?
        # The prompt says: "It supports --smoke (1 epoch on a small fraction)."
        # Ultralytics doesn't easily support "small fraction" without modifying the dataset YAML or using fractional datasets.
        # But YOLO has fraction argument? `fraction=0.1` maybe? Let me check ultralytics docs or just use 1 epoch and small fraction if possible.
        pass

if __name__ == '__main__':
    main()
