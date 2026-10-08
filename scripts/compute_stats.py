import json
from pathlib import Path
from ml.config import load_config
from ml.data.stats import compute_dataset_stats

def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    manifest_path = processed_dir / "manifest.csv"
    crops_dir = Path(config["paths"]["crops_dir"])
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Run build_crops first.")
        return
        
    stats = compute_dataset_stats(
        manifest_path=manifest_path,
        crops_dir=crops_dir,
        img_size=config["model"]["img_size"]
    )
    
    stats_out = processed_dir / "stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Dataset stats computed and saved to {stats_out}")
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    print(f"K (classes): {stats['k']}")

if __name__ == "__main__":
    main()
