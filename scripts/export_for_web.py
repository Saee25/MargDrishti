import os
import shutil
import json
from pathlib import Path
import pandas as pd

def safe_copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)

def main():
    out_dir = Path("backend/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    
    # 1. Models
    safe_copy("experiments/custom_v5_margnet/best.pt", out_dir / "models/margnet.pt")
    safe_copy("experiments/resnet50_finetune/best.pt", out_dir / "models/resnet50.pt")
    
    # 2. JSONs
    safe_copy("data/processed/class_map.json", out_dir / "class_map.json")
    safe_copy("data/processed/stats.json", out_dir / "dataset_stats.json")
    
    # 3. Per-model metrics
    metrics = {}
    for run, name in [("custom_v5_margnet", "margnet"), ("resnet50_frozen", "resnet50_frozen"), ("resnet50_finetune", "resnet50_finetune")]:
        sum_path = Path(f"experiments/{run}/summary.json")
        if sum_path.exists():
            with open(sum_path, 'r') as f:
                metrics[name] = json.load(f)
                
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 4. Comparison, Benchmark, Ablation
    safe_copy("reports/comparison/comparison_table.json", out_dir / "comparison.json")
    safe_copy("reports/comparison/benchmark.json", out_dir / "benchmark.json")
    safe_copy("reports/ablation/ablation.json", out_dir / "ablation.json")
    
    # 5. Layer descriptions
    safe_copy("reports/model_summaries/custom_v5_layers.json", out_dir / "custom_v5_layers.json")
    safe_copy("reports/model_summaries/resnet50_default_layers.json", out_dir / "resnet50_default_layers.json")
    
    # 6. Figures
    for fig in Path("reports/comparison").glob("*.png"):
        safe_copy(fig, out_dir / f"figures/{fig.name}")
        
    for fig in Path("experiments/custom_v5_margnet").glob("*.png"):
        safe_copy(fig, out_dir / f"figures/margnet_{fig.name}")
        
    for fig in Path("experiments/resnet50_finetune").glob("*.png"):
        safe_copy(fig, out_dir / f"figures/resnet50_{fig.name}")
        
    # 7. Samples
    sample_dir = Path("data/samples")
    if sample_dir.exists():
        samples_list = []
        for img_path in sample_dir.glob("*.jpg"):
            safe_copy(img_path, out_dir / f"samples/{img_path.name}")
            samples_list.append(img_path.name)
        with open(out_dir / "samples/samples.json", 'w') as f:
            json.dump({"samples": samples_list}, f, indent=2)
            
    # 8. Manifest
    import datetime
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "available_artifacts": [p.name for p in out_dir.iterdir() if p.is_file()],
        "models": [p.name for p in (out_dir / "models").iterdir()],
        "figures": [p.name for p in (out_dir / "figures").iterdir()]
    }
    with open(out_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print("Export for web complete.")

if __name__ == '__main__':
    main()
