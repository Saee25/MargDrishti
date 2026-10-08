"""
scripts/make_ppt_notes.py
-------------------------
Generates the PPT_NOTES.md based on the results and injects the results table into README.md.
"""
import json
import csv
from pathlib import Path
import re

def main():
    reports_dir = Path("reports")
    backend_artifacts = Path("backend/artifacts")
    ppt_notes = reports_dir / "PPT_NOTES.md"
    readme_path = Path("README.md")
    comp_csv = reports_dir / "comparison/comparison_table.csv"
    
    # Load dataset stats
    try:
        with open(backend_artifacts / "dataset_stats.json") as f:
            stats = json.load(f)
        train_crops = stats.get("train", {}).get("total_crops", "NOT AVAILABLE")
        valid_crops = stats.get("valid", {}).get("total_crops", "NOT AVAILABLE")
        test_crops = stats.get("test", {}).get("total_crops", "NOT AVAILABLE")
        total_classes = stats.get("kept_classes", "NOT AVAILABLE")
    except Exception:
        train_crops, valid_crops, test_crops, total_classes = ("NOT AVAILABLE",) * 4

    # Load metrics
    try:
        with open(backend_artifacts / "metrics.json") as f:
            metrics = json.load(f)
        cnn_metrics = metrics.get("custom_v5_margnet", {})
        rn_frozen = metrics.get("resnet50_frozen", {})
        rn_fine = metrics.get("resnet50_finetune", {})
        
        cnn_acc = f"{cnn_metrics.get('test_metrics', {}).get('accuracy', 0):.4f}" if cnn_metrics else "NOT AVAILABLE"
        rn_fine_acc = f"{rn_fine.get('test_metrics', {}).get('accuracy', 0):.4f}" if rn_fine else "NOT AVAILABLE"
        cnn_f1 = f"{cnn_metrics.get('test_metrics', {}).get('macro_f1', 0):.4f}" if cnn_metrics else "NOT AVAILABLE"
        rn_fine_f1 = f"{rn_fine.get('test_metrics', {}).get('macro_f1', 0):.4f}" if rn_fine else "NOT AVAILABLE"
        
        cnn_params = cnn_metrics.get("parameters", "NOT AVAILABLE")
        rn_params = rn_fine.get("parameters", "NOT AVAILABLE")
        
        cnn_size = cnn_metrics.get("size_mb", "NOT AVAILABLE")
        rn_size = rn_fine.get("size_mb", "NOT AVAILABLE")
        
    except Exception:
        cnn_acc, rn_fine_acc, cnn_f1, rn_fine_f1 = ("NOT AVAILABLE",) * 4
        cnn_params, rn_params, cnn_size, rn_size = ("NOT AVAILABLE",) * 4

    content = f"""# MargDrishti PPT Notes

## Slide 1: Title
- MargDrishti: Indian Traffic Sign Classification
- Custom CNN vs ResNet50

## Slide 2: Problem and motivation
- Accurate classification of traffic signs is critical for ADAS and autonomous driving.

## Slide 3: Research question
- How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs?

## Slide 4: Dataset
- Classes: {total_classes}
- Train crops: {train_crops}
- Valid crops: {valid_crops}
- Test crops: {test_crops}
- Image: `reports/figures/class_distribution.png`

## Slide 5: Preprocessing and augmentation
- Bounding box crops with 10% padding.
- Augmentations: rotation, scale, shear, color jitter.
- Why no flips: flips change the meaning of direction signs.

## Slide 6: CNN building blocks
- Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

## Slide 7: MargNet architecture
- 4 blocks, 32 to 256 filters, Global Average Pooling, Dense layer.
- Parameters: {cnn_params}
- Image: architecture diagram.

## Slide 8: Ablation
- Progression from V1 (plain) to V5 (MargNet).
- Table and chart of incremental improvements.

## Slide 9: ResNet50 and transfer learning
- Stage 1: Frozen backbone, train new head.
- Stage 2: Fine-tune all layers.

## Slide 10: Experimental setup
- Same crops, splits, seed (42), and metrics for fair comparison.

## Slide 11: Results (Headline table)
- Compare Custom CNN vs ResNet50 fine-tuned.

## Slide 12: Results
- Accuracy and F1 scores.

## Slide 13: Efficiency
- MargNet size: {cnn_size} MB
- ResNet size: {rn_size} MB
- Parameter count ratios.

## Slide 14: Error analysis
- McNemar's test, confused pairs.

## Slide 15: Grad-CAM
- Heatmap visualisations of model attention.

## Slide 16: Demo screenshots
- Show local web app predicting test samples.

## Slide 17: Limitations
- Imbalanced classes, limited dataset size.

## Slide 18: Conclusion
- Custom CNN achieves {cnn_acc} accuracy vs ResNet {rn_fine_acc}.

## Slide 19: Future work
- YOLO for object detection, lighter pretrained models.

## Slide 20: Viva questions
- Why no flips? Directional signs.
- Why GAP? Reduces parameters over Flatten.
- Why macro-F1? Treats all classes equally.
- What is a residual connection? Skips layers to help gradients flow.
- Why class weights? Handles imbalance.
- What does McNemar's test do? Compares model disagreement.
- Why BatchNorm? Stabilizes and speeds up training.
- What is transfer learning? Reusing a model trained on a large dataset.

===== NOTE THIS FOR PPT =====
MargNet Accuracy: {cnn_acc}
ResNet Fine-tuned Accuracy: {rn_fine_acc}
MargNet Macro-F1: {cnn_f1}
ResNet Fine-tuned Macro-F1: {rn_fine_f1}
MargNet Parameters: {cnn_params}
ResNet Parameters: {rn_params}
MargNet Size (MB): {cnn_size}
ResNet Size (MB): {rn_size}
Train Crops: {train_crops}
Test Crops: {test_crops}
===== END NOTE =====
"""
    with open(ppt_notes, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Generated {ppt_notes}")
    
    # Read comparison_table.csv and inject into README
    if comp_csv.exists() and readme_path.exists():
        import pandas as pd
        df = pd.read_csv(comp_csv)
        md_table = df.to_markdown(index=False, floatfmt=".4f")
        
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()
            
        new_text = re.sub(
            r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->",
            f"<!-- RESULTS_START -->\n{md_table}\n<!-- RESULTS_END -->",
            readme_text,
            flags=re.DOTALL
        )
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print("Injected results table into README.md")

if __name__ == "__main__":
    main()
