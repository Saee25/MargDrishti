# MargDrishti PPT Notes

## Slide 1: Title
- MargDrishti: Indian Traffic Sign Classification
- Custom CNN vs ResNet50

## Slide 2: Problem and motivation
- Accurate classification of traffic signs is critical for ADAS and autonomous driving.

## Slide 3: Research question
- How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs?

## Slide 4: Dataset
- Classes: NOT AVAILABLE
- Train crops: NOT AVAILABLE
- Valid crops: NOT AVAILABLE
- Test crops: NOT AVAILABLE
- Image: `reports/figures/class_distribution.png`

## Slide 5: Preprocessing and augmentation
- Bounding box crops with 10% padding.
- Augmentations: rotation, scale, shear, color jitter.
- Why no flips: flips change the meaning of direction signs.

## Slide 6: CNN building blocks
- Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

## Slide 7: MargNet architecture
- 4 blocks, 32 to 256 filters, Global Average Pooling, Dense layer.
- Parameters: NOT AVAILABLE
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
- MargNet size: NOT AVAILABLE MB
- ResNet size: 90.23098373413086 MB
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
- Custom CNN achieves NOT AVAILABLE accuracy vs ResNet 0.9832.

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
MargNet Accuracy: NOT AVAILABLE
ResNet Fine-tuned Accuracy: 0.9832
MargNet Macro-F1: NOT AVAILABLE
ResNet Fine-tuned Macro-F1: 0.9786
MargNet Parameters: NOT AVAILABLE
ResNet Parameters: 23653511
MargNet Size (MB): NOT AVAILABLE
ResNet Size (MB): 90.23098373413086
Train Crops: NOT AVAILABLE
Test Crops: NOT AVAILABLE
===== END NOTE =====
