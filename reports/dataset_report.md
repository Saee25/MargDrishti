# Dataset Report

## Overview
This report summarizes the dataset processing, statistics, and exploration findings.

### Dataset Stats
- **Classes Kept (K):** 71
- **Classes Dropped:** 4
- **Leakage Check:** No leakage detected across splits.
- **Channel Mean:** [0.5892273783683777, 0.5254560112953186, 0.537609338760376]
- **Channel Std:** [0.3375380337238312, 0.3405824303627014, 0.34753215312957764]

### Splits
{'train': 2932, 'test': 835, 'valid': 422}

### Dropped Classes
|    | class               |   train |   valid |   test |   total | reason              |
|---:|:--------------------|--------:|--------:|-------:|--------:|:--------------------|
|  0 | cycle_prohibited    |      28 |       3 |      8 |      39 | Total crops 39 < 40 |
|  1 | road_widens_ahead   |      26 |       5 |      8 |      39 | Total crops 39 < 40 |
|  2 | speed_limit_15      |      23 |       4 |     11 |      38 | Total crops 38 < 40 |
|  3 | straight_prohibited |      18 |       1 |      3 |      22 | Total crops 22 < 40 |

## Observations
1. **Scene Composition:** Based on the box area share distribution (average share 0.695), these images primarily represent close-up photos.
2. **Class Imbalance:** There is a significant class imbalance with the largest class being 2.24x larger than the smallest kept class in the training set.
3. **Small Signs:** Many crops are significantly smaller than the 64x64 target, indicating the model will need to handle low-resolution features effectively.
4. **Similar Classes:** There are numerous look-alike classes (e.g., left/right turn pairs, speed limit increments) that will challenge the model's fine-grained classification capabilities.

## Figures
- **Class Distribution:** `reports/figures/dataset/class_distribution.png`
- **Split Sizes:** `reports/figures/dataset/split_sizes.png`
- **Crop Sizes:** `reports/figures/dataset/crop_size_hist.png`
- **Box Area Share:** `reports/figures/dataset/box_area_share_hist.png`
- **Samples:** `reports/figures/dataset/sample_grid.png`
- **Smallest Crops:** `reports/figures/dataset/smallest_crops.png`
