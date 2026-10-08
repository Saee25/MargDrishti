# Ablation Summary

The step that helped the most was **+ MargNet (V5, Label Smoothing)** with a Macro F1 of 0.9545.
The following steps decreased the F1 score: + BatchNorm (V2), + Augmentation (V3), + Class Weights (V4). Note that a step which does not improve accuracy is still a valid finding in an ablation study.

## Ablation Table

| step               | run_name           | description                     |   parameters |   best_epoch |   train_acc |   val_acc |   test_acc |   test_macro_f1 |   test_top3_acc |   overfit_gap |   train_time_min |   acc_change_pct_pt |   f1_change_pct_pt |
|:-------------------|:-------------------|:--------------------------------|-------------:|-------------:|------------:|----------:|-----------:|----------------:|----------------:|--------------:|-----------------:|--------------------:|-------------------:|
| custom_v1_plain    | custom_v1_plain    | Baseline CNN (V1)               |      2208903 |           27 |      0.9908 |    0.9265 |     0.9401 |          0.9319 |          0.9725 |        0.0643 |           0.9374 |              0.0000 |             0.0000 |
| custom_v2_bn       | custom_v2_bn       | + BatchNorm (V2)                |      2209127 |           28 |      0.0481 |    0.0900 |     0.1162 |          0.0736 |          0.2120 |       -0.0420 |           0.9528 |            -82.3952 |           -85.8351 |
| custom_v3_aug      | custom_v3_aug      | + Augmentation (V3)             |      2209127 |            1 |      0.0123 |    0.0261 |     0.0156 |          0.0069 |          0.0431 |       -0.0138 |           0.8346 |            -10.0599 |            -6.6630 |
| custom_v4_weighted | custom_v4_weighted | + Class Weights (V4)            |      2209127 |            1 |      0.0113 |    0.0118 |     0.0120 |          0.0062 |          0.0491 |       -0.0006 |           0.8251 |             -0.3593 |            -0.0752 |
| custom_v5_margnet  | custom_v5_margnet  | + MargNet (V5, Label Smoothing) |      1191463 |           37 |      0.9625 |    0.9621 |     0.9605 |          0.9545 |          0.9844 |        0.0004 |           3.4443 |             94.8503 |            94.8281 |