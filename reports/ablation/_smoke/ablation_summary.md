# Ablation Summary

The step that helped the most was **+ BatchNorm (V2)** with a Macro F1 of 0.0246.
The following steps decreased the F1 score: + MargNet (V5, Label Smoothing). Note that a step which does not improve accuracy is still a valid finding in an ablation study.

## Ablation Table

| step              | run_name                 | description                     |   parameters |   best_epoch |   train_acc |   val_acc |   test_acc |   test_macro_f1 |   test_top3_acc |   overfit_gap |   train_time_min |   acc_change_pct_pt |   f1_change_pct_pt |
|:------------------|:-------------------------|:--------------------------------|-------------:|-------------:|------------:|----------:|-----------:|----------------:|----------------:|--------------:|-----------------:|--------------------:|-------------------:|
| custom_v1_plain   | _smoke_custom_v1_plain   | Baseline CNN (V1)               |      2208903 |            2 |      0.0000 |    0.0101 |     0.0273 |          0.0178 |          0.0710 |       -0.0101 |           0.5543 |              0.0000 |             0.0000 |
| custom_v2_bn      | _smoke_custom_v2_bn      | + BatchNorm (V2)                |      2209127 |            2 |      0.0158 |    0.0277 |     0.0401 |          0.0246 |          0.0765 |       -0.0119 |           0.6249 |              1.2750 |             0.6785 |
| custom_v5_margnet | _smoke_custom_v5_margnet | + MargNet (V5, Label Smoothing) |      1191463 |            2 |      0.0176 |    0.0479 |     0.0364 |          0.0138 |          0.0929 |       -0.0303 |           0.7386 |             -0.3643 |            -1.0778 |