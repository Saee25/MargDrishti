

### Imported Run: custom_v1_plain at 2026-10-08 21:08:22
```text
===== NOTE THIS FOR PPT =====
Run Name       : custom_v1_plain
Model & Variant: custom v1
Description    : variant v1, augment false, class_weights false, label_smoothing 0, dropout 0.5
Parameters     : 2,208,903
Model Size     : 8.43 MB
Epochs Run     : 35 (Best: 27)
Total Time     : 0.9 mins
Device         : cuda

--- Validation ---
Accuracy : 0.9265
Macro F1 : 0.9115

--- Test ---
Accuracy : 0.9401
Top-3 Acc: 0.9725
Macro P  : 0.9394
Macro R  : 0.9370
Macro F1 : 0.9319
Weight F1: 0.9406

Worst F1 Classes: staggered_intersection (0.62), unguarded_level_crossing (0.67), steep_descent (0.77)
Most Confused   : height_limit->staggered_intersection (3), school_ahead->side_road_right (2), all_motor_vehicle_prohibited->overtaking_prohibited (1)
===== END NOTE =====
```


### Imported Run: custom_v2_bn at 2026-10-08 21:08:23
```text
===== NOTE THIS FOR PPT =====
Run Name       : custom_v2_bn
Model & Variant: custom v2
Description    : variant v2, augment false, class_weights false, label_smoothing 0, dropout 0.5
Parameters     : 2,209,127
Model Size     : 8.43 MB
Epochs Run     : 36 (Best: 28)
Total Time     : 1.0 mins
Device         : cuda

--- Validation ---
Accuracy : 0.0900
Macro F1 : 0.0732

--- Test ---
Accuracy : 0.1162
Top-3 Acc: 0.2120
Macro P  : 0.0690
Macro R  : 0.1085
Macro F1 : 0.0736
Weight F1: 0.0775

Worst F1 Classes: all_motor_vehicle_prohibited (0.00), axle_load_limit (0.00), bullock_cart_and_hand_cart_prohibited (0.00)
Most Confused   : speed_limit_30->no_parking (19), school_ahead->no_parking (18), all_motor_vehicle_prohibited->no_parking (17)
===== END NOTE =====
```


### Imported Run: custom_v3_aug at 2026-10-08 21:08:23
```text
===== NOTE THIS FOR PPT =====
Run Name       : custom_v3_aug
Model & Variant: custom v3
Description    : variant v3, augment true, class_weights false, label_smoothing 0, dropout 0.5
Parameters     : 2,209,127
Model Size     : 8.43 MB
Epochs Run     : 9 (Best: 1)
Total Time     : 0.8 mins
Device         : cuda

--- Validation ---
Accuracy : 0.0261
Macro F1 : 0.0173

--- Test ---
Accuracy : 0.0156
Top-3 Acc: 0.0431
Macro P  : 0.0057
Macro R  : 0.0147
Macro F1 : 0.0069
Weight F1: 0.0081

Worst F1 Classes: axle_load_limit (0.00), bullock_cart_and_hand_cart_prohibited (0.00), cattle_ahead (0.00)
Most Confused   : axle_load_limit->u_turn_prohibited (8), compulsary_ahead->narrow_road_ahead (7), compulsary_turn_left_ahead->steep_ascent (7)
===== END NOTE =====
```


### Imported Run: custom_v4_weighted at 2026-10-08 21:08:23
```text
===== NOTE THIS FOR PPT =====
Run Name       : custom_v4_weighted
Model & Variant: custom v4
Description    : variant v4, augment true, class_weights true, label_smoothing 0, dropout 0.5
Parameters     : 2,209,127
Model Size     : 8.43 MB
Epochs Run     : 9 (Best: 1)
Total Time     : 0.8 mins
Device         : cuda

--- Validation ---
Accuracy : 0.0118
Macro F1 : 0.0068

--- Test ---
Accuracy : 0.0120
Top-3 Acc: 0.0491
Macro P  : 0.0092
Macro R  : 0.0082
Macro F1 : 0.0062
Weight F1: 0.0083

Worst F1 Classes: all_motor_vehicle_prohibited (0.00), axle_load_limit (0.00), bullock_cart_and_hand_cart_prohibited (0.00)
Most Confused   : guarded_level_crossing->left_turn_prohibited (10), narrow_road_ahead->left_turn_prohibited (10), u_turn->left_turn_prohibited (10)
===== END NOTE =====
```


### Imported Run: custom_v5_margnet at 2026-10-08 21:08:23
```text
===== NOTE THIS FOR PPT =====
Run Name       : custom_v5_margnet
Model & Variant: custom v5
Description    : variant v5, augment true, class_weights true, label_smoothing 0.1, dropout 0.4
Parameters     : 1,191,463
Model Size     : 4.55 MB
Epochs Run     : 40 (Best: 37)
Total Time     : 3.4 mins
Device         : cuda

--- Validation ---
Accuracy : 0.9621
Macro F1 : 0.9589

--- Test ---
Accuracy : 0.9605
Top-3 Acc: 0.9844
Macro P  : 0.9554
Macro R  : 0.9588
Macro F1 : 0.9545
Weight F1: 0.9606

Worst F1 Classes: steep_descent (0.77), y_intersection (0.77), staggered_intersection (0.80)
Most Confused   : all_motor_vehicle_prohibited->truck_prohibited (2), bullock_cart_and_hand_cart_prohibited->overtaking_prohibited (2), height_limit->y_intersection (2)
===== END NOTE =====
```


### Imported Run: resnet50_finetune at 2026-10-08 22:41:00
```text
===== NOTE THIS FOR PPT =====
Run Name       : resnet50_finetune
Model & Variant: resnet50 default
Description    : 
Parameters     : 23,653,511
Model Size     : 90.23 MB
Epochs Run     : 9 (Best: 5)
Total Time     : 5.5 mins
Device         : cuda

--- Validation ---
Accuracy : 0.9858
Macro F1 : 0.9834

--- Test ---
Accuracy : 0.9832
Top-3 Acc: 0.9964
Macro P  : 0.9799
Macro R  : 0.9805
Macro F1 : 0.9786
Weight F1: 0.9832

Worst F1 Classes: steep_descent (0.80), length_limit (0.83), bullock_cart_and_hand_cart_prohibited (0.86)
Most Confused   : bullock_cart_and_hand_cart_prohibited->pedestrian_prohibited (2), all_motor_vehicle_prohibited->truck_prohibited (1), compulsary_sound_horn->horn_prohibited (1)
===== END NOTE =====
```


### Imported Run: resnet50_frozen at 2026-10-08 22:41:00
```text
===== NOTE THIS FOR PPT =====
Run Name       : resnet50_frozen
Model & Variant: resnet50 default
Description    : 
Parameters     : 23,653,511
Model Size     : 90.23 MB
Epochs Run     : 6 (Best: 6)
Total Time     : 2.8 mins
Device         : cuda

--- Validation ---
Accuracy : 0.5924
Macro F1 : 0.5705

--- Test ---
Accuracy : 0.5880
Top-3 Acc: 0.8000
Macro P  : 0.6421
Macro R  : 0.5961
Macro F1 : 0.5750
Weight F1: 0.5844

Worst F1 Classes: u_turn_prohibited (0.11), side_road_right (0.15), roundabout (0.17)
Most Confused   : speed_limit_30->speed_limit_50 (12), left_turn_prohibited->horn_prohibited (8), falling_rocks->unguarded_level_crossing (6)
===== END NOTE =====
```
