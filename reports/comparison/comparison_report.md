# Detailed Comparison

===== NOTE THIS FOR PPT =====
Headline Table:
                 Model  Test Accuracy  Parameters  CPU Latency (ms)
0              MargNet            0.0     1191463           7.76300
1      ResNet50 frozen            0.0    23653511          81.19845
2  ResNet50 fine-tuned            0.0    23653511          78.25765

Ratios: MargNet has 19.9x fewer params, is 19.8x smaller, and 10.1x faster than ResNet50 FT.
Accuracy Gap: 0.0 percentage points.

Agreement: Both correct 793, MargNet only 5, ResNet FT only 28, Both wrong 9.
McNemar p-value: 0.0001. The difference in accuracy is statistically significant (p = 0.0001).

Bootstrap 95% CIs:
MargNet Acc: [0.9413, 0.9689]
ResNet FT Acc: [0.9737, 0.9916]
Diff Acc: [-0.0419, -0.0144]

MargNet wins 9 classes, ResNet FT wins heavily on 18 classes.
===== END NOTE =====