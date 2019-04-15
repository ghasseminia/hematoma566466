After testing, the best parameter used for mortality and expansion modeling are: 
mortality: max_depth = 10, min_samples_leaf = 15, max_leaf_nodes = 4 (gini)
           max_depth = 3, min_samples_leaf = 6, max_leaf_nodes = 4 (entropy)
expansion: max_depth = 10, min_samples_leaf = 6, max_leaf_nodes = 8 (gini)
           max_depth = 3, min_samples_leaf = 8, max_leaf_nodes = 8 (entropy)


----mortality result after t=10000:----
Gini:
average accuracy:  84.93000000000059
average accuracy if choosing the most common:  85.29533333333326
for  15  samples in test, there are on average  0.9915 false_positives
for  15  samples in test, there are on average  1.269 false_negatives
for  15  samples in test, there are on average  0.9367 correct_positives
for  15  samples in test, there are on average  11.8028 correct_negatives
the top ten most important features are:
- Heart Rate  0.5745296002741171
- Acute_IVH_MEAN 0.13112233528851205
- Mortality 0.09767724050751962
- Time Sx onset - CTP (hours) 0.0581779593389712
- Baseline Systolic BP 0.046882902049974993
- Gender 0.0251223305781502
- 2 hour GCS 0.023741175778570944
- Enalapril 0.016086849389144897
-----------------------------------------
Entropy:
average accuracy:  82.05000000000014
average accuracy if choosing the most common:  85.43066666666638
for  15  samples in test, there are on average  1.1477 false_positives
for  15  samples in test, there are on average  1.5448 false_negatives
for  15  samples in test, there are on average  0.6406 correct_positives
for  15  samples in test, there are on average  11.6669 correct_negatives
the top ten most important features are:
- Acute_IVH_MEAN 0.46109005891617855
- Heart Rate  0.21281916186739264
- Baseline Systolic BP 0.08469684132373309
- Anticoag (Y/N) 0.05201698622378152
- Mortality 0.047301242470451164
- Time Sx onset - CTP (hours) 0.03598483307775689
- Baseline GCS 0.021356301249414143
- Gender 0.018241271193128968


----expansion result after t=10000:----
Gini:
average accuracy:  77.58785714285966
average accuracy if choosing the most common:  74.38857142857154
for  14  samples in test, there are on average  1.0885 false_positives
for  14  samples in test, there are on average  2.0492 false_negatives
for  14  samples in test, there are on average  1.5364 correct_positives
for  14  samples in test, there are on average  9.3259 correct_negatives
the top ten most important features are:
- 2 hour NIHSS 0.5936993502180039
- Time Sx Onset - Randomization (hours) 0.10578946179360084
- Baseline NIHSS 0.0734228298476119
- Acute_IPH_vol_MEAN 0.06354800852081374
- Baseline Systolic BP 0.03315251342357231
- Baseline Diastolic BP 0.029959007519679358
- Baseline MAP 0.026626387555978546
- Age 0.020634289056221552
-----------------------------------------
Entropy:
average accuracy:  75.51785714285874
average accuracy if choosing the most common:  74.39214285714337
for  14  samples in test, there are on average  1.22 false_positives
for  14  samples in test, there are on average  2.2075 false_negatives
for  14  samples in test, there are on average  1.3776 correct_positives
for  14  samples in test, there are on average  9.1949 correct_negatives
the top ten most important features are:
- 2 hour NIHSS 0.4328229801118633
- Baseline NIHSS 0.15862976099271694
- Time Sx Onset - Randomization (hours) 0.11207640722485752
- Acute_IPH_vol_MEAN 0.04354677988791732
- Baseline Systolic BP 0.03804463913563131
- Baseline Diastolic BP 0.0356249854650101
- Baseline MAP 0.03205929840530993
- Age 0.0313294846726127