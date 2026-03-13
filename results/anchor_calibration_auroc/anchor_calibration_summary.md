# Anchor Calibration Summary

| selection_mode | model | prompt | n | baseline_test_auroc | calibrated_test_auroc | baseline_test_ece | calibrated_test_ece | baseline_test_brier | calibrated_test_brier | recommended_self_anchors | recommended_adoption_anchors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| auroc_first | Qwen/Qwen3-8B | hierarchical | 378 | 0.6803 | 0.6731 | 0.1199 | 0.1376 | 0.1536 | 0.1679 | VL=0.144, L=0.277, M=0.553, H=0.831, VH=1.000 | VL=0.092, L=0.224, M=0.553, H=0.786, VH=0.999 |
| auroc_first | Qwen/Qwen3-8B | sequential | 378 | 0.7103 | 0.7163 | 0.1980 | 0.1068 | 0.1861 | 0.1635 | VL=0.001, L=0.286, M=0.421, H=0.748, VH=1.000 | VL=0.001, L=0.230, M=0.505, H=0.679, VH=0.978 |
| auroc_first | google/gemma-3-12b-it | hierarchical | 378 | 0.5405 | 0.5528 | 0.3377 | 0.1590 | 0.3218 | 0.2120 | VL=0.118, L=0.188, M=0.292, H=0.609, VH=0.970 | VL=0.167, L=0.372, M=0.622, H=0.752, VH=0.861 |
| auroc_first | google/gemma-3-12b-it | sequential | 378 | 0.6850 | 0.6798 | 0.1635 | 0.1133 | 0.2179 | 0.2048 | VL=0.095, L=0.305, M=0.515, H=0.804, VH=1.000 | VL=0.142, L=0.325, M=0.515, H=0.773, VH=1.000 |
