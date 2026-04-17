# Sensitivity Analysis: LLM-as-a-Judge and Text Generation Metrics

## Table 1. Model Configurations

| Parameter | e4dw86nh (MedGemma 1.0) | v93n471z (MedGemma 1.0) | 2s3ll02e (MedGemma 1.5) |
|---|---|---|---|
| **LLM backbone** | google/medgemma-4b-it | google/medgemma-4b-it | google/medgemma-1.5-4b-it |
| **Stage 1 checkpoint** | j4bb0w33 | j4bb0w33 | z8wuml4t |
| **Q-Former layers** | 10 | 10 | 6 |
| **Q-Former heads** | 12 | 12 | 8 |
| **Query tokens** | 32 | 32 | 32 |
| **Codebooks** | 8 | 8 | 8 |
| **Bridge dim** | 768 | 768 | 768 |
| **LoRA r / alpha** | 32 / 64 | 96 / 192 | 32 / 64 |
| **LoRA dropout** | 0.05 | 0.0 | 0.05 |
| **LoRA targets** | q,k,v,o | q,k,v,o,gate,up,down | q,k,v,o |
| **LLM LR** | 5e-5 | 5e-5 | 5e-5 |
| **Adapter LR** | 5e-4 | 5e-4 | 5e-4 |
| **Batch size (eff.)** | 256 | 256 | 256 |
| **Epochs** | 1 | 1 | 1 |
| **Test samples** | 19,904 | 47,873 | 49,776 |

## Table 2. Overall Text Generation Metrics (All Tasks)

| Metric | e4dw86nh (MedGemma 1.0) | v93n471z (MedGemma 1.0) | 2s3ll02e (MedGemma 1.5) |
|---|---|---|---|
| **ROUGE-1** | 0.637 | 0.759 | 0.753 |
| **ROUGE-2** | 0.503 | 0.670 | 0.662 |
| **ROUGE-L** | 0.611 | 0.751 | 0.745 |
| **BLEU-1** | 0.555 | 0.730 | 0.727 |
| **BLEU-4** | 0.440 | 0.629 | 0.624 |
| **METEOR** | 0.590 | 0.767 | 0.766 |
| **n** | 19,904 | 47,873 | 49,776 |

## Table 3. Interpretation Task — Text Generation Metrics

| Metric | e4dw86nh (MedGemma 1.0) | v93n471z (MedGemma 1.0) | 2s3ll02e (MedGemma 1.5) |
|---|---|---|---|
| **ROUGE-1** | 0.692 | 0.689 | 0.699 |
| **ROUGE-2** | 0.546 | 0.546 | 0.558 |
| **ROUGE-L** | 0.673 | 0.671 | 0.681 |
| **BLEU-1** | 0.634 | 0.632 | 0.646 |
| **BLEU-4** | 0.475 | 0.476 | 0.494 |
| **METEOR** | 0.689 | 0.691 | 0.704 |
| **n** | 4,000 | 8,055 | 10,000 |

## Table 4. LLM-as-a-Judge — Overall Scores

| Model | Judge Score | n |
|---|---|---|
| e4dw86nh (MedGemma 1.0, LoRA r=32, q/k/v/o) | **0.707** | 49,776 |
| v93n471z (MedGemma 1.0, LoRA r=96, +MLP targets) | 0.702 | 47,873 |

## Table 5. LLM-as-a-Judge — Per-Category Breakdown

| Category | e4dw86nh (0.707) | v93n471z (0.702) |
|---|---|---|
| **category_pericarditis** | 1.000 (7) | 1.000 (10) |
| **localization_qrs_axis** | 1.000 (3) | 1.000 (3) |
| **ecg_interval** | 0.854 (684) | 0.688 (16) |
| **category_chamber_enlarg.** | 0.856 (223) | 0.848 (275) |
| **lvef** | 0.820 (2,656) | 0.843 (2,656) |
| **category_other** | 0.813 (521) | 0.700 (1,283) |
| **category_rhythm** | 0.805 (4,697) | 0.792 (6,101) |
| **category_conduction** | 0.721 (1,401) | 0.725 (1,553) |
| **category_infarct_ischem.** | 0.701 (1,739) | 0.679 (2,597) |
| **structural_heart_disease** | 0.664 (3,039) | 0.627 (3,039) |
| **afib_risk** | 0.639 (3,121) | 0.603 (3,121) |
| **json_interpretation** | 0.606 (9,595) | 0.758 (9,595) |
| **acs_severity** | 0.593 (1,060) | 0.569 (1,021) |
| **localization_st_elev.** | 0.583 (291) | -- |
| **culprit_artery** | 0.573 (146) | 0.561 (185) |
| **localization_t_wave** | 0.628 (563) | 0.480 (5) |
| **interpretation** | 0.552 (8,935) | 0.612 (8,055) |
| **urgency_assessment** | 0.518 (558) | 0.000 (1) |
| **classification** | 0.498 (10,537) | 0.501 (8,343) |

> Numbers in parentheses indicate per-category sample count. "--" indicates category not present in that evaluation. Category sample counts vary across evaluations due to different test set compositions.

## Table 6. Continuous Metrics (LLM-as-a-Judge Extraction)

| Metric | e4dw86nh | v93n471z |
|---|---|---|
| **Heart Rate MAE (bpm)** | -- | 2.20 |
| **Heart Rate Pearson r** | -- | 0.943 |
| **Heart Rate ICC** | -- | 0.943 |
| **LVEF MAE (%)** | 7.50 | 7.64 |
| **LVEF Pearson r** | -- | 0.552 |

## Table 7. MedGemma 1.0 vs 1.5 — Per-Category Text Metrics (ROUGE-L / BLEU-4 / METEOR)

| Category | e4dw86nh (MedGemma 1.0) | 2s3ll02e (MedGemma 1.5) | Delta ROUGE-L |
|---|---|---|---|
| **interpretation** | 0.673 / 0.475 / 0.689 | 0.681 / 0.494 / 0.704 | +0.008 |
| **json_interpretation** | 0.190 / 0.005 / 0.047 | 0.757 / 0.735 / 0.843 | **+0.567** |
| **classification** | 0.470 / 0.258 / 0.502 | 0.611 / 0.461 / 0.583 | **+0.141** |
| **category_rhythm** | 0.869 / 0.798 / 0.883 | 0.869 / 0.801 / 0.887 | +0.000 |
| **lvef** | 0.819 / 0.632 / 0.844 | 0.826 / 0.658 / 0.870 | +0.007 |
| **category_infarct_ischem.** | 0.749 / 0.649 / 0.737 | 0.745 / 0.644 / 0.732 | -0.004 |
| **category_conduction** | 0.794 / 0.669 / 0.780 | 0.788 / 0.663 / 0.775 | -0.006 |
| **acs_severity** | 0.588 / 0.375 / 0.536 | 0.700 / 0.573 / 0.702 | **+0.112** |
| **structural_heart_disease** | 0.858 / 0.689 / 0.838 | 0.858 / 0.702 / 0.857 | +0.000 |
| **category_chamber_enlarg.** | 0.725 / 0.518 / 0.721 | 0.915 / 0.715 / 0.897 | **+0.190** |

## Table 8. Training Progression — e4dw86nh LLM-as-a-Judge by Step (Validation Set, n~624)

| Step | Overall | Rhythm | Interpretation | Classification | LVEF | JSON Interp |
|---|---|---|---|---|---|---|
| 3,000 | 0.588 | 0.720 | 0.449 | 0.339 | 0.879 | 0.564 |
| 4,000 | 0.629 | 0.806 | 0.472 | 0.377 | 0.867 | 0.555 |
| 5,000 | 0.627 | 0.831 | 0.493 | 0.377 | 0.902 | 0.543 |
| 6,000 | 0.641 | 0.814 | 0.480 | 0.414 | 0.885 | 0.579 |
| 7,000 | 0.630 | 0.810 | 0.494 | 0.359 | 0.912 | 0.586 |
| 8,000 | 0.618 | 0.816 | 0.501 | 0.438 | 0.929 | 0.594 |
| 9,000 | 0.674 | 0.824 | 0.516 | 0.416 | 0.873 | 0.594 |
| 10,000 | 0.676 | 0.824 | 0.497 | 0.430 | 0.889 | 0.587 |
| 11,000 | 0.661 | 0.828 | 0.514 | 0.402 | 0.896 | 0.599 |
| 12,000 | 0.663 | 0.847 | 0.523 | 0.419 | 0.900 | 0.611 |
| 13,000 | 0.665 | 0.843 | 0.523 | 0.447 | 0.894 | 0.611 |
| 14,000 | 0.679 | 0.843 | 0.534 | 0.457 | 0.883 | 0.613 |
