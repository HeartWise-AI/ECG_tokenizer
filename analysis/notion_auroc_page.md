# Per-Diagnosis AUROC Analysis & Binary QA Training

## Overview

Per-diagnosis binary AUROC evaluation of the ECG tokenizer generative model (e4dw86nh), comparing P(Yes) logit extraction across 77 diagnoses against ground-truth labels. Includes sensitivity analysis with binary QA fine-tuning and synonym-based prompting.

- **Date:** 2026-04-08 to 2026-04-13
- **Model:** e4dw86nh (MedGemma 1.0, 8CB)
- **Test set:** 10,000 ECGs (MIMIC+MHI)

---

## Method: P(Yes) Logit Extraction

For each ECG and each diagnosis, we ask:
> "Is {diagnosis} present in this ECG? Answer Yes or No."

We extract `P(Yes) = softmax(logit_Yes, logit_No)[0]` from the **first generated token** logits. This gives a continuous [0,1] score per diagnosis per ECG, enabling standard AUROC computation.

**Script:** `scripts/binary_auroc_eval.py` (synonym version: `scripts/binary_auroc_synonym_eval.py`)

---

## Aggregate Results

| Method | Macro AUROC | Micro AUROC | Macro AUPRC | Micro AUPRC |
|---|---|---|---|---|
| P(Yes) logit — e4dw86nh baseline | 0.606 | 0.633 | 0.130 | 0.107 |
| P(Yes) logit — zskx6r5y (+binary QA, 10k steps) | **0.628** | **0.693** | **0.154** | **0.162** |
| Synonym text extraction from interpretation | 0.698 | - | - | - |

> 22 diagnoses in "OTHER" category + 3 normal markers were **never used as QA training targets**. Excluding them: Macro AUROC = 0.633 (baseline).

---

## Binary QA Fine-tuning

### Training Details

| Parameter | Value |
|---|---|
| wandb Run ID | `zskx6r5y` (fast-plant-404) |
| wandb URL | https://wandb.ai/mhi_ai/ECG_tokenizer_MedGemma/runs/0dq63o40 |
| Base checkpoint | e4dw86nh (resumed via `resume_checkpoint_path`) |
| Config | `config/llm_finetuning/medgemma/e4dw86nh_binary_qa.yaml` |
| Binary QA data | 188,780 balanced Yes/No pairs across 52 diagnoses |
| Dataset mix | 95% original weighted QA + 5% binary diagnosis QA |
| Prompt variations | Enabled (3,990 variations for 129 prompts, per-prompt level) |
| GPU | 1x H200 (GPU 2) |
| Steps completed | ~160,000 / 466,446 (34%) before process died |
| Best checkpoint | `checkpoint_step_10000.pt` (val rougeL=0.736, loss=0.157) |

### Binary QA Data Generation

- **Script:** `dataset_generation/generate_binary_qa.py`
- **Output:** `/volume/ECG_tokenizer/output/binary_diagnosis_qa_train.parquet`
- 52 non-OTHER diagnoses, 2000 pos + 2000 neg per diagnosis (balanced)
- 5 question templates, 3 Yes templates, 3 No templates for variety

---

## Per-Diagnosis Comparison (e4dw86nh vs zskx6r5y)

**40/50 diagnoses improved** after just 10k steps. Mean delta: +0.053.

### Biggest Improvements

| Diagnosis | e4dw86nh | zskx6r5y | Delta |
|---|---|---|---|
| ST elevation (inferior) | 0.571 | 0.871 | **+0.300** |
| ST elevation (lateral) | 0.637 | 0.867 | **+0.230** |
| 2nd degree AVB mobitz 1 | 0.479 | 0.708 | **+0.229** |
| Q wave (posterior) | 0.413 | 0.609 | **+0.195** |
| SVT | 0.460 | 0.637 | **+0.178** |
| ST depression (inferior) | 0.448 | 0.610 | **+0.162** |
| 2nd degree AVB mobitz 2 | 0.725 | 0.875 | **+0.150** |
| PVC | 0.616 | 0.757 | **+0.142** |
| Irregularly irregular | 0.705 | 0.813 | **+0.108** |

### Regressions

| Diagnosis | e4dw86nh | zskx6r5y | Delta | Note |
|---|---|---|---|---|
| LV pacing | 0.852 | 0.367 | -0.485 | Only 7 positives — unstable |
| Bradycardia | 0.760 | 0.630 | -0.130 | |
| NIVCD | 0.590 | 0.520 | -0.070 | |

---

## Vocabulary Mismatch

| Label Name | How it appears in GT/model text |
|---|---|
| Atrial paced | "A-V sequential pacemaker", "Atrial-ventricular dual-paced" |
| Junctional rhythm | "Accelerated junctional rhythm", "idioventricular rhythm" |
| PAC | "PAC(s)", "Atrial premature complex" |
| Delta wave | "Wolff-Parkinson-White", "ventricular pre-excitation" |
| NIVCD | "IV conduction defect", "IVCD" |

---

## DeepECG Data Gap Analysis

**784,238 ECGs (54%) in DeepECG splits NOT in our training data.**

| Diagnosis | + missing | - missing | + in ours | - in ours | % pos missing | AUROC |
|---|---|---|---|---|---|---|
| Third Degree AV Block | 3,297 | 780,941 | 1,017 | 668,680 | 76.4% | 0.429 |
| RVH | 8,749 | 775,489 | 3,982 | 665,715 | 68.7% | 0.706 |
| Ectopic atrial rhythm | 7,428 | 776,810 | 3,492 | 666,205 | 68.0% | 0.480 |
| Acute MI | 16,895 | 767,343 | 9,436 | 660,261 | 64.2% | 0.925 |
| SVT | 4,675 | 779,563 | 2,698 | 666,999 | 63.4% | 0.460 |
| PAC | 36,095 | 748,143 | 22,036 | 647,661 | 62.1% | 0.542 |
| Regularly irregular | 48,928 | 735,310 | 31,726 | 637,971 | 60.7% | 0.502 |
| Junctional rhythm | 12,830 | 771,408 | 8,461 | 661,236 | 60.3% | 0.461 |
| PVC | 64,377 | 719,861 | 46,947 | 622,750 | 57.8% | 0.616 |
| Bradycardia | 168,574 | 615,664 | 129,398 | 540,299 | 56.6% | 0.760 |

### DeepECG Dataset Summary

- **Total:** 1,704,012 (metadata) / 1,453,935 (with splits)
- **Splits:** train=1,017,719 | test=287,039 | val=149,177 | None=250,077
- **GT labels:** 93 diagnoses (76 overlap with our 77)
- **In our data:** 669,697 (46%) | **Missing:** 784,238 (54%)
- **Source:** `/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet`

---

## Output Files

- `analysis/binary_auroc/binary_auroc_results.json` — e4dw86nh AUROC
- `analysis/binary_auroc_zskx6r5y/binary_auroc_results.json` — zskx6r5y AUROC
- `analysis/binary_auroc_synonyms/` — Synonym AUROC (in progress)
- `analysis/binary_auroc/raw_predictions.npz` — Raw predictions
- `output/binary_diagnosis_qa_train.parquet` — Binary QA training data
- `config/llm_finetuning/medgemma/e4dw86nh_binary_qa.yaml` — Training config
- `analysis/manuscript_sensitivity_tables.md` — Sensitivity tables

## Next Steps

- [ ] Complete synonym-based P(Yes) logit eval
- [ ] Resume binary QA training from step 10k checkpoint
- [ ] Add 784k missing DeepECG ECGs to training data
- [ ] Evaluate on DeepECG 287k test set for apples-to-apples comparison
- [ ] Add 17 extra DeepECG diagnoses to label ontology
