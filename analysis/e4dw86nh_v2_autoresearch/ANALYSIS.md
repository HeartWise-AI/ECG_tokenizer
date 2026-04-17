# e4dw86nh_v2: Champion + Autoresearch Improvements

## Overview

Full training run combining the e4dw86nh champion config with hyperparameter improvements found during 101 autoresearch experiments.

**Base model**: MedGemma 1.0 (`google/medgemma-4b-it`)
**Bridge**: j4bb0w33 (10 Q-Former layers, 12 heads, codebook_offset=-1)
**Data**: 7.27M rows (`combined_train_qa_m5000k_h5000k_weighted.parquet`)
**GPUs**: 1,2 (2x H200, DDP)

## Changes from e4dw86nh Champion

### LoRA Configuration (biggest change)

| Parameter | e4dw86nh | v2 | Rationale |
|-----------|----------|-----|-----------|
| lora_r | 32 | **96** | 3x more capacity |
| lora_alpha | 64 | **192** | Maintain alpha/r = 2 |
| lora_dropout | 0.05 | **0.0** | No dropout helps short runs |
| lora_top_k_layers | 12 | **24** | Train top 24 of 34 layers |
| lora_target_modules | 4 (q,k,v,o) | **7** (+gate,up,down) | Cover MLP layers too |

**LoRA params**: ~8.9M effective → ~70M+ effective (roughly 8x increase)

### Learning Rates

| Parameter | e4dw86nh | v2 | Change |
|-----------|----------|-----|--------|
| llm_lr | 5e-5 | **4e-4** | 8x higher |
| adapter_lr | 5e-4 | **1.5e-3** | 3x higher |
| num_warmup_percent | 0.256 (25.6%) | **0.01** (1%) | Much faster ramp |
| adapter_weight_decay | 1e-5 | **5e-5** | 5x higher |

### Bridge Regularization

| Parameter | e4dw86nh | v2 | Change |
|-----------|----------|-----|--------|
| bridge_dropout | 0.1 | **0.0** | Remove bridge dropout |
| bridge_codebook_dropout | 0.0 | **0.05** | Add codebook regularization |
| instruction_dropout | 0.2 | **0.05** | Less instruction masking |

### Training Dynamics

| Parameter | e4dw86nh | v2 | Change |
|-----------|----------|-----|--------|
| gradient_accumulation_steps | 16 | **8** | 2x more frequent updates |
| Effective batch size | 512 (16×16×2) | **256** (16×8×2) | Smaller effective batch |
| Steps per epoch | ~28,400 | ~56,800 | 2x more optimizer steps |

### Generation Parameters

| Parameter | e4dw86nh | v2 | Change |
|-----------|----------|-----|--------|
| max_new_tokens | 96 | **48** | Tighter output (exp89: improved both models) |
| repetition_penalty | 1.10 | **1.15** | Slightly stronger (exp92: improved both models) |

### Training Phases

| | e4dw86nh | v2 |
|--|----------|-----|
| Phase 1 (alignment) | 1 epoch | **0 epochs** (skip — bridge already aligned from Stage-1) |
| Phase 2 (LoRA finetuning) | 1 epoch | **1 epoch** |

## Risk Assessment

1. **Higher LRs from 100-step runs may not transfer**: autoresearch optimized LRs for 100 optimizer steps. With ~56K steps in a full epoch, the cosine decay will keep LR high much longer. Risk: gradient explosion or catastrophic forgetting.

2. **No Phase 1 alignment**: e4dw86nh used 1 epoch of Phase 1 (bridge alignment with LLM frozen). v2 skips this entirely, relying on the Stage-1 pretrained bridge. Risk: bridge-LLM misalignment in early training.

3. **Higher LoRA capacity**: 7 targets × 24 layers × r=96 = much more trainable params. More expressive but harder to train. Could overfit on smaller categories.

## Monitoring Checklist

- [ ] Watch for loss explosion in first 500 steps (LR may be too high)
- [ ] Compare val_loss at step 1000 vs e4dw86nh step 1000
- [ ] Check gradient norms (should be <10, warn if >100)
- [ ] Verify LoRA key count: expect ~1300+ keys (vs 434 in e4dw86nh)
- [ ] Monitor GPU memory: 2x H200 should handle r=96 with 7 targets
- [ ] Compare ROUGE/BLEU/METEOR at validation snapshots

## Autoresearch Experiment History

101 experiments tested on 100-step training runs. Only 4 improvements found:

| Exp | Change | Result |
|-----|--------|--------|
| 74 | max_new_tokens 96→64 | MedGemma 1.0 improved |
| 79 | num_warmup_percent 0.05→0.01 | MedGemma 1.5 improved |
| 89 | max_new_tokens 64→48 | Both improved |
| 92 | repetition_penalty 1.2→1.15 | Both improved |

All 4 improvements were generation parameters. Training hyperparameters were at a local optimum for 100-step runs.

## Previous Results to Beat

### e4dw86nh (champion)
- **Overall judge score: 0.707**
- LVEF: Pearson 0.559, MAE 8.38 (in-distribution)
- LVEF on EchoNext: Pearson 0.10-0.16, MAE 10-12 (out-of-distribution)
- Worst categories: classification (0.497), interpretation (0.552), urgency (0.518)
- Best categories: chamber_enlargement (0.856), ecg_interval (0.854), rhythm (0.805)

## EchoNext LVEF Generalization Analysis

We ran 8 decoding strategies on 200 EchoNext test samples. All performed near-random:

| Strategy | Pearson | ICC | MAE | #Unique Predictions |
|----------|---------|-----|-----|---------------------|
| greedy_default | 0.100 | 0.098 | 12.06 | 6 |
| sample_t0.7 (best) | 0.162 | 0.148 | 10.69 | 4 |

**Root cause**: Model only predicts ~10 discrete LVEF values (10-60, multiples of 5). Never predicts 55%, 65%, 70%, 75%. This is a model limitation, not an inference issue.

## Files

- `config_e4dw86nh_v2.yaml` — Training config
- `run_training.sh` — Training launcher (GPUs 1,2)
- `ANALYSIS.md` — This file
- `echonext_lvef_decoding_comparison.json` — EchoNext LVEF inference results (in e4dw86nh model dir)
