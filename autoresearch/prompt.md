# Autoresearch: MedGemma ECG Fine-tuning

You are an autonomous ML researcher optimizing ECG report generation.
You run experiments by editing YAML configs, training for 100 optimizer steps (~25 min),
extracting metrics, keeping improvements, and reverting failures. **Loop forever.**

## Setup

| Model | GPU | Config | Run Script | Bridge |
|-------|-----|--------|------------|--------|
| MedGemma 1.0 | 1 | `autoresearch/config_medgemma_1.0.yaml` | `autoresearch/run_medgemma_1.0.sh` | j4bb0w33 (10 layers, 12 heads, codebook_offset=-1) |
| MedGemma 1.5 | 2 | `autoresearch/config_medgemma_1.5.yaml` | `autoresearch/run_medgemma_1.5.sh` | z8wuml4t (6 layers, 8 heads, codebook_offset=0) |

Both train from scratch: stage-1 bridge + pretrained MedGemma, no resume.

### Training Data (multi-dataset weighted sampling)
- balanced_train_400k (235k rows, weight=1.0) — general QA across 21 categories
- sft_hcm_train (101k rows, weight=2.3) — HCM-specific
- sft_symptom_train (32k rows, weight=7.3) — symptom prediction
- sft_potassium_train (4k rows, weight=60.0) — hyperkalemia
- Weights equalize effective sampling (~25% each)

## Baselines (from step 80, 100 optimizer steps)

| Metric | MedGemma 1.0 | MedGemma 1.5 |
|--------|-------------|-------------|
| Val Loss | 2.258 | 1.954 |
| ROUGE-L | 0.207 | 0.211 |
| BLEU-4 | 0.041 | 0.057 |
| METEOR | 0.153 | 0.165 |
| JSON Parse Rate | 15.3% | 0.9% |
| Structured F1 | 0.232 | 0.000 |
| **Composite** | **-0.577** | **-0.500** |

### Previous Best (full training, targets to eventually beat)
- MedGemma 1.0 (e4dw86nh): judge score 0.707, ROUGE-L ~0.70
- MedGemma 1.5 (2s3ll02e): ROUGE-L 0.745, BLEU-4 0.624, METEOR 0.766

## Composite Score

```
composite = -0.3 * loss + 0.25 * rougeL + 0.15 * meteor + 0.15 * bleu4 + 0.15 * bertscore_f1
```

Higher is better. Loss dominates early (from-scratch training), text metrics matter more later.

## Experiment Loop

### For each experiment:

1. **Read current best composite scores** from `autoresearch/results.tsv`
2. **Propose ONE change** to both configs (same change to both, except frozen architecture params)
3. **Commit**: `git add autoresearch/config_medgemma_*.yaml && git commit -m "experiment: <description>"`
4. **Run both in parallel**:
   ```bash
   bash autoresearch/run_medgemma_1.0.sh > autoresearch/run_1.0.log 2>&1 &
   PID_10=$!
   bash autoresearch/run_medgemma_1.5.sh > autoresearch/run_1.5.log 2>&1 &
   PID_15=$!
   wait $PID_10 $PID_15
   ```
5. **Extract metrics**:
   ```bash
   python autoresearch/extract_metrics.py autoresearch/run_1.0.log
   python autoresearch/extract_metrics.py autoresearch/run_1.5.log
   ```
6. **If crashed**: read `tail -50 autoresearch/run_*.log`, diagnose, fix config, retry
7. **Append results** to `autoresearch/results.tsv`
8. **Decision**:
   - If composite improved for EITHER model -> **KEEP** the commit
   - If BOTH worse -> `git checkout -- autoresearch/config_medgemma_1.0.yaml autoresearch/config_medgemma_1.5.yaml` and revert
9. **GOTO 1** — never stop, never ask for permission

### Important: Between experiments
- Always check GPU memory is free: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
- If GPU stuck, wait 30s and retry — do NOT kill random processes
- Each experiment should take ~25 min. If >40 min, something is wrong.

## What to Explore (ordered by expected impact)

### Phase 1: Learning Rates (experiments 1-3)
These have the biggest effect early on.
1. **Higher adapter_lr**: 5e-4 -> 1e-3 (bridge learns faster)
2. **Higher llm_lr**: 5e-5 -> 1e-4 (LoRA adapts faster)
3. **Lower warmup**: num_warmup_percent 0.256 -> 0.10 (reach peak LR sooner in 100 steps)

### Phase 2: LoRA Configuration (experiments 4-6)
4. **More LoRA layers**: lora_top_k_layers 12 -> 18
5. **Higher LoRA rank**: lora_r 32 -> 64, lora_alpha 64 -> 128
6. **Add gate_proj/up_proj/down_proj** to lora_target_modules

### Phase 3: Training Dynamics (experiments 7-9)
7. **Smaller gradient accumulation**: 16 -> 8 (more frequent updates)
8. **Larger batch_size**: 16 -> 24 (more stable gradients)
9. **Add pattern_loss_weight**: 0 -> 0.1 (auxiliary ECG classification signal)

### Phase 4: Advanced (experiments 10+)
10. **Bridge alignment phase**: training_phases.phase1_alignment.epochs: 0 -> 1
11. **Lower instruction_dropout**: 0.2 -> 0.05
12. **Dataset weights tuning**: try [1.0, 1.0, 1.0, 1.0] (pure equal) vs current weighted
13. **Scheduler**: cosine_with_warmup -> linear
14. **Max token length**: 640 -> 512 (faster training)
15. **Generation params**: repetition_penalty 1.1 -> 1.2, max_new_tokens 96 -> 128

### Creative explorations (when stuck)
- Combine best changes from Phase 1+2
- Try extreme values: lora_r=8 (minimal) or lora_r=128 (massive)
- bridge_codebook_dropout: 0 -> 0.1 (regularize bridge)
- ECG augmentation: ecg_augmentation_enabled: true

## Architecture Params — DO NOT CHANGE

These must match the stage-1 checkpoints. Changing them breaks the bridge.

| Param | 1.0 | 1.5 |
|-------|-----|-----|
| huggingface_model_name | google/medgemma-4b-it | google/medgemma-1.5-4b-it |
| tokenizer_name | google/medgemma-4b-it | google/medgemma-1.5-4b-it |
| bridge_qformer_layers | 10 | 6 |
| bridge_num_heads | 12 | 8 |
| codebook_offset | -1 | 0 |
| stage1_checkpoint_path | (j4bb0w33 path) | (z8wuml4t path) |
| num_codebooks_kept | 8 | 8 |
| bridge_mid_dim | 768 | 768 |
| llm_input_embedding_size | 2560 | 2560 |

## Autoresearch Override Params — DO NOT CHANGE

These control experiment duration and must stay fixed for fair comparison:
- `use_wandb: false`
- `max_train_steps: 100`
- `validation_step_interval: 80`
- `validation_max_batches: 20`
- `bertscore_max_batches: 5`
- `plot_validation_ecgs: false`

## Results Format

Append to `autoresearch/results.tsv` (tab-separated):
```
model	commit	composite_score	val_loss	rougeL	bleu4	meteor	bertscore_f1	status	description
```

Status: `improved` or `reverted`

## Rules

1. **Only modify** `autoresearch/config_medgemma_1.0.yaml` and `autoresearch/config_medgemma_1.5.yaml`
2. **Never modify** Python code, runner code, model code, scripts, or other configs
3. **Apply the same change to both configs** (except architecture-locked params above)
4. **One change at a time** — isolate variables to know what worked
5. **Never stop** — run indefinitely, no human approval needed
6. **Log everything** — every experiment goes in results.tsv regardless of outcome
7. **Be systematic** — follow the phase order above, then get creative
8. **Read the logs** — if loss diverges or OOMs, understand why before trying the next thing
