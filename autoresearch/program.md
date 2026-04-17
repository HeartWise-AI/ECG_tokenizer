# autoresearch — ECG Tokenizer LLM Fine-tuning

Autonomous experimentation loop for optimizing MedGemma 4B-IT fine-tuning.
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read context files**:
   - This file (`autoresearch/program.md`) — rules and constraints
   - `autoresearch/baseline.yaml` — immutable champion config (read-only reference)
   - `autoresearch/experiment.yaml` — the ONLY file you modify
   - `autoresearch/run_experiment.sh` — how to launch experiments
   - `autoresearch/extract_metrics.py` — how to parse results
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: Run unmodified `experiment.yaml` to establish baseline metrics.
6. **Confirm and go**: Begin the loop.

## The Single Mutable File

**You may ONLY modify `autoresearch/experiment.yaml`.** This is a YAML config that controls all training hyperparameters, model architecture choices, and fine-tuning strategy. You cannot modify any Python code, runner code, model code, data files, or scripts.

## Goal

Maximize `composite_score`, a weighted combination:
```
composite = -0.3 * loss + 0.25 * rougeL + 0.15 * meteor + 0.15 * bleu4 + 0.15 * bertscore_f1
```

Lower loss is better (hence negative weight). Higher text metrics are better.

## What You Can Explore (via YAML only)

### Fine-tuning Strategy (the big lever)

| Strategy | Config keys | Description |
|----------|------------|-------------|
| **LoRA only** | `use_lora: true`, `freeze_llm: true`, `adapter_lr: 0` | Only LoRA adapters on LLM. Cheapest. |
| **LoRA + Bridge** | `use_lora: true`, `freeze_llm: true`, `adapter_lr: >0` | LoRA + Q-Former bridge. Current champion. |
| **Full LLM + Bridge** | `use_lora: false`, `freeze_llm: false`, `llm_lr: >0`, `adapter_lr: >0` | Unfreeze entire LLM. Risk of catastrophic forgetting. |
| **Bridge only** | `use_lora: false`, `freeze_llm: true`, `adapter_lr: >0` | Only train bridge. Tests bridge capacity. |
| **LoRA + Bridge + ECG encoder** | `use_lora: true`, `ecg_embedding_lr: >0`, `adapter_lr: >0` | Also unfreeze ECG encoder. |

### Hyperparameters

| Category | Parameters | Typical Range |
|----------|-----------|---------------|
| **Component LRs** | `llm_lr`, `adapter_lr`, `ecg_embedding_lr`, `cross_attention_lr` | 1e-6 to 1e-3 |
| **LoRA arch** | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_top_k_layers`, `lora_target_modules` | r: 8-64, alpha: 16-128 |
| **Scheduler** | `scheduler_type`, `num_warmup_percent` | cosine/linear/cosine_with_warmup |
| **Batch/accum** | `batch_size`, `gradient_accumulation_steps` | bs: 8-32, accum: 4-32 |
| **Weight decay** | `llm_weight_decay`, `adapter_weight_decay` | 0 to 0.1 |
| **Bridge arch** | `instruction_dropout`, `bridge_dropout`, `num_query_tokens`, `bridge_qformer_layers` | dropout: 0-0.3 |
| **Generation** | `max_new_tokens`, `repetition_penalty`, `no_repeat_ngram_size` | tokens: 64-128 |
| **Sequence length** | `max_token_length` | 384-768 |
| **Auxiliary loss** | `pattern_loss_weight` | 0 to 1.0 |

### Training Phases

The `training_phases` dict controls per-phase freeze/unfreeze:
- `phase1_alignment`: typically frozen LLM, trains bridge. Set `epochs: 0` to skip.
- Phase 2 (default): unfreezes LLM (with or without LoRA)

## Running Experiments

```bash
bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1
```

Then extract metrics:
```bash
python autoresearch/extract_metrics.py autoresearch/run.log
```

**Time budget**: ~15-20 minutes per experiment (200 optimizer steps + validation).
**Timeout**: The script has a 1500s (25 min) timeout. If exceeded, treat as crash.

## Logging Results

Log to `results.tsv` (tab-separated). Columns:
```
commit	composite_score	val_loss	rougeL	bleu4	meteor	bertscore_f1	memory_gb	status	description
```

- `status`: `keep`, `discard`, or `crash`
- Use `0.000000` for metrics on crashes
- Do NOT commit `results.tsv` or `run.log` (they're gitignored)

## The Experiment Loop

LOOP FOREVER:

1. Review current state: best composite score, recent experiments, what's been tried
2. Propose a hypothesis — what YAML change might improve composite_score?
3. Edit `autoresearch/experiment.yaml` with the change
4. `git commit -m "descriptive message"`
5. Run: `bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1`
6. Extract: `python autoresearch/extract_metrics.py autoresearch/run.log`
7. If extraction fails, check `tail -n 50 autoresearch/run.log` for errors
8. Log results to `results.tsv`
9. If composite improved → keep the commit, advance the branch
10. If composite is equal or worse → `git reset --hard HEAD~1` to revert
11. Go to step 1

## Constraints

- **GPUs**: Always use GPUs 1,2 (`CUDA_VISIBLE_DEVICES=1,2` in `run_experiment.sh`)
- **Only modify**: `autoresearch/experiment.yaml`
- **Cannot modify**: Python code, runner code, model code, data, scripts
- **Cannot**: install packages, change dependencies
- **NEVER STOP**: Once the loop begins, run indefinitely. Do not ask the human if you should continue. The human may be away and expects you to keep going.

## Crash Handling

- Typos/easy fixes in YAML: fix and re-run
- OOM: reduce batch_size or gradient_accumulation_steps, or simplify model
- Fundamental issues: log as crash, revert, move on
- If stuck after 3 consecutive crashes: revert to last known-good state and try a different direction

## Tips

- Start with small changes — one variable at a time
- The champion config is already well-tuned, so improvements may be incremental
- Keep notes in commit messages about what you're testing and why
- If a direction shows promise (even small gain), explore it further before moving on
- If you plateau, try more radical changes (different fine-tuning strategy, very different LR ranges)
