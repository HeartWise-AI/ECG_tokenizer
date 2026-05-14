# RFT (#1) and OpenRLHF GRPO (#2) — Execution Plan

## Why both, in this order
- **#1 RFT** captures the cheap, almost-certain win first: distill the existing best-of-5 wins into the weights so greedy decoding stops being lossy. ~2–4 h wall-clock, low risk of regression because it's SFT on judge-validated targets.
- **#2 OpenRLHF GRPO** is the principled follow-up: train on *new* samples + judge reward signal, including cases RFT can't fix. Higher engineering cost (~2–3 days adapter work + tuning), higher upside, but #1 is a strict subset of the data #2 would generate, so #1 is also the right warm-start for #2.

## Hardware budget
- GPU 0: free (143 GB)
- GPU 1: free (143 GB) — *the inference / vLLM GPU*
- GPU 2: in use by another job (~10 GB, 26 % util) — leave alone

Plan uses GPUs 0 + 1 only.

---

## Plan #1 — Rejection-sampling fine-tune (RFT)

### Concept
We already have 700 best-of-5 generations in `analysis/rlvr_eval/bestofn_1k/generations_bestof5_1k.csv` with judge scores. Keep only the high-confidence wins and SFT the same SFT checkpoint on them — model learns to put probability mass directly on the winning samples.

### Steps

#### Step 1.1 — Build the RFT parquet
Script: `scripts/build_rft_dataset.py` (new, ~40 lines).
- Read `generations_bestof5_1k.csv`.
- Filter `bestof_picked_score ≥ 0.7` (≈ 80 % of rows qualify; tune later if too aggressive).
- Drop ERROR rows.
- Output columns: `waveform_path_psa`, `prompt`, `target_text` (= the chosen best-of-5 generation), `prompt_category`, `bestof_picked_score`.
- Save as `data/rft_bestof5_v1.parquet`.

Optional augmentation: also include rows from the 170-row eval (~140 more), seed the parquet with the original SFT ground-truth for unmodified rows where `bestof_picked_score=1.0` and `baseline_gen ≠ ground_truth` (these are the high-value flips).

#### Step 1.2 — Wire it into the existing SFT pipeline
We don't need a new project. Re-use `projects/llm_finetuning_project.py` with a config that points at the new parquet, `prompt_column=prompt`, `report_column=target_text`. The model architecture is identical (same `ECG_Tokenizer_Wrapper`, same LoRA config from the SFT checkpoint).

Config file: `config/llm_finetuning/medgemma/2wjwbk0b_rft_bestof5_v1.yaml`. Key overrides vs the SFT base config:
- `pretrained_tokenizer_path: /volume/ECG_tokenizer/checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt`
- `train_dataset_path: data/rft_bestof5_v1.parquet`
- `lr: 1e-5` (low — we're not changing the model, just sharpening it)
- `epochs: 2`
- `lora_r/alpha`: inherit from checkpoint (this is the critical bit — see [[lora-checkpoint-loading-bug]] in memory)
- `batch_size: 4`, `grad_accum: 4` → effective 16
- `selected_gpus: 0` (single GPU, the LoRA adapter is small and fits easily)

#### Step 1.3 — Train
```bash
bash scripts/runner.sh \
  --base_config config/llm_finetuning/medgemma/2wjwbk0b_rft_bestof5_v1.yaml \
  --selected_gpus 0 --use_wandb true --run_mode train
```
ETA ≈ 90–120 min for 2 epochs over ~550 rows.

#### Step 1.4 — Eval (greedy decoding, no best-of-N)
On GPU 1, run the standard eval script against the locked 700-row subset:
```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/rlvr_eval_subset.py \
  --checkpoint <new_rft_ckpt> \
  --subset_parquet analysis/rlvr_eval/eval_subset_50per_cat.parquet \
  --output_dir analysis/rlvr_eval/rft_v1 \
  --device cuda:0 --run_judge
```
- **Success criterion**: greedy overall ≥ 0.55 (vs baseline 0.4530 → +0.10).
- **Stretch**: ≥ 0.65 (best-of-5 ceiling is 0.81).

#### Step 1.5 — Iterate if needed
- If RFT model plateaus, run **best-of-5 on the RFT model**, regenerate RFT data from those new wins, re-train. This is "iterative RFT" / "STaR". Each round shrinks the gap to the judge ceiling.
- If RFT plateaus *and* best-of-5 on the RFT model also plateaus → the model has exhausted what supervised distillation can give. Move to #2.

---

## Plan #2 — OpenRLHF GRPO with hybrid engine

### Concept
Real RL: generate candidates online, judge-as-reward, gradient update with GRPO advantage. OpenRLHF's hybrid engine lets vLLM rollouts and policy training share GPUs, which is required on a 2-GPU budget.

### Key OpenRLHF features we'll use (confirmed via DeepWiki)
- `--train.colocate_all` — actor + ref + vLLM share GPUs via Ray placement groups
- `--vllm.enable_sleep` + `--ds.enable_sleep` — vLLM/DeepSpeed sleep when the other phase is active (memory-sharing trick)
- `--algo.advantage.estimator group_norm` — that's GRPO (no critic needed)
- `--algo.kl.init_coef 0` — skip the ref model entirely if we want max memory (we'll start with a low coef like 1e-3 and only zero it if we hit OOM)
- `--rollout.n_samples_per_prompt 5` — same N=5 as best-of-N
- `--algo.dynamic_filtering_enable` + `--algo.dynamic_filtering_range 0.05 0.95` — drop prompts where all candidates are 0 or all are 1 (no learning signal)
- `--reward.remote_url <python_file>` — load our LLM-judge as a custom reward function

### Engineering work (the hard part)
OpenRLHF expects `AutoModelForCausalLM` text-in/text-out. Our model is ECG-signal-in/text-out. We need three adapters:

#### Adapter A — `ECGCausalLMWrapper` (~150 lines)
File: `models/openrlhf_adapter.py`.
- Subclasses `transformers.PreTrainedModel`, exposes `forward(input_ids, attention_mask, **mm_inputs)`.
- `mm_inputs` carries the pre-encoded ECG soft tokens (numpy → bf16 tensors).
- Internally calls our existing `ECG_Tokenizer_Wrapper.forward()` with the soft tokens injected at the right position.
- Re-uses the existing LoRA adapter on the MedGemma decoder.

#### Adapter B — ECG signal pre-encoder service (~80 lines)
File: `services/ecg_preencode.py`.
- Loads encoder + quantizer + bridge from the checkpoint.
- Reads `waveform_path_psa`, returns soft tokens as `mm_inputs` dict.
- Called by a custom `SingleTurnAgentExecutor` *before* the prompt enters vLLM.

#### Adapter C — Custom reward function (~60 lines)
File: `services/openrlhf_judge_reward.py`.
- Loads `/volume/LLM_JUDGE` registry (we already have this code in `scripts/rlvr_eval_bestofn.py:judge_score`).
- Function signature OpenRLHF expects: `def reward_fn(queries, responses, labels) -> {"rewards": [...], "scores": [...]}`.
- Plugged in via `--reward.remote_url services/openrlhf_judge_reward.py`.

### GPU layout (2-GPU hybrid engine)
```
GPU 0  |  vLLM engine (1 engine, TP=1) + actor (DeepSpeed Zero 3)
GPU 1  |  vLLM engine (1 engine, TP=1) + actor (DeepSpeed Zero 3) + judge HTTP client
```
With `--vllm.enable_sleep`: when training step runs, vLLM is asleep; when rollout runs, training is asleep. Memory: actor ≈ 8 GB (4B model with LoRA), vLLM ≈ 30 GB, ref skipped → ~38 GB peak per GPU, comfortably in 143 GB.

### Command (filled-in for our setup)
```bash
PYTHONPATH=/volume/ECG_tokenizer CUDA_VISIBLE_DEVICES=0,1 \
python3 -m openrlhf.cli.train_ppo_ray \
  --ref.num_nodes 1 --ref.num_gpus_per_node 2 \
  --actor.num_nodes 1 --actor.num_gpus_per_node 2 \
  --vllm.num_engines 2 --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.55 \
  --vllm.enable_sleep --ds.enable_sleep \
  --vllm.sync_backend nccl --vllm.enforce_eager \
  --train.colocate_all \
  --algo.kl.init_coef 1e-3 --algo.kl.estimator k3 --algo.kl.use_loss \
  --algo.advantage.estimator group_norm \
  --algo.dynamic_filtering_enable \
  --algo.dynamic_filtering_range 0.05 0.95 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.adam.lr 5e-7 \
  --actor.gradient_checkpointing_enable \
  --actor.model_name_or_path /volume/ECG_tokenizer/checkpoints/.../best_model.pt \
  --actor.model_loader_path models/openrlhf_adapter.py \
  --reward.remote_url services/openrlhf_judge_reward.py \
  --train.agent_func_path services/ecg_preencode.py \
  --train.micro_batch_size 2 --train.batch_size 32 \
  --rollout.micro_batch_size 4 --rollout.batch_size 32 \
  --rollout.n_samples_per_prompt 5 \
  --train.max_epochs 1 \
  --data.max_len 2048 --data.max_samples 5000 \
  --ds.zero_stage 3 --ds.param_dtype bf16 \
  --ckpt.output_dir checkpoints/openrlhf_rft_v1 \
  --ckpt.save_steps 50 --ckpt.save_hf
```

### Success criterion
On the same 700-row eval subset with greedy decoding: ≥ 0.55 overall (≥ +0.10 over baseline). Stretch ≥ 0.65.

### Risks / abort conditions
- If reward collapses (training-time mean reward drops below baseline 0.45 for 10+ steps in a row) → abort, lower LR to 2e-7, raise KL coef to 1e-2, re-start.
- If gradient norms persistently > 10 → adapter bug (likely signal injection misaligned with token positions). Stop and debug, do not power through.
- If the OpenRLHF adapter takes longer than 3 days to write/debug → stop, ship RFT-only results, escalate to a separate work item.

---

## Run order and decision tree

```
Plan #1 (RFT) ──► greedy ≥ 0.55?
                  │
                  ├── YES, ≥ 0.65 → SHIP. Optionally one more iterative RFT round.
                  │
                  ├── YES, 0.55–0.65 → SHIP, then start #2 in parallel for upside.
                  │
                  └── NO (< 0.55)  → Investigate (data quality, LR, epochs).
                                     If still flat → escalate to #2.
```

## What to commit first
1. `scripts/build_rft_dataset.py` (data prep, ~40 lines)
2. `config/llm_finetuning/medgemma/2wjwbk0b_rft_bestof5_v1.yaml` (RFT config)
3. After Plan #1 result: VICTORY/POST-MORTEM markdown in `analysis/rlvr_eval/rft_v1/`.
