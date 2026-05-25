# OpenRLHF GRPO integration — status

## Done ✓
1. **`services/openrlhf_judge_reward.py`** — OpenRLHF-compatible reward
   callback that loads the `/volume/LLM_JUDGE` registry and scores each
   (prediction, ground_truth, category) tuple. Smoke-tested with a
   classification prompt; returns the judge score in [0, 1] and also exposes
   it as the `scores` field so `--algo.dynamic_filtering_enable` can drop
   prompts where every sample is degenerate.
2. **`scripts/build_openrlhf_dataset.py`** — Converts our 7.27M-row weighted
   parquet into OpenRLHF JSONL format (`prompt`, `label`, `signal_path`).
   Labels carry `{"category", "ground_truth"}` as JSON so the reward func
   can recover both at scoring time. Produced
   `data/openrlhf_train_v1.jsonl` (4610 rows, balanced across 21 categories).
3. **`scripts/run_openrlhf_grpo.sh`** — Launch script with the right flag
   set for our 2-GPU split engine (group_norm advantage = GRPO,
   actor+reference colocated on one training GPU, one vLLM rollout GPU,
   low LR 5e-7, KL beta 1e-2, dynamic filtering).

## Still required ⏳
The single hard piece is wrapping `ECG_Tokenizer_Wrapper` so OpenRLHF's
PolicyModelActor + vLLM can load it. The script will not run end-to-end
without this. Two-part build:

### Part A: `models/openrlhf_ecg_wrapper.py` (~150 LOC)
Subclass of `transformers.PreTrainedModel` that:
- Owns the existing `ECG_Tokenizer_Wrapper` as a submodule.
- Exposes `forward(input_ids, attention_mask, **mm_inputs)` where
  `mm_inputs["ecg_soft_tokens"]` is a `(B, T_soft, D_llm)` tensor of
  pre-encoded soft tokens.
- Injects those tokens at the embedding stage of the underlying MedGemma
  decoder.
- Implements `save_pretrained` / `from_pretrained` so OpenRLHF's checkpoint
  loader works.
- Registers a `vision_config` stub so OpenRLHF's `is_vlm` auto-detection
  fires and it loads via `AutoModelForImageTextToText`.

### Part B: `services/openrlhf_agent.py` (~80 LOC)
Custom `SingleTurnAgentExecutor` that:
- Reads `signal_path` from each batch row.
- Loads the .npy and runs encoder + quantizer + bridge to get soft tokens.
- Caches encoded signals in memory (the same waveform appears many times in
  the dataset across different prompts).
- Hands the soft tokens to vLLM as `mm_inputs` during rollout.

Currently `services/ecg_preencode.py` has the signal loader and the
checkpoint loader, but `batch_encode()` raises `NotImplementedError`
because `ECG_Tokenizer_Wrapper` doesn't yet expose its encoder output
publicly — that's the third TODO:

### Part C: expose `encode_to_soft_tokens(signal) -> Tensor` on `ECG_Tokenizer_Wrapper`
The hook is missing because the existing pipeline always goes through
`generate_report_with_question()` which conflates encoding + generation.

## Expected runtime when done
- Engineering effort to finish A+B+C: ~2–3 days of dedicated work.
- Once it runs: likely several hours for one GRPO epoch over 5000 prompts
  (N=5 samples/prompt) on 2× H200 with one dedicated vLLM rollout GPU and
  one actor/reference training GPU.

## GPU placement
For ECG we should prefer split placement over `--train.colocate_all`:
- Training GPU: actor + reference via `--train.colocate_actor_ref`,
  `--actor.num_gpus_per_node 1`, `--ref.num_gpus_per_node 1`.
- Rollout GPU: vLLM via `--vllm.num_engines 1`,
  `--vllm.tensor_parallel_size 1`.
- Avoid `--vllm.enable_sleep` / `--ds.enable_sleep` in this mode; OpenRLHF
  disables vLLM sleep unless `--train.colocate_all` is active.

This costs some throughput versus colocated hybrid scheduling, but it keeps
ECG rollout/eval memory off the training GPU and makes failures easier to
diagnose.

## Risk profile
The previous gradient-RLVR attempts on this checkpoint all regressed
(see `feedback_rlvr_lessons.md`). OpenRLHF GRPO is structurally similar:
PPO-style ratio + KL anchor. Even with conservative settings (lr 5e-7,
KL 1e-2, dynamic filtering), this checkpoint may still be too fragile.
Fall-back if v1 regresses: lower LR to 1e-7, raise KL to 1e-1, restrict
to one category at a time.

## How to verify the reward function is wired correctly (no model needed)
```bash
PYTHONPATH=/volume/ECG_tokenizer python3 services/openrlhf_judge_reward.py
# expected: 'judge registry initialized, 26 categories'
# and a tensor reward in [0, 1]
```
