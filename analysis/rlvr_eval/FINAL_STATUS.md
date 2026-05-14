# Final session status — RLVR / RFT / OpenRLHF

## What's verifiably done

### Best-of-N (the only working method on this checkpoint)
- 170-row eval: 0.6674 → 0.8085 (**+14.11 pt**)
- 700-row eval: 0.4530 → 0.8132 (**+36.02 pt**)
- Method: N=5 candidates at temp=1.0, judge selects winner
- Script: `scripts/rlvr_eval_bestofn.py`

### RFT v1 — failed (this session)
- Trained on 425 best-of-5 wins (score≥0.7, held out the 170-row eval)
- Greedy decode on 170-row: 0.6674 → **0.4387 (−22.87 pt)** ❌
- Root causes documented in `analysis/rlvr_eval/rft_v1/POSTMORTEM.md`:
  - Catastrophic forgetting on categories with 0 RFT training rows
  - Format artifacts inherited from temp=1.0 sampling noise

### OpenRLHF integration — scaffolding complete, end-to-end blocked
**Done (committed `a91d506`):**
- `services/openrlhf_judge_reward.py` — reward callback, validated on a
  diverse batch (perfect candidates → 1.0, mismatching → 0.0, gibberish → 0.0)
- `scripts/build_openrlhf_dataset.py` — built `data/openrlhf_train_v1.jsonl`
  (4610 rows balanced across 21 categories)
- `scripts/run_openrlhf_grpo.sh` — 2-GPU hybrid-engine launcher with
  conservative GRPO settings (lr=5e-7, beta=1e-2, group_norm, dynamic
  filtering)
- `scripts/test_openrlhf_reward.py` — passes
- `models/openrlhf_ecg_wrapper.py` — config class done, model class
  stubbed with TODO

**Blocked on (multi-day):** the `ECGCausalLM` HuggingFace wrapper. OpenRLHF
loads its actor via `AutoModelForCausalLM.from_pretrained(path,
trust_remote_code=True)`. To make our `ECG_Tokenizer_Wrapper` loadable
that way, we need:
1. A HuggingFace model directory layout: `config.json` + `modeling_ecg.py`
   (subclass of `PreTrainedModel`) + `pytorch_model.bin` (state dict from
   our `.pt` checkpoint).
2. `forward(input_ids, attention_mask, **mm_inputs)` that splices ECG soft
   tokens into MedGemma input embeddings at the placeholder position
   (currently hidden inside `generate_report_with_question`).
3. `generate()` that does the same.
4. A custom dataset class to route `signal_path` from the OpenRLHF JSONL
   row through to the model (OpenRLHF's stock prompt dataset doesn't know
   about ECG signals).
5. (Optional, for vLLM) a way to feed soft tokens through vLLM's mm input
   path — non-trivial because vLLM's image handlers expect pixels, not
   pre-encoded embeddings.

Estimated effort: 2–3 days of focused work.

## Recommended next steps (in priority order)

### Option A — ship best-of-N as the production inference path
+36 pt at 700-row scale is a real product win. No more training needed,
just deploy `rlvr_eval_bestofn.py` (with batched judge calls) as the
serving stack. Inference cost is 5× decode + 1× judge call per query.

### Option B — RFT v2 with the lessons from v1
1. Mix RFT data with broad SFT anchor (70%/30%).
2. Strip prefix artifacts from targets.
3. lr=1e-6, 1 epoch only.
4. Per-category minimum quota (≥10 rows/category).
~4-hour experiment. May or may not match best-of-N ceiling.

### Option C — finish the OpenRLHF model wrapper
2–3 days of engineering. Highest upside (can theoretically exceed
best-of-N ceiling) but the highest risk on this checkpoint, which has
already failed three gradient-RL attempts and one supervised RFT
attempt.

## Memory + Notion
- `feedback_rlvr_lessons.md` updated with RFT v1 lesson.
- Notion page (`3600855e87b68106b694f2983b96fff2`) has RFT post-mortem and
  OpenRLHF scaffolding status.
- All work committed and pushed to `rb/openrlhf-rlvr`:
  - `90783ca` 700-row best-of-N victory
  - `7e37492` RFT v1 + postmortem
  - `68fd2d0` OpenRLHF scaffolding
  - `a91d506` wrapper stub + integration test
