# NEXT STEPS - ECG Tokenizer / Bridge / LLM (hand-off brief)

Paste-into-a-new-session brief. Live tracker: Notion
[NEXT STEPS](https://app.notion.com/p/3ba0855e87b681caad72dd0faf7398f0) ·
measurements: *Information Cascade* page · reproduce: `REPRODUCE.md` (same folder).

**Last updated Aug 24 2026, after the Step 3.2 judge verdict.** Steps 1, 2, 3.1 and 3.2 are
all DONE. The current best model is the **time-axis bridge**. Two of the numbers the original
plan was built on turned out to be measurement artifacts - read §7 Hazards before trusting any
historical figure.

---

## 1. Where we are

| stage | current best | note |
|---|---|---|
| tokenizer | `x1_split` (ScalableEncoder 6M + split codebooks) | unchanged since Aug 3 |
| bridge | **time-axis Q-Former** (`bridge_token_axis: time` + `concat_linear`) | Stage-1 run `6aeuhpxo` |
| LLM | **`72g2pvq9`** = MedGemma-4B, e4d recipe, time-axis bridge | ckpt `.../72g2pvq9_20260822-174049/best_model.pt` |

**Judge (s3000 sample, REGEN GT, current MiniMax judge - all rows comparable):**

| model | example-wt | macro | json_interp | interp | classif | AFib | LVEF | SHD | ACS |
|---|---|---|---|---|---|---|---|---|---|
| e4d (production, corrected) | 0.6187 | 0.616 | **0.608** | 0.573 | 0.443 | 0.652 | 0.845 | 0.634 | 0.536 |
| kept8 champion (softmax) | 0.6181 | 0.658 | 0.518 | 0.546 | 0.485 | 0.751 | 0.843 | 0.639 | 0.528 |
| concat_linear (Step 1) | 0.6353 | 0.644 | 0.551 | 0.562 | 0.498 | 0.756 | 0.848 | 0.628 | 0.536 |
| **time-axis (Step 3.2)** | **0.6598** | **0.686** | 0.560 | **0.610** | **0.527** | **0.772** | **0.858** | **0.656** | **0.541** |

Time-axis wins **both weightings** and **11 of 13 categories**. `json_interpretation` is the only
category still below production e4d (−0.048).

**Deterministic endpoints.** Full test (49,776 rows, concat_linear): LVEF≤40 **AUROC 0.83
(95% CI 0.82–0.85)** · AFib-5y 0.73 (0.71–0.75) · SHD 0.68 (0.66–0.69) · ACS 0.73 (0.71–0.76).
s3000 (time-axis): AFib **0.77 (0.70–0.83)** best of any model; LVEF 0.84 (0.73–0.92); SHD 0.65;
ACS 0.72. Endpoints move ~0 between bridges - they are tokenizer-capped (hazard R4).

## 2. Decision rule (unchanged, and it has teeth)

Accept or reject on **generated-output** metrics only - never probe AUROC, never Stage-1
retrieval. Primary = `json_interpretation` + per-label ST/Q-wave; secondary = judge overall
(**quote BOTH example-weighted and macro**) + the four deterministic endpoints. Re-score the
baseline with the **same judge AND the same GT version** on the **same sample**.

Empirically validated Aug 24: *generation* metrics (Stage-1 ETG next-acc, Stage-3 snapshot
ROUGE/BLEU) predicted the judge correctly at all 19 checkpoints; *retrieval* metrics (SigLIP
r@1/r@5/tail) favoured the losing model and would have caused a wrong rejection.

## 3. Ranked next steps

### 🥇 STEP A - per-endpoint readout policy - **OPEN DECISION, zero compute**
**Effort:** none (already measured) · **Risk:** product decision, not technical
ACS greedy text gives sens **0.48** / spec 0.98. The *same frozen model* scored by log-prob
margin, calibrated on train-side ECGs and applied to the full test set, gives
sens **0.70** / spec 0.84, **AUROC 0.85 (95% CI 0.82–0.87)**. Best readout is endpoint-specific:
numeric-EF parsing for LVEF (0.83 vs margin 0.75), generated text for AFib (0.73 vs 0.61),
margin for ACS; SHD unresolved (ranking better, threshold sample-sensitive: sens swung
0.48→0.72 between draws). Prompt-ensembling is NOT uniformly good - helps LVEF<50/ACS, hurts
AFib (0.61→0.57) and SHD (0.73→0.70). ACS ECE is 0.170, so use the locked threshold as a
decision rule and do **not** report ACS probabilities as calibrated.
**Decision needed:** does a binary answer come from the generated sentence, the calibrated
score, or both - and what is shown when they disagree? Deployable (threshold locked at
calibration, no GT at inference) and distinct from the BANNED best-of-N judge selection.

### 🥈 STEP B - commit + PR the two-week working tree - **housekeeping, urgent**
Uncommitted on `main`: bridge `mix_strategy` {softmax|sum|concat_linear} and
`token_axis` {channel|time}; `scripts/probe_axis_views.py`;
`scripts/calibrate_endpoints_on_train.py`; `scripts/run_fulltest_gen_resumable.sh`;
`scripts/run_timeaxis_eval_chain.sh`; configs for every run above; and **five eval-loader
fixes**, each of which caught a silent-failure bug (three `bridge_mix_strategy` threading,
one Trie-shim, one `score_deterministic` dedup). Tests: `tests/models/test_qformer_mix_strategy.py`
(8) + `test_qformer_token_axis.py` (7), all green.

### 🥉 STEP C - close the last `json_interpretation` gap (−0.048 vs e4d)
The only category still behind production. Two candidate levers, cheapest first:
1. **q64 rerun of the winning bridge** (~1 day Stage-1 + ~2 days Stage-3). Step 2 showed q64 is
   the query-count optimum (R@1 0.605 vs 0.561 @q32 / 0.546 @q128) but every bridge to date
   uses **q32** - an untested, cheap capacity lever on the current champion.
2. **Step D (below)** if q64 does not move it.

### STEP D - split/parallel RVQ on the next tokenizer - tokenizer retrain, medium risk
`x1_split` supervises codebooks [0,1] *of a single residual chain*, so levels 2–8 must encode
the residual after endpoint supervision and morphology gets crowded out. Moshi/Mimi documented
exactly this and fixed it with a **parallel** VQ (semantics in a separate quantizer, not a
prefix). Note Step 3.2 already bypasses the discrete interface for kv, so a tokenizer redesign
should also decide whether the bridge stays continuous-kv or returns to true time-indexed tokens.

### STEP E - de-binarise SHD - data-side, do LAST
SFT parquet has only binary structural answers (149,153 No / 105,307 Yes, zero multi-condition).
Regenerate with EchoNext 7-label GT so answers enumerate present conditions. **Confounds
architecture comparisons** - only after the architecture is frozen.

## 4. Runnable commands

```bash
# Stage-1 bridge (channel|time axis; concat_linear fusion)
MASTER_PORT=29715 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/runner.sh --base_config config/ecg_text_stage1/winner_cb8_fullETG_timeaxis.yaml \
  --selected_gpus 0,1 --use_wandb true --run_mode train

# MedGemma retrain off a Stage-1 checkpoint (set stage1_checkpoint_path + BOTH bridge knobs)
bash scripts/runner.sh --base_config config/llm_finetuning/medgemma/e4d_x1split_cb8_timeaxis_retrain.yaml \
  --selected_gpus 0,1,2 --use_wandb true --run_mode train

# Judge eval - generate on the EXACT s3000 subset, then score. PATHS MUST BE ABSOLUTE (see R8).
# Replace all three identity placeholders with immutable provider-side revisions.
PYTHONPATH=/volume/ECG_tokenizer python scripts/eval_judge_csv.py --checkpoint <best_model.pt> \
  --subset_parquet /volume/ECG_tokenizer/analysis/x1split_judge/eval_subset_s3000.parquet \
  --output_csv /volume/ECG_tokenizer/analysis/x1split_judge/<tag>_s3000_generations.csv \
  --device cuda:0 --batch_size 16
CSV=/volume/ECG_tokenizer/analysis/x1split_judge/<tag>_s3000_generations.csv \
OUTPUT=/volume/ECG_tokenizer/analysis/x1split_judge/judge_<tag>_s3000.json \
OUT_DIR=/volume/ECG_tokenizer/analysis/x1split_judge/<tag>_judge_shards \
JUDGE_MODEL_ID='<immutable-model-revision>' \
JUDGE_DEPLOYMENT_ID='<immutable-deployment-revision>' \
JUDGE_DECODING_ID='<immutable-decoding-configuration>' \
SHARDS=16 MAX_PARALLEL=4 bash scripts/run_csv_llm_judge_sharded.sh

# Full-test deterministic endpoints (resumable, 8 shards, auto-scores)
bash scripts/run_fulltest_gen_resumable.sh
python scripts/score_deterministic.py "<glob-or-csv>" --tag <TAG>

# Calibrated per-endpoint readout - fit on TRAIN, evaluate on the FULL test set
PYTHONPATH=/volume/ECG_tokenizer python scripts/calibrate_endpoints_on_train.py \
  --checkpoint <best_model.pt> --device cuda:2 --n_calib 1200

# Probes
python scripts/probe_tokenizer.py --ckpt <tok.pt> --tag <TAG> --pool tok_mean \
  --group_col new_PatientID --device 0 --n 10000
python scripts/probe_axis_views.py --ckpt <tok.pt> --tag <TAG> --device 0 --n 10000
```

## 5. Explicitly rejected (do not re-propose without new evidence)

| rejected | why |
|---|---|
| Swap Q-Former for MLP / token-preserving projection | The MLP-wins literature fully fine-tunes the LLM. In frozen+LoRA - ours - every clean ablation reverses (Flamingo 70.7 vs 66.6; Idefics2 **+8.5** from *adding* the resampler; Garg & Bas +7.4 frozen vs +1.7 trained). |
| Query-orthogonality regulariser (ORCA) | Ruled out empirically - query off-diagonal cosine 0.243/0.167 vs 0.923 collapsed. Queries are healthy. |
| More codebooks / bigger token budget | Saturated: 2→8 fixed ACS but not json. Depth probe: pooled linear decodability saturates at cb1–2 (sheets 1→8 macro 0.854→0.865). |
| 16-codebook tokenizer (x2_depth16) | Ties on endpoints, doubles bits/token ⇒ mandatory bridge rebuild. |
| Unfreezing the tokenizer | Prismatic: full FT of the visual backbone degrades localisation (p=0.00381). LoRA only, if ever. |
| Best-of-N with judge selection | BANNED - circular (judge-select then judge-score), degenerate on binary, not deployable. |
| Softmax codebook fusion | Retired Aug 18. Measured on the trained champion: 99% of gate weight on cb0–cb1 (770× disparity), mixed-vector norm 32.7 vs 174.8 for the sum. It had silently re-created `kept≈2`. |

## 6. Standing hazards

- **R1 Judge versions.** Never compare across judge versions (old e4d CSV = spurious +0.16).
- **R1b GT versions.** The original three-way table was contaminated - e4d scored on **pre-REGEN**
  GT while kept2/kept8 used REGEN (23.6% of rows had a different answer key; json GT 96% changed).
  The famous json "−0.27 catastrophe" was **two-thirds artifact**; the real gap was −0.09, now
  −0.048. **Hash the `ground_truth` column before any cross-model comparison.**
- **R1c Macro vs example-weighted.** The headline macro is a fixed-weight mean over categories; a
  20-row category moves it as much as a 700-row one. It has already inverted one verdict. Quote both.
- **R2 Probe/retrieval ≠ generation.** Never accept on probe or SigLIP metrics. Proven twice: the
  kept=2 probe led on every endpoint and collapsed downstream; the time-axis bridge LOST every
  Stage-1 retrieval metric at e10 and won the judge decisively.
- **R3 More resolution can hurt coarse tasks.** Always report rhythm/AFib when sweeping.
- **R4 The tokenizer is the endpoint ceiling.** Deterministic endpoints moved ~0 across three
  bridges. Bridge work buys text/fine-detail, not endpoints.
- **R5 Small n.** ACS n=1,021 (357 pos); chamber n=20 in the judge sample; snapshot LVEF n=47 has
  swung 0.83–0.94 within single runs. Trust full-test numbers with CIs.
- **R6 Silent weight-drop on eval paths.** Every new structural bridge knob must be threaded into
  *all* loaders. `bridge_mix_strategy` needed patching in three separate eval scripts; the Stage-3
  mismatch guard caught it each time. Keep the guard hard-failing; never downgrade it to a warning.
- **R7 Unreviewed preprints.** LePaX 2607.06909, CheXpercept 2606.21020, CARE-X 2608.03890,
  ORCA 2607.06014, PARCEL 2605.30126 - re-read before manuscript use.
- **R8 Judge wrapper needs ABSOLUTE paths.** `run_csv_llm_judge_sharded.sh` `cd`s into
  `/volume/LLM_JUDGE`, so relative CSV/OUTPUT/OUT_DIR make every shard fail with FileNotFoundError.
  Also verify the exit code AND a non-empty output json - a wrapper echoing success unconditionally
  hid a total failure on Aug 24.
- **R9 Patient-ID formats differ across parquets.** Train stores `'337511.0'`, test stores
  `'338306'` - a raw string leak-check matches nothing and passes **vacuously**. Normalise via
  `pd.to_numeric(...).astype('Int64').astype(str)`. (True overlap verified: 0 patients, 0 ECGs.)
- **R10 Rare endpoints starve uniform sampling.** ACS is labelled on only 1.9% of train ECGs, so a
  1,500-ECG uniform draw yielded 68 labels and the fit was silently skipped. Stratify per endpoint.

## 7. Suggested order from here

1. **Step A** - decide the readout policy (zero compute, largest immediate clinical gain).
2. **Step B** - commit and PR the working tree (two weeks of undefended work).
3. **Step C.1** - q64 rerun of the time-axis bridge (cheapest remaining lever on `json_interpretation`).
4. **Step D** - parallel RVQ, only if q64 does not move it.
5. **Step E** - de-binarise SHD last, once the architecture is frozen.
