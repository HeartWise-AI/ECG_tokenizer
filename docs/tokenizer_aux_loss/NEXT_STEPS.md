# NEXT STEPS — ECG Tokenizer / Bridge / LLM (hand-off brief)

Paste-into-a-new-session brief. Live tracker: Notion
[NEXT STEPS](https://app.notion.com/p/3ba0855e87b681caad72dd0faf7398f0) ·
measurements: *Information Cascade* page · reproduce: `REPRODUCE.md` (same folder).

**Last updated Aug 26 2026.** Steps 1, 2, 3.1 and 3.2 are all DONE and the current best model is
the **time-axis bridge**. Two of the numbers the original plan was built on turned out to be
measurement artifacts — read §6 Hazards before trusting any historical figure. The original
Aug-2026 hand-off brief that this file grew out of is preserved verbatim-in-substance in
**Appendix A**; where the two disagree, §1–§7 win and Appendix A records why.

---

## 1. Where we are

| stage | current best | note |
|---|---|---|
| tokenizer | `x1_split` (ScalableEncoder 6M + split codebooks) | unchanged since Aug 3 |
| bridge | **time-axis Q-Former** (`bridge_token_axis: time` + `concat_linear`) | Stage-1 run `6aeuhpxo` |
| LLM | **`72g2pvq9`** = MedGemma-4B, e4d recipe, time-axis bridge | ckpt `.../72g2pvq9_20260822-174049/best_model.pt` |

**Judge scoreboard (s3000 sample, REGEN GT, current MiniMax judge — all rows comparable):**

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
ACS 0.72. Endpoints move ~0 between bridges — they are tokenizer-capped (hazard R4).

## 2. Decision rule (unchanged, and it has teeth)

Accept or reject on **generated-output** metrics only — never probe AUROC, never Stage-1
retrieval. Primary = `json_interpretation` + per-label ST/Q-wave; secondary = judge overall
(**quote BOTH example-weighted and macro**) + the four deterministic endpoints. Re-score the
baseline with the **same judge AND the same GT version** on the **same sample**.

Empirically validated Aug 24: *generation* metrics (Stage-1 ETG next-acc, Stage-3 snapshot
ROUGE/BLEU) predicted the judge correctly at all 19 checkpoints; *retrieval* metrics (SigLIP
r@1/r@5/tail) favoured the losing model and would have caused a wrong rejection.

## 3. Ranked next steps

> **ID namespace warning.** `R1…R11` in §6 below are **standing hazards**. `R0a…R3` in the
> research hub are **proposed experiments**. Same letter, unrelated lists — always say which.

### 🥇 STEP A0 — R1 axis/content disentanglement — **RUNNING, and it gates the write-up**
**Effort:** 2× Stage-1 (one per arm) · **Risk:** none to production; this is a measurement
The Step 3.2 champion changed **two** things at once — kv positions became 82 time slices AND
kv content became continuous `z` instead of discrete code embeddings — so its **+0.025 cannot
be attributed to the axis**. A new `kv_source` knob makes content independent of axis and
completes the 2×2. Arm A (time + discrete) launched Aug 26 on GPU 2, wandb `u12zld0r`; arm B
(channel + continuous) is queued behind q64. Design, reading rules and the three possible
verdicts: `docs/tokenizer_aux_loss/R1_AXIS_DISENTANGLEMENT.md`.
Also settled while implementing it: `bridge_mix_strategy` is **inert** on the time axis —
`72g2pvq9` records `concat_linear` but its state dict holds **no `mix_proj.*` keys**. The
fusion fix and the axis change never composed.

### 🥇 STEP A — per-endpoint readout policy — **OPEN DECISION, zero compute**
**Effort:** none (already measured) · **Risk:** product decision, not technical
ACS greedy text gives sens **0.48** / spec 0.98. The *same frozen model* scored by log-prob
margin, calibrated on train-side ECGs and applied to the full test set, gives
sens **0.70** / spec 0.84, **AUROC 0.85 (95% CI 0.82–0.87)**. Best readout is endpoint-specific:
numeric-EF parsing for LVEF (0.83 vs margin 0.75), generated text for AFib (0.73 vs 0.61),
margin for ACS; SHD unresolved (ranking better, threshold sample-sensitive: sens swung
0.48→0.72 between draws). Prompt-ensembling is NOT uniformly good — helps LVEF<50/ACS, hurts
AFib (0.61→0.57) and SHD (0.73→0.70). ACS ECE is 0.170, so use the locked threshold as a
decision rule and do **not** report ACS probabilities as calibrated.
**Decision needed:** does a binary answer come from the generated sentence, the calibrated
score, or both — and what is shown when they disagree? Deployable (threshold locked at
calibration, no GT at inference) and distinct from the BANNED best-of-N judge selection.

### 🥈 STEP B — commit + PR the two-week working tree — **housekeeping, urgent**
Uncommitted on `main`: bridge `mix_strategy` {softmax|sum|concat_linear} and
`token_axis` {channel|time}; `scripts/probe_axis_views.py`;
`scripts/calibrate_endpoints_on_train.py`; `scripts/run_fulltest_gen_resumable.sh`;
`scripts/run_timeaxis_eval_chain.sh`; configs for every run above; and **five eval-loader
fixes**, each of which caught a silent-failure bug (three `bridge_mix_strategy` threading,
one Trie-shim, one `score_deterministic` dedup). Tests: `tests/models/test_qformer_mix_strategy.py`
(8) + `test_qformer_token_axis.py` (7), all green.

### 🥉 STEP C — close the last `json_interpretation` gap (−0.048 vs e4d)
The only category still behind production. Two candidate levers, cheapest first:
1. **q64 rerun of the winning bridge** (~1 day Stage-1 + ~2 days Stage-3). Step 2 showed q64 is
   the query-count optimum (R@1 0.605 vs 0.561 @q32 / 0.546 @q128) but every bridge to date
   uses **q32** — an untested, cheap capacity lever on the current champion.
2. **Step D (below)** if q64 does not move it.

### STEP D — split/parallel RVQ on the next tokenizer — tokenizer retrain, medium risk
`x1_split` supervises codebooks [0,1] *of a single residual chain*, so levels 2–8 must encode
the residual after endpoint supervision and morphology gets crowded out. Moshi/Mimi documented
exactly this and fixed it with a **parallel** VQ (semantics in a separate quantizer, not a
prefix). Note Step 3.2 already bypasses the discrete interface for kv, so a tokenizer redesign
should also decide whether the bridge stays continuous-kv or returns to true time-indexed tokens.

### STEP E — de-binarise SHD — data-side, do LAST
SFT parquet has only binary structural answers (149,153 No / 105,307 Yes, zero multi-condition).
Regenerate with EchoNext 7-label GT so answers enumerate present conditions. **Confounds
architecture comparisons** — only after the architecture is frozen.

## 4. Runnable commands

```bash
# Stage-1 bridge (channel|time axis; concat_linear fusion)
MASTER_PORT=29715 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/runner.sh --base_config config/ecg_text_stage1/winner_cb8_fullETG_timeaxis.yaml \
  --selected_gpus 0,1 --use_wandb true --run_mode train

# MedGemma retrain off a Stage-1 checkpoint (set stage1_checkpoint_path + BOTH bridge knobs)
bash scripts/runner.sh --base_config config/llm_finetuning/medgemma/e4d_x1split_cb8_timeaxis_retrain.yaml \
  --selected_gpus 0,1,2 --use_wandb true --run_mode train

# Judge eval — generate on the EXACT s3000 subset, then score. PATHS MUST BE ABSOLUTE (see R8).
PYTHONPATH=/volume/ECG_tokenizer python scripts/eval_judge_csv.py --checkpoint <best_model.pt> \
  --subset_parquet /volume/ECG_tokenizer/analysis/x1split_judge/eval_subset_s3000.parquet \
  --output_csv /volume/ECG_tokenizer/analysis/x1split_judge/<tag>_s3000_generations.csv \
  --device cuda:0 --batch_size 16
CSV=/volume/ECG_tokenizer/analysis/x1split_judge/<tag>_s3000_generations.csv \
OUTPUT=/volume/ECG_tokenizer/analysis/x1split_judge/judge_<tag>_s3000.json \
OUT_DIR=/volume/ECG_tokenizer/analysis/x1split_judge/<tag>_judge_shards \
SHARDS=16 MAX_PARALLEL=4 bash scripts/run_csv_llm_judge_sharded.sh

# Full-test deterministic endpoints (shard-level resume, 8 shards, auto-scores).
# Use THIS, not a bare `generate_all_qa_pairs.py --resume` — see hazard R11.
bash scripts/run_fulltest_gen_resumable.sh
python scripts/score_deterministic.py "<glob-or-csv>" --tag <TAG>

# Calibrated per-endpoint readout — fit on TRAIN, evaluate on the FULL test set
PYTHONPATH=/volume/ECG_tokenizer python scripts/calibrate_endpoints_on_train.py \
  --checkpoint <best_model.pt> --device cuda:2 --n_calib 1200

# Probes. probe_axis_views.py is the CONTROLLED axis probe (see Appendix A, Step 3.1):
# it holds the representation fixed and varies only which axis is attended over.
python scripts/probe_tokenizer.py --ckpt <tok.pt> --tag <TAG> --pool tok_mean \
  --group_col new_PatientID --device 0 --n 10000
python scripts/probe_axis_views.py --ckpt <tok.pt> --tag <TAG> --device 0 --n 10000
```

## 5. Explicitly rejected (do not re-propose without new evidence)

| rejected | why |
|---|---|
| Swap Q-Former for MLP / token-preserving projection | The MLP-wins literature fully fine-tunes the LLM. In frozen+LoRA — ours — every clean ablation reverses (Flamingo 70.7 vs 66.6; Idefics2 **+8.5** from *adding* the resampler; frozen-Whisper speech 2.28 vs 3.00 WER; Garg & Bas +7.4 frozen vs +1.7 trained). |
| Query-orthogonality regulariser (ORCA) | Ruled out empirically — query off-diagonal cosine 0.243/0.167 vs 0.923 collapsed. Queries are healthy. |
| More codebooks / bigger token budget | Saturated: 2→8 fixed ACS but not json. Depth probe: pooled linear decodability saturates at cb1–2 (sheets 1→8 macro 0.854→0.865). |
| 16-codebook tokenizer (x2_depth16) | Ties on endpoints, doubles bits/token ⇒ mandatory bridge rebuild. |
| Unfreezing the tokenizer | Prismatic: full FT of the visual backbone degrades localisation (p=0.00381). LoRA only, if ever. |
| Best-of-N with judge selection | BANNED — circular (judge-select then judge-score), degenerate on binary, not deployable. |
| Softmax codebook fusion | Retired Aug 18. Measured on the trained champion: 99% of gate weight on cb0–cb1 (770× disparity), mixed-vector norm 32.7 vs 174.8 for the sum. It had silently re-created `kept≈2`. |

## 6. Standing hazards

- **R1 Judge versions.** Never compare across judge versions (old e4d CSV = spurious +0.16).
- **R1b GT versions.** The original three-way table was contaminated — e4d scored on **pre-REGEN**
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
  ORCA 2607.06014, PARCEL 2605.30126 — re-read before manuscript use.
- **R8 Judge wrapper needs ABSOLUTE paths.** `run_csv_llm_judge_sharded.sh` `cd`s into
  `/volume/LLM_JUDGE`, so relative CSV/OUTPUT/OUT_DIR make every shard fail with FileNotFoundError.
  Also verify the exit code AND a non-empty output json — a wrapper echoing success unconditionally
  hid a total failure on Aug 24.
- **R9 Patient-ID formats differ across parquets.** Train stores `'337511.0'`, test stores
  `'338306'` — a raw string leak-check matches nothing and passes **vacuously**. Normalise via
  `pd.to_numeric(...).astype('Int64').astype(str)`. (True overlap verified: 0 patients, 0 ECGs.)
- **R10 Rare endpoints starve uniform sampling.** ACS is labelled on only 1.9% of train ECGs, so a
  1,500-ECG uniform draw yielded 68 labels and the fit was silently skipped. Stratify per endpoint.
- **R11 `generate_all_qa_pairs.py --resume` counts OUTPUTS, not INPUTS.** The loop `continue`s past
  any row whose waveform fails to load or whose generation raises
  (`inference/generate_all_qa_pairs.py`, the `except Exception` at the bottom of the row loop), but
  resume sets `start_idx = len(results)`. After *k* skipped rows a restart rewinds *k* rows too far
  and **duplicates** already-generated samples in the CSV. Use the shard-level
  `scripts/run_fulltest_gen_resumable.sh` (each shard is validated and regenerated whole, never
  partially continued); `scripts/score_deterministic.py` additionally de-duplicates on
  `waveform_name`/`question`/`json_key`/`prompt` as a second line of defence. Do not advertise the
  per-row `--resume` flag as safe until it persists the input index.

## 7. Suggested order from here

1. **Step A** — decide the readout policy (zero compute, largest immediate clinical gain).
2. **Step B** — commit and PR the working tree (two weeks of undefended work).
3. **Step C.1** — q64 rerun of the time-axis bridge (cheapest remaining lever on `json_interpretation`).
4. **Step D** — parallel RVQ, only if q64 does not move it.
5. **Step E** — de-binarise SHD last, once the architecture is frozen.

---

## Appendix A — the original hand-off brief (Aug 2026, superseded)

Kept for provenance: this is the plan the work above was executed against, with the reasoning that
motivated each step. **Its status labels are stale** — Steps 1, 2, 3.1 and 3.2 are done — and two of
its headline numbers were later shown to be measurement artifacts (R1b). Corrections raised in the
PR #134 review are applied inline and marked **[corrected]**.

<details>
<summary><b>A.1 — Original context table and the "one unsolved problem"</b></summary>

The `x1_split` tokenizer (reconstruction + auxiliary losses + split codebooks) with a full-ETG
Q-Former bridge reading **all 8 codebooks** was the **first configuration to beat production e4d**:

| stage | metric | e4d | kept=2 | **kept=8** |
|---|---|---|---|---|
| tokenizer codes | LVEF / AFib | 0.75 / 0.60 | — | **0.85 / 0.78** |
| bridge output | LVEF / AFib | 0.74 / **0.55** | 0.78 / 0.73 | 0.76 / 0.69 |
| LLM (full 49,776 test) | LVEF / AFib / SHD / ACS | 0.80 / 0.65 / 0.67 / 0.72 | 0.82 / 0.72 / 0.68 / 0.62 | **0.83 / 0.73 / 0.68 / 0.74** |
| LLM judge | overall | 0.643 | 0.582 | **0.658** |

Probes: frozen encoder/quantizer/bridge, patient-grouped split (`new_PatientID`), millivolts,
n≈10k. Age+sex floors: AFib 0.62 · LVEF 0.53 · SHD 0.45.

**The then-unsolved problem:** `json_interpretation` was **0.518 vs e4d 0.784 (−0.27)** and it did
**not** recover going from 2 → 8 codebooks (0.546 → 0.518).
**[corrected]** Two-thirds of that −0.27 was a GT-version artifact (hazard R1b): the e4d column was
scored on pre-REGEN ground truth. The honest gap was −0.09 at the time and is **−0.048** today
against the time-axis bridge. The `0.643` / `0.658` judge overalls in the table above are likewise
**not** comparable to the §1 scoreboard — use §1.

</details>

<details>
<summary><b>A.2 — Original evaluation commands (superseded by §4)</b></summary>

```bash
# Generation, three shards in parallel — one process per shard.
# [corrected] The original brief wrote a single command with `regen_test_shard{0,1,2}.parquet`
# and `--device {i}`. Bash brace-expansion produces THREE positional values, but
# `--validation_parquet` takes exactly one, and `{i}` is not a valid int for `--device`,
# so that command could never run. Loop instead:
for i in 0 1 2; do
  python inference/generate_all_qa_pairs.py --checkpoint <best_model.pt> \
    --validation_parquet "output/regen_test_shards/regen_test_shard${i}.parquet" \
    --output_dir "<out>/shard${i}" --device "${i}" --save_interval 5000 &
done
wait
# [corrected] `--resume` is deliberately omitted: it counts outputs, not inputs, and duplicates
# rows after any skipped sample (hazard R11). Prefer scripts/run_fulltest_gen_resumable.sh.

# judge-free endpoint scoring (fast, unconfounded)
python scripts/score_deterministic.py "<merged>.csv" --tag <NAME>

# LLM judge — MUST use the same sample+seed for every model compared
cd /volume/LLM_JUDGE && .venv/bin/python judge_eval.py --csv <merged>.csv \
  --sample 3000 --stratify --seed 42 --output <out>.json
```

**[corrected]** The review flagged that `scripts/score_deterministic.py` tested for the substring
`present` before checking `not present` / `no structural` / a leading `no`, so
"structural heart disease is not present" scored as a positive. That was true of the version this
brief was written against and is **fixed on `main`**: `parse_bin()` now calls
`has_binary_negation()` (`utils/endpoint_readout.py`) **first** for every binary task, and that
helper matches leading negatives, `no/without <term>`, `no evidence of <term>`,
`does not show <term>`, `<term> is/was not present`, `not <term>`, `low risk of <term>` and
`<term> … unlikely`. No action outstanding.

</details>

<details>
<summary><b>A.3 — STEP 1 (DONE): fix the codebook fusion</b></summary>

The softmax gate in `models/bridge/bridge.py` collapsed the 8 RVQ codebooks with a **convex
combination** (an average) although RVQ is **additive**. No surveyed system does this (MusicGen,
Moshi, VALL-E, RQ-Transformer, VILA-U all sum).

Measured on the trained champion at the time of writing:
- gate weights **0.305 → 0.017** across cb0→cb7 (**18.4× disparity**)
- mixed-vector norm **22.6** vs **172.7** for the plain sum (**7.6× amplitude loss**)
- learned bias monotone *against* the deep codebooks that carry morphology
- our own probe: **concat beats fixed-weight collapse on all four endpoints**

(A later re-measurement, recorded in §5, put the disparity at 770× and the norms at 32.7 vs 174.8.
Both measurements point the same way; §5 is the one to quote.)

**Change, as shipped:** replace the gate with `concat(8×768) → Linear(6144→768)` whose per-codebook
768×768 blocks are initialised to the identity; drop `bias_last_codebook` from the path. ~4.7M
params on a 200M bridge. Plain unweighted `sum` is retained as the ablation floor
(`mix_strategy: sum`).

**[corrected] The original "safe initialisation" rationale was wrong, and it matters.** The brief
claimed the identity-sum init "reproduces current behaviour at init and can only improve". It does
not reproduce the *current* behaviour:

1. The incumbent fusion is an **input-dependent softmax** produced by the `mix_gate` MLP plus
   `bias_last_codebook`. A single static linear map cannot express a per-example gate, so identity
   init reproduces the **plain sum**, i.e. the ablation floor — not the trained champion.
2. The 768-d vectors being summed are the bridge's **own learned `embed_tables`**
   (`nn.Embedding(vocab, d_mid)`), not the tokenizer's 82-d RVQ codebook vectors that the
   tokenizer's decoder consumes. RVQ additivity therefore does **not** license "summing them is
   equivalent"; the additivity argument is an analogy, not an identity.

Consequence: switching fusion is **not** behaviour-preserving and **requires a retrain** — treat it
as a new arm, never as a free in-place swap of a trained checkpoint. That is how it was in fact run
(`concat_linear` is a separately trained Stage-1 + Stage-3 pair), and it won on its own merits (§1).
Related, from STEP A0: `mix_strategy` is **inert** when `token_axis: time`, so the fusion fix and
the axis change have never actually composed.

</details>

<details>
<summary><b>A.4 — STEP 2 (DONE): score the abandoned query-count ablation</b></summary>

`checkpoints/stage2_qformer_layer_sensitivity/8cb_2layers_h512_q{32,64,128}` existed with **empty
rows** in `analysis/stage2_qformer_layer_sensitivity/..._summary.csv` — set up, never scored.
Now scored: **q64 is the optimum** (R@1 0.605 vs 0.561 @q32 / 0.546 @q128), which is what makes
Step C.1 the cheapest remaining lever.

</details>

<details>
<summary><b>A.5 — STEP 3 (DONE): expose TIME as the token axis — the architectural argument</b></summary>

**The defect.** `Conv1d` returns `(B, C, L)`. Our encoder: body → (B,384,79) →
`adaptive_avg_pool1d(·,82)` → (B,384,**82**) → `proj` → **(B, 128 channels, 82 timesteps)**.
`ResidualVQ(dim=82)` quantises the **last** dim, so it treats **channels as the token axis** and
the **time course as the feature vector**. The Q-Former's 128 key/value positions are
*channels*; `_time_pe` encodes **channel index**.

**Consequence:** the LLM has **no time axis to attend over**. "ST elevation in V2–V4, 200–300 ms
after QRS onset" has no coordinate system. Adding codebooks adds *information* but not
*addressing* — which is why kept=8 fixed binary ACS (0.62→0.74) but not the
enumeration-of-localised-findings task. **That dissociation was the strongest evidence for the
hypothesis**, and Step 3.2 went on to win the judge (§1).

Three stages, cheapest first — **do not jump to a full retrain**:

1. **Probe-only prototype (hours, no training).** Frozen tokenizer, ask whether localised
   (ST/Q/T-wave) signal is more addressable along time than along channels.
   **[corrected]** The original brief said "transpose the tensor and run the standard linear
   probe". That test is void: in `scripts/probe_tokenizer.py`, `--pool flat` makes the transpose a
   pure feature permutation (identical linear problem), while the default `concat` pooling changes
   *which* axis is mean/max-pooled and even the feature dimensionality — confounding axis semantics
   with pooling capacity. The valid test needs a probe that **attends over one nominated axis** with
   representation and pooling capacity held fixed. That is what shipped as
   `scripts/probe_axis_views.py`: `flat` as the transposition-invariant information ceiling,
   `marg_time` / `marg_chan` as the marginal profiles, and `mil_time_*` / `mil_chan_*` MIL heads
   (per-token linear scorer + LSE/max pool) as the attention analog of a Q-Former query.
   Result at n=10k: **time-token MIL 0.86 vs channel-token MIL 0.58** (localised battery 0.82 vs
   0.54) — hypothesis confirmed, Step 3.2 authorised.
2. **Bridge-only change (days).** Keep the tokenizer; feed the bridge a time-major sequence
   (82 time positions) with a real time positional encoding; retrain Stage-1 + MedGemma.
   Isolates "can the bridge use time?" from "can the tokenizer represent time?". Shipped as
   `bridge_token_axis: time`; it is the current champion — but see **STEP A0**, because it also
   changed kv *content* from discrete code embeddings to continuous `z`, so the win is not yet
   attributable to the axis alone.
3. **Tokenizer redesign (weeks).** Retokenize per **time patch** so each code is a time slice.
   Invalidates every existing checkpoint — only if (1) and (2) point that way.

</details>

<details>
<summary><b>A.6 — STEP 4 / STEP 5 (now STEP D / STEP E) and the original rejection list</b></summary>

**Split / parallel RVQ (now STEP D).** `x1_split` supervises codebooks **[0,1] of a single residual
chain**, so the remaining 6 levels encode the residual *after* endpoint supervision and morphology
gets crowded out. Moshi/Mimi documented exactly this conflict (*"higher-order quantizers operate on
the residual of the first one, the latter needs to trade audio quality for phonetic
discriminability"*: ABX 23.3→6.5% but MUSHRA 65.9→**57.8**) and fixed it with a **parallel/split
RVQ** — semantics in a *separate* VQ, not a prefix of the chain — recovering MUSHRA to **64.0** at
equal semantics. Predicts our kept=2 failure precisely.

**De-binarise SHD (now STEP E).** The SFT parquet has **only** binary structural answers
(149,153 No / 105,307 Yes, **zero** multi-condition), so the model can only ever say Yes/No — part
of why SHD is stuck at ~0.68. Regenerate with the EchoNext 7-label GT, **only after the architecture
is settled**, otherwise a win is un-attributable.

**Original rejection list.** Superseded by §5, which carries the same five rows plus
*Best-of-N with judge selection* and *Softmax codebook fusion*. The one detail worth keeping from
the original wording, now folded into §5: the frozen-encoder speech ablation (frozen-Whisper
2.28 vs 3.00 WER) alongside Flamingo 70.7 vs 66.6, Idefics2 **+8.5** from *adding* the resampler,
and Garg & Bas **+7.4 frozen vs +1.7 trained**.

**Original standing hazards 1–6** map onto §6 as: 1 → R1, 2 → R2, 3 → R3, 4 → R4, 5 → R5, 6 → R7.
Nothing from that list was dropped; R1b, R1c and R6, R8–R11 are additions learned since.

</details>

<details>
<summary><b>A.7 — Original suggested order (historical)</b></summary>

**Step 2** (free) → **Step 1** (cheap, independent) → **Step 3.1** (hours; falsifies or confirms
the biggest hypothesis) → Step 3.2 / Step 4 depending on 3.1 → **Step 5 last**, once the
architecture is frozen.

Executed as written. All four of Step 2, Step 1, Step 3.1 and Step 3.2 are complete; the live
ordering is §7.

</details>
