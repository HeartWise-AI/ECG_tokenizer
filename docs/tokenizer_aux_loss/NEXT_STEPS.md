# NEXT STEPS — ECG Tokenizer / Bridge / LLM

> Hand-off brief. Written to be pasted straight into a new session as the task prompt.
> Full measurements + literature refs: Notion "Information Cascade: Tokenizer → Q-Former → LLM"
> and "NEXT STEPS — ECG Tokenizer / Bridge / LLM (execution plan)". PR: #132.

## Context (what is already true — do not re-litigate)

The `x1_split` tokenizer (reconstruction + auxiliary losses + split codebooks) with a full-ETG
Q-Former bridge reading **all 8 codebooks** is the **first configuration to beat production e4d**:

| stage | metric | e4d | kept=2 | **kept=8** |
|---|---|---|---|---|
| tokenizer codes | LVEF / AFib | 0.75 / 0.60 | — | **0.85 / 0.78** |
| bridge output | LVEF / AFib | 0.74 / **0.55** | 0.78 / 0.73 | 0.76 / 0.69 |
| LLM (full 49,776 test) | LVEF / AFib / SHD / ACS | 0.80 / 0.65 / 0.67 / 0.72 | 0.82 / 0.72 / 0.68 / 0.62 | **0.83 / 0.73 / 0.68 / 0.74** |
| LLM judge | overall | 0.643 | 0.582 | **0.658** |

Probes: frozen encoder/quantizer/bridge, patient-grouped split (`new_PatientID`), millivolts,
n≈10k. Age+sex floors: AFib 0.62 · LVEF 0.53 · SHD 0.45.

**The one unsolved problem:** `json_interpretation` is **0.518 vs e4d 0.784 (−0.27)** and it did
**not** recover going from 2 → 8 codebooks (0.546 → 0.518). More information did not help.

## Decision rule (applies to every experiment below)

Accept/reject on **generated-output** metrics, never on probe AUROC — we proved the probe
misleads (kept=2 probed best on every endpoint and still collapsed downstream).

- **Primary:** `json_interpretation` + per-label ST/Q-wave AUROC, full 49,776-row REGEN test.
- **Secondary:** judge overall + the four deterministic endpoints.
- **Always** re-score the baseline with the *current* judge on the *same* sample.

```bash
# generation (3-way sharded, saves every 5000 rows, resumable)
python inference/generate_all_qa_pairs.py --checkpoint <best_model.pt> \
  --validation_parquet output/regen_test_shards/regen_test_shard{0,1,2}.parquet \
  --output_dir <out>/shard{i} --device {i} --save_interval 5000 --resume

# judge-free endpoint scoring (fast, unconfounded)
python scripts/score_deterministic.py "<merged>.csv" --tag <NAME>

# LLM judge — MUST use the same sample+seed for every model compared
cd /volume/LLM_JUDGE && .venv/bin/python judge_eval.py --csv <merged>.csv \
  --sample 3000 --stratify --seed 42 --output <out>.json
```

---

## STEP 1 — Fix the codebook fusion  *(~1 day, low risk, do first)*

`models/bridge/bridge.py:677` collapses the 8 RVQ codebooks with a softmax **convex
combination** (an average) although RVQ is **additive** — `Σ_d e(code_d)` *is* the vector the
tokenizer's own decoder consumes. No surveyed system does this (MusicGen, Moshi, VALL-E,
RQ-Transformer, VILA-U all sum).

Measured on the trained champion:
- gate weights **0.305 → 0.017** across cb0→cb7 (**18.4× disparity**)
- mixed-vector norm **22.6** vs **172.7** for the plain sum (**7.6× amplitude loss**)
- learned bias monotone *against* the deep codebooks that carry morphology
- our own probe: **concat beats fixed-weight collapse on all four endpoints**

**Change:** replace the gate with `concat(8×768) → Linear(6144→768)` **initialised to the
identity-sum** (each 768×768 block = I); delete `bias_last_codebook`. Strictly generalises plain
sum and any fixed mixing, so at init it reproduces current behaviour and can only improve.
~4.7M params on a 200M bridge. **Also run plain unweighted sum as the ablation floor** — if
concat→Linear doesn't beat it, drop the parameters.

---

## STEP 2 — Score the abandoned query-count ablation  *(free)*

`checkpoints/stage2_qformer_layer_sensitivity/8cb_2layers_h512_q{32,64,128}` exist but have
**empty rows** in `analysis/stage2_qformer_layer_sensitivity/..._summary.csv`. Set up, never
scored. Free scaling curve.

---

## STEP 3 — 🔴 Expose TIME as the token axis  *(highest value, high risk)*

**The defect.** `Conv1d` returns `(B, C, L)`. Our encoder: body → (B,384,79) →
`adaptive_avg_pool1d(·,82)` → (B,384,**82**) → `proj` → **(B, 128 channels, 82 timesteps)**.
`ResidualVQ(dim=82)` quantises the **last** dim, so it treats **channels as the token axis** and
the **time course as the feature vector**. The Q-Former's 128 key/value positions are
*channels*; `_time_pe` encodes **channel index**.

**Consequence:** the LLM has **no time axis to attend over**. "ST elevation in V2–V4, 200–300 ms
after QRS onset" has no coordinate system. Adding codebooks adds *information* but not
*addressing* — which is exactly why kept=8 fixed binary ACS (0.62→0.74) but not the
enumeration-of-localised-findings task (`json_interpretation` 0.546→0.518). **That dissociation
is the strongest evidence for this hypothesis.**

Three stages, cheapest first — **do not jump to a full retrain**:

1. **Probe-only prototype (hours, no training).** Existing frozen tokenizer, transpose to
   `(B, 82 timesteps, 128 channels)`, run the standard frozen probe. Does a time-indexed view
   carry more ST/localised signal than the channel-indexed one? Cheap falsification first.
2. **Bridge-only change (days).** Keep the tokenizer; feed the bridge the transposed sequence
   (82 time positions) with a real time positional encoding; retrain Stage-1 + MedGemma.
   Isolates "can the bridge use time?" from "can the tokenizer represent time?".
3. **Tokenizer redesign (weeks).** Re-tokenise per **time patch** so each code is a time slice.
   Invalidates every existing checkpoint — only if (1) and (2) point that way.

---

## STEP 4 — Split / parallel RVQ on the next tokenizer

`x1_split` supervises codebooks **[0,1] of a single residual chain**, so the remaining 6 levels
encode the residual *after* endpoint supervision and morphology gets crowded out. Moshi/Mimi
documented exactly this conflict (*"higher-order quantizers operate on the residual of the first
one, the latter needs to trade audio quality for phonetic discriminability"*: ABX 23.3→6.5% but
MUSHRA 65.9→**57.8**) and fixed it with a **parallel/split RVQ** — semantics in a *separate* VQ,
not a prefix of the chain — recovering MUSHRA to **64.0** at equal semantics. Predicts our
kept=2 failure precisely.

---

## STEP 5 — De-binarise SHD  *(data-side, do last)*

The SFT parquet has **only** binary structural answers (149,153 No / 105,307 Yes, **zero**
multi-condition), so the model can only ever say Yes/No — part of why SHD is stuck at ~0.68.
Regenerate with the EchoNext 7-label GT. **Only after the architecture is settled**, otherwise a
win is un-attributable.

---

## Explicitly NOT doing

| rejected | reason |
|---|---|
| Swap the Q-Former for an MLP / token-preserving projection | The MLP-beats-Q-Former literature (Honeybee, Cambrian, MM1, DeCo) **fully fine-tunes the LLM**. In a **frozen + LoRA** regime — ours — every clean ablation reverses: Flamingo 70.7 vs 66.6; Idefics2 **+8.5** from *adding* the resampler; frozen-Whisper speech 2.28 vs 3.00 WER; Garg & Bas **+7.4 frozen vs +1.7 trained**. |
| Query-orthogonality regulariser (ORCA) | Ruled out empirically: query off-diagonal cosine **0.243 / 0.167** vs collapsed reference 0.923. |
| More codebooks / bigger token budget | Saturated. 2→8 fixed ACS but made `json_interpretation` slightly *worse*. Bottleneck is addressing, not capacity. |
| 16-codebook tokenizer (x2_depth16) | Ties x1_split on endpoints but doubles bits/token ⇒ mandatory bridge rebuild, not drop-in. |
| Unfreezing the tokenizer | Prismatic: full fine-tuning of the visual backbone *"dramatically degrades performance… especially on localization tasks"* (p=0.00381). LoRA only, if ever. |

## Standing hazards

1. **Judge versions.** The old e4d judge CSV is Jan-2026 (older judge, smaller ontology) → spurious **+0.16**. Re-score baselines with the current judge, same sample.
2. **Probe ≠ generation.** Never accept an architecture on probe AUROC.
3. **More resolution can hurt coarse tasks.** Chest X-ray: higher resolution helps small findings (nodule +0.042) but *degrades* global ones (cardiomegaly 0.82→0.80). Prefer multi-scale; always report rhythm/AFib when sweeping.
4. **The tokenizer may still be the ceiling.** Raw codes cap at LVEF 0.85 / AFib 0.78; removing *all* bridge loss recovers ~0.07 and no more.
5. **Small n.** ACS n=1,021 (357 pos), LVEF n=2,656. The validation-snapshot LVEF of 0.93 was a balanced-subset artifact; the honest full-test number is **0.83**.
6. **Unreviewed preprints** (LePaX 2607.06909, CheXpercept 2606.21020, CARE-X 2608.03890, ORCA 2607.06014, PARCEL 2605.30126) — re-read before manuscript use.

## Suggested order

**Step 2** (free) → **Step 1** (cheap, independent) → **Step 3.1** (hours; falsifies or confirms
the biggest hypothesis) → Step 3.2 / Step 4 depending on 3.1 → **Step 5 last**, once the
architecture is frozen.
