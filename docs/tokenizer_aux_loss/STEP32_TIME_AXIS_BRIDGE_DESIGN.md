# Step 3.2 - Time-indexed bridge (design, pre-implementation)

Status: DESIGN ONLY (drafted Aug 14 while the Step-1 MedGemma retrain `hcjs6vk9` occupies all
GPUs). Implementation + launch gated on the Step-1 judge verdict (~Aug 17-18).

## Why (evidence from Step 3.1, Aug 12-14)

- Axis probe (`analysis/tokenizer_axis_probe/axis_probe_X1_SPLIT.json`, n=10k patient-grouped):
  linear MIL scorers over **time tokens** (82 positions × 128-d channel vector) reach macro
  **0.86** / localised ST-Q-T **0.82**; the same scorers over **channel tokens** (what the
  Q-Former sees today) get **0.58** / **0.54** - barely above chance. Per-label:
  ST-elev-inferior 0.92 vs 0.64, Q-wave-anterior 0.90 vs 0.52.
- Mechanism: attention over channel tokens (shared per-token projections, simplex weights)
  cannot form cross-channel contrasts; those are exactly the readable directions.
- `marg_chan` (plain mean over time) ties MIL selection → the *feature space* per time slice is
  what matters, not fancy temporal selection. Findings recur every beat.

## The constraint that kills the naive version

`ResidualVQ(dim=82)` assigns codes **per channel** (grid = 128 channels × 8 levels). A
per-timestep code does not exist, so "transpose the ids" is meaningless. The time axis exists
only in the continuous post-quant tensor z (B, 128, 82) = Σ_d codebook_d[ids].

## Design: rebuild z inside the bridge, time-major

Keep the data pipeline (code ids, shape (B, 128, 8)) untouched. New bridge input mode:

```
ids (B,128,8) ── codebook lookup (frozen tables, init from quantizer, 8×512×82 ≈ 336K params)
    → per-level z_d (B,128,82)
    → transpose each to (B,82,128)                      # tokens = 82 time slices
    → concat levels → (B,82,8×128=1024)
    → Linear(1024→d_mid) init to identity-sum blocks    # Step-1 fusion, time-major form
    → + sinusoidal PE over the 82 REAL time positions   # ~122 ms per step (10 s / 82)
    → Q-Former kv (all 82 positions valid; no padding)
```

- Composes Step 1 and Step 3.2 in one clean operator: identity-sum init ⇒ at step 0 the kv
  features are exactly `zᵀ + PE` (the probe-validated view); training can only improve on it.
- Queries, `_QFormerBlock`s, Stage-1 heads (ETC/ETM/ETG), `to_llm` are unchanged - only the kv
  construction swaps. num_steps 128 → 82.
- Codebook tables FROZEN (init from `x1_split_adapted` quantizer): keeps information identical
  to the code path, isolating "addressing" from "information". (Trainable variant = ablation
  later, not first run.)

## Config surface

- `bridge_token_axis: channel | time` (default `channel` - full back-compat; `time` implies the
  additive/concat fusion path; `softmax` gate is channel-axis-only).
- Threading mirrors `bridge_mix_strategy` exactly (dataclasses, stage1/llm projects, wrapper,
  medgemma_decoder pop + `bridge_config` + hard mismatch guard vs the Stage-1 `config.yaml`).
- New adapter needs the quantizer checkpoint at bridge init: pass codebook tensors from the
  wrapper's already-loaded quantizer (do NOT re-read the file inside the bridge).

## Attribution / sequencing

The time-axis bridge inherits the additive fusion by construction, so its baseline is whichever
of {champion softmax 0.658, concat_linear (verdict pending)} stands after Step 1's judge run.
Decision matrix when `hcjs6vk9` verdict lands:
- concat_linear **wins** → 3.2 run vs concat_linear as the new baseline (isolated change:
  token axis only).
- concat_linear **loses/ties** → 3.2 still justified (probe evidence is independent of fusion),
  vs champion; fusion inside 3.2 stays identity-sum-init concat (it subsumes sum).

## Test plan

1. Unit: rebuilt Σ_d codebook_d[ids] equals the quantizer's z on real batches (atol 1e-5);
   transpose semantics; identity-init ⇒ kv == zᵀ+PE at step 0; old checkpoints load (channel
   mode untouched).
2. Stage-1 (~1 day, 2 GPUs, clone of `winner_cb8_fullETG_concatmix.yaml` + `bridge_token_axis:
   time`, new wandb project): watch next-acc (beat 0.884) AND tail retrieval; epoch-1 ETG
   readiness telling (concat_linear hit 0.835 at e1).
3. MedGemma (champion recipe + both knobs) → judge on same sample. Accept iff
   `json_interpretation` closes a meaningful share of the −0.27 gap, no AFib/LVEF regression.

## Risks

- 82 kv positions < 128: less kv capacity for 32 queries - probe says the view is strictly more
  readable, but watch rhythm/AFib (hazard R3: global findings can suffer when local resolution
  improves).
- PE scale: kv features are z-valued (norm ~175 pre-norm) vs embed-table-valued before;
  `input_norm` (RMSNorm) handles magnitude, but verify PE isn't drowned (PE added pre-norm,
  same as today).
- json_interpretation may need more than addressing (enumeration formatting, GT quirks) - the
  accept rule only requires closing a share of the gap, not all of it.
