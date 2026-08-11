# ECG Tokenizer — Auxiliary-Loss Program (reproduce)

Make the frozen VQ tokenizer carry the physiologic information that a reconstruction-only
objective discards, and prove it with a **frozen linear probe** before touching the LLM.

- Notion (live tracker): https://app.notion.com/p/3af0855e87b6817c91f0c74c505cf2cf
- All probe metrics: `analysis/tokenizer_probe/probe_*.json`
- Signal representation: **millivolts** (`npy_path` × per-source scale, MHI 0.00488) — never
  `waveform_path_psa` (FFT-normalized, needs batch stats, does not generalize).
- Split: **patient-grouped** (`GroupShuffleSplit`, grouped on `new_PatientID`, 0 patient
  overlap). Row-level splitting leaks patients (~1.29 ECG/patient) and inflates AUROC 0.02–0.07.

## Headline result

Auxiliary losses on the **post-quantization** codes (so the codebook is forced to carry the
info) raise the code-probe ceiling substantially, **unsupervised** for the echo/prediction
endpoints. Winner = **x1_split** (wider encoder + split/semantic codebooks).

| tokenizer | encoder | diag macro | LVEF≤40 | AFib-5y | SHD | ACS | axis L |
|---|---|---|---|---|---|---|---|
| CONTROL (recon-only) | 173K | 0.764 | 0.74 | 0.61 | 0.63 | 0.73 | 0.80 |
| E0 aux (recon+diag+phys) | 173K | 0.856 | 0.84 | 0.65 | 0.64 | 0.80 | 0.85 |
| E1 enc-6M | 6.0M | 0.887 | 0.83 | 0.67 | 0.68 | 0.83 | 0.86 |
| E1 enc-13M | 13.4M | 0.896 | 0.81 | 0.67 | 0.69 | **0.89** | **0.88** |
| E2 endpoint-heads | 6.0M | 0.859 | 0.83 | 0.78 | **0.72** | 0.79 | 0.80 |
| E3 aux_w=0.02 | 6.0M | 0.847 | 0.80 | 0.64 | 0.69 | 0.81 | 0.82 |
| **x1_split (WINNER)** | 6.0M | 0.865 | **0.85** | **0.78** | 0.69 | 0.83 | 0.85 |
| x2_depth16 (16 cb) | 6.0M | 0.872 | 0.85 | 0.79 | 0.73 | 0.81 | 0.83 |
| x3_baseline_split | 173K | 0.843 | 0.80 | 0.72 | 0.68 | 0.79 | 0.83 |

All values are AUROC, frozen linear probe, patient-grouped, mV, n≈10k MHI ECGs. Age+sex-only
floors on this split: **AFib 0.62 · LVEF 0.53 · SHD 0.45** (so every number above is real ECG
signal). Benchmarks: LVEF≤40 target 0.90–0.92 (ECG-FM linear 0.91); AFib-5y 0.78 (Jabbour 2024
EHJ, our MHI cohort) — **matched** by x1_split/x2.

## What each experiment established

- **E0 — aux losses work.** recon-only → +aux lifts LVEF 0.74→0.84 and AFib 0.61→0.65, both
  **never supervised** (EF/AFib coverage too sparse to train a head) → genuine info gain in the
  codes. Reconstruction *improved* (MAE 0.19→0.046), not degraded.
- **E1 — encoder scaling is a clean dissociation.** 173K→13.4M lifts morphology (diag
  0.856→0.896, axis→0.88, ACS→0.89) but NOT the echo/prediction endpoints (LVEF, AFib flat).
  Encoder capacity is not the binding constraint for LVEF/AFib.
- **E2 — endpoint heads: best endpoints, morphology cost.** Directly supervising LVEF/AFib/SHD
  gives AFib **0.78** (= Jabbour on a frozen probe) and SHD 0.72, but costs morphology
  (axis 0.88→0.80, diag→0.859). The two objectives fight over the shared 128-d code space.
- **E3 — aux-weight sweep NEGATIVE.** λ=0.02 worse than λ=0.3 across the board. Keep λ=0.3.
- **x1_split — resolves the trade-off.** Split/semantic codebooks: endpoint heads read the
  **first 2 of 8** codebooks; reconstruction uses all 8. Keeps E2's endpoints (AFib 0.78,
  LVEF 0.85) **and** recovers morphology (axis 0.85), with the **8-codebook LLM interface
  unchanged**. Winner.
- **x2_depth16 — not worth it.** 8→16 codebooks ties x1 on endpoints (SHD +0.04) but doubles
  the bits/token → **mandatory bridge rebuild, not drop-in**. Rejected.
- **x3_baseline_split — proves the encoder is load-bearing.** x1's recipe on the 173K baseline
  encoder regresses (AFib 0.72 vs 0.78, LVEF 0.80 vs 0.85) → the 6M ScalableEncoder is needed.

## Reproduce

Env: `pip install uv && uv sync && source .venv/bin/activate`. GPUs: DDP via torchrun.

### 1. Build auxiliary targets (HR/intervals + endpoint labels, sentinels cleaned)
```bash
python scripts/build_tokenizer_aux_targets.py    # -> output/tokenizer_aux_targets_{train,val}.parquet
```

### 2. Train tokenizers (self-contained; does NOT touch the shared runner)
`scripts/train_tokenizer_aux.py` — recon + masked aux losses on the post-quant tensor. Key args:
`--manifest`, `--path_col npy_path`, `--scale 0.00488`, `--labels_from <REGEN parquet>`,
`--enc_width` (0 = baseline Residual_Conv_Encoder 173K; >0 = ScalableEncoder width-N),
`--sem_codebooks N` (split latent: endpoint heads read first-N codebooks),
`--aux_weight 0.3`, `--num_quantizers`.

```bash
MAN=output/mhi_mv_manifest.parquet
LBL=output/combined_train_qa_m5000k_h5000k_weighted.REGEN.parquet
COMMON="--manifest $MAN --path_col npy_path --scale 0.00488 --labels_from $LBL \
        --aux_weight 0.3 --num_quantizers 8 --codebook_size 512 --epochs 3 --batch_size 32 --lr 3e-4 --wandb"

# CONTROL (recon-only baseline)          — omit aux via --aux_weight 0
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --aux_weight 0 --enc_width 0 --out checkpoints/tok_mv_ctrl
# E0 aux (baseline encoder + aux)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 0 --out checkpoints/tok_mv_aux
# E1 encoder scaling
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 64 --out checkpoints/e1_enc6M
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 96 --out checkpoints/e1_enc13M
# E2 endpoint heads (endpoints supervised; enc_width 64)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 64 --endp_weight 1.0 --out checkpoints/e2_endpoint
# E3 aux-weight sweep (negative)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 64 --aux_weight 0.02 --out checkpoints/e3_auxw002
# x1_split WINNER (split codebooks: endpoint heads read first 2 of 8)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 64 --endp_weight 1.0 --sem_codebooks 2 --out checkpoints/x1_split
# x2_depth16 (16 codebooks — single GPU, breaks LLM interface)
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_tokenizer_aux.py $COMMON --enc_width 64 --endp_weight 1.0 --sem_codebooks 2 --num_quantizers 16 --out checkpoints/x2_depth16
# x3_baseline_split (drop-in test: baseline encoder + split)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_tokenizer_aux.py $COMMON --enc_width 0 --endp_weight 1.0 --sem_codebooks 2 --out checkpoints/x3_baseline_split
```

### 3. Probe (frozen, patient-grouped) — the acceptance criterion
```bash
python scripts/probe_tokenizer.py --ckpt checkpoints/x1_split/tokenizer_aux_final.pt \
  --tag X1_SPLIT_PT --pool tok_mean --group_col new_PatientID --device 0 --n 10000
# -> analysis/tokenizer_probe/probe_X1_SPLIT_PT.json  (diagnostic_macro + LVEF/AFib/SHD/ACS/axis AUROC + 95% CI)
```

### 4. Integrate the winner into the MedGemma / Q-Former pipeline
The winner uses `ScalableEncoder` (now registered in `models/scalable_encoder.py`). Convert the
standalone checkpoint to the production `pretrained_tokenizer_path` format (validates enc/quant
keys; guarantees a drop-in load):
```bash
python scripts/adapt_tokenizer_ckpt.py --in checkpoints/x1_split/tokenizer_aux_final.pt \
  --out checkpoints/x1_split_adapted/tokenizer_prod_format.pt
```

### 5. Q-Former bridge sweep (Stage-1) on the winner
`config/ecg_text_stage1/winner_base_config.yaml` (MedGemma-4B text encoder) + 3 codebook-selection
variants (`winner_sweep_{cb2_off0,cb8_all,cb1_last}.yaml`; the split tokenizer front-loads endpoint
info into coarse codebooks [0,1] → sweep `num_codebooks_kept × codebook_offset`). batch 120,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
```bash
bash scripts/runner.sh --base_config config/ecg_text_stage1/winner_sweep_cb2_off0.yaml \
  --selected_gpus 0,1 --use_wandb true --run_mode train
```

### 6. MedGemma retrain (e4d recipe, new tokenizer + bridge)
`config/llm_finetuning/medgemma/e4d_x1split_retrain.yaml` — same REGEN dataset; set the 3 TODOs
(`stage1_checkpoint_path`, `num_codebooks_kept`, `codebook_offset`) from the bridge-sweep winner.
```bash
bash scripts/runner.sh --base_config config/llm_finetuning/medgemma/e4d_x1split_retrain.yaml \
  --selected_gpus 0,1,2 --use_wandb true --run_mode train
```
**LLM transfer is the pre-registered gate** — every prior representation win died here; no success
claim until Stage-2/3 beats the current e4d on the LLM-judge eval.

## Files (this branch)
- Code: `models/scalable_encoder.py`, `models/__init__.py`,
  `scripts/{train_tokenizer_aux,probe_tokenizer,build_tokenizer_aux_targets,adapt_tokenizer_ckpt}.py`
- Configs: `config/ecg_text_stage1/winner_*.yaml`,
  `config/llm_finetuning/medgemma/e4d_x1split_retrain.yaml`

---

## Downstream results: tokenizer → bridge → LLM

Full write-up (with literature references): Notion "Information Cascade: Tokenizer → Q-Former → LLM".

### Stage 1 — tokenizer (raw codes, frozen probe, patient-grouped, mV)
| tokenizer | diag | LVEF≤40 | AFib 5y | SHD | ACS |
|---|---|---|---|---|---|
| e4d's `tfq5q94l` (recon-only) | 0.758 | 0.75 | 0.60 | 0.65 | 0.72 |
| DeepECG-SSL v2 WCR (external SSL, frozen) | 0.82 | 0.80 | 0.63 | 0.62 | 0.84 |
| **`x1_split` (recon+aux+split)** | **0.865** | **0.85** | **0.78** | **0.69** | 0.83 |

### Stage 2 — Q-Former bridge output (what the LLM receives)
| bridge | diag | LVEF≤40 | AFib 5y | SHD | ACS |
|---|---|---|---|---|---|
| e4d bridge (`j4bb0w33`) | 0.808 | 0.74 | **0.55** (≈age+sex floor 0.62) | 0.65 | 0.78 |
| kept=2 (coarse codebooks [0,1]) | 0.851 | 0.78 | 0.73 | 0.68 | 0.81 |
| kept=8 (all codebooks) | 0.828 | 0.76 | 0.69 | 0.67 | 0.78 |

Bridge loss vs raw codes: LVEF −0.07, AFib −0.05. Query collapse ruled out (off-diagonal cosine 0.24 / 0.17 vs 0.923 collapsed).
⚠️ The probe does **not** predict generation: kept=2 probes best on every endpoint yet collapses on ACS/JSON at the LLM stage.

### Stage 3 — LLM generated outputs (full 49,776-row REGEN test)
Deterministic (`scripts/score_deterministic.py`; LVEF = AUROC from parsed EF, others = balanced accuracy):
| endpoint | e4d | kept=2 | **kept=8** |
|---|---|---|---|
| LVEF ≤40 | 0.80 | 0.82 | **0.83** |
| AFib 5y | 0.65 | 0.72 | **0.73** |
| SHD | 0.67 | 0.68 | **0.68** |
| ACS-acute | 0.72 | 0.62 | **0.74** |

LLM-judge (all re-scored with the **current** MiniMax judge, stratified-3000 seed 42):
**overall e4d 0.643 · kept=2 0.582 · kept=8 0.658** — kept=8 is the first configuration to beat production e4d.
Biggest gains: chamber +0.19, AFib +0.10, conduction +0.04. **Holdout: `json_interpretation` 0.518 vs e4d 0.784 (−0.27), which did NOT recover with more codebooks.**

⚠️ **Never compare across judge versions.** The pre-existing e4d judge CSV is from Jan 2026 (older judge + smaller ontology) and produces a spurious +0.16. Always re-score the baseline with the current judge on the same sample.

### Two defects found in the bridge (see Notion §5)
1. `models/bridge/bridge.py:677` collapses the 8 RVQ codebooks with a **softmax convex combination** (an average) although RVQ is **additive**. Measured: gate weights 0.305→0.017 (18.4× disparity), mixed-vector norm 22.6 vs 172.7 for the plain sum. Fix: `concat(8×d) → Linear` initialised to the identity-sum.
2. The encoder emits `(B, 128 channels, 82 timesteps)` and `ResidualVQ(dim=82)` quantises the last dim, so the bridge's 128 positions are **channels, not time** — `_time_pe` encodes channel index and no query can address a temporal window. This is the leading explanation for the unrecovered `json_interpretation` deficit.
