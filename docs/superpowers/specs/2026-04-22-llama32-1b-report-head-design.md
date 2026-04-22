# Llama-3.2-1B Report-Generation Head — Design Spec

**Date:** 2026-04-22
**Branch context:** `rb/gpt2_large_training`
**Author:** Rohan Banerjee
**Status:** Approved for implementation plan

## 1. Goal

Attach a Llama-3.2-1B report-generation head to the ECG tokenizer, mirroring the
working GPT-2 recipe but wired as a clean, parallel module that shares no code
with the GPT-2 path. Training recipe must be fair-comparable to the ECG-Byte
baseline (same LoRA target set, same trainable-parameter footprint) and produce
clinical interpretation reports from 12-lead PSA-adjusted ECGs.

## 2. Why replace the existing Llama code

The current Llama files in the repo are confirmed broken or unused:

- `models/llama32_tokenizer_decoder.py` calls
  `.generate(inputs_embeds=...)`, which is broken in transformers 4.56.2 (re-feeds
  the original embeds instead of continuing from generated tokens). There is no
  manual autoregressive fallback.
- `config/llm_finetuning/llama32_1b/lora_config.yaml` defines LoRA keys, but no
  code path ever calls `peft.get_peft_model(...)`, so LoRA is never applied.
- `config/llm_finetuning/llama32_1b/sequence_token_progressive_config.yaml`
  references `Llama32_SequenceTokenAdapter` / `LLAMA32_SEQUENCE_TOKEN_ADAPTER`
  that don't exist in `utils/enums.py`, and imports a missing
  `models/sequence_token_adapter.py`.
- `legacy_prefix_tuning_changes/llama32_tokenizer_decoder.py` is an abandoned
  prefix-tuning variant with TODO comments in cross-attention.

These files are deleted, not patched. See Section 6 for the full list.

## 3. Scope

### In scope

- A single LoRA recipe on Llama-3.2-1B: `q_proj` + `v_proj` only, rank 16,
  alpha 32, dropout 0, bias none. 16 layers × 4 LoRA tensors = 64 tensors,
  ~1.7M trainable params, matching ECG-Byte's PEFT default exactly.
- ECG conditioning via a single `<ecg>` special token whose embedding row is
  replaced at runtime by the `SequenceAdapter` output, fed through
  `inputs_embeds` — mirroring the GPT-2 path.
- Manual autoregressive generation loop with KV cache (copied pattern from
  `gpt2_tokenizer_decoder.py:299-359`, re-shaped for Llama / PEFT). Not an
  import; a re-implementation in the Llama file.
- Single-GPU training on H200, bf16, batch size 16, 1 epoch over 1,491,307 rows.
- Inference script that emits per-row predictions + ROUGE/BLEU/METEOR on the
  10k-row patient-disjoint test parquet.
- Unit tests (pytest) and one integration smoke test mirroring the GPT-2 side.

### Out of scope (explicit non-goals)

- Multi-GPU DDP / FSDP / Q-LoRA.
- Full fine-tuning variant.
- Sequence-token-progressive / prefix-tuning variants.
- ECG-Byte's discrete 10,261-signal-token vocabulary scheme (we use our
  adapter-based continuous `<ecg>` embedding; matches GPT-2, not ECG-Byte).
- LLM-judge scoring (can layer on top of `predictions.jsonl` later).
- Any changes to GPT-2 code.

## 4. Architecture

### 4.1 Model stack (inference shape)

```
ECG waveform (B, 12, 2500)      raw PSA-adjusted 12-lead @ 250Hz × 10s
      │
      ▼
ECG tokenizer wrapper (FROZEN)   models/ecg_tokenizer_wrapper.py
  encoder + quantizer            same checkpoint as the GPT-2 head
      │
      ▼
Quantized features (B, 128, 82)
      │
      ▼
SequenceAdapter (TRAINABLE)      models/adapters.py (reused, unchanged)
  attn + pos-embed + linear      (128, 82) → (B, 2048)
      │
      ▼
Prompt: [<ecg>, t1, ..., tN]
  token_embeds = embed_tokens(ids)           # (B, N+1, 2048)
  token_embeds[:, 0, :] = adapter_out        # replace <ecg> row
      │
      ▼
LlamaForCausalLM (base FROZEN;    models/llama32_report_decoder.py
LoRA on q_proj + v_proj,          (new, parallel to gpt2_tokenizer_decoder.py)
r=16, α=32, 16 layers)
      │
      ▼
CE loss / next-token logits
```

### 4.2 Trainable-parameter budget

| Component                        | Params     | Trainable |
|----------------------------------|-----------:|:---------:|
| ECG encoder + quantizer          |  (varies)  | ❌        |
| `SequenceAdapter`                | ~0.5–1M    | ✅        |
| Llama-3.2-1B base                | 1.26B      | ❌        |
| LoRA q_proj + v_proj × 16 layers | ~1.7M      | ✅        |
| `<ecg>` row of `embed_tokens`    | 2,048      | ✅        |
| **Total trainable**              | **~2.5–3M (~0.2%)** | |

Only the `<ecg>` row of the embedding table is unfrozen; the other ~128k rows
stay at the pretrained weights.

### 4.3 Training step

```python
waveform, input_ids, labels, attention_mask = batch

with torch.no_grad():
    features = ecg_tokenizer(waveform)               # (B, 128, 82)
ecg_embed = adapter(features)                        # (B, 2048)

token_embeds = decoder.get_input_embeddings()(input_ids)   # (B, N+1, 2048)
token_embeds[:, 0, :] = ecg_embed                          # replace <ecg>
# labels[:, 0] must be -100 (do not supervise the ECG placeholder)

outputs = decoder(inputs_embeds=token_embeds,
                  attention_mask=attention_mask,
                  labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

### 4.4 Generation loop (manual, KV-cached)

```python
@torch.inference_mode()
def generate_report(self, ecg_embed, max_new_tokens=256,
                    do_sample=False, temperature=1.0, top_p=1.0,
                    eos_token_id=None):
    device = ecg_embed.device
    B = ecg_embed.size(0)
    eos = eos_token_id or self.tokenizer.eos_token_id

    cur_embeds = ecg_embed.unsqueeze(1)              # (B, 1, 2048)
    attn = torch.ones(B, 1, dtype=torch.long, device=device)
    past_kv = None
    generated = torch.empty(B, 0, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        out = self.decoder(
            inputs_embeds=cur_embeds,
            attention_mask=attn,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past_kv = out.past_key_values

        if do_sample:
            logits = logits / max(temperature, 1e-5)
            # (optional top-p filtering here)
            next_tok = torch.multinomial(F.softmax(logits, -1), 1)
        else:
            next_tok = logits.argmax(-1, keepdim=True)

        next_tok = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_tok, eos),
            next_tok,
        )
        generated = torch.cat([generated, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == eos)
        if finished.all():
            break

        cur_embeds = self.decoder.get_input_embeddings()(next_tok)
        attn = torch.cat(
            [attn, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1,
        )

    return generated
```

Rationale for the manual loop: HF `generate(inputs_embeds=...)` is broken in
transformers 4.56.2. A wrapper-level re-implementation is already validated on
the GPT-2 side. Pattern is copied, not imported, because Llama's embedding
accessor differs from GPT-2's (`model.embed_tokens` vs `transformer.wte`, and
when wrapped by PEFT, `base_model.model.model.embed_tokens`).

## 5. File layout

### 5.1 New files

| Path | Purpose |
|---|---|
| `models/llama32_report_decoder.py` | `Llama32ReportDecoder` wrapper — mirror of `gpt2_tokenizer_decoder.py`. Holds `LlamaForCausalLM`, applies `peft.LoraConfig`, resizes embeddings for `<ecg>`, freezes everything except LoRA + adapter + `<ecg>` row, exposes `forward()` and `generate_report()`. |
| `config/llm_finetuning/llama32_1b/lora_config.yaml` | Single canonical config (rewrite of the existing file, see Section 7). |
| `scripts/train_llama32_report.py` | Training entrypoint. Mirrors the GPT-2 launcher; hands off to `LLMFinetuningRunner`. |
| `inference/generate_llama32_reports.py` | Inference script — loads checkpoint, runs greedy decode over 10k test parquet, writes `predictions.jsonl` + `metrics.json`. |
| `inference/test_val_generation_llama32.py` | Smoke script — loads real ckpt, prints 4 sample predictions. Matches the GPT-2 equivalent. |
| `tests/models/test_llama32_report_decoder.py` | Pytest unit tests (Section 8). |

### 5.2 Files deleted

| Path | Reason |
|---|---|
| `models/llama32_tokenizer_decoder.py` | Broken `generate(inputs_embeds=...)`; replaced by `llama32_report_decoder.py`. |
| `config/llm_finetuning/llama32_1b/base_config.yaml` | Full-FT variant; not wanted. |
| `config/llm_finetuning/llama32_1b/sequence_token_progressive_config.yaml` | References missing adapter class + bad enum. |
| `config/llm_finetuning/llama32_1b/lora_config.yaml` (old contents) | Rewritten at same path; see Section 7. |
| `legacy_prefix_tuning_changes/llama32_tokenizer_decoder.py` | Abandoned. |
| `test_sequence_token_implementation.py` (repo root) | Imports missing modules. |

### 5.3 Files reused unchanged

- `models/ecg_tokenizer_wrapper.py`
- `models/adapters.py` (the existing `SequenceAdapter` already supports
  `out_dim=2048`)
- `data/ecg_clinical_report_dataset.py`
- `runners/llm_finetuning_runner.py` (already supports two param groups via
  `adapter_lr` / `llm_lr`)
- `utils/metrics/llm_metrics.py` (ROUGE / BLEU / METEOR)
- `utils/wandb_wrapper.py`, `utils/ddp.py`, `utils/enums.py`

### 5.4 Registry / enum touch-ups (minimal, shared)

- Confirm `LLAMA32_SEQUENCE_ADAPTER` exists in `utils/enums.py` (it does).
  **Do not add** `LLAMA32_SEQUENCE_TOKEN_ADAPTER` — that was part of the deleted
  variant.
- Add `Llama32_Report_Decoder` to the model registry/factory used by
  `LLMFinetuningRunner`. The exact registration site will be identified during
  implementation; no GPT-2 entry is touched.

## 6. Config

`config/llm_finetuning/llama32_1b/lora_config.yaml`:

```yaml
# --- identity ---
pipeline_project: llm_finetuning
model_name: Llama32_Report_Decoder
runner_name: LLMFinetuningRunner
run_mode: train

# --- experiment bookkeeping ---
num_epochs: 1
seed: 42
base_checkpoint_path: null
wandb_project: ecg-llama32-report
wandb_entity: heartwise
use_wandb: true

# --- LLM backbone ---
decoder_mode: llm
decoder_name: Llama32_Report_Decoder
huggingface_model_name: meta-llama/Llama-3.2-1B
tokenizer_name: meta-llama/Llama-3.2-1B
llm_input_embedding_size: 2048
max_token_length: 256

# --- LoRA (matches ECG-Byte exactly) ---
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
lora_bias: none
lora_target_modules: [q_proj, v_proj]

# --- ECG tokenizer ---
ecg_tokenizer_checkpoint: <copied from GPT-2 config during implementation>
ecg_waveform_length: 2500
ecg_num_leads: 12
ecg_freeze: true

# --- adapter ---
adapter_name: Llama32_SequenceAdapter    # enum: LLAMA32_SEQUENCE_ADAPTER
adapter_input_shape: [128, 82]
adapter_output_dim: 2048

# --- ECG token ---
ecg_special_token: "<ecg>"
ecg_token_trainable: true

# --- optimization ---
optimizer: adamw
llm_lr: 2.0e-4          # LoRA params
adapter_lr: 2.0e-4      # adapter + <ecg> row
llm_weight_decay: 0.0
adapter_weight_decay: 0.0
scheduler_type: linear_warmup
num_warmup_percent: 0.03
grad_clip: 1.0

# --- batching / precision ---
batch_size: 16
num_workers: 8
precision: bf16
grad_accum_steps: 1

# --- data ---
train_dataset_path: combined_train_qa_interpretation_only.parquet
validation_dataset_path: paper/inputs/combined_test_m25kh25k_interpretation_10k.parquet

# --- eval / checkpointing ---
metrics: [rouge, bleu, meteor]
val_every_n_steps: 5000
save_every_n_steps: 5000
keep_last_n_ckpts: 3
save_best_metric: rouge_l

# --- generation defaults ---
generation:
  max_new_tokens: 256
  do_sample: false
  temperature: 1.0
  top_p: 1.0
  eos_token_id: null
```

Single config file. No `base_config.yaml`, no progressive-variant config — those
are deleted.

## 7. Dataset

- **Train:** `combined_train_qa_interpretation_only.parquet` — 1,491,307 rows,
  65.4% MHI / 34.6% MIMIC, PSA-adjusted 12-lead @ 250 Hz × 10 s,
  interpretation-only.
- **Test:** `paper/inputs/combined_test_m25kh25k_interpretation_10k.parquet` —
  10,000 rows, 50/50 MIMIC/MHI, patient-disjoint from train, same PSA
  preprocessing.

Both served through the existing `ECGClinicalReportDataset`.

## 8. Training operations

Total steps: `ceil(1,491,307 / 16) = 93,207`. Warmup: `round(0.03 × 93,207) =
2,796` steps. Validation: every 5,000 steps (≈18 validation passes per epoch).
Checkpoint retention: last 3 + best-by-rouge_l.

Two optimizer param groups:

- **Group A (adapter):** `adapter.parameters()` + `<ecg>` embedding row →
  `adapter_lr = 2e-4`, wd = 0.
- **Group B (LoRA):** all `p` with `requires_grad=True` not in Group A →
  `llm_lr = 2e-4`, wd = 0.

Both LRs are equal today; the split is preserved so they can diverge later
without code changes.

Launch protocol (honors the project's "run long jobs in tmux" rule):

```
tmux new -s llama32_report
uv run python scripts/train_llama32_report.py \
    --config config/llm_finetuning/llama32_1b/lora_config.yaml
```

Before the real run, implementation must include a one-shot memory probe to
confirm bs=16 fits on H200 at `max_token_length=256`, bf16. If it doesn't, bump
`grad_accum_steps` rather than reducing effective batch size.

Estimated wall-clock: ~93k steps × ~0.6 s/step ≈ 15–18 h for training, plus
~5 h total for the 18 validation passes. Under 24 h.

## 9. Testing plan

### 9.1 Unit tests — `tests/models/test_llama32_report_decoder.py`

All tests use a tiny `LlamaConfig` (1 layer, hidden=128, 2 heads) to stay fast
and avoid any Hub download in CI.

| # | Test | What it catches |
|---|---|---|
| 1 | `test_forward_shapes_and_loss` | `forward(inputs_embeds, attention_mask, labels)` returns scalar `loss` (grad-connected) and `(B, N+1, vocab+1)` logits. First label masked to -100. | Wrong label masking, adapter→LLM shape bugs. |
| 2 | `test_lora_wiring` | Only LoRA + adapter + `<ecg>` row are trainable. LoRA tensors exist on `q_proj` and `v_proj` of every layer; absent on `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. | The class of bug the old config had (LoRA defined but never applied). |
| 3 | `test_ecg_token_row_trainable` | Exactly one row of `embed_tokens.weight` has `requires_grad=True`. | Guards against training the full ~128k×2048 embed table. |
| 4 | `test_generate_report_shape_and_eos` | `generate_report(..., max_new_tokens=10)` returns `(B, ≤10)` longs; forcing EOS at step 3 via a logits hook stops one batch element there. | Manual loop correctness, EOS logic, post-EOS padding. |
| 5 | `test_generate_uses_kv_cache` | Second iteration inside the loop receives non-None `past_key_values` and length-1 `inputs_embeds`. | Guards against regressing to full-sequence re-feeding. |
| 6 | `test_generate_vs_forward_equivalence` | Greedy-decoding 5 tokens matches the 5th-token argmax of a single full-sequence forward over `<ecg>` + first 4 decoded tokens. | End-to-end correctness of the incremental loop — the exact failure mode of broken HF `generate(inputs_embeds=...)`. |
| 7 | `test_peft_wrapper_embedding_access` | `decoder.get_input_embeddings().weight` has `vocab+1` rows through the PEFT wrapper. | PEFT proxy regressions. |

### 9.2 Integration smoke test — `inference/test_val_generation_llama32.py`

Loads a real trained checkpoint, runs `generate_report` on 4 val samples, prints
predictions + ground truth. Not in pytest (needs GPU + checkpoint).

## 10. Open items handed to the implementation plan

- Exact path of the ECG tokenizer checkpoint to reuse (copy from the current
  GPT-2 config).
- Exact name of the model-registry / factory entry point (discovered from the
  GPT-2 decoder's registration site).
- Run the memory probe before launch; adjust `grad_accum_steps` if bs=16 does
  not fit.

These are implementation details, not design choices. They do not block writing
the plan.
