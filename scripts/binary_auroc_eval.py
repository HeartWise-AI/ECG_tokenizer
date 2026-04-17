#!/usr/bin/env python3
"""
Binary Yes/No AUROC evaluation for per-diagnosis ECG classification.

For each unique ECG in the test set, asks "Is there {diagnosis} present?"
for all 77 diagnoses, extracts P(Yes) from first-token logits, and computes
per-diagnosis AUROC.

Usage:
    PYTHONPATH=/volume/ECG_tokenizer python scripts/binary_auroc_eval.py \
        --checkpoint /media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt \
        --device cuda:2 --max_ecgs 5
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import ECG_PATTERNS
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper

# Transformers 5.x merged GemmaTokenizerFast into GemmaTokenizer
# and removed Trie from tokenization_utils. Patch both for checkpoint compat.
import importlib, types, sys

_gemma_mod = importlib.import_module("transformers.models.gemma")
if not hasattr(_gemma_mod, "tokenization_gemma_fast"):
    shim = types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
    from transformers import GemmaTokenizer
    shim.GemmaTokenizerFast = GemmaTokenizer
    sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = shim
    _gemma_mod.tokenization_gemma_fast = shim

import transformers.tokenization_utils as _tok_utils
if not hasattr(_tok_utils, "Trie"):
    class _Trie:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, state): pass
    _tok_utils.Trie = _Trie


# ---------------------------------------------------------------------------
# 1. Model loading (mirrors llm_finetuning_project._setup_inference_objects)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    from transformers import AutoTokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    hf_name = getattr(cfg, "huggingface_model_name", "google/medgemma-4b-it")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Reconstruct lora_config from individual fields (critical – see MEMORY)
    use_lora = any(
        "lora_A" in k or "lora_B" in k or "base_layer" in k
        for k in ckpt["model_state_dict"]
    )
    lora_config = None
    if use_lora:
        lora_config = {
            "r": getattr(cfg, "lora_r", 32),
            "lora_alpha": getattr(cfg, "lora_alpha", 64),
            "lora_dropout": getattr(cfg, "lora_dropout", 0.05),
            "target_modules": getattr(cfg, "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            "bias": getattr(cfg, "lora_bias", "none"),
            "top_k_layers": getattr(cfg, "lora_top_k_layers", None),
        }

    model = ECG_Tokenizer_Wrapper(
        encoder_name=getattr(cfg, "encoder_name", "Residual_Conv_Encoder"),
        quantizer_name=getattr(cfg, "quantizer_name", "ECG_Tokenizer_Quantizer"),
        decoder_name=getattr(cfg, "decoder_name", "MedGemmaDecoder"),
        num_quantizers=getattr(cfg, "num_quantizers", 8),
        codebook_size=getattr(cfg, "codebook_size", 512),
        decoder_mode="llm",
        bridge_name=getattr(cfg, "bridge_name", "InstructionAwareECGQFormerBridge"),
        huggingface_model_name=hf_name,
        llm_input_embedding_size=getattr(cfg, "llm_input_embedding_size", 2560),
        tokenizer=tokenizer,
        num_visual_tokens=getattr(cfg, "num_query_tokens", 32),
        bridge_mid_dim=getattr(cfg, "bridge_mid_dim", 768),
        bridge_num_heads=getattr(cfg, "bridge_num_heads", 12),
        bridge_dropout=getattr(cfg, "bridge_dropout", 0.1),
        bridge_num_special_tokens=getattr(cfg, "bridge_num_special_tokens", 4),
        bridge_qformer_layers=getattr(cfg, "bridge_qformer_layers", 10),
        bridge_text_hidden_size=getattr(cfg, "bridge_text_hidden_size", 768),
        bridge_bias_last_codebook=getattr(cfg, "bridge_bias_last_codebook", 0.5),
        bridge_codebook_dropout=getattr(cfg, "bridge_codebook_dropout", 0.0),
        bridge_cross_every=getattr(cfg, "bridge_cross_every", 2),
        instruction_dropout=0.0,
        use_lora=use_lora,
        lora_config=lora_config,
        num_codebooks_kept=getattr(cfg, "num_codebooks_kept", 8),
        codebook_offset=getattr(cfg, "codebook_offset", -1),
        prefix_tuning=False,
        stage1_checkpoint_path=getattr(cfg, "stage1_checkpoint_path", None),
    ).to(device)

    model._load_state_dict(ckpt["model_state_dict"], strict=True)
    if use_lora:
        model.set_lora_inference_mode(True)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Prompt construction (MedGemma chat format)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are an expert cardiologist. You interpret ECGs and answer "
    "in a concise, structured way."
)


def build_prompt_text(diagnosis_name: str) -> str:
    return (
        f"<start_of_turn>system\n{SYSTEM_MSG}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"<start_of_image>\n\n"
        f"Question: Is {diagnosis_name} present in this ECG? Answer Yes or No.\n\n"
        f"Respond concisely with the key finding or answer."
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def pretokenize_prompts(
    tokenizer, diagnoses: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-tokenize all 77 prompts, pad to equal length, return stacked tensors."""
    encodings = []
    for diag in diagnoses:
        text = build_prompt_text(diag)
        enc = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        encodings.append((enc.input_ids[0], enc.attention_mask[0]))

    max_len = max(ids.size(0) for ids, _ in encodings)
    pad_id = tokenizer.pad_token_id or 0

    all_ids, all_masks = [], []
    for ids, mask in encodings:
        pad_len = max_len - ids.size(0)
        all_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        all_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))

    return torch.stack(all_ids), torch.stack(all_masks)  # [77, L]


# ---------------------------------------------------------------------------
# 3. Core: get P(Yes) for one ECG across all diagnoses
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_p_yes_for_ecg(
    model: ECG_Tokenizer_Wrapper,
    ecg_signal: torch.Tensor,        # [1, 12, 2500]
    prompt_ids: torch.Tensor,        # [D, L]  (D = num diagnoses)
    prompt_mask: torch.Tensor,       # [D, L]
    yes_id: int,
    no_id: int,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Return P(Yes) array of shape [D] for one ECG."""
    ecg_signal = ecg_signal.to(device=device, dtype=torch.float32)

    # Encode ECG once
    features = model.encoder(ecg_signal)
    quantized, indices, _ = model.quantizer(features)
    codes = ECG_Tokenizer_Wrapper._extract_primary_codes(
        indices, model.num_codebooks_kept, model.codebook_offset
    )

    D = prompt_ids.size(0)
    p_yes = np.zeros(D, dtype=np.float32)
    decoder = model.decoder

    for start in range(0, D, batch_size):
        end = min(start + batch_size, D)
        B = end - start

        # Expand ECG features for this mini-batch
        q_feat = quantized.expand(B, -1, -1)
        q_codes = codes.expand(B, -1) if codes.dim() == 2 else codes.expand(B, -1, -1)

        b_ids = prompt_ids[start:end].to(device)
        b_mask = prompt_mask[start:end].to(device)

        # Build inputs_embeds with ECG injection
        inputs_embeds, attn_mask, _ = decoder._prepare_inputs_for_generation(
            b_ids, b_mask, q_feat, q_codes, detach_soft_prompts=True,
        )

        model_dtype = decoder.llm_model.get_input_embeddings().weight.dtype
        outputs = decoder.llm_model(
            inputs_embeds=inputs_embeds.to(model_dtype),
            attention_mask=attn_mask,
            return_dict=True,
        )

        logits = outputs.logits  # [B, seq, vocab]

        for b in range(B):
            seq_len = int(attn_mask[b].sum().item())
            last_logit = logits[b, seq_len - 1, :].float()
            probs = torch.softmax(
                torch.stack([last_logit[yes_id], last_logit[no_id]]), dim=0
            )
            p_yes[start + b] = probs[0].item()

    return p_yes


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------

def load_ecg_signal(path: str) -> torch.Tensor:
    """Load and format ECG signal to [1, 12, 2500]."""
    wav = np.load(path)
    if wav.ndim == 3:
        wav = wav.squeeze(-1)
    # Ensure (samples, leads) ordering
    if wav.shape == (12, 2500):
        wav = wav.T  # -> (2500, 12)
    # Crop/pad to 2500
    target = 2500
    n = wav.shape[0]
    if n >= target:
        s = (n - target) // 2
        wav = wav[s : s + target]
    else:
        pad_b = (target - n) // 2
        pad_a = target - n - pad_b
        wav = np.pad(wav, ((pad_b, pad_a), (0, 0)), mode="edge")
    return torch.from_numpy(wav.T).unsqueeze(0).float()  # [1, 12, 2500]


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt")
    parser.add_argument("--test_parquet", default="/media/data1/datasets/ECG_Tokenizer/combined_test_qa_m25k_h25k.parquet")
    parser.add_argument("--output_dir", default="/volume/ECG_tokenizer/analysis/binary_auroc")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_diagnoses", type=int, default=8)
    parser.add_argument("--max_ecgs", type=int, default=None)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.device)

    # Verify Yes/No tokens
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token IDs — Yes: {yes_id} ({tokenizer.decode([yes_id])}), "
          f"No: {no_id} ({tokenizer.decode([no_id])})")

    # Load test data — deduplicate to unique ECGs
    print("Loading test data...")
    df = pd.read_parquet(args.test_parquet)
    df = df.drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    diagnoses = [d for d in ECG_PATTERNS if d in df.columns]
    print(f"Unique ECGs: {len(df)}, Diagnoses: {len(diagnoses)}")

    if args.max_ecgs:
        df = df.head(args.max_ecgs)
        print(f"  (limited to {args.max_ecgs} ECGs for testing)")

    # Pre-tokenize all prompts
    print("Pre-tokenizing prompts...")
    prompt_ids, prompt_mask = pretokenize_prompts(tokenizer, diagnoses)
    print(f"Prompt shape: {prompt_ids.shape}")

    # Run inference
    all_preds = np.zeros((len(df), len(diagnoses)), dtype=np.float32)
    all_labels = np.zeros((len(df), len(diagnoses)), dtype=np.int32)

    for i in tqdm(range(len(df)), desc="ECGs"):
        row = df.iloc[i]
        ecg = load_ecg_signal(row["waveform_path_psa"])

        p_yes = get_p_yes_for_ecg(
            model, ecg, prompt_ids, prompt_mask,
            yes_id, no_id, args.device, args.batch_diagnoses,
        )
        all_preds[i] = p_yes

        for j, diag in enumerate(diagnoses):
            all_labels[i, j] = 1 if row.get(diag, 0) >= 1 else 0

    # Save raw predictions first (so we never lose them)
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "raw_predictions.npz"),
        preds=all_preds, labels=all_labels,
        diagnoses=np.array(diagnoses, dtype=object),
        waveform_paths=df["waveform_path_psa"].values,
    )
    print(f"Saved raw predictions to {args.output_dir}/raw_predictions.npz")

    # Compute full metrics per diagnosis
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, accuracy_score,
        confusion_matrix,
    )

    results = {}
    header = f"{'Label':<45} {'PPV':>6} {'NPV':>6} {'Spec':>6} {'Sens':>6} {'Acc':>6} {'AUROC':>6} {'AUPRC':>6} {'F1':>6} {'Prev':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for j, diag in enumerate(diagnoses):
        preds = all_preds[:, j]
        labels = all_labels[:, j]
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        binary_preds = (preds >= 0.5).astype(int)

        if n_pos == 0 or n_neg == 0:
            results[diag] = {
                "ppv": float("nan"), "npv": float("nan"),
                "specificity": float("nan"), "sensitivity": float("nan"),
                "accuracy": float("nan"), "auroc": float("nan"),
                "auprc": float("nan"), "f1": float("nan"),
                "n_positive": n_pos, "n_negative": n_neg,
                "prevalence": n_pos / (n_pos + n_neg),
                "mean_p_yes": float(preds.mean()),
            }
            print(f"  {diag:<43} {'SKIPPED (no pos or neg)':>60}")
            continue

        # Confusion matrix: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()

        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        acc = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds, zero_division=0.0)
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
        prev = n_pos / (n_pos + n_neg)

        results[diag] = {
            "ppv": ppv, "npv": npv,
            "specificity": spec, "sensitivity": sens,
            "accuracy": acc, "auroc": auroc,
            "auprc": auprc, "f1": f1,
            "n_positive": n_pos, "n_negative": n_neg,
            "prevalence": prev,
            "mean_p_yes": float(preds.mean()),
        }
        print(f"  {diag:<43} {ppv:>6.3f} {npv:>6.3f} {spec:>6.3f} {sens:>6.3f} {acc:>6.3f} {auroc:>6.3f} {auprc:>6.3f} {f1:>6.3f} {prev:>7.3f}")

    # Macro / Micro AUROC
    valid_aurocs = [r["auroc"] for r in results.values() if not np.isnan(r["auroc"])]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    valid_auprcs = [r["auprc"] for r in results.values() if not np.isnan(r["auprc"])]
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    # Micro: pool all pairs
    micro_labels, micro_preds = [], []
    for j, diag in enumerate(diagnoses):
        lj, pj = all_labels[:, j], all_preds[:, j]
        if int(lj.sum()) > 0 and int((1 - lj).sum()) > 0:
            micro_labels.extend(lj.tolist())
            micro_preds.extend(pj.tolist())
    micro_auroc = roc_auc_score(micro_labels, micro_preds) if micro_labels else float("nan")
    micro_auprc = average_precision_score(micro_labels, micro_preds) if micro_labels else float("nan")

    print(f"\nMacro AUROC: {macro_auroc:.4f}  |  Micro AUROC: {micro_auroc:.4f}")
    print(f"Macro AUPRC: {macro_auprc:.4f}  |  Micro AUPRC: {micro_auprc:.4f}")
    print(f"Evaluated: {len(valid_aurocs)}/{len(diagnoses)} diagnoses")

    # Save JSON
    output = {
        "macro_auroc": macro_auroc, "micro_auroc": micro_auroc,
        "macro_auprc": macro_auprc, "micro_auprc": micro_auprc,
        "n_ecgs": len(df),
        "n_diagnoses_evaluated": len(valid_aurocs),
        "n_diagnoses_skipped": len(diagnoses) - len(valid_aurocs),
        "checkpoint": args.checkpoint,
        "per_diagnosis": results,
    }
    out_json = os.path.join(args.output_dir, "binary_auroc_results.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    # Save markdown table
    md_lines = [
        "| Label | PPV | NPV | Specificity | Sensitivity | Accuracy | AUROC | AUPRC | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for diag in diagnoses:
        r = results[diag]
        if np.isnan(r["auroc"]):
            continue
        md_lines.append(
            f"| {diag} "
            f"| {r['ppv']:.3f} | {r['npv']:.3f} "
            f"| {r['specificity']:.3f} | {r['sensitivity']:.3f} "
            f"| {r['accuracy']:.3f} | {r['auroc']:.3f} "
            f"| {r['auprc']:.3f} | {r['f1']:.3f} |"
        )
    md_lines.append("")
    md_lines.append(f"**Macro AUROC: {macro_auroc:.4f} | Micro AUROC: {micro_auroc:.4f} | Macro AUPRC: {macro_auprc:.4f} | Micro AUPRC: {micro_auprc:.4f}**")
    md_lines.append(f"\nEvaluated on {len(df)} ECGs, {len(valid_aurocs)}/{len(diagnoses)} diagnoses with sufficient positive/negative samples.")

    md_path = os.path.join(args.output_dir, "binary_auroc_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved to:\n  {out_json}\n  {md_path}")


if __name__ == "__main__":
    main()
