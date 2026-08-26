#!/usr/bin/env python3
"""
Binary Yes/No AUROC with synonym prompts.

For each diagnosis, asks P(Yes) using multiple clinical synonyms and takes
the MAX P(Yes) across synonyms. This matches how the model actually learned
these conditions in free-text training.

Usage:
    PYTHONPATH=/volume/ECG_tokenizer python scripts/binary_auroc_synonym_eval.py \
        --checkpoint /media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt \
        --device cuda:2
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import ECG_PATTERNS, DEEPECG_CATEGORIES
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper

# Transformers 5.x compat shims
import importlib, types
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


# ── Synonym map: label → list of clinical names the model knows ──
DIAGNOSIS_SYNONYMS = {
    "Atrial paced": ["A-V sequential pacemaker", "atrial-ventricular dual-paced rhythm", "atrial pacing", "DDD pacemaker rhythm"],
    "Junctional rhythm": ["junctional rhythm", "accelerated junctional rhythm", "junctional escape rhythm", "accelerated idioventricular rhythm", "nodal rhythm"],
    "Prolonged QT": ["Prolonged QT interval", "QT prolongation", "long QT"],
    "Ectopic atrial rhythm (< 100 BPM)": ["ectopic atrial rhythm", "wandering atrial pacemaker", "multifocal atrial rhythm"],
    "Left posterior fascicular block": ["left posterior fascicular block", "LPFB", "left posterior hemiblock"],
    "Nonspecific intraventricular conduction delay": ["IV conduction defect", "IVCD", "intraventricular conduction delay", "nonspecific intraventricular conduction delay", "QRS widening"],
    "Premature atrial complex": ["PAC", "atrial premature complex", "premature atrial contraction", "atrial ectopy"],
    "Delta wave": ["delta wave", "Wolff-Parkinson-White", "pre-excitation", "WPW syndrome", "ventricular pre-excitation"],
    "Ventricular paced": ["ventricular paced rhythm", "ventricular pacing", "pacemaker rhythm", "V-paced rhythm"],
    "Premature ventricular complex": ["PVC", "ventricular premature complex", "premature ventricular contraction", "ventricular ectopy"],
    "Atrial flutter": ["atrial flutter", "flutter waves"],
    "Regularly irregular": ["regularly irregular rhythm", "Wenckebach", "group beating"],
    "Left axis deviation": ["left axis deviation", "leftward axis", "LAD"],
    "Bradycardia": ["bradycardia", "sinus bradycardia"],
    "Left bundle branch block": ["left bundle branch block", "LBBB"],
    "Right bundle branch block": ["right bundle branch block", "RBBB", "incomplete RBBB"],
    "Afib": ["atrial fibrillation", "AFib"],
    "Irregularly irregular": ["irregularly irregular rhythm", "atrial fibrillation"],
    "Atrial tachycardia (>= 100 BPM)": ["atrial tachycardia", "sinus tachycardia", "SVT"],
    "Left anterior fascicular block": ["left anterior fascicular block", "LAFB", "anterior hemiblock"],
    "Left ventricular hypertrophy": ["left ventricular hypertrophy", "LVH", "LV hypertrophy"],
    "Right ventricular hypertrophy": ["right ventricular hypertrophy", "RVH", "RV hypertrophy", "RV strain"],
    "1st degree AV block": ["1st degree A-V block", "first degree AV block", "prolonged PR interval", "PR prolongation"],
    "Right axis deviation": ["right axis deviation", "rightward axis", "RAD"],
    "Left atrial enlargement": ["left atrial enlargement", "left atrial abnormality", "LAE", "P mitrale"],
    "Right atrial enlargement": ["right atrial enlargement", "right atrial abnormality", "RAE", "P pulmonale"],
    "Acute MI": ["acute MI", "acute myocardial infarction", "STEMI", "acute ST elevation MI"],
    "Acute pericarditis": ["pericarditis", "acute pericarditis", "diffuse ST elevation with PR depression"],
    "Supraventricular tachycardia": ["supraventricular tachycardia", "SVT", "PSVT", "paroxysmal SVT"],
    "ST elevation (anterior - V3-V4)": ["anterior ST elevation", "ST elevation in anterior leads", "anterior injury"],
    "ST elevation (septal - V1-V2)": ["septal ST elevation", "ST elevation in V1-V2"],
    "ST elevation (inferior - II, III, aVF)": ["inferior ST elevation", "ST elevation in inferior leads", "inferior injury"],
    "ST elevation (lateral - I, aVL, V5-V6)": ["lateral ST elevation", "ST elevation in lateral leads"],
    "ST depression (inferior - II, III, aVF)": ["inferior ST depression", "ST depression in inferior leads", "inferior ST-T changes"],
    "ST depression (lateral - I, avL, V5-V6)": ["lateral ST depression", "ST depression in lateral leads", "lateral ST-T changes"],
    "Q wave (inferior - II, III, aVF)": ["inferior Q waves", "inferior infarct", "inferior myocardial infarction", "old inferior infarct"],
    "Q wave (anterior - V3-V4)": ["anterior Q waves", "anterior infarct", "anteroseptal infarct", "old anterior infarct"],
    "Q wave (septal- V1-V2)": ["septal Q waves", "septal infarct", "anteroseptal infarct"],
    "Q wave (lateral- I, aVL, V5-V6)": ["lateral Q waves", "lateral infarct"],
    "Q wave (posterior - V7-V9)": ["posterior Q waves", "posterior infarct", "posterior MI"],
    "Ventricular tachycardia": ["ventricular tachycardia", "V-tach", "VT"],
    "Wolff-Parkinson-White (Pre-excitation syndrome)": ["Wolff-Parkinson-White", "WPW", "pre-excitation syndrome", "ventricular pre-excitation"],
    "Third Degree AV Block": ["third degree AV block", "complete heart block", "complete AV block", "3rd degree AV block"],
    "2nd degree AV block - mobitz 1": ["2nd degree AV block Mobitz type 1", "Wenckebach", "Mobitz type I", "2:1 AV block"],
    "2nd degree AV block - mobitz 2": ["2nd degree AV block Mobitz type 2", "Mobitz type II", "Mobitz 2 block"],
    "Bi-atrial enlargement": ["biatrial enlargement", "combined atrial enlargement", "bi-atrial enlargement"],
    "LV pacing": ["LV pacing", "biventricular pacing", "cardiac resynchronization", "CRT pacing"],
    "Ventricular Rhythm": ["ventricular rhythm", "idioventricular rhythm", "ventricular escape rhythm"],
    "U wave": ["U wave", "prominent U waves"],
    "Sinusal": ["sinus rhythm", "normal sinus rhythm"],
    "Regular": ["regular rhythm", "regular rate"],
}

# Excluded from training — skip in eval
OTHER_LABELS = set(DEEPECG_CATEGORIES.get("OTHER", []))
NORMAL_LABELS = {"Sinusal", "Regular", "Monomorph"}


# ── Model loading (same as binary_auroc_eval.py) ──

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    from transformers import AutoTokenizer
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hf_name = getattr(cfg, "huggingface_model_name", "google/medgemma-4b-it")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    use_lora = any("lora_A" in k or "lora_B" in k or "base_layer" in k for k in ckpt["model_state_dict"])
    lora_config = None
    if use_lora:
        lora_config = {
            "r": getattr(cfg, "lora_r", 32), "lora_alpha": getattr(cfg, "lora_alpha", 64),
            "lora_dropout": getattr(cfg, "lora_dropout", 0.05),
            "target_modules": getattr(cfg, "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            "bias": getattr(cfg, "lora_bias", "none"), "top_k_layers": getattr(cfg, "lora_top_k_layers", None),
        }
    model = ECG_Tokenizer_Wrapper(
        encoder_name=getattr(cfg, "encoder_name", "Residual_Conv_Encoder"),
        quantizer_name=getattr(cfg, "quantizer_name", "ECG_Tokenizer_Quantizer"),
        decoder_name=getattr(cfg, "decoder_name", "MedGemmaDecoder"),
        num_quantizers=getattr(cfg, "num_quantizers", 8), codebook_size=getattr(cfg, "codebook_size", 512),
        decoder_mode="llm", bridge_name=getattr(cfg, "bridge_name", "InstructionAwareECGQFormerBridge"),
        huggingface_model_name=hf_name, llm_input_embedding_size=getattr(cfg, "llm_input_embedding_size", 2560),
        tokenizer=tokenizer, num_visual_tokens=getattr(cfg, "num_query_tokens", 32),
        bridge_mid_dim=getattr(cfg, "bridge_mid_dim", 768), bridge_num_heads=getattr(cfg, "bridge_num_heads", 12),
        bridge_dropout=getattr(cfg, "bridge_dropout", 0.1), bridge_num_special_tokens=getattr(cfg, "bridge_num_special_tokens", 4),
        bridge_qformer_layers=getattr(cfg, "bridge_qformer_layers", 10),
        bridge_text_hidden_size=getattr(cfg, "bridge_text_hidden_size", 768),
        bridge_bias_last_codebook=getattr(cfg, "bridge_bias_last_codebook", 0.5),
        bridge_codebook_dropout=getattr(cfg, "bridge_codebook_dropout", 0.0),
        bridge_mix_strategy=getattr(cfg, "bridge_mix_strategy", None),
        bridge_token_axis=getattr(cfg, "bridge_token_axis", None),
        bridge_cross_every=getattr(cfg, "bridge_cross_every", 2), instruction_dropout=0.0,
        use_lora=use_lora, lora_config=lora_config,
        num_codebooks_kept=getattr(cfg, "num_codebooks_kept", 8),
        codebook_offset=getattr(cfg, "codebook_offset", -1), prefix_tuning=False,
        stage1_checkpoint_path=getattr(cfg, "stage1_checkpoint_path", None),
    ).to(device)
    model._load_state_dict(ckpt["model_state_dict"], strict=True)
    if use_lora:
        model.set_lora_inference_mode(True)
    model.eval()
    return model, tokenizer


SYSTEM_MSG = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."

def build_prompt_text(name: str) -> str:
    return (
        f"<start_of_turn>system\n{SYSTEM_MSG}<end_of_turn>\n"
        f"<start_of_turn>user\n<start_of_image>\n\n"
        f"Question: Is {name} present in this ECG? Answer Yes or No.\n\n"
        f"Respond concisely with the key finding or answer.<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def load_ecg_signal(path: str) -> torch.Tensor:
    wav = np.load(path)
    if wav.ndim == 3: wav = wav.squeeze(-1)
    if wav.shape == (12, 2500): wav = wav.T
    target = 2500
    n = wav.shape[0]
    if n >= target:
        s = (n - target) // 2
        wav = wav[s:s+target]
    else:
        pad_b = (target - n) // 2
        wav = np.pad(wav, ((pad_b, target - n - pad_b), (0, 0)), mode="edge")
    return torch.from_numpy(wav.T).unsqueeze(0).float()


@torch.no_grad()
def get_p_yes_batch(model, ecg_signal, prompt_ids, prompt_mask, yes_id, no_id, device, batch_size=8):
    """Return P(Yes) array for all prompts given one ECG."""
    ecg_signal = ecg_signal.to(device=device, dtype=torch.float32)
    features = model.encoder(ecg_signal)
    quantized, indices, _ = model.quantizer(features)
    codes = ECG_Tokenizer_Wrapper._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)
    D = prompt_ids.size(0)
    p_yes = np.zeros(D, dtype=np.float32)
    decoder = model.decoder
    for start in range(0, D, batch_size):
        end = min(start + batch_size, D)
        B = end - start
        q_feat = quantized.expand(B, -1, -1)
        q_codes = codes.expand(B, -1) if codes.dim() == 2 else codes.expand(B, -1, -1)
        b_ids = prompt_ids[start:end].to(device)
        b_mask = prompt_mask[start:end].to(device)
        inputs_embeds, attn_mask, _ = decoder._prepare_inputs_for_generation(
            b_ids, b_mask, q_feat, q_codes, detach_soft_prompts=True)
        model_dtype = decoder.llm_model.get_input_embeddings().weight.dtype
        outputs = decoder.llm_model(
            inputs_embeds=inputs_embeds.to(model_dtype), attention_mask=attn_mask, return_dict=True)
        logits = outputs.logits
        for b in range(B):
            seq_len = int(attn_mask[b].sum().item())
            last_logit = logits[b, seq_len - 1, :].float()
            probs = torch.softmax(torch.stack([last_logit[yes_id], last_logit[no_id]]), dim=0)
            p_yes[start + b] = probs[0].item()
    return p_yes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt")
    parser.add_argument("--test_parquet", default="/media/data1/datasets/ECG_Tokenizer/combined_test_qa_m25k_h25k.parquet")
    parser.add_argument("--output_dir", default="/volume/ECG_tokenizer/analysis/binary_auroc_synonyms")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_ecgs", type=int, default=None)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.device)
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Yes={yes_id}, No={no_id}")

    # Load test data
    df = pd.read_parquet(args.test_parquet)
    df = df.drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    diagnoses = [d for d in ECG_PATTERNS if d in df.columns and d not in OTHER_LABELS and d not in NORMAL_LABELS]
    print(f"ECGs: {len(df)}, Diagnoses: {len(diagnoses)}")
    if args.max_ecgs:
        df = df.head(args.max_ecgs)
        print(f"  (limited to {args.max_ecgs})")

    # Build synonym prompt groups: for each diagnosis, tokenize ALL synonym prompts
    # Structure: diag_synonyms[diag] = (prompt_ids [S, L], prompt_mask [S, L])
    print("Pre-tokenizing synonym prompts...")
    diag_synonyms = {}
    total_prompts = 0
    for diag in diagnoses:
        synonyms = DIAGNOSIS_SYNONYMS.get(diag, [diag])
        encodings = []
        for syn in synonyms:
            text = build_prompt_text(syn)
            enc = tokenizer(text, add_special_tokens=True, return_tensors="pt")
            encodings.append((enc.input_ids[0], enc.attention_mask[0]))
        max_len = max(ids.size(0) for ids, _ in encodings)
        pad_id = tokenizer.pad_token_id or 0
        all_ids, all_masks = [], []
        for ids, mask in encodings:
            pad_len = max_len - ids.size(0)
            all_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            all_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
        diag_synonyms[diag] = (torch.stack(all_ids), torch.stack(all_masks))
        total_prompts += len(synonyms)
    print(f"Total synonym prompts: {total_prompts} across {len(diagnoses)} diagnoses")

    # Run inference: for each ECG, for each diagnosis, run all synonym prompts and take max P(Yes)
    all_preds = np.zeros((len(df), len(diagnoses)), dtype=np.float32)
    all_labels = np.zeros((len(df), len(diagnoses)), dtype=np.int32)

    for i in tqdm(range(len(df)), desc="ECGs"):
        row = df.iloc[i]
        ecg = load_ecg_signal(row["waveform_path_psa"])

        for j, diag in enumerate(diagnoses):
            syn_ids, syn_mask = diag_synonyms[diag]
            p_yes_arr = get_p_yes_batch(
                model, ecg, syn_ids, syn_mask, yes_id, no_id, args.device, args.batch_size)
            all_preds[i, j] = p_yes_arr.max()  # MAX across synonyms
            all_labels[i, j] = 1 if row.get(diag, 0) >= 1 else 0

    # Save raw
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.output_dir, "raw_predictions.npz"),
        preds=all_preds, labels=all_labels, diagnoses=np.array(diagnoses, dtype=object))

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score

    results = {}
    print(f"\n  {'Diagnosis':<48} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Sens':>6} {'Spec':>6} {'pos':>5}")
    print("  " + "-" * 90)

    for j, diag in enumerate(diagnoses):
        preds = all_preds[:, j]
        labels = all_labels[:, j]
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        binary_preds = (preds >= 0.5).astype(int)
        if n_pos == 0 or n_neg == 0:
            results[diag] = {"auroc": float("nan"), "n_positive": n_pos, "n_negative": n_neg}
            continue
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = f1_score(labels, binary_preds, zero_division=0.0)
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
        results[diag] = {
            "auroc": auroc, "auprc": auprc, "ppv": ppv, "npv": npv,
            "specificity": spec, "sensitivity": sens, "f1": f1,
            "n_positive": n_pos, "n_negative": n_neg, "mean_p_yes": float(preds.mean()),
        }
        print(f"  {diag:<48} {auroc:>7.4f} {auprc:>7.4f} {f1:>6.3f} {sens:>6.3f} {spec:>6.3f} {n_pos:>5}")

    valid_aurocs = [r["auroc"] for r in results.values() if not np.isnan(r["auroc"])]
    valid_auprcs = [r["auprc"] for r in results.values() if "auprc" in r and not np.isnan(r.get("auprc", float("nan")))]
    macro_auroc = float(np.mean(valid_aurocs))
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    # Micro
    micro_l, micro_p = [], []
    for j, diag in enumerate(diagnoses):
        lj, pj = all_labels[:, j], all_preds[:, j]
        if int(lj.sum()) > 0 and int((1 - lj).sum()) > 0:
            micro_l.extend(lj.tolist()); micro_p.extend(pj.tolist())
    micro_auroc = roc_auc_score(micro_l, micro_p)
    micro_auprc = average_precision_score(micro_l, micro_p)

    print(f"\n  Macro AUROC: {macro_auroc:.4f}  |  Micro AUROC: {micro_auroc:.4f}")
    print(f"  Macro AUPRC: {macro_auprc:.4f}  |  Micro AUPRC: {micro_auprc:.4f}")
    print(f"  Evaluated: {len(valid_aurocs)}/{len(diagnoses)}")

    output = {
        "macro_auroc": macro_auroc, "micro_auroc": micro_auroc,
        "macro_auprc": macro_auprc, "micro_auprc": micro_auprc,
        "n_ecgs": len(df), "n_diagnoses_evaluated": len(valid_aurocs),
        "checkpoint": args.checkpoint, "per_diagnosis": results,
    }
    with open(os.path.join(args.output_dir, "binary_auroc_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Markdown table
    md = ["| Label | PPV | NPV | Specificity | Sensitivity | Accuracy | AUROC | AUPRC | F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for diag in diagnoses:
        r = results[diag]
        if np.isnan(r["auroc"]): continue
        acc = (r["sensitivity"] * r["n_positive"] + r["specificity"] * r["n_negative"]) / (r["n_positive"] + r["n_negative"])
        md.append(f"| {diag} | {r['ppv']:.3f} | {r['npv']:.3f} | {r['specificity']:.3f} | {r['sensitivity']:.3f} | {acc:.3f} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['f1']:.3f} |")
    md.append(f"\n**Macro AUROC: {macro_auroc:.4f} | Micro AUROC: {micro_auroc:.4f} | Macro AUPRC: {macro_auprc:.4f} | Micro AUPRC: {micro_auprc:.4f}**")
    with open(os.path.join(args.output_dir, "binary_auroc_table.md"), "w") as f:
        f.write("\n".join(md))

    print(f"\n  Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
