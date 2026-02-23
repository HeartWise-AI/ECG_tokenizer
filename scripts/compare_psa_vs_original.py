#!/usr/bin/env python3
"""
Compare QA inference quality between PSA (pre-adjusted) and Original (raw/Docker-converted)
signal paths across two checkpoints: pnsncqqv (new, augmentation-trained) vs e4dw86nh (old baseline).

Tests 4 conditions:
  1. e4dw86nh + waveform_path_psa       (baseline, direct PSA)
  2. e4dw86nh + waveform_path_original   (baseline, Docker-style conversion)
  3. pnsncqqv + waveform_path_psa        (new, direct PSA)
  4. pnsncqqv + waveform_path_original   (new, Docker-style conversion)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_psa_vs_original.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import re
from typing import Optional, Any

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
# Force sdpa/eager attention to avoid flash_attention_2 bug in transformers 5.x
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shim: transformers >=5.x merged tokenization_gemma_fast into tokenization_gemma
# Checkpoints saved with older versions pickle the old module path + class name.
import types as _types, importlib as _importlib
_gemma_tok = _importlib.import_module("transformers.models.gemma.tokenization_gemma")
_shim = _types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
_shim.GemmaTokenizerFast = _gemma_tok.GemmaTokenizer
sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = _shim

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml

# Monkey-patch: force medgemma loader to skip flash_attention_2 (broken in transformers 5.x)
import models.decoder.medgemma_decoder as _medgemma_mod
_orig_load = _medgemma_mod._load_medgemma_model
def _patched_load(model_name, torch_dtype=None):
    from transformers import AutoModelForImageTextToText
    for attn_impl in ("sdpa", "eager", None):
        try:
            kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            if attn_impl:
                print(f"   Using attention: {attn_impl}")
            return model
        except Exception:
            continue
    return _orig_load(model_name, torch_dtype)
_medgemma_mod._load_medgemma_model = _patched_load

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "e4dw86nh": {
        "path": "/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt",
        "config_yaml": "/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/config.yaml",
    },
    "pnsncqqv": {
        "path": "/volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/pnsncqqv_20260217-041628/best_model.pt",
        "config_yaml": "/volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/pnsncqqv_20260217-041628/config.yaml",
    },
}

VAL_PARQUET = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet"
NUM_SAMPLES = 10
DEVICE_ID = 0
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def load_model(checkpoint_path: str, config_yaml_path: str, device: torch.device):
    """Load model from checkpoint, using its own config.yaml."""
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {os.path.basename(os.path.dirname(checkpoint_path))}")
    print(f"  Weights: {checkpoint_path}")
    print(f"  Config:  {config_yaml_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

    yaml_config = None
    if os.path.exists(config_yaml_path):
        yaml_config = load_yaml(config_yaml_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))

    # Infer num_codebooks_kept from checkpoint tensor shapes
    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", None)
    if num_codebooks_kept is None:
        num_codebooks_kept = _cfg_get(config, "num_codebooks_kept", None)

    checkpoint_state_dict = checkpoint_data["model_state_dict"]
    mix_gate_key = "decoder.bridge.mix_gate.0.weight"
    if mix_gate_key in checkpoint_state_dict:
        mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
        hidden_dim = mix_gate_shape[0]
        inferred_codebooks = mix_gate_shape[1] // hidden_dim
        if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
            print(f"  OVERRIDE: config num_codebooks_kept={num_codebooks_kept} → checkpoint has {inferred_codebooks}")
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))
    num_quantizers = int(getattr(config, "num_quantizers", 8))
    print(f"  Bridge: num_codebooks_kept={num_codebooks_kept}, codebook_offset={codebook_offset}")
    print(f"  HF model: {config.huggingface_model_name}")

    # Reconstruct LoRA config
    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and bool(getattr(config, "use_lora", False)):
        lora_config = {
            "r": int(getattr(config, "lora_r", 16)),
            "lora_alpha": int(getattr(config, "lora_alpha", getattr(config, "lora_r", 16))),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.0)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or []),
            "bias": str(getattr(config, "lora_bias", "none")),
        }
        try:
            setattr(config, "lora_config", lora_config)
        except Exception:
            pass

    model = ECG_Tokenizer_Wrapper(
        encoder_name=config.encoder_name,
        quantizer_name=config.quantizer_name,
        decoder_name=config.decoder_name,
        num_quantizers=num_quantizers,
        codebook_size=int(getattr(config, "codebook_size", 512)),
        decoder_mode=decoder_mode,
        huggingface_model_name=config.huggingface_model_name,
        llm_input_embedding_size=int(config.llm_input_embedding_size),
        bridge_name=config.bridge_name,
        num_visual_tokens=num_visual_tokens,
        bridge_mid_dim=int(getattr(config, "bridge_mid_dim", 512)),
        bridge_num_heads=int(getattr(config, "bridge_num_heads", 8)),
        bridge_dropout=float(getattr(config, "bridge_dropout", 0.1)),
        bridge_num_special_tokens=int(getattr(config, "bridge_num_special_tokens", 4)),
        bridge_qformer_layers=getattr(config, "bridge_qformer_layers", None),
        bridge_text_hidden_size=getattr(config, "bridge_text_hidden_size", None),
        bridge_bias_last_codebook=_cfg_get(yaml_config, "bridge_bias_last_codebook", _cfg_get(config, "bridge_bias_last_codebook", None)),
        bridge_codebook_dropout=_cfg_get(yaml_config, "bridge_codebook_dropout", _cfg_get(config, "bridge_codebook_dropout", None)),
        bridge_cross_every=_cfg_get(yaml_config, "bridge_cross_every", _cfg_get(config, "bridge_cross_every", None)),
        instruction_dropout=0.0,  # DETERMINISTIC: no instruction dropout at inference
        stage1_checkpoint_path=None,
        use_lora=bool(getattr(config, "use_lora", False)),
        lora_config=lora_config,
        tokenizer=tokenizer,
        ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
        num_codebooks_kept=num_codebooks_kept,
        codebook_offset=codebook_offset,
    )

    state_dict = checkpoint_data["model_state_dict"]
    model._load_state_dict(state_dict, strict=False)

    try:
        if bool(getattr(config, "use_lora", False)):
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(device)
    print(f"  Model loaded and set to eval mode (deterministic)")
    return model, tokenizer, config


def load_ecg_psa(waveform_path: str, target_length: int = 2500) -> Optional[np.ndarray]:
    """Load pre-adjusted PSA signal (already normalized, 2500 samples)."""
    try:
        waveform = np.load(waveform_path)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(-1)
        if waveform.shape[-1] == 12:
            pass
        elif waveform.shape[0] == 12:
            waveform = waveform.T
        current_length = waveform.shape[0]
        if current_length >= target_length:
            start = (current_length - target_length) // 2
            waveform = waveform[start:start + target_length, :]
        else:
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")
        return waveform.astype(np.float32)
    except Exception as e:
        print(f"  ERROR loading PSA {waveform_path}: {e}")
        return None


def load_ecg_original(waveform_path: str, target_length: int = 2500) -> Optional[np.ndarray]:
    """
    Load original (raw/Docker-converted) signal and apply basic PSA-like conversion:
      - Downsample from 5000→2500 if needed
      - Scale by /1000 to approximate PSA normalization
      - Cast to float32
    This simulates what the Docker XML→PSA pipeline produces.
    """
    try:
        waveform = np.load(waveform_path)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(-1)
        if waveform.shape[-1] == 12:
            pass
        elif waveform.shape[0] == 12:
            waveform = waveform.T

        # Downsample if needed (5000 → 2500)
        current_length = waveform.shape[0]
        if current_length == 2 * target_length:
            waveform = waveform[::2, :]  # Take every 2nd sample
        elif current_length > target_length:
            start = (current_length - target_length) // 2
            waveform = waveform[start:start + target_length, :]
        elif current_length < target_length:
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")

        # Scale: raw MIMIC signals are ~1000x larger than PSA-normalized
        # PSA range ≈ [-1.5, 1.2], Original range ≈ [-1490, 1210]
        waveform = waveform / 1000.0

        return waveform.astype(np.float32)
    except Exception as e:
        print(f"  ERROR loading Original {waveform_path}: {e}")
        return None


def generate_answer(model, tokenizer, ecg_tensor, question, device, config) -> str:
    """Generate answer for a single (ECG, question) pair. Deterministic (no sampling)."""
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    user_content = f"<start_of_image>\n\nQuestion: {question}\n\nRespond concisely with the key finding or answer."
    prompt_text = (
        f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    encoding = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
    prompt_ids = encoding.input_ids.to(device)
    prompt_mask = encoding.attention_mask.to(device)

    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
        eos_ids.append(end_of_turn_id)

    # DETERMINISTIC generation - override to ensure no sampling
    generation_kwargs = dict(getattr(config, "default_generation_kwargs", {}) or {})
    generation_kwargs["do_sample"] = False
    generation_kwargs["temperature"] = 0.0
    generation_kwargs.setdefault("max_new_tokens", 96)
    generation_kwargs.setdefault("no_repeat_ngram_size", 5)
    generation_kwargs.setdefault("repetition_penalty", 1.1)
    generation_kwargs["eos_token_id"] = eos_ids
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            max_token_length=int(getattr(config, "max_token_length", 640)),
            **generation_kwargs,
        )

    generation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    for pattern in [r'<end_of_turn>', r'<start_of_turn>', r'<\|eot_id\|>', r'model\s*$', r'^model\s*']:
        generation = re.sub(pattern, '', generation)
    return generation.strip()


def compute_metrics(results: list[dict]) -> dict:
    """Compute ROUGE, BLEU, METEOR over a list of {generation, ground_truth} dicts."""
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction()

    rouge1, rouge2, rougeL = [], [], []
    bleu1_scores, bleu4_scores = [], []
    meteor_scores = []

    for r in results:
        gen = r.get("generation", "")
        ref = r.get("ground_truth", "")
        if not gen or not ref:
            continue
        scores = scorer.score(ref, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
        try:
            ref_tokens = word_tokenize(ref.lower())
            gen_tokens = word_tokenize(gen.lower())
            bleu1_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother.method1))
            bleu4_scores.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1))
            meteor_scores.append(meteor_score([ref_tokens], gen_tokens))
        except Exception:
            pass

    n = len(rouge1)
    return {
        "n": n,
        "ROUGE-1": np.mean(rouge1) if rouge1 else 0,
        "ROUGE-2": np.mean(rouge2) if rouge2 else 0,
        "ROUGE-L": np.mean(rougeL) if rougeL else 0,
        "BLEU-1": np.mean(bleu1_scores) if bleu1_scores else 0,
        "BLEU-4": np.mean(bleu4_scores) if bleu4_scores else 0,
        "METEOR": np.mean(meteor_scores) if meteor_scores else 0,
    }


def sample_diverse_qa(val_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n QA pairs with diverse prompt categories."""
    rng = np.random.RandomState(seed)
    categories = val_df['prompt_category'].unique()
    # Try to get at least 1 sample from each major category
    samples = []
    for cat in categories:
        cat_df = val_df[val_df['prompt_category'] == cat]
        if len(cat_df) > 0 and len(samples) < n:
            samples.append(cat_df.sample(1, random_state=rng))
    # Fill remaining slots randomly
    remaining = n - len(samples)
    if remaining > 0:
        already_idx = set()
        for s in samples:
            already_idx.update(s.index.tolist())
        pool = val_df[~val_df.index.isin(already_idx)]
        if len(pool) >= remaining:
            samples.append(pool.sample(remaining, random_state=rng))
    result = pd.concat(samples).head(n)
    return result


def main():
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Sample validation data ───────────────────────────────────────────
    print(f"\nLoading validation parquet: {VAL_PARQUET}")
    val_df = pd.read_parquet(VAL_PARQUET)
    print(f"Total QA pairs: {len(val_df)}, Unique waveforms: {val_df['waveform_name'].nunique()}")

    # Filter to samples that have BOTH paths available
    has_both = val_df['waveform_path_psa'].notna() & val_df['waveform_path_original'].notna()
    val_both = val_df[has_both].copy()
    print(f"Samples with both PSA + Original paths: {len(val_both)}")

    sampled = sample_diverse_qa(val_both, NUM_SAMPLES, SEED)
    print(f"\nSampled {len(sampled)} QA pairs:")
    for i, (_, row) in enumerate(sampled.iterrows()):
        print(f"  [{i+1}] {row['prompt_category']:30s} | {row['waveform_name']} | Q: {row['prompt'][:60]}...")

    # ── 2. Verify both signal paths exist for all samples ───────────────────
    print("\nVerifying signal files exist...")
    valid_rows = []
    for _, row in sampled.iterrows():
        psa_ok = os.path.exists(row['waveform_path_psa'])
        orig_ok = os.path.exists(row['waveform_path_original'])
        if psa_ok and orig_ok:
            valid_rows.append(row)
        else:
            print(f"  SKIP {row['waveform_name']}: PSA={'OK' if psa_ok else 'MISSING'}, Orig={'OK' if orig_ok else 'MISSING'}")
    sampled = pd.DataFrame(valid_rows)
    print(f"Valid samples: {len(sampled)}")
    if len(sampled) == 0:
        print("ERROR: No valid samples found!")
        return

    # ── 3. Run inference for each checkpoint × signal path ──────────────────
    all_results = {}  # key: "checkpoint_name/signal_type" → list of result dicts

    for ckpt_name, ckpt_info in CHECKPOINTS.items():
        print(f"\n{'#'*70}")
        print(f"# CHECKPOINT: {ckpt_name}")
        print(f"{'#'*70}")

        model, tokenizer, config = load_model(ckpt_info["path"], ckpt_info["config_yaml"], device)

        for signal_type, loader_fn, path_col in [
            ("PSA", load_ecg_psa, "waveform_path_psa"),
            ("Original", load_ecg_original, "waveform_path_original"),
        ]:
            condition_key = f"{ckpt_name}/{signal_type}"
            print(f"\n  ── {condition_key} ──")
            results = []

            for i, (_, row) in enumerate(sampled.iterrows()):
                waveform = loader_fn(row[path_col])
                if waveform is None:
                    print(f"    [{i+1}] FAILED to load {row[path_col]}")
                    continue

                ecg_tensor = torch.from_numpy(waveform).T.unsqueeze(0).to(device)
                generation = generate_answer(model, tokenizer, ecg_tensor, row['prompt'], device, config)
                ground_truth = row['generated_answer']

                results.append({
                    "waveform_name": row['waveform_name'],
                    "prompt_category": row['prompt_category'],
                    "question": row['prompt'],
                    "ground_truth": ground_truth,
                    "generation": generation,
                })

                # Print sample output for first 3
                if i < 3:
                    print(f"    [{i+1}] Q: {row['prompt'][:60]}...")
                    print(f"        GT:  {ground_truth[:80]}...")
                    print(f"        Gen: {generation[:80]}...")

            all_results[condition_key] = results

        # Free GPU memory before loading next checkpoint
        del model
        torch.cuda.empty_cache()

    # ── 4. Compute metrics ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("METRICS COMPARISON")
    print(f"{'='*70}")

    metrics_table = {}
    for condition_key, results in all_results.items():
        metrics = compute_metrics(results)
        metrics_table[condition_key] = metrics

    # Print table
    conditions = list(metrics_table.keys())
    metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR"]

    # Header
    col_w = 22
    print(f"\n{'Metric':<12}", end="")
    for c in conditions:
        print(f"{c:>{col_w}}", end="")
    print()
    print("-" * (12 + col_w * len(conditions)))

    # Rows
    for m in metric_names:
        print(f"{m:<12}", end="")
        for c in conditions:
            val = metrics_table[c][m]
            print(f"{val:>{col_w}.4f}", end="")
        print()

    # N row
    print(f"{'N':<12}", end="")
    for c in conditions:
        print(f"{metrics_table[c]['n']:>{col_w}d}", end="")
    print()

    # ── 5. Delta analysis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DELTA ANALYSIS (PSA vs Original for each checkpoint)")
    print(f"{'='*70}")

    for ckpt_name in CHECKPOINTS:
        psa_key = f"{ckpt_name}/PSA"
        orig_key = f"{ckpt_name}/Original"
        if psa_key in metrics_table and orig_key in metrics_table:
            print(f"\n  {ckpt_name}:")
            for m in metric_names:
                psa_val = metrics_table[psa_key][m]
                orig_val = metrics_table[orig_key][m]
                delta = orig_val - psa_val
                pct = (delta / psa_val * 100) if psa_val > 0 else 0
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"    {m:<10}: PSA={psa_val:.4f}  Original={orig_val:.4f}  Δ={delta:+.4f} ({pct:+.1f}%) {arrow}")

    # ── 6. Cross-checkpoint comparison ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-CHECKPOINT COMPARISON (pnsncqqv vs e4dw86nh)")
    print(f"{'='*70}")

    for signal_type in ["PSA", "Original"]:
        old_key = f"e4dw86nh/{signal_type}"
        new_key = f"pnsncqqv/{signal_type}"
        if old_key in metrics_table and new_key in metrics_table:
            print(f"\n  Signal: {signal_type}")
            for m in metric_names:
                old_val = metrics_table[old_key][m]
                new_val = metrics_table[new_key][m]
                delta = new_val - old_val
                pct = (delta / old_val * 100) if old_val > 0 else 0
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"    {m:<10}: e4dw86nh={old_val:.4f}  pnsncqqv={new_val:.4f}  Δ={delta:+.4f} ({pct:+.1f}%) {arrow}")

    # ── 7. Save detailed results ────────────────────────────────────────────
    output_path = "/volume/ECG_tokenizer/output/psa_vs_original_comparison.json"
    save_data = {
        "config": {
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "checkpoints": {k: v["path"] for k, v in CHECKPOINTS.items()},
            "val_parquet": VAL_PARQUET,
        },
        "metrics": metrics_table,
        "detailed_results": {k: v for k, v in all_results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
