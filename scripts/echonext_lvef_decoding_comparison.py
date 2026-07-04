#!/usr/bin/env python3
"""
Compare different decoding strategies for LVEF prediction on EchoNext dataset.

Loads the e4dw86nh MedGemma model, runs inference on 200 EchoNext test samples
with various generation configurations, extracts LVEF values, and computes
Pearson correlation, ICC, and MAE for each strategy.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/echonext_lvef_decoding_comparison.py
"""

import os
import sys
import re
import json
import warnings
import types as _types
import importlib as _importlib

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Tokenizer shim for transformers 5.x
_gemma_tok = _importlib.import_module("transformers.models.gemma.tokenization_gemma")
_shim = _types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
_shim.GemmaTokenizerFast = _gemma_tok.GemmaTokenizer
sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = _shim

import torch
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Dict, Tuple
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml

# ─── Config helpers ───────────────────────────────────────────────────────────

def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def _coerce_config(config_obj: Any) -> Any:
    if isinstance(config_obj, dict):
        return SimpleNamespace(**{k: _coerce_config(v) for k, v in config_obj.items()})
    if isinstance(config_obj, list):
        return [_coerce_config(item) for item in config_obj]
    return config_obj


# ─── Model loading (from verify_inference_reproducibility.py) ─────────────────

def load_model(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _coerce_config(checkpoint_data["config"])

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = None
    if os.path.exists(config_yaml_path):
        yaml_config = load_yaml(config_yaml_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))

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
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))
    num_quantizers = int(getattr(config, "num_quantizers", 8))

    has_lora_weights = any('lora_A' in k or 'lora_B' in k for k in checkpoint_state_dict.keys())
    use_lora = bool(getattr(config, "use_lora", False)) or has_lora_weights

    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and use_lora:
        lora_config = {
            "r": int(getattr(config, "lora_r", 32)),
            "lora_alpha": int(getattr(config, "lora_alpha", 64)),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.05)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
            "bias": str(getattr(config, "lora_bias", "none")),
        }
        try:
            setattr(config, "lora_config", lora_config)
            setattr(config, "use_lora", True)
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
        instruction_dropout=_cfg_get(yaml_config, "instruction_dropout", _cfg_get(config, "instruction_dropout", 0.0)),
        stage1_checkpoint_path=None,
        use_lora=use_lora,
        lora_config=lora_config,
        tokenizer=tokenizer,
        ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
        num_codebooks_kept=num_codebooks_kept,
        codebook_offset=codebook_offset,
    )

    model._load_state_dict(checkpoint_state_dict, strict=False)
    try:
        if use_lora:
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(device)
    return model, tokenizer, config


# ─── EchoNext data loading ────────────────────────────────────────────────────

def load_echonext_samples(n_samples: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load EchoNext test samples with LVEF labels.

    Returns:
        waveforms: (N, 2500, 12) float32 array
        lvef_values: (N,) ground truth LVEF
        metadata: DataFrame with sample info
    """
    metadata = pd.read_csv("/media/data1/datasets/EchoNext/echonext_metadata_100k.csv")
    test_mask = metadata["split"] == "test"
    test_df = metadata[test_mask].copy()
    test_df = test_df[test_df["lvef_value"].notna()].copy()

    # Stratified sampling to get good LVEF distribution
    np.random.seed(seed)
    test_df = test_df.reset_index(drop=True)

    # Create position-in-test-array index
    # The test waveform array corresponds to ALL test rows (not just those with LVEF)
    test_all = metadata[test_mask].reset_index(drop=True)
    test_df["test_array_idx"] = test_df.index.map(
        lambda i: test_all.index[test_all["ecg_key"] == test_df.loc[i, "ecg_key"]].tolist()[0]
        if i < len(test_df) else -1
    )

    # Actually, the row order in metadata matches the npy array order per split
    # So test rows in metadata (filtered by split==test) map 1:1 to test npy rows
    # Let me redo this properly
    test_all_with_lvef = test_all[test_all["lvef_value"].notna()].copy()
    test_all_with_lvef["npy_idx"] = test_all_with_lvef.index  # position in the test npy array

    sample = test_all_with_lvef.sample(n=min(n_samples, len(test_all_with_lvef)), random_state=seed)

    # Load waveforms
    all_waveforms = np.load("/media/data1/datasets/EchoNext/EchoNext_test_waveforms.npy", mmap_mode="r")
    # Shape: (5442, 1, 2500, 12)

    waveforms = []
    for idx in sample["npy_idx"].values:
        wf = all_waveforms[idx, 0, :, :]  # squeeze the extra dim -> (2500, 12)
        waveforms.append(wf.astype(np.float32))

    waveforms = np.stack(waveforms)  # (N, 2500, 12)
    lvef_values = sample["lvef_value"].values

    print(f"Loaded {len(waveforms)} EchoNext test samples")
    print(f"  LVEF range: {lvef_values.min():.1f} - {lvef_values.max():.1f}")
    print(f"  LVEF mean: {lvef_values.mean():.1f}, std: {lvef_values.std():.1f}")
    print(f"  Waveform shape: {waveforms.shape}, range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")

    return waveforms, lvef_values, sample.reset_index(drop=True)


# ─── LVEF prompts ─────────────────────────────────────────────────────────────

LVEF_QUESTIONS = [
    "What is the LVEF based on echocardiography?",
    "What is the patient's left ventricular ejection fraction?",
    "What is the ejection fraction?",
]


def build_prompt(question: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build MedGemma-format prompt."""
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    user_content = f"<start_of_image>\n\nQuestion: {question}\n\nRespond concisely with the key finding or answer."
    prompt_text = (
        f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    encoding = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
    return encoding.input_ids, encoding.attention_mask


# ─── Generation strategies ────────────────────────────────────────────────────

def get_decoding_strategies() -> Dict[str, dict]:
    """Define different decoding strategies to compare."""
    return {
        # Baseline: greedy (same as e4dw86nh inference)
        "greedy_default": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 96,
            "no_repeat_ngram_size": 5,
            "repetition_penalty": 1.1,
        },
        # Greedy with shorter output (scalar task profile)
        "greedy_scalar24": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Greedy, no repetition penalty (removes bias against repeated digits)
        "greedy_no_rep_penalty": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 96,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Sampling with low temperature
        "sample_t0.3": {
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Sampling with moderate temperature
        "sample_t0.5": {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Sampling with higher temperature
        "sample_t0.7": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Sampling with temperature 1.0
        "sample_t1.0": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        # Greedy with no ngram blocking or rep penalty
        "greedy_clean": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 24,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
    }


# ─── LVEF extraction ─────────────────────────────────────────────────────────

PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUM_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def extract_lvef_from_text(text: str) -> Optional[float]:
    """Extract LVEF numeric value from generation text."""
    if not text:
        return None

    # Try percentage pattern first
    match = PCT_PATTERN.search(text)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 100:
            return val

    # Fall back to any number in reasonable LVEF range
    for match in NUM_PATTERN.finditer(text):
        val = float(match.group(1))
        if 5 <= val <= 95:
            return val

    return None


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_icc(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute ICC(3,1) - two-way mixed, single measures, consistency."""
    n = len(predictions)
    if n < 3:
        return float("nan")

    # ICC(3,1) using two-way mixed model
    grand_mean = (predictions.mean() + ground_truth.mean()) / 2

    # Between-subjects variance
    subject_means = (predictions + ground_truth) / 2
    ss_between = 2 * np.sum((subject_means - grand_mean) ** 2)

    # Within-subjects variance
    ss_within = np.sum((predictions - subject_means) ** 2) + np.sum((ground_truth - subject_means) ** 2)

    # Rater variance
    rater_means = np.array([predictions.mean(), ground_truth.mean()])
    ss_rater = n * np.sum((rater_means - grand_mean) ** 2)

    # Error
    ss_error = ss_within - ss_rater

    ms_between = ss_between / (n - 1)
    ms_error = ss_error / (n - 1) if (n - 1) > 0 else 1e-10

    icc = (ms_between - ms_error) / (ms_between + ms_error)
    return float(np.clip(icc, -1, 1))


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute Pearson, Spearman, ICC, MAE, and RMSE."""
    valid_mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
    preds = predictions[valid_mask]
    gt = ground_truth[valid_mask]

    n_valid = len(preds)
    n_total = len(predictions)
    extraction_rate = n_valid / n_total if n_total > 0 else 0

    if n_valid < 3:
        return {
            "n_valid": n_valid,
            "n_total": n_total,
            "extraction_rate": extraction_rate,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "icc": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mean_pred": float("nan"),
            "std_pred": float("nan"),
            "n_unique_pred": 0,
        }

    from scipy.stats import pearsonr, spearmanr

    pearson_r, _ = pearsonr(preds, gt)
    spearman_r, _ = spearmanr(preds, gt)
    icc = compute_icc(preds, gt)
    mae = float(np.abs(preds - gt).mean())
    rmse = float(np.sqrt(np.mean((preds - gt) ** 2)))

    return {
        "n_valid": n_valid,
        "n_total": n_total,
        "extraction_rate": extraction_rate,
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "icc": icc,
        "mae": mae,
        "rmse": rmse,
        "mean_pred": float(preds.mean()),
        "std_pred": float(preds.std()),
        "n_unique_pred": len(np.unique(np.round(preds, 1))),
    }


# ─── Run inference ────────────────────────────────────────────────────────────

def run_single_inference(
    model, tokenizer, ecg_tensor: torch.Tensor,
    question: str, device: torch.device,
    gen_kwargs: dict, seed: int = 42
) -> str:
    """Generate answer for a single ECG+question pair."""
    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    prompt_ids, prompt_mask = build_prompt(question, tokenizer)
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
        eos_ids.append(end_of_turn_id)

    kwargs = dict(gen_kwargs)
    kwargs.pop("_num_samples", None)  # remove our custom key
    kwargs["eos_token_id"] = eos_ids
    kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            max_token_length=640,
            **kwargs,
        )

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    for pattern in [r'<end_of_turn>', r'<start_of_turn>', r'<\|eot_id\|>', r'model\s*$', r'^model\s*']:
        text = re.sub(pattern, '', text)
    return text.strip()


def run_strategy(
    model, tokenizer, waveforms: np.ndarray, lvef_gt: np.ndarray,
    strategy_name: str, gen_kwargs: dict, device: torch.device
) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    """Run a decoding strategy on all samples and return predictions + metrics."""
    n = len(waveforms)
    num_samples = gen_kwargs.get("_num_samples", 1)
    question = LVEF_QUESTIONS[0]  # Use consistent question

    predictions = np.full(n, np.nan)
    raw_texts = []

    for i in range(n):
        wf = waveforms[i]  # (2500, 12)
        # Transpose to (12, 2500) and add batch dim -> (1, 12, 2500)
        ecg_tensor = torch.from_numpy(wf.T.copy()).float().unsqueeze(0).to(device)

        if num_samples > 1:
            # Multiple samples, take median
            sample_vals = []
            for s in range(num_samples):
                text = run_single_inference(
                    model, tokenizer, ecg_tensor, question, device,
                    gen_kwargs, seed=42 + s
                )
                val = extract_lvef_from_text(text)
                if val is not None:
                    sample_vals.append(val)

            if sample_vals:
                predictions[i] = float(np.median(sample_vals))
                raw_texts.append(f"[median of {len(sample_vals)}/{num_samples}] " +
                               f"vals={[f'{v:.0f}' for v in sample_vals]}")
            else:
                raw_texts.append("[all failed]")
        else:
            text = run_single_inference(
                model, tokenizer, ecg_tensor, question, device,
                gen_kwargs, seed=42
            )
            val = extract_lvef_from_text(text)
            if val is not None:
                predictions[i] = val
            raw_texts.append(text)

        if (i + 1) % 25 == 0:
            valid_so_far = ~np.isnan(predictions[:i+1])
            print(f"  [{strategy_name}] {i+1}/{n} done, "
                  f"{valid_so_far.sum()} valid extractions")

    metrics = compute_metrics(predictions, lvef_gt)
    return predictions, metrics, raw_texts


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = "/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt"
    model, tokenizer, config = load_model(checkpoint_path, device)

    # Load EchoNext samples
    waveforms, lvef_gt, sample_df = load_echonext_samples(n_samples=200, seed=42)

    # Get decoding strategies
    strategies = get_decoding_strategies()

    # Run all strategies
    all_results = {}
    all_predictions = {}
    all_texts = {}

    for name, kwargs in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name}")
        print(f"  Config: { {k:v for k,v in kwargs.items() if not k.startswith('_')} }")
        print(f"{'='*60}")

        preds, metrics, texts = run_strategy(
            model, tokenizer, waveforms, lvef_gt, name, kwargs, device
        )

        all_results[name] = metrics
        all_predictions[name] = preds
        all_texts[name] = texts

        print(f"\n  Results for {name}:")
        print(f"    Extraction rate: {metrics['extraction_rate']:.1%}")
        print(f"    Pearson:  {metrics['pearson']:.4f}")
        print(f"    Spearman: {metrics['spearman']:.4f}")
        print(f"    ICC:      {metrics['icc']:.4f}")
        print(f"    MAE:      {metrics['mae']:.2f}")
        print(f"    RMSE:     {metrics['rmse']:.2f}")
        print(f"    Unique predictions: {metrics['n_unique_pred']}")
        print(f"    Pred mean/std: {metrics['mean_pred']:.1f} / {metrics['std_pred']:.1f}")

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY: All Decoding Strategies")
    print(f"{'='*100}")
    print(f"{'Strategy':<30} {'Pearson':>8} {'ICC':>8} {'MAE':>8} {'RMSE':>8} {'Extr%':>8} {'#Unique':>8} {'PredMean':>9}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")

    for name in strategies:
        m = all_results[name]
        print(f"{name:<30} {m['pearson']:>8.4f} {m['icc']:>8.4f} {m['mae']:>8.2f} "
              f"{m['rmse']:>8.2f} {m['extraction_rate']:>7.1%} {m['n_unique_pred']:>8} {m['mean_pred']:>9.1f}")

    print(f"\nGround truth: mean={lvef_gt.mean():.1f}, std={lvef_gt.std():.1f}, "
          f"range=[{lvef_gt.min():.1f}, {lvef_gt.max():.1f}]")

    # ─── Save detailed results ────────────────────────────────────────────
    output_dir = "/volume/ECG_tokenizer/autoresearch"
    output_path = os.path.join(output_dir, "echonext_lvef_decoding_comparison.json")

    save_data = {
        "ground_truth_stats": {
            "mean": float(lvef_gt.mean()),
            "std": float(lvef_gt.std()),
            "min": float(lvef_gt.min()),
            "max": float(lvef_gt.max()),
            "n_samples": len(lvef_gt),
        },
        "strategies": {},
    }

    for name in strategies:
        save_data["strategies"][name] = {
            "config": {k: v for k, v in strategies[name].items()},
            "metrics": all_results[name],
            "sample_predictions": [
                {
                    "gt_lvef": float(lvef_gt[i]),
                    "pred_lvef": float(all_predictions[name][i]) if not np.isnan(all_predictions[name][i]) else None,
                    "raw_text": all_texts[name][i][:200],
                }
                for i in range(min(20, len(lvef_gt)))  # Save first 20 examples
            ],
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
