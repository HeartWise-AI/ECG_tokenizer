#!/usr/bin/env python3
"""
Verify inference reproducibility by comparing fresh generations against stored predictions.

This script:
1. Randomly samples N examples from the results file
2. Runs inference using the same model/settings with FULL DETERMINISM
3. Compares outputs to verify reproducibility (should get 100% ROUGE/BLEU/METEOR)

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/verify_inference_reproducibility.py \
        --checkpoint /media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt \
        --results_csv /media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/results_e4dw86nh_best_e4dw86nh_20251220_20251220_enhanced.csv \
        --validation_parquet /volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet \
        --num_samples 10 \
        --device 0
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Optional, Any
import random

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# ============================================================
# DETERMINISTIC SETTINGS - Set BEFORE any CUDA operations
# ============================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic cuBLAS
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA for reproducibility


def set_deterministic_mode(seed: int = 42):
    """Set all random seeds and enable deterministic algorithms for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+ deterministic mode
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[DETERMINISTIC] Seed set to {seed}")
    print(f"[DETERMINISTIC] cudnn.deterministic={torch.backends.cudnn.deterministic}")
    print(f"[DETERMINISTIC] cudnn.benchmark={torch.backends.cudnn.benchmark}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shim for transformers compatibility
import types as _types, importlib as _importlib
_gemma_tok = _importlib.import_module("transformers.models.gemma.tokenization_gemma")
_shim = _types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
_shim.GemmaTokenizerFast = _gemma_tok.GemmaTokenizer
sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = _shim

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml
from utils.checkpoint_structure import resolve_checkpoint_bridge_option  # noqa: E402
import re


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    """Get config value from dict or object."""
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def _coerce_config(config_obj: Any) -> Any:
    """Convert dict config to SimpleNamespace for attribute access."""
    from types import SimpleNamespace
    if isinstance(config_obj, dict):
        return SimpleNamespace(**{k: _coerce_config(v) for k, v in config_obj.items()})
    if isinstance(config_obj, list):
        return [_coerce_config(item) for item in config_obj]
    return config_obj


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint (same as generate_all_qa_pairs.py)."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _coerce_config(checkpoint_data["config"])

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = None
    if os.path.exists(config_yaml_path):
        print(f"Loading config.yaml from {config_yaml_path}...")
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
            print(f"   OVERRIDE: Config num_codebooks_kept={num_codebooks_kept} but checkpoint has {inferred_codebooks}")
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))

    num_quantizers = int(getattr(config, "num_quantizers", 8))
    if num_codebooks_kept is not None:
        print(f"   Bridge config: num_codebooks_kept={num_codebooks_kept}, codebook_offset={codebook_offset}, num_quantizers={num_quantizers}")

    has_lora_weights = any('lora_A' in k or 'lora_B' in k for k in checkpoint_state_dict.keys())
    use_lora = bool(getattr(config, "use_lora", False)) or has_lora_weights

    if has_lora_weights and not getattr(config, "use_lora", False):
        print(f"   OVERRIDE: Config use_lora=False but checkpoint has LoRA weights, enabling LoRA")

    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and use_lora:
        default_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        lora_config = {
            "r": int(getattr(config, "lora_r", 32)),
            "lora_alpha": int(getattr(config, "lora_alpha", 64)),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.05)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or default_target_modules),
            "bias": str(getattr(config, "lora_bias", "none")),
        }
        print(f"   Using LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, targets={lora_config['target_modules']}")
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
        bridge_mix_strategy=resolve_checkpoint_bridge_option(config, yaml_config, "bridge_mix_strategy"),
        bridge_token_axis=resolve_checkpoint_bridge_option(config, yaml_config, "bridge_token_axis"),
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

    state_dict = checkpoint_data["model_state_dict"]
    model._load_state_dict(state_dict, strict=False)

    try:
        if use_lora:
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(device)

    return model, tokenizer, config


def generate_answer(
    model,
    tokenizer,
    ecg_tensor: torch.Tensor,
    question: str,
    device: torch.device,
    config,
    seed: int = 42
) -> str:
    """Generate answer for a single (ECG, question) pair with FULL DETERMINISM."""
    # Reset seeds before EACH generation for full reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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

    # FORCE DETERMINISTIC GENERATION - override any stochastic settings
    generation_kwargs = dict(getattr(config, "default_generation_kwargs", {}) or {})
    generation_kwargs["do_sample"] = False  # Greedy decoding
    generation_kwargs["temperature"] = 1.0  # Ignored when do_sample=False but set for safety
    generation_kwargs["top_p"] = 1.0
    generation_kwargs["top_k"] = 0
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


def load_ecg_waveform(waveform_path: str, target_length: int = 2500) -> Optional[np.ndarray]:
    """Load and preprocess ECG waveform."""
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

        return waveform
    except Exception as e:
        print(f"Error loading {waveform_path}: {e}")
        return None


def compute_metrics(predictions: list, references: list):
    """Compute ROUGE, BLEU, METEOR metrics."""
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

    metrics = {
        'rouge1': [], 'rouge2': [], 'rougeL': [],
        'bleu1': [], 'bleu4': [], 'meteor': [],
        'exact_match': []
    }

    for pred, ref in zip(predictions, references):
        if pred and ref:
            # Exact match
            metrics['exact_match'].append(1.0 if pred.strip() == ref.strip() else 0.0)

            # ROUGE
            scores = scorer.score(ref, pred)
            metrics['rouge1'].append(scores['rouge1'].fmeasure)
            metrics['rouge2'].append(scores['rouge2'].fmeasure)
            metrics['rougeL'].append(scores['rougeL'].fmeasure)

            # BLEU & METEOR
            try:
                ref_tokens = word_tokenize(ref.lower())
                pred_tokens = word_tokenize(pred.lower())
                metrics['bleu1'].append(sentence_bleu([ref_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=smoother.method1))
                metrics['bleu4'].append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother.method1))
                metrics['meteor'].append(meteor_score([ref_tokens], pred_tokens))
            except:
                pass

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Verify inference reproducibility")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results CSV with stored generations")
    parser.add_argument("--validation_parquet", type=str,
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet",
                        help="Path to validation parquet")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to verify")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and generation")
    args = parser.parse_args()

    # ============================================================
    # ENABLE FULL DETERMINISM
    # ============================================================
    set_deterministic_mode(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load stored results
    print(f"\nLoading stored results from {args.results_csv}...")
    results_df = pd.read_csv(args.results_csv)
    print(f"Total stored predictions: {len(results_df)}")

    # Load validation parquet for waveform paths
    print(f"Loading validation parquet from {args.validation_parquet}...")
    val_df = pd.read_parquet(args.validation_parquet)

    # Create mapping from (waveform_name, prompt) -> waveform_path
    # First, extract waveform_name from waveform_path
    val_df['waveform_name'] = val_df['waveform_path_psa'].apply(lambda x: os.path.basename(x))
    waveform_path_map = dict(zip(val_df['waveform_name'], val_df['waveform_path_psa']))

    # Extract waveform_name from json_key (format: {waveform_name}_{idx})
    results_df['waveform_name'] = results_df['json_key'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x)

    # Sample random examples
    sample_indices = random.sample(range(len(results_df)), min(args.num_samples, len(results_df)))
    sampled_results = results_df.iloc[sample_indices].copy()

    print(f"\nSampled {len(sampled_results)} examples for verification:")
    for i, row in sampled_results.iterrows():
        print(f"  - {row['json_key']}: {row['prompt_category']}")

    # Load model
    model, tokenizer, config = load_model(args.checkpoint, device)

    # Run inference on sampled examples
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON SAMPLED EXAMPLES")
    print("=" * 80)

    fresh_predictions = []
    stored_predictions = []

    for idx, row in sampled_results.iterrows():
        waveform_name = row['waveform_name']
        question = row['prompt']
        stored_generation = row['generation']

        # Find waveform path
        if waveform_name in waveform_path_map:
            waveform_path = waveform_path_map[waveform_name]
        else:
            # Try matching in val_df
            matches = val_df[val_df['waveform_name'] == waveform_name]
            if len(matches) > 0:
                waveform_path = matches.iloc[0]['waveform_path_psa']
            else:
                print(f"  WARNING: Could not find waveform path for {waveform_name}")
                continue

        # Load waveform
        waveform = load_ecg_waveform(waveform_path)
        if waveform is None:
            print(f"  WARNING: Could not load waveform from {waveform_path}")
            continue

        # Run inference with deterministic seeding
        ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)
        fresh_generation = generate_answer(model, tokenizer, ecg_tensor, question, device, config, seed=args.seed)

        fresh_predictions.append(fresh_generation)
        stored_predictions.append(stored_generation)

        # Print comparison
        print(f"\n{'='*60}")
        print(f"Example: {row['json_key']}")
        print(f"Category: {row['prompt_category']}")
        print(f"Question: {question}")
        print(f"\nSTORED:  {stored_generation}")
        print(f"FRESH:   {fresh_generation}")
        exact_match = "✓ EXACT MATCH" if stored_generation.strip() == fresh_generation.strip() else "✗ MISMATCH"
        print(f"\n{exact_match}")

    # Compute metrics
    print("\n" + "=" * 80)
    print("VERIFICATION METRICS (Fresh vs Stored)")
    print("=" * 80)

    metrics = compute_metrics(fresh_predictions, stored_predictions)

    print(f"\nExact Match Rate: {metrics['exact_match']*100:.2f}%")
    print(f"ROUGE-1:  {metrics['rouge1']:.4f}")
    print(f"ROUGE-2:  {metrics['rouge2']:.4f}")
    print(f"ROUGE-L:  {metrics['rougeL']:.4f}")
    print(f"BLEU-1:   {metrics['bleu1']:.4f}")
    print(f"BLEU-4:   {metrics['bleu4']:.4f}")
    print(f"METEOR:   {metrics['meteor']:.4f}")

    if metrics['exact_match'] == 1.0:
        print("\n✓ SUCCESS: All fresh predictions exactly match stored predictions!")
        print("  This confirms inference reproducibility.")
    else:
        print(f"\n⚠ WARNING: {(1-metrics['exact_match'])*100:.1f}% of predictions differ from stored values.")
        print("  Possible causes:")
        print("  - Different generation parameters (temperature, sampling)")
        print("  - Model weight differences")
        print("  - Numerical precision / GPU differences")

    return metrics


if __name__ == "__main__":
    main()
