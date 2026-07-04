#!/usr/bin/env python3
"""
Run Choice-Free (CF) evaluation on a trained ECG tokenizer + LLM checkpoint.

Usage example:
python scripts/evaluate_cf.py \
    --cf_dataset_path ecg_cf_eval/test_cf.json \
    --model_path /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/<run_id> \
    --output_dir results/cf_eval_full \
    --max_samples 0

This will compute per-category and overall accuracies and save detailed
predictions to <output_dir>/predictions.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure repo root on path when running from scripts/
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Minimal distributed env for config loader
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from utils.metrics.cf_evaluator import CFEvaluator
from tqdm import tqdm
from utils.config.llm_finetuning_config import LLMFinetuningConfig
from utils.enums import DecoderMode
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper


def _build_lora_config(config: LLMFinetuningConfig) -> Optional[dict[str, Any]]:
    """Assemble LoRA config from flat fields (single source of truth)."""
    if not getattr(config, "use_lora", False):
        return None
    cfg: dict[str, Any] = {
        "r": getattr(config, "lora_r", 16),
        "lora_alpha": getattr(config, "lora_alpha", 32),
        "lora_dropout": getattr(config, "lora_dropout", 0.05),
        "target_modules": getattr(config, "lora_target_modules", None),
        "bias": getattr(config, "lora_bias", "none"),
    }
    top_k = getattr(config, "lora_top_k_layers", None)
    if top_k is not None:
        cfg["top_k_layers"] = top_k
    return cfg


def _find_checkpoint_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "best_model.pt",
    ]
    if (run_dir / "checkpoint.pt").exists():
        candidates.append(run_dir / "checkpoint.pt")
    # Fallback to latest checkpoint_step_*.pt
    step_ckpts = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if step_ckpts:
        candidates.append(step_ckpts[-1])
    for p in candidates:
        if p.exists():
            return p
    # As a last resort, pick any .pt file
    all_pts = list(run_dir.glob("*.pt"))
    if not all_pts:
        raise FileNotFoundError(f"No checkpoint .pt file found in {run_dir}")
    return sorted(all_pts)[-1]


def _build_model_from_run(model_dir: Path) -> ECG_Tokenizer_Wrapper:
    """Recreate ECG_Tokenizer_Wrapper and load weights from a run directory."""
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run directory: {model_dir}")
    config = LLMFinetuningConfig.from_yaml(str(config_path))
    ckpt_path = _find_checkpoint_file(model_dir)

    # Load state dict with safe globals for tokenizers
    from torch.serialization import add_safe_globals
    import tokenizers
    from tokenizers import AddedToken
    from tokenizers.models import Model as _TokModel
    from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast

    add_safe_globals([LLMFinetuningConfig, GemmaTokenizerFast, tokenizers.Tokenizer, _TokModel, AddedToken])
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Prefer Stage-1 path from config if present
    stage1_path = getattr(config, "stage1_checkpoint_path", None)
    if stage1_path is not None:
        stage1_path = str(stage1_path)

    model = ECG_Tokenizer_Wrapper(
        encoder_name=getattr(config, "encoder_name", "Residual_Conv_Encoder"),
        quantizer_name=getattr(config, "quantizer_name", "ECG_Tokenizer_Quantizer"),
        decoder_name=config.decoder_name,
        num_quantizers=config.num_quantizers,
        codebook_size=config.ecg_codebook_size,
        decoder_mode=DecoderMode(config.decoder_mode),
        huggingface_model_name=config.huggingface_model_name,
        llm_input_embedding_size=config.llm_input_embedding_size,
        bridge_name=config.bridge_name,
        adapter_dropout=getattr(config, "adapter_dropout", 0.1),
        num_visual_tokens=(getattr(config, 'num_query_tokens', None)
                           or config.bridge_num_visual_tokens
                           or config.num_ecg_tokens),
        bridge_mid_dim=config.bridge_mid_dim,
        bridge_num_heads=config.bridge_num_heads,
        bridge_dropout=config.bridge_dropout,
        bridge_num_special_tokens=config.bridge_num_special_tokens,
        bridge_qformer_layers=(config.bridge_qformer_layers if config.bridge_qformer_layers is not None else 6),
        bridge_text_hidden_size=(config.bridge_text_hidden_size if config.bridge_text_hidden_size is not None else config.bridge_mid_dim),
        bridge_bias_last_codebook=(config.bridge_bias_last_codebook if config.bridge_bias_last_codebook is not None else 0.5),
        bridge_codebook_dropout=(config.bridge_codebook_dropout if config.bridge_codebook_dropout is not None else 0.0),
        bridge_cross_every=(config.bridge_cross_every if config.bridge_cross_every is not None else 2),
        instruction_dropout=config.instruction_dropout,
        use_lora=config.use_lora,
        lora_config=_build_lora_config(config),
        ecg_waveform_length=config.ecg_waveform_length,
        ecg_num_leads=config.ecg_num_leads,
        default_generation_kwargs=config.default_generation_kwargs,
        prefix_tuning=config.prefix_tuning,
        stage1_checkpoint_path=stage1_path,
    )
    model._load_state_dict(state_dict, strict=False)

    if getattr(config, "use_lora", False):
        model.set_lora_inference_mode(True)
        # Attempt to merge adapters where supported
        try:
            from peft import PeftModel  # type: ignore
            host = getattr(model.decoder, "llm_model", getattr(model.decoder, "llm", None))
            if isinstance(host, PeftModel):
                merged = host.merge_and_unload()
                if hasattr(model.decoder, "llm_model"):
                    model.decoder.llm_model = merged
                else:
                    model.decoder.llm = merged
        except Exception:
            pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def run_full_cf_evaluation(cf_path: Path, model_dir: Path, output_dir: Path, max_samples: int) -> Dict[str, Any]:
    evaluator = CFEvaluator(str(cf_path))
    model = _build_model_from_run(model_dir)
    tokenizer = getattr(model.decoder, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Decoder tokenizer not available")

    # Optionally collect detailed predictions
    data = evaluator.cf_data
    if max_samples and max_samples > 0 and len(data) > max_samples:
        import random
        data = random.sample(data, max_samples)

    predictions: List[Dict[str, Any]] = []
    results_by_category: Dict[str, List[bool]] = {}

    for record in tqdm(data, desc="CF Eval (full)"):
        try:
            signal = evaluator._load_ecg_signal(record.get("signal_path", ""))
        except Exception:
            # Skip missing/corrupt
            continue
        candidates = list(record["candidate_answers"])  # copy
        scores = evaluator.score_all_candidates_batched(
            model=model,
            tokenizer=tokenizer,
            ecg_signal=signal,
            question=record["question"],
            candidates=candidates,
        )
        pred_idx = int(torch.tensor(scores).argmax().item()) if scores else -1
        correct = bool(pred_idx == int(record.get("ground_truth_index", -1)))
        cat = str(record.get("category", "unknown"))
        results_by_category.setdefault(cat, []).append(correct)
        predictions.append({
            "ecg_id": record.get("ecg_id"),
            "category": cat,
            "question": record.get("question"),
            "candidates": candidates,
            "ground_truth_index": int(record.get("ground_truth_index", -1)),
            "pred_index": pred_idx,
            "correct": correct,
        })

    # Aggregate metrics
    metrics: Dict[str, float] = {}
    all_results: List[bool] = []
    for cat, res in results_by_category.items():
        if not res:
            continue
        acc = float(sum(res)) / float(len(res))
        key = f"cf_{cat.lower().replace(' ', '_').replace(',', '').replace('/', '_')}_acc"
        metrics[key] = acc
        all_results.extend(res)
    metrics["cf_overall_acc"] = float(sum(all_results)) / float(len(all_results)) if all_results else 0.0

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "predictions": predictions,
        }, f, indent=2)

    return {"metrics": metrics, "predictions_path": str(output_dir / "predictions.json")}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Choice-Free (CF) evaluation on a checkpoint")
    p.add_argument("--cf_dataset_path", type=str, required=True, help="Path to CF dataset JSON")
    p.add_argument("--model_path", type=str, required=True, help="Path to training run directory with config.yaml and checkpoint")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    p.add_argument("--max_samples", type=int, default=0, help="Subsample size (0 = full dataset)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cf_path = Path(args.cf_dataset_path)
    model_dir = Path(args.model_path)
    output_dir = Path(args.output_dir)

    result = run_full_cf_evaluation(cf_path, model_dir, output_dir, max_samples=int(args.max_samples or 0))
    metrics = result["metrics"]

    print("\nCF Evaluation Results:")
    print(f"  Overall Accuracy: {metrics.get('cf_overall_acc', 0.0) * 100:.1f}%\n")

    print("  Per-Category Accuracy:")
    for k, v in sorted(metrics.items()):
        if k == "cf_overall_acc":
            continue
        print(f"    {k.replace('cf_', '').replace('_', ' ').upper()}: {v * 100:.1f}%")

    print(f"\n  Saved detailed results to: {result['predictions_path']}")


if __name__ == "__main__":
    main()
