#!/usr/bin/env python3
"""
Generate comprehensive JSON report for all validation samples.

Example:
    python scripts/generate_validation_json.py \
        --checkpoint checkpoints/.../checkpoint_epoch_2.pt \
        --validation_parquet output/combined_test_qa_m5k_h5k.parquet \
        --output_json validation_inference_epoch2.json \
        --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
from tqdm import tqdm

import random
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

# Import all necessary functions from generate_ecg_answer
from scripts.generate_ecg_answer import (
    _prepare_tokenizer,
    _instantiate_model,
    _split_state_dict,
    _maybe_attach_or_merge_lora,
    _build_prompt_tensors,
    _load_waveform,
    _extract_answer,
)


DEFAULT_DETERMINISTIC_SEED = 42


def _set_global_seed(seed: int = DEFAULT_DETERMINISTIC_SEED) -> None:
    """Set seeds for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch.backends.cudnn as cudnn  # type: ignore

        cudnn.benchmark = False
        cudnn.deterministic = True  # type: ignore[attr-defined]
    except Exception:
        # CPU-only builds or stripped backends may not expose cudnn helpers.
        pass


def _normalize_eos_ids(eos_value: Any) -> List[int]:
    """Return a list of EOS token ids from a config value."""
    if eos_value is None:
        return []
    if isinstance(eos_value, (list, tuple, set)):
        return [int(v) for v in eos_value if v is not None]
    try:
        return [int(eos_value)]
    except (TypeError, ValueError):
        return []


def _trim_sequence_to_eos(ids: torch.Tensor, eos_ids: List[int]) -> torch.Tensor:
    """Trim token ids at the first EOS occurrence (exclusive)."""
    if not eos_ids or ids.numel() == 0:
        return ids
    if ids.dim() != 1:
        ids = ids.view(-1)
    id_list = ids.tolist()
    trim_idx = None
    for idx, token_id in enumerate(id_list):
        if token_id in eos_ids:
            trim_idx = idx
            break
    if trim_idx is None:
        return ids
    if trim_idx == 0:
        return ids.new_empty(0)
    return ids[:trim_idx]


def _build_deterministic_generation_kwargs(tokenizer, base_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge config kwargs with deterministic overrides."""
    merged: Dict[str, Any] = dict(base_kwargs or {})
    merged.update({
        "do_sample": False,
        "top_p": 1.0,
        "num_beams": 1,
        "num_return_sequences": 1,
    })
    merged.pop("temperature", None)
    if tokenizer is not None:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, (list, tuple)):
            eot_id = eot_id[0] if eot_id else None
        try:
            eot_id_int = int(eot_id) if eot_id is not None else None
        except (TypeError, ValueError):
            eot_id_int = None
        if eot_id_int is not None and eot_id_int >= 0:
            merged["eos_token_id"] = eot_id_int
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            merged.setdefault("eos_token_id", int(tokenizer.eos_token_id))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            merged.setdefault("pad_token_id", int(pad_id))
    return merged


def load_validation_data(parquet_path: str, num_samples: int = None) -> pd.DataFrame:
    """Load validation parquet and optionally sample."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} validation samples from {parquet_path}")

    if num_samples and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} examples for inference")

    return df


def setup_model_once(checkpoint: str, device_str: str = None):
    """Load model and tokenizer once using functions from generate_ecg_answer."""
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if device_str:
        target_device = torch.device(device_str)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]
    if target_device.type == "cuda":
        config.device = target_device.index or 0
    else:
        config.device = 0
    config.world_size = 1
    config.is_ref_device = True

    # Use functions from generate_ecg_answer
    tokenizer, ecg_token_start_id = _prepare_tokenizer(config)
    model = _instantiate_model(config, tokenizer, ecg_token_start_id)

    state_dict = checkpoint_data["model_state_dict"]
    base_sd, lora_sd = _split_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)

    if missing:
        print(f"Warning: missing {len(missing)} keys when loading state dict")
    if unexpected:
        print(f"Warning: unexpected keys (first 10): {sorted(unexpected)[:10]}")

    _maybe_attach_or_merge_lora(model, lora_sd, config)

    model.eval()
    model.to(target_device)

    return model, tokenizer, config, target_device


def generate_batch_answers(
    model,
    tokenizer,
    config,
    target_device,
    df: pd.DataFrame,
    waveform_column: str = "waveform_path_psa",
    question_column: str = "prompt",
    answer_column: str = "generated_answer",
    category_column: str = "prompt_category",
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Generate answers for all samples using functions from generate_ecg_answer."""
    results = []
    base_generation_kwargs = getattr(config, "default_generation_kwargs", {}) or {}
    generation_kwargs = _build_deterministic_generation_kwargs(tokenizer, base_generation_kwargs)
    generation_kwargs.setdefault("max_new_tokens", 256)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    total_ctx = int(getattr(config, "max_token_length", 640))

    batch_items: List[Dict[str, Any]] = []
    pad_token = getattr(tokenizer, "pad_token_id", None)
    if pad_token is None:
        pad_token = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = int(pad_token)
    eos_base_ids = _normalize_eos_ids(generation_kwargs.get("eos_token_id"))
    eot_id_extra = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id_extra, (list, tuple)):
        eot_id_extra = eot_id_extra[0] if eot_id_extra else None
    try:
        eot_int = int(eot_id_extra) if eot_id_extra is not None else None
    except (TypeError, ValueError):
        eot_int = None
    if eot_int is not None and eot_int >= 0 and eot_int not in eos_base_ids:
        eos_base_ids.append(eot_int)

    def _flush_batch():
        nonlocal batch_items, results
        if not batch_items:
            return

        max_prompt_len = max(item["prompt_ids"].size(0) for item in batch_items)
        allow_tokens = [item["max_new_tokens"] for item in batch_items]
        batch_max_new = max(1, min(allow_tokens))
        min_new_tokens_target = int(generation_kwargs.get("min_new_tokens", 0) or 0)
        effective_min_new = max(0, min(min_new_tokens_target, batch_max_new))

        prompt_id_tensors = []
        prompt_mask_tensors = []
        ecg_tensors = []

        for item in batch_items:
            ids = item["prompt_ids"]
            mask = item["prompt_mask"]

            if ids.size(0) < max_prompt_len:
                pad_length = max_prompt_len - ids.size(0)
                pad_ids = torch.full((pad_length,), pad_token_id, dtype=ids.dtype)
                pad_mask = torch.zeros(pad_length, dtype=mask.dtype)
                ids = torch.cat([ids, pad_ids], dim=0)
                mask = torch.cat([mask, pad_mask], dim=0)

            prompt_id_tensors.append(ids)
            prompt_mask_tensors.append(mask)
            ecg_tensors.append(item["ecg_tensor"])

        prompt_batch = torch.stack(prompt_id_tensors, dim=0).to(target_device)
        mask_batch = torch.stack(prompt_mask_tensors, dim=0).to(target_device)
        ecg_batch = torch.stack(ecg_tensors, dim=0).to(target_device)

        deterministic_kwargs = dict(generation_kwargs)
        deterministic_kwargs["max_new_tokens"] = batch_max_new
        deterministic_kwargs["min_new_tokens"] = effective_min_new
        deterministic_kwargs.pop("min_length", None)

        try:
            with torch.no_grad():
                generated_batch = model.generate_report(
                    x=ecg_batch,
                    prompt_input_ids=prompt_batch,
                    prompt_attention_mask=mask_batch,
                    max_token_length=total_ctx,
                    **deterministic_kwargs,
                )
        except Exception as batch_exc:
            print(f"\nBatch generation failed for {len(batch_items)} samples: {batch_exc}")
            # Fallback to individual processing
            for fallback_item in batch_items:
                try:
                    single_prompt = fallback_item["prompt_ids"].unsqueeze(0).to(target_device)
                    single_mask = fallback_item["prompt_mask"].unsqueeze(0).to(target_device)
                    single_ecg = fallback_item["ecg_tensor"].unsqueeze(0).to(target_device)
                    single_kwargs = dict(deterministic_kwargs)
                    with torch.no_grad():
                        single_generated = model.generate_report(
                            x=single_ecg,
                            prompt_input_ids=single_prompt,
                            prompt_attention_mask=single_mask,
                            max_token_length=total_ctx,
                            **single_kwargs,
                        )
                    generated_ids_single = single_generated[0].to("cpu")
                    eos_ids_single = list(eos_base_ids)
                    trimmed_single = _trim_sequence_to_eos(generated_ids_single, eos_ids_single)
                    if trimmed_single.numel() == 0:
                        trimmed_single = generated_ids_single
                    decoded_single = tokenizer.decode(trimmed_single, skip_special_tokens=True)
                    answer_single = _extract_answer(decoded_single, fallback_item["question"])
                    if not answer_single:
                        answer_single = decoded_single.strip()
                    results.append({
                        "waveform_id": fallback_item["waveform_id"],
                        "waveform_path": fallback_item["waveform_path"],
                        "category": fallback_item["category"],
                        "question": fallback_item["question"],
                        "reference_answer": fallback_item["reference_answer"],
                        "generated_answer": answer_single,
                        "raw_decoded": decoded_single,
                        "success": True,
                        "error": None,
                    })
                except Exception as single_exc:
                    results.append({
                        "waveform_id": fallback_item["waveform_id"],
                        "waveform_path": fallback_item["waveform_path"],
                        "category": fallback_item["category"],
                        "question": fallback_item["question"],
                        "reference_answer": fallback_item["reference_answer"],
                        "generated_answer": "",
                        "raw_decoded": "",
                        "success": False,
                        "error": str(single_exc),
                    })
            batch_items = []
            return

        generated_batch = generated_batch.to("cpu")
        for idx, item in enumerate(batch_items):
            waveform_id = item["waveform_id"]
            eos_ids = list(eos_base_ids)
            gen_ids = generated_batch[idx]
            trimmed_ids = _trim_sequence_to_eos(gen_ids, eos_ids)
            if trimmed_ids.numel() == 0:
                trimmed_ids = gen_ids
            decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            answer = _extract_answer(decoded, item["question"])
            if not answer:
                answer = decoded.strip()

            results.append({
                "waveform_id": waveform_id,
                "waveform_path": item["waveform_path"],
                "category": item["category"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "generated_answer": answer,
                "raw_decoded": decoded,
                "success": True,
                "error": None,
            })

        batch_items = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        waveform_path = row[waveform_column]
        question = row[question_column]
        reference_answer = row.get(answer_column, "")
        category = row.get(category_column, "unknown")

        waveform_id = Path(waveform_path).stem

        try:
            prompt_ids, prompt_mask, prompt_text = _build_prompt_tensors(
                question,
                tokenizer,
                config,
                int(generation_kwargs["max_new_tokens"]),
            )

            prompt_len = int(prompt_ids.numel())
            current_max_new = generation_kwargs["max_new_tokens"]
            if prompt_len + current_max_new > total_ctx:
                current_max_new = max(1, total_ctx - prompt_len)

            ecg_tensor = _load_waveform(
                waveform_path,
                target_length=int(getattr(config, "ecg_waveform_length", 2500)),
                num_leads=int(getattr(config, "ecg_num_leads", 12)),
            ).to(target_device).squeeze(0)

            batch_items.append({
                "waveform_id": waveform_id,
                "waveform_path": waveform_path,
                "category": category,
                "question": question,
                "reference_answer": reference_answer,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "ecg_tensor": ecg_tensor,
                "max_new_tokens": current_max_new,
            })

            if len(batch_items) >= batch_size:
                _flush_batch()

        except Exception as e:
            print(f"\nError processing {waveform_id}: {e}")
            results.append({
                "waveform_id": waveform_id,
                "waveform_path": waveform_path,
                "category": category,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": "",
                "raw_decoded": "",
                "success": False,
                "error": str(e),
            })

    _flush_batch()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate validation JSON report.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--validation_parquet", required=True, help="Path to validation parquet file.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run (default: all).")
    parser.add_argument("--device", default=None, help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("--waveform_column", default="waveform_path_psa", help="Column name for waveform paths.")
    parser.add_argument("--question_column", default="prompt", help="Column name for questions.")
    parser.add_argument("--answer_column", default="generated_answer", help="Column name for reference answers.")
    parser.add_argument("--category_column", default="prompt_category", help="Column name for prompt categories.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to generate per batch.")

    args = parser.parse_args()

    # Load validation data
    _set_global_seed(DEFAULT_DETERMINISTIC_SEED)
    df = load_validation_data(args.validation_parquet, args.num_samples)

    # Setup model once using functions from generate_ecg_answer
    print("\nLoading model and checkpoint...")
    model, tokenizer, config, target_device = setup_model_once(args.checkpoint, args.device)
    print(f"Model loaded on {target_device}")

    # Generate answers using functions from generate_ecg_answer
    print("\nGenerating answers...")
    results = generate_batch_answers(
        model,
        tokenizer,
        config,
        target_device,
        df,
        waveform_column=args.waveform_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
        category_column=args.category_column,
        batch_size=max(1, args.batch_size),
    )

    # Save to JSON
    output_data = {
        "checkpoint": args.checkpoint,
        "validation_parquet": args.validation_parquet,
        "num_samples": len(results),
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
        "results": results,
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Generated {len(results)} answers")
    print(f"✓ Success rate: {output_data['success_rate']:.1%}")
    print(f"✓ Saved to {args.output_json}")

    # Show sample results
    print("\n=== Sample Results ===")
    for i, result in enumerate(results[:3]):
        print(f"\n[{i+1}] {result['waveform_id']}")
        print(f"Category: {result['category']}")
        print(f"Question: {result['question']}")
        print(f"Generated: {result['generated_answer'][:200]}...")
        print(f"Reference: {result['reference_answer'][:200]}...")


if __name__ == "__main__":
    main()
