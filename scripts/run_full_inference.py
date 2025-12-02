#!/usr/bin/env python3
"""
Run inference on all samples from a parquet dataset using a finetuned checkpoint.

Usage:
    python scripts/run_full_inference.py \
        --checkpoint checkpoints/.../checkpoint_step_XXXX.pt \
        --parquet /volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet \
        --output results/full_inference.json \
        --device cuda:0 \
        --batch_size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from inference.generate_ecg_answer import (
    _prepare_tokenizer,
    _instantiate_model,
    _split_state_dict,
    _maybe_attach_or_merge_lora,
    _build_prompt_tensors,
    _load_waveform,
    _extract_answer,
)


def collate_batch(
    batch_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str, str]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], List[str], List[str]]:
    """Collate a batch of samples with padding."""
    ecg_tensors = []
    prompt_ids_list = []
    prompt_mask_list = []
    keys = []
    waveform_names = []
    questions = []
    ground_truths = []
    
    for ecg, prompt_ids, prompt_mask, key, wf_name, question, gt in batch_data:
        ecg_tensors.append(ecg)
        prompt_ids_list.append(prompt_ids)
        prompt_mask_list.append(prompt_mask)
        keys.append(key)
        waveform_names.append(wf_name)
        questions.append(question)
        ground_truths.append(gt)
    
    # Stack ECG tensors (all same size)
    ecg_batch = torch.stack(ecg_tensors, dim=0).to(device)
    
    # Pad prompt tensors
    prompt_ids_padded = pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_token_id).to(device)
    prompt_mask_padded = pad_sequence(prompt_mask_list, batch_first=True, padding_value=0).to(device)
    
    return ecg_batch, prompt_ids_padded, prompt_mask_padded, keys, waveform_names, questions, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Run inference on full parquet dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--parquet", required=True, help="Path to parquet dataset")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="cuda:0", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--waveform_column", default="waveform_path_psa", help="Column with waveform paths")
    parser.add_argument("--question_column", default="prompt", help="Column with questions/prompts")
    parser.add_argument("--answer_column", default="generated_answer", help="Column with ground truth answers")
    parser.add_argument("--waveform_name_column", default="waveform_name", help="Column with waveform names")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (None = all)")
    parser.add_argument("--batch_save_interval", type=int, default=1000, help="Save progress every N samples")
    args = parser.parse_args()

    # Set up environment
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]
    
    if device.type == "cuda":
        config.device = device.index or 0
    else:
        config.device = 0
    config.world_size = 1
    config.is_ref_device = True

    # Prepare tokenizer and model
    print("Preparing tokenizer and model...")
    tokenizer, ecg_token_start_id = _prepare_tokenizer(config)
    model = _instantiate_model(config, tokenizer, ecg_token_start_id)
    
    state_dict = checkpoint_data["model_state_dict"]
    base_sd, lora_sd = _split_state_dict(state_dict)
    model.load_state_dict(base_sd, strict=False)
    _maybe_attach_or_merge_lora(model, lora_sd, config)
    
    model.eval()
    model.to(device)

    # Build generation kwargs
    generation_kwargs = dict(getattr(config, "default_generation_kwargs", {}) or {})
    generation_kwargs.setdefault("max_new_tokens", 96)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    generation_kwargs.setdefault("no_repeat_ngram_size", 5)
    generation_kwargs.setdefault("repetition_penalty", 1.1)
    
    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
        eos_ids.append(end_of_turn_id)
    generation_kwargs.setdefault("eos_token_id", eos_ids)

    # Load parquet
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"Total samples in parquet: {len(df)}")

    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Processing first {len(df)} samples")

    # Prepare output
    results: dict[str, dict[str, Any]] = {}
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check for existing progress
    if os.path.exists(args.output):
        print(f"Found existing output file, loading progress...")
        with open(args.output, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    # Process samples
    ecg_waveform_length = int(getattr(config, "ecg_waveform_length", 2500))
    ecg_num_leads = int(getattr(config, "ecg_num_leads", 12))
    max_token_length = int(getattr(config, "max_token_length", 640))
    max_new_tokens = int(generation_kwargs["max_new_tokens"])

    errors = []
    processed = 0
    skipped = 0
    
    # Prepare all valid samples first
    print("\nPreparing samples...")
    samples_to_process = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        waveform_name = str(row.get(args.waveform_name_column, f"sample_{idx}"))
        question = str(row.get(args.question_column, ""))
        ground_truth = str(row.get(args.answer_column, ""))
        waveform_path = str(row.get(args.waveform_column, ""))

        # Create unique key
        key = f"{waveform_name}_{hash(question) % 10000}"
        
        # Skip if already processed
        if key in results:
            skipped += 1
            continue

        if not waveform_path or not os.path.exists(waveform_path):
            errors.append({"idx": idx, "waveform_name": waveform_name, "error": f"Waveform not found: {waveform_path}"})
            continue

        if not question:
            errors.append({"idx": idx, "waveform_name": waveform_name, "error": "Empty question"})
            continue

        samples_to_process.append({
            "idx": idx,
            "key": key,
            "waveform_name": waveform_name,
            "question": question,
            "ground_truth": ground_truth,
            "waveform_path": waveform_path,
        })

    print(f"Samples to process: {len(samples_to_process)}")
    print(f"Already processed: {skipped}")
    print(f"Errors during prep: {len(errors)}")

    # Process in batches
    print(f"\nStarting batched inference (batch_size={args.batch_size})...")
    num_batches = (len(samples_to_process) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Inference batches"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(samples_to_process))
        batch_samples = samples_to_process[batch_start:batch_end]
        
        # Prepare batch data
        batch_data = []
        batch_errors = []
        
        for sample in batch_samples:
            try:
                # Load waveform
                ecg_tensor = _load_waveform(
                    sample["waveform_path"],
                    target_length=ecg_waveform_length,
                    num_leads=ecg_num_leads,
                ).squeeze(0)  # Remove batch dim for now
                
                # Build prompt
                prompt_ids, prompt_mask, _ = _build_prompt_tensors(
                    sample["question"],
                    tokenizer,
                    config,
                    max_new_tokens,
                )
                
                batch_data.append((
                    ecg_tensor,
                    prompt_ids,
                    prompt_mask,
                    sample["key"],
                    sample["waveform_name"],
                    sample["question"],
                    sample["ground_truth"],
                ))
            except Exception as e:
                batch_errors.append({
                    "idx": sample["idx"],
                    "waveform_name": sample["waveform_name"],
                    "error": str(e)
                })
        
        errors.extend(batch_errors)
        
        if not batch_data:
            continue
        
        # Collate batch
        ecg_batch, prompt_ids_batch, prompt_mask_batch, keys, waveform_names, questions, ground_truths = collate_batch(
            batch_data, tokenizer.pad_token_id, device
        )
        
        try:
            # Generate for batch
            with torch.no_grad():
                generated_ids = model.generate_report(
                    x=ecg_batch,
                    prompt_input_ids=prompt_ids_batch,
                    prompt_attention_mask=prompt_mask_batch,
                    max_token_length=max_token_length,
                    **generation_kwargs,
                )
            
            # Decode each sample in batch
            for i in range(len(batch_data)):
                gen_ids = generated_ids[i].to("cpu")
                decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                generation = _extract_answer(decoded, questions[i])
                if not generation:
                    generation = decoded.strip()
                
                results[keys[i]] = {
                    "waveform_name": waveform_names[i],
                    "Question": questions[i],
                    "Generation": generation,
                    "Ground truth": ground_truths[i],
                }
                processed += 1
                
        except Exception as e:
            # If batch fails, try one by one
            for i, (ecg, prompt_ids, prompt_mask, key, wf_name, question, gt) in enumerate(batch_data):
                try:
                    with torch.no_grad():
                        gen_ids = model.generate_report(
                            x=ecg.unsqueeze(0).to(device),
                            prompt_input_ids=prompt_ids.unsqueeze(0).to(device),
                            prompt_attention_mask=prompt_mask.unsqueeze(0).to(device),
                            max_token_length=max_token_length,
                            **generation_kwargs,
                        )
                    
                    decoded = tokenizer.decode(gen_ids[0].to("cpu").tolist(), skip_special_tokens=True)
                    generation = _extract_answer(decoded, question)
                    if not generation:
                        generation = decoded.strip()
                    
                    results[key] = {
                        "waveform_name": wf_name,
                        "Question": question,
                        "Generation": generation,
                        "Ground truth": gt,
                    }
                    processed += 1
                except Exception as e2:
                    errors.append({"waveform_name": wf_name, "error": str(e2)})

        # Save progress periodically
        if processed > 0 and processed % args.batch_save_interval == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"Saved progress: {processed} processed, {len(errors)} errors")

    # Final save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total results: {len(results)}")
    print(f"  Output saved to: {args.output}")

    if errors:
        error_file = args.output.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"  Errors saved to: {error_file}")


if __name__ == "__main__":
    main()

