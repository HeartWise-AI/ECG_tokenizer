#!/usr/bin/env python3
"""
Test batch inference consistency by comparing:
1. Single-sample inference (batch_size=1)
2. Batched inference (batch_size=20) with target ECG at random positions
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Setup environment
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.generate_ecg_answer import (
    _prepare_tokenizer,
    _instantiate_model,
    _split_state_dict,
    _maybe_attach_or_merge_lora,
    _build_prompt_tensors,
    _load_waveform,
    _extract_answer,
)


def load_model_and_tokenizer(checkpoint_path, device):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

    if device.type == "cuda":
        config.device = device.index or 0
    else:
        config.device = 0
    config.world_size = 1
    config.is_ref_device = True

    tokenizer, ecg_token_start_id = _prepare_tokenizer(config)
    model = _instantiate_model(config, tokenizer, ecg_token_start_id)

    state_dict = checkpoint_data["model_state_dict"]
    base_sd, lora_sd = _split_state_dict(state_dict)
    model.load_state_dict(base_sd, strict=False)
    _maybe_attach_or_merge_lora(model, lora_sd, config)

    model.eval()
    model.to(device)

    return model, tokenizer, config


def generate_single(model, tokenizer, config, ecg_tensor, question, device):
    """Generate answer for single sample (batch_size=1)."""
    prompt_ids, prompt_mask, _ = _build_prompt_tensors(
        question, tokenizer, config, 96
    )

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

    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor.unsqueeze(0).to(device),
            prompt_input_ids=prompt_ids.unsqueeze(0).to(device),
            prompt_attention_mask=prompt_mask.unsqueeze(0).to(device),
            max_token_length=640,
            **generation_kwargs,
        )

    decoded = tokenizer.decode(generated_ids[0].to("cpu").tolist(), skip_special_tokens=True)
    answer = _extract_answer(decoded, question)
    if not answer:
        answer = decoded.strip()

    return answer


def generate_batched(model, tokenizer, config, ecg_tensors, questions, device, target_idx):
    """Generate answers for batch of samples."""
    from torch.nn.utils.rnn import pad_sequence
    
    batch_size = len(questions)

    # Build prompts for all samples
    prompt_ids_list = []
    prompt_mask_list = []
    for question in questions:
        prompt_ids, prompt_mask, _ = _build_prompt_tensors(
            question, tokenizer, config, 96
        )
        prompt_ids_list.append(prompt_ids)
        prompt_mask_list.append(prompt_mask)

    # Pad to same length
    prompt_ids_batch = pad_sequence(prompt_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    prompt_mask_batch = pad_sequence(prompt_mask_list, batch_first=True, padding_value=0)

    # Stack ECG tensors
    ecg_batch = torch.stack(ecg_tensors, dim=0).to(device)
    prompt_ids_batch = prompt_ids_batch.to(device)
    prompt_mask_batch = prompt_mask_batch.to(device)

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

    print(f"  [Generating batch_size={batch_size}, target at position {target_idx}]")

    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_batch,
            prompt_input_ids=prompt_ids_batch,
            prompt_attention_mask=prompt_mask_batch,
            max_token_length=640,
            **generation_kwargs,
        )

    # Extract answer for target sample
    decoded = tokenizer.decode(generated_ids[target_idx].to("cpu").tolist(), skip_special_tokens=True)
    answer = _extract_answer(decoded, questions[target_idx])
    if not answer:
        answer = decoded.strip()

    return answer


def create_batch_with_target(num_samples, target_ecg, target_question, target_idx):
    """Create a batch with target ECG at specified position."""
    dummy_questions = [
        "What is the heart rate?",
        "Is the rhythm regular?",
        "Are there any ST changes?",
        "What is the QRS duration?",
        "Is there any atrial fibrillation?",
        "What is the PR interval?",
        "Are there any Q waves?",
        "Is there left ventricular hypertrophy?",
        "What is the QT interval?",
        "Are there any T wave abnormalities?",
        "Is there right bundle branch block?",
        "What is the axis deviation?",
        "Are there any U waves?",
        "Is there ventricular hypertrophy?",
        "What does this ECG show?",
        "Is this ECG normal?",
        "What are the main findings?",
        "Is there anything abnormal?",
        "Describe this ECG.",
        "What is your interpretation?",
    ]

    # Create dummy ECGs (small variations of target)
    ecg_tensors = []
    questions = []

    for i in range(num_samples):
        if i == target_idx:
            # Insert target
            ecg_tensors.append(target_ecg)
            questions.append(target_question)
        else:
            # Create slight variation
            noise = torch.randn_like(target_ecg) * 0.01
            dummy_ecg = target_ecg + noise
            ecg_tensors.append(dummy_ecg)
            questions.append(dummy_questions[i % len(dummy_questions)])

    return ecg_tensors, questions


def main():
    # Configuration
    checkpoint_path = "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/d8389lsr_20251129-073725/checkpoint_step_5500.pt"
    target_ecg_path = "/media/data1/datasets/MHI/adjusted_signals/test/0011420_04-09-2017_15-36-00.npy"
    target_question = "Is there anything wrong with this ECG?"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("BATCH CONSISTENCY TEST - Random Position")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target ECG: {target_ecg_path}")
    print(f"Target question: {target_question}\n")

    # Load model
    model, tokenizer, config = load_model_and_tokenizer(checkpoint_path, device)

    # Load target ECG
    target_ecg = _load_waveform(target_ecg_path, 2500, 12).squeeze(0)

    print("-" * 80)
    print("BASELINE: Single-sample inference (batch_size=1)")
    print("-" * 80)

    single_answer = generate_single(model, tokenizer, config, target_ecg, target_question, device)
    print(f"Output: {single_answer}\n")

    # Test batched inference at different positions
    batch_sizes = [5, 20]
    num_trials = 3

    results_summary = []

    for batch_size in batch_sizes:
        print("-" * 80)
        print(f"TEST: Batched inference (batch_size={batch_size})")
        print("-" * 80)

        batch_matches = 0
        for trial in range(num_trials):
            # Random position for target ECG
            target_idx = random.randint(0, batch_size - 1)

            # Create batch
            ecg_tensors, questions = create_batch_with_target(
                batch_size, target_ecg, target_question, target_idx
            )

            # Generate
            batch_answer = generate_batched(
                model, tokenizer, config, ecg_tensors, questions, device, target_idx
            )

            # Compare
            match = (single_answer.strip() == batch_answer.strip())
            
            print(f"\n  Trial {trial + 1}: Target at position {target_idx}/{batch_size-1}")
            print(f"    Single: {single_answer}")
            print(f"    Batch:  {batch_answer}")
            
            if match:
                print(f"    Result: ✅ MATCH")
                batch_matches += 1
            else:
                print(f"    Result: ❌ MISMATCH")
                # Calculate similarity
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, single_answer.lower(), batch_answer.lower()).ratio()
                print(f"    Similarity: {similarity*100:.1f}%")

        results_summary.append({
            'batch_size': batch_size,
            'matches': batch_matches,
            'trials': num_trials,
            'success_rate': batch_matches / num_trials * 100
        })
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results_summary:
        print(f"Batch size {result['batch_size']:2d}: {result['matches']}/{result['trials']} matches ({result['success_rate']:.0f}%)")
    print()

    # Overall result
    all_match = all(r['success_rate'] == 100 for r in results_summary)
    if all_match:
        print("✅ ALL TESTS PASSED - Batch inference is consistent!")
    else:
        print("❌ SOME TESTS FAILED - Batch inference has issues")


if __name__ == "__main__":
    main()
