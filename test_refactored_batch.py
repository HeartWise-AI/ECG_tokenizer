#!/usr/bin/env python3
"""Test refactored batch inference with TRUE batched generation for uniform tasks."""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from inference.generate_ecg_answer import (
    _prepare_tokenizer,
    _instantiate_model,
    _split_state_dict,
    _maybe_attach_or_merge_lora,
    _build_prompt_tensors,
    _load_waveform,
    _extract_answer,
)

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

checkpoint_path = "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/d8389lsr_20251129-073725/checkpoint_step_5500.pt"
device = torch.device("cuda:0")

print("Loading model...")
checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint_data["config"]
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

ecg_waveform_length = int(getattr(config, "ecg_waveform_length", 2500))
ecg_num_leads = int(getattr(config, "ecg_num_leads", 12))
max_token_length = int(getattr(config, "max_token_length", 640))
max_new_tokens = int(generation_kwargs["max_new_tokens"])

# Select 4 JSON questions (uniform task)
df = pd.read_parquet('/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet')

samples = []
seen_ecgs = set()

for idx, row in df.iterrows():
    if 'json' in str(row['prompt']).lower():
        ecg = row['waveform_path_psa']
        if ecg not in seen_ecgs and len(samples) < 4:
            samples.append({
                'waveform': ecg,
                'question': row['prompt'],
            })
            seen_ecgs.add(ecg)

print(f"\nTesting REFACTORED code with 4 uniform-task (JSON) ECGs")
print("This should use TRUE batched generation (fast path)\n")

# Individual processing
print("STEP 1: Individual Processing")
individual_results = []

with torch.no_grad():
    for i, sample in enumerate(samples):
        ecg_tensor = _load_waveform(sample['waveform'], ecg_waveform_length, ecg_num_leads)
        prompt_ids, prompt_mask, _ = _build_prompt_tensors(sample['question'], tokenizer, config, max_new_tokens)

        generated_ids = model.generate_report(
            x=ecg_tensor.to(device),
            prompt_input_ids=prompt_ids.unsqueeze(0).to(device),
            prompt_attention_mask=prompt_mask.unsqueeze(0).to(device),
            max_token_length=max_token_length,
            **generation_kwargs,
        )

        gen_ids = generated_ids[0].to("cpu")
        decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        generation = _extract_answer(decoded, sample['question'])
        if not generation:
            generation = decoded.strip()

        individual_results.append(generation)
        print(f"  Sample {i+1}: {generation[:60]}...")

# Batch processing
print("\nSTEP 2: Batch Processing (TRUE batched generation)")

batch_ecg_tensors = []
batch_prompt_ids = []
batch_prompt_masks = []

for sample in samples:
    ecg_tensor = _load_waveform(sample['waveform'], ecg_waveform_length, ecg_num_leads)
    prompt_ids, prompt_mask, _ = _build_prompt_tensors(sample['question'], tokenizer, config, max_new_tokens)
    batch_ecg_tensors.append(ecg_tensor.squeeze(0))
    batch_prompt_ids.append(prompt_ids)
    batch_prompt_masks.append(prompt_mask)

ecg_batch = torch.stack(batch_ecg_tensors, dim=0).to(device)
prompt_ids_batch = pad_sequence(batch_prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
prompt_mask_batch = pad_sequence(batch_prompt_masks, batch_first=True, padding_value=0).to(device)

print(f"Batch shapes: ECG={ecg_batch.shape}, Prompts={prompt_ids_batch.shape}")

batch_results = []

with torch.no_grad():
    generated_ids_batch = model.generate_report(
        x=ecg_batch,
        prompt_input_ids=prompt_ids_batch,
        prompt_attention_mask=prompt_mask_batch,
        max_token_length=max_token_length,
        **generation_kwargs,
    )

    for i in range(len(samples)):
        gen_ids = generated_ids_batch[i].to("cpu")
        decoded = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        generation = _extract_answer(decoded, samples[i]['question'])
        if not generation:
            generation = decoded.strip()

        batch_results.append(generation)
        print(f"  Sample {i+1}: {generation[:60]}...")

# Compare
print(f"\n{'='*80}")
print("STEP 3: EQUIVALENCE CHECK")
print(f"{'='*80}\n")

all_match = True
for i in range(len(samples)):
    match = individual_results[i] == batch_results[i]
    status = "✅ MATCH" if match else "❌ MISMATCH"
    print(f"Sample {i+1}: {status}")
    if not match:
        print(f"  Individual: {individual_results[i][:60]}...")
        print(f"  Batch:      {batch_results[i][:60]}...")
        all_match = False

print(f"\n{'='*80}")
if all_match:
    print("✅ SUCCESS: Refactored code produces IDENTICAL outputs!")
    print("TRUE batched generation is working correctly for uniform tasks.")
else:
    print("❌ FAILURE: Outputs differ")
print(f"{'='*80}")
