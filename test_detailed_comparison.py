#!/usr/bin/env python3
"""Detailed comparison of individual vs batch outputs."""

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

# Select 4 JSON questions
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

print(f"\nTesting {len(samples)} JSON ECGs\n")

# Individual processing
print("="*80)
print("INDIVIDUAL PROCESSING")
print("="*80)
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
        print(f"\nSample {i+1} (prompt_len={len(prompt_ids)}):")
        print(f"  {generation}")

# Batch processing
print("\n" + "="*80)
print("BATCH PROCESSING (GROUPED)")
print("="*80)

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
        print(f"\nSample {i+1}:")
        print(f"  {generation}")

# Detailed comparison
print("\n" + "="*80)
print("DETAILED COMPARISON")
print("="*80)

for i in range(len(samples)):
    individual = individual_results[i]
    batch = batch_results[i]

    match = individual == batch
    status = "✅ EXACT MATCH" if match else "❌ DIFFER"

    print(f"\n{'='*80}")
    print(f"Sample {i+1}: {status}")
    print(f"{'='*80}")

    if match:
        print(f"Both produce: {individual[:100]}...")
    else:
        print(f"\nINDIVIDUAL ({len(individual)} chars):")
        print(f"{individual}")
        print(f"\nBATCH ({len(batch)} chars):")
        print(f"{batch}")

        # Character-by-character comparison for first difference
        min_len = min(len(individual), len(batch))
        for j in range(min_len):
            if individual[j] != batch[j]:
                print(f"\nFirst difference at position {j}:")
                print(f"  Individual: ...{individual[max(0,j-20):j+20]}...")
                print(f"  Batch:      ...{batch[max(0,j-20):j+20]}...")
                break

        if len(individual) != len(batch):
            print(f"\nLength difference: Individual={len(individual)}, Batch={len(batch)}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
matches = sum(1 for i in range(len(samples)) if individual_results[i] == batch_results[i])
print(f"Matches: {matches}/{len(samples)}")
print(f"Success rate: {matches/len(samples)*100:.1f}%")
