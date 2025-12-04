#!/usr/bin/env python3
"""Test if batch processing is deterministic (same seed → same output)."""

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

# Select first JSON ECG
df = pd.read_parquet('/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet')

for idx, row in df.iterrows():
    if 'json' in str(row['prompt']).lower():
        sample = {
            'waveform': row['waveform_path_psa'],
            'question': row['prompt'],
        }
        break

print(f"\nTesting determinism for Sample 1 (first JSON ECG)\n")

ecg_tensor = _load_waveform(sample['waveform'], ecg_waveform_length, ecg_num_leads)
prompt_ids, prompt_mask, _ = _build_prompt_tensors(sample['question'], tokenizer, config, max_new_tokens)

# Run batch inference 3 times with same seed
results = []

for run in range(3):
    print(f"Run {run+1}:")

    # Set seed for determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    with torch.no_grad():
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

        results.append(generation)
        print(f"  {generation}\n")

# Check determinism
print("="*80)
print("DETERMINISM CHECK")
print("="*80)

all_same = all(r == results[0] for r in results)

if all_same:
    print("✅ ALL 3 RUNS PRODUCE IDENTICAL OUTPUT")
    print(f"\nOutput: {results[0]}")
else:
    print("❌ RUNS PRODUCE DIFFERENT OUTPUTS")
    for i, r in enumerate(results):
        print(f"\nRun {i+1}: {r}")
