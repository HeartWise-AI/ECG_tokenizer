#!/usr/bin/env python3
"""
Generate diverse inference candidates for DPO training.

Runs multiple inferences with temperature sampling on specified samples,
producing candidates that can be ranked by ground truth or LLM judge.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_dpo_candidates.py \
        --checkpoint checkpoints/.../best_model.pt \
        --waveforms "0010322_09-19-2007_11-48-50.npy,0420851_11-21-2014_13-20-04.npy" \
        --validation_parquet /volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet \
        --num_samples 10 \
        --temperature 0.7
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Optional, List
import re

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml
from utils.artifact_provenance import atomic_write_json
from utils.checkpoint_structure import resolve_checkpoint_bridge_option


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

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

    # Get num_codebooks_kept from yaml or config
    def _cfg_get(container, key, fallback=None):
        if container is None:
            return fallback
        if isinstance(container, dict) and key in container:
            return container[key]
        if hasattr(container, key):
            val = getattr(container, key)
            return val if val is not None else fallback
        return fallback

    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", _cfg_get(config, "num_codebooks_kept", None))

    # Infer from checkpoint tensor shapes
    checkpoint_state_dict = checkpoint_data["model_state_dict"]
    mix_gate_key = "decoder.bridge.mix_gate.0.weight"
    if mix_gate_key in checkpoint_state_dict:
        mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
        hidden_dim = mix_gate_shape[0]
        inferred_codebooks = mix_gate_shape[1] // hidden_dim
        if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))

    # Reconstruct LoRA config if needed
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
        num_quantizers=int(getattr(config, "num_quantizers", 8)),
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

    return model, tokenizer, config


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


def generate_diverse(
    model,
    tokenizer,
    ecg_tensor: torch.Tensor,
    question: str,
    device: torch.device,
    config,
    num_samples: int = 10,
    temperature: float = 0.7,
) -> List[str]:
    """Generate multiple diverse answers for a single (ECG, question) pair."""
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    user_content = (
        "<start_of_image>\n\n"
        f"Question: {question}\n\n"
        "Respond concisely with the key finding or answer."
    )
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

    generation_kwargs = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 96,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.1,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Derive task hint
    generations = []
    for i in range(num_samples):
        with torch.no_grad():
            generated_ids = model.generate_report(
                x=ecg_tensor,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                max_token_length=640,
                **generation_kwargs,
            )

        generation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        for pattern in [r'<end_of_turn>', r'<start_of_turn>', r'<\|eot_id\|>', r'model\s*$', r'^model\s*']:
            generation = re.sub(pattern, '', generation)
        generations.append(generation.strip())

    return generations


def main():
    parser = argparse.ArgumentParser(description="Generate diverse candidates for DPO")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--waveforms", type=str, required=True, help="Comma-separated list of waveform names")
    parser.add_argument("--validation_parquet", type=str,
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet",
                        help="Path to validation parquet")
    parser.add_argument("--reference_json", type=str, default=None,
                        help="Path to training validation JSON for ground truth")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per waveform")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output", type=str, default="/tmp/dpo_candidates.json", help="Output JSON path")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer, config = load_model(args.checkpoint, device)

    # Parse waveform names
    waveform_names = [w.strip() for w in args.waveforms.split(",")]
    print(f"\nProcessing {len(waveform_names)} waveforms with {args.num_samples} samples each (temp={args.temperature})")

    # Load validation parquet
    val_df = pd.read_parquet(args.validation_parquet)

    # Load reference JSON if provided
    ref_data = {}
    if args.reference_json:
        with open(args.reference_json) as f:
            ref_data = json.load(f)

    results = {}
    for waveform_name in waveform_names:
        print(f"\n{'='*60}")
        print(f"Processing: {waveform_name}")
        print('='*60)

        # Find matching rows in parquet
        matches = val_df[val_df['waveform_name'] == waveform_name]
        if len(matches) == 0:
            # Try with .npy suffix
            matches = val_df[val_df['waveform_name'] == waveform_name.replace('.npy', '') + '.npy']
        if len(matches) == 0:
            matches = val_df[val_df['waveform_name'] == waveform_name.replace('.npy', '')]

        if len(matches) == 0:
            print(f"Warning: No match found for {waveform_name}")
            continue

        row = matches.iloc[0]
        waveform_path = row['waveform_path_psa']
        question = row['prompt']
        ground_truth = row.get('generated_answer', '')
        category = row.get('prompt_category', 'unknown')

        # Get reference from training validation if available
        ref_key = waveform_name if waveform_name in ref_data else f"{waveform_name}.npy"
        if ref_key not in ref_data:
            ref_key = waveform_name.replace('.npy', '')
        training_generation = ref_data.get(ref_key, {}).get('Generation', '')

        print(f"Question: {question[:80]}...")
        print(f"Category: {category}")
        print(f"Ground Truth: {ground_truth[:100]}...")
        if training_generation:
            print(f"Training Gen: {training_generation[:100]}...")

        # Load ECG
        waveform = load_ecg_waveform(waveform_path)
        if waveform is None:
            print(f"Error loading waveform")
            continue

        ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)

        # Generate diverse samples
        print(f"\nGenerating {args.num_samples} diverse samples...")
        generations = generate_diverse(
            model, tokenizer, ecg_tensor, question, device, config,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )

        results[waveform_name] = {
            'question': question,
            'category': category,
            'ground_truth': ground_truth,
            'training_generation': training_generation,
            'diverse_generations': generations,
        }

        print(f"\nGenerated {len(generations)} candidates:")
        for i, gen in enumerate(generations):
            print(f"  [{i+1}] {gen[:100]}...")

    # Save results
    atomic_write_json(output_path, results)
    print(f"\n{'='*60}")
    print(f"Saved results to {args.output}")
    print('='*60)


if __name__ == "__main__":
    main()
