#!/usr/bin/env python3
"""
Run custom questions on a subset of ECGs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference/run_custom_questions.py \
        --checkpoint checkpoints/BEST_LLM/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt \
        --output_dir checkpoints/BEST_LLM/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/ \
        --num_ecgs 100 \
        --temperature 0.5
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from typing import Optional, Any

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml
from utils.checkpoint_structure import resolve_checkpoint_bridge_option


CUSTOM_QUESTIONS = [
    "What is the rhythm? Is there anything else I should know or do?",
    "Are there any ST segment changes? If so, describe them.",
]


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


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

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


def generate_answer(
    model,
    tokenizer,
    ecg_tensor: torch.Tensor,
    question: str,
    device: torch.device,
    temperature: float = 0.5,
) -> str:
    """Generate answer for a single (ECG, question) pair."""
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

    generation_kwargs = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.9,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.1,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }

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


def main():
    parser = argparse.ArgumentParser(description="Run custom questions on ECGs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--validation_parquet", type=str,
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet",
                        help="Path to validation parquet (for ECG paths)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_ecgs", type=int, default=100, help="Number of unique ECGs to process")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ECG selection")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Temperature: {args.temperature}")

    model, tokenizer, config = load_model(args.checkpoint, device)

    print(f"\nLoading validation data from {args.validation_parquet}...")
    val_df = pd.read_parquet(args.validation_parquet)

    # Get unique ECGs
    unique_ecgs = val_df[['waveform_name', 'waveform_path_psa']].drop_duplicates()
    print(f"Total unique ECGs: {len(unique_ecgs)}")

    # Sample random ECGs
    np.random.seed(args.seed)
    if len(unique_ecgs) > args.num_ecgs:
        unique_ecgs = unique_ecgs.sample(n=args.num_ecgs, random_state=args.seed)
    print(f"Selected {len(unique_ecgs)} ECGs")

    results = []
    errors = 0

    print(f"\nGenerating answers for {len(CUSTOM_QUESTIONS)} questions per ECG...")
    print(f"Questions:")
    for i, q in enumerate(CUSTOM_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()

    for idx, row in tqdm(unique_ecgs.iterrows(), total=len(unique_ecgs), desc="Processing ECGs"):
        waveform_name = row['waveform_name']
        waveform_path = row['waveform_path_psa']

        try:
            waveform = load_ecg_waveform(waveform_path)
            if waveform is None:
                errors += 1
                continue

            ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)

            for question in CUSTOM_QUESTIONS:
                generation = generate_answer(
                    model, tokenizer, ecg_tensor, question, device,
                    temperature=args.temperature
                )

                results.append({
                    'waveform_name': waveform_name,
                    'waveform_path': waveform_path,
                    'question': question,
                    'answer': generation,
                })

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\nError on {waveform_name}: {e}")
            continue

    print(f"\nGenerated {len(results)} answers, {errors} errors")

    # Save results
    results_df = pd.DataFrame(results)

    output_prefix = f"custom_questions_{args.num_ecgs}ecgs_temp{args.temperature}"

    csv_path = os.path.join(args.output_dir, f"{output_prefix}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Save as JSON grouped by waveform
    json_data = {}
    for waveform_name, group in results_df.groupby('waveform_name'):
        json_data[waveform_name] = []
        for _, row in group.iterrows():
            json_data[waveform_name].append({
                'Question': row['question'],
                'Answer': row['answer'],
            })

    json_path = os.path.join(args.output_dir, f"{output_prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON to {json_path}")

    # Print a few samples
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS")
    print("=" * 80)

    sample_ecgs = results_df['waveform_name'].unique()[:3]
    for ecg_name in sample_ecgs:
        print(f"\n--- {ecg_name} ---")
        ecg_results = results_df[results_df['waveform_name'] == ecg_name]
        for _, row in ecg_results.iterrows():
            print(f"\nQ: {row['question']}")
            print(f"A: {row['answer']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
