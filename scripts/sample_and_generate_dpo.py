#!/usr/bin/env python3
"""
Sample from training parquet and generate DPO candidates with focused category sampling.

This script:
1. Samples from training parquet with category-based stratified sampling
2. Applies bucket-based regex filtering (e.g., ST elevation prompts)
3. Excludes specified categories
4. Generates diverse candidates for each sample using temperature sampling
5. Outputs JSONL ready for DPO ranking

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_and_generate_dpo.py \
        --checkpoint checkpoints/.../best_model.pt \
        --train_parquet output/combined_train_qa_m5000k_h5000k_weighted.parquet \
        --sampling_config config/dpo_sampling_50k.json \
        --bucket_config config/dpo_bucket_sampling_50k.json \
        --output_dir output/dpo_focused_50k \
        --num_generations 5 \
        --temperature 0.7
"""

import os
import sys
import json
import argparse
import re
import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from tqdm import tqdm

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml


# Categories to exclude (already good or noisy)
EXCLUDED_CATEGORIES = {
    "category_other",
    "category_chamber_enlargement",
    "category_pericarditis",
    "random_finding_question",
    "localization_qrs_axis",
    "localization_q_wave",
    "localization_t_wave",
    "localization_st_depression",
    "ecg_interval",
}


def _cfg_get(container, key, fallback=None):
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
        yaml_config = load_yaml(config_yaml_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))

    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", _cfg_get(config, "num_codebooks_kept", None))

    checkpoint_state_dict = checkpoint_data["model_state_dict"]
    mix_gate_key = "decoder.bridge.mix_gate.0.weight"
    if mix_gate_key in checkpoint_state_dict:
        mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
        hidden_dim = mix_gate_shape[0]
        inferred_codebooks = mix_gate_shape[1] // hidden_dim
        if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))

    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and bool(getattr(config, "use_lora", False)):
        lora_config = {
            "r": int(getattr(config, "lora_r", 16)),
            "lora_alpha": int(getattr(config, "lora_alpha", getattr(config, "lora_r", 16))),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.0)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or []),
            "bias": str(getattr(config, "lora_bias", "none")),
        }

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

    model._load_state_dict(checkpoint_data["model_state_dict"], strict=False)

    if bool(getattr(config, "use_lora", False)):
        model.set_lora_inference_mode(True)

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
        return None


def sample_from_parquet(
    df: pd.DataFrame,
    sampling_config: Dict[str, int],
    bucket_config: List[Dict[str, Any]],
    excluded_categories: set,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample from dataframe with category-based and bucket-based sampling."""
    np.random.seed(seed)

    # Filter out excluded categories
    df_filtered = df[~df['prompt_category'].isin(excluded_categories)].copy()
    print(f"After excluding categories: {len(df_filtered):,} rows")

    sampled_dfs = []
    used_indices = set()

    # 1. Category-based sampling
    print("\n=== Category Sampling ===")
    for category, n_samples in sampling_config.items():
        cat_df = df_filtered[df_filtered['prompt_category'] == category]
        available = len(cat_df)
        actual_samples = min(n_samples, available)

        if actual_samples > 0:
            sampled = cat_df.sample(n=actual_samples, random_state=seed)
            sampled_dfs.append(sampled)
            used_indices.update(sampled.index.tolist())
            print(f"  {category}: requested={n_samples}, available={available}, sampled={actual_samples}")
        else:
            print(f"  {category}: requested={n_samples}, available={available}, sampled=0 (SKIP)")

    # 2. Bucket-based sampling (regex filtering)
    print("\n=== Bucket Sampling ===")
    for bucket in bucket_config:
        name = bucket.get("name", "unnamed")
        regex = bucket.get("prompt_regex", "")
        ignore_case = bucket.get("ignore_case", True)
        category_in = bucket.get("category_in", [])
        n_samples = bucket.get("n_samples", 0)
        set_prompt_category = bucket.get("set_prompt_category", False)

        # Filter by category if specified
        if category_in:
            bucket_df = df_filtered[df_filtered['prompt_category'].isin(category_in)]
        else:
            bucket_df = df_filtered

        # Apply regex filter on prompt
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(regex, flags)
        mask = bucket_df['prompt'].apply(lambda x: bool(pattern.search(str(x))))
        bucket_df = bucket_df[mask]

        # Exclude already sampled
        bucket_df = bucket_df[~bucket_df.index.isin(used_indices)]

        available = len(bucket_df)
        actual_samples = min(n_samples, available)

        if actual_samples > 0:
            sampled = bucket_df.sample(n=actual_samples, random_state=seed)
            if set_prompt_category:
                sampled = sampled.copy()
                sampled['prompt_category'] = name
            sampled_dfs.append(sampled)
            used_indices.update(sampled.index.tolist())
            print(f"  {name}: regex='{regex}', categories={category_in}, requested={n_samples}, available={available}, sampled={actual_samples}")
        else:
            print(f"  {name}: requested={n_samples}, available={available}, sampled=0 (SKIP)")

    # Combine all samples
    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
        # Shuffle the combined result
        result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"\n=== Total sampled: {len(result):,} rows ===")
        print("\nFinal category distribution:")
        print(result['prompt_category'].value_counts())
        return result
    else:
        return pd.DataFrame()


def generate_diverse(
    model,
    tokenizer,
    ecg_tensor: torch.Tensor,
    question: str,
    device: torch.device,
    num_samples: int = 5,
    temperatures: List[float] = None,
) -> List[Dict[str, Any]]:
    """Generate multiple diverse answers for a single (ECG, question) pair.

    Uses varied temperatures for diversity:
    - Low temp (0.3): Conservative, safe outputs
    - Med temp (0.7): Balanced diversity
    - High temp (1.0): Creative, riskier outputs

    Returns list of dicts with 'text' and 'temperature' keys.
    """
    if temperatures is None:
        # Default: varied temperatures for DPO diversity
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]

    # Extend or truncate to match num_samples
    if len(temperatures) < num_samples:
        # Cycle through temperatures
        temperatures = (temperatures * ((num_samples // len(temperatures)) + 1))[:num_samples]
    elif len(temperatures) > num_samples:
        temperatures = temperatures[:num_samples]

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

    base_generation_kwargs = {
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 96,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.1,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }

    generations = []
    for temp in temperatures:
        generation_kwargs = {**base_generation_kwargs, "temperature": temp}
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
        generations.append({
            "text": generation.strip(),
            "temperature": temp,
        })

    return generations


def main():
    parser = argparse.ArgumentParser(description="Sample and generate DPO candidates")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--train_parquet", type=str,
                        default="output/combined_train_qa_m5000k_h5000k_weighted.parquet",
                        help="Path to training parquet")
    parser.add_argument("--sampling_config", type=str, required=True,
                        help="Path to category sampling config JSON")
    parser.add_argument("--bucket_config", type=str, required=True,
                        help="Path to bucket sampling config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_generations", type=int, default=5, help="Generations per sample")
    parser.add_argument("--temperatures", type=str, default="0.3,0.5,0.7,0.9,1.0",
                        help="Comma-separated temperatures for diverse generation (default: 0.3,0.5,0.7,0.9,1.0)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint interval")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--sample_only", action="store_true", help="Only sample, don't generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configs
    with open(args.sampling_config) as f:
        sampling_config = json.load(f)
    with open(args.bucket_config) as f:
        bucket_config = json.load(f)

    print(f"\n=== Sampling Config ===")
    print(json.dumps(sampling_config, indent=2))
    print(f"\n=== Bucket Config ===")
    print(json.dumps(bucket_config, indent=2))

    # Load and sample from parquet
    sampled_parquet_path = os.path.join(args.output_dir, "sampled_data.parquet")

    if args.resume and os.path.exists(sampled_parquet_path):
        print(f"\nLoading existing sampled data from {sampled_parquet_path}")
        sampled_df = pd.read_parquet(sampled_parquet_path)
    else:
        print(f"\nLoading training parquet from {args.train_parquet}...")
        df = pd.read_parquet(args.train_parquet)
        print(f"Loaded {len(df):,} rows")

        sampled_df = sample_from_parquet(
            df, sampling_config, bucket_config, EXCLUDED_CATEGORIES, seed=args.seed
        )

        # Save sampled data
        sampled_df.to_parquet(sampled_parquet_path)
        print(f"Saved sampled data to {sampled_parquet_path}")

    if args.sample_only:
        print("\n--sample_only flag set, exiting without generation")
        return

    # Load model
    model, tokenizer, config = load_model(args.checkpoint, device)

    # Check for resume checkpoint
    output_jsonl = os.path.join(args.output_dir, "dpo_generations.jsonl")
    checkpoint_path = os.path.join(args.output_dir, "generation_checkpoint.json")

    start_idx = 0
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
            start_idx = ckpt.get("last_idx", 0) + 1
        print(f"Resuming from index {start_idx}")

    # Parse temperatures
    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    print(f"\n=== Generating {len(temperatures)} candidates per sample ===")
    print(f"Temperatures: {temperatures}")

    mode = "a" if args.resume and start_idx > 0 else "w"

    with open(output_jsonl, mode) as f:
        for idx in tqdm(range(start_idx, len(sampled_df)), desc="Generating"):
            row = sampled_df.iloc[idx]

            waveform_path = row['waveform_path_psa']
            question = row['prompt']
            ground_truth = row['generated_answer']
            category = row['prompt_category']
            waveform_name = row.get('waveform_name', os.path.basename(waveform_path))

            # Load ECG
            waveform = load_ecg_waveform(waveform_path)
            if waveform is None:
                continue

            ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)

            # Generate diverse samples with varied temperatures
            generations = generate_diverse(
                model, tokenizer, ecg_tensor, question, device,
                num_samples=len(temperatures),
                temperatures=temperatures,
            )

            # Write result
            result = {
                "waveform_path": waveform_path,
                "waveform_name": waveform_name,
                "prompt": question,
                "ground_truth": ground_truth,
                "category": category,
                "generations": generations,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

            # Save checkpoint periodically
            if (idx + 1) % args.save_interval == 0:
                with open(checkpoint_path, "w") as ckpt_f:
                    json.dump({"last_idx": idx}, ckpt_f)
                print(f"  Checkpoint saved at index {idx}")

    # Save final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump({"last_idx": len(sampled_df) - 1, "complete": True}, f)

    print(f"\n=== Generation complete ===")
    print(f"Output: {output_jsonl}")
    print(f"Total samples: {len(sampled_df)}")


if __name__ == "__main__":
    main()
