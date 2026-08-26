#!/usr/bin/env python3
"""
Generate multiple outputs per sample for DPO training - batched and efficient.
Based on generate_all_qa_pairs.py but with multi-generation support.
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
from typing import Optional, Any, List, Dict

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, "/volume/ECG_tokenizer")

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml
from utils.checkpoint_structure import resolve_checkpoint_bridge_option


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
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


def generate_answer_with_temp(
    model, tokenizer, ecg_tensor: torch.Tensor, question: str, 
    device: torch.device, config, temperature: float = 0.3
) -> str:
    """Generate answer for a single (ECG, question) pair with specified temperature."""
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
        "max_new_tokens": 96,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.1,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": 0.9 if temperature > 0 else 1.0,
    }
    
    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            max_token_length=int(getattr(config, "max_token_length", 640)),
            **generation_kwargs,
        )
    
    generation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    for pattern in [r'<end_of_turn>', r'<start_of_turn>', r'<\|eot_id\|>', r'model\s*$', r'^model\s*']:
        generation = re.sub(pattern, '', generation)
    
    return generation.strip()


def generate_multi_outputs(
    model, tokenizer, ecg_tensor: torch.Tensor, question: str,
    device: torch.device, config, temperatures: List[float]
) -> List[Dict]:
    """Generate multiple outputs with different temperatures."""
    outputs = []
    for i, temp in enumerate(temperatures):
        try:
            output = generate_answer_with_temp(model, tokenizer, ecg_tensor, question, device, config, temp)
            outputs.append({
                "idx": i,
                "temperature": temp,
                "output": output
            })
        except Exception as e:
            outputs.append({
                "idx": i,
                "temperature": temp,
                "output": f"ERROR: {e}"
            })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", default="dpo_multi")
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--temperatures", type=str, default="0.1,0.3,0.3,0.5,0.5")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    
    temperatures = [float(t) for t in args.temperatures.split(",")]
    # Extend or truncate to match num_generations
    while len(temperatures) < args.num_generations:
        temperatures.append(temperatures[-1])
    temperatures = temperatures[:args.num_generations]
    
    print(f"Device: {device}")
    print(f"Generating {args.num_generations} outputs per sample")
    print(f"Temperatures: {temperatures}")
    
    model, tokenizer, config = load_model(args.checkpoint, device)
    
    print(f"Loading data from {args.validation_parquet}")
    df = pd.read_parquet(args.validation_parquet)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    print(f"Total samples: {len(df)}")
    
    # Resume logic
    output_file = os.path.join(args.output_dir, f"{args.output_prefix}_generations.json")
    checkpoint_file = os.path.join(args.output_dir, f"{args.output_prefix}_checkpoint.json")
    
    results = []
    start_idx = 0
    
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
            start_idx = ckpt.get("last_idx", 0) + 1
            results = ckpt.get("results", [])
        print(f"Resuming from index {start_idx}")
    
    # Cache waveforms for efficiency (group by waveform)
    waveform_cache = {}
    
    for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df), desc="Generating"):
        row = df.iloc[idx]
        waveform_path = row['waveform_path_psa']
        
        # Load waveform (with caching)
        if waveform_path not in waveform_cache:
            waveform = load_ecg_waveform(waveform_path)
            if waveform is not None:
                waveform_cache[waveform_path] = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)
            else:
                waveform_cache[waveform_path] = None
        
        ecg_tensor = waveform_cache.get(waveform_path)
        
        if ecg_tensor is None:
            results.append({"idx": idx, "error": "Failed to load waveform"})
            continue
        
        try:
            generations = generate_multi_outputs(
                model, tokenizer, ecg_tensor, row['prompt'],
                device, config, temperatures
            )
            
            result = {
                "idx": idx,
                "waveform_path": waveform_path,
                "prompt": row['prompt'],
                "prompt_category": row['prompt_category'],
                "ground_truth": row['generated_answer'],
                "generations": generations
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            results.append({"idx": idx, "error": str(e)})
        
        # Checkpoint
        if (idx + 1) % args.save_interval == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_idx": idx, "results": results}, f)
            print(f"\nCheckpoint saved at {idx + 1}")
            
            # Clear waveform cache periodically to manage memory
            if len(waveform_cache) > 5000:
                waveform_cache.clear()
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(results)} samples to {output_file}")
    
    # Summary stats
    errors = sum(1 for r in results if "error" in r)
    print(f"Errors: {errors}/{len(results)}")


if __name__ == "__main__":
    main()
