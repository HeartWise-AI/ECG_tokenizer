#!/usr/bin/env python3
"""
Generate an ECG report answer from a finetuned MedGemma checkpoint.

Example:
    python scripts/generate_ecg_answer.py \
        --checkpoint checkpoints/.../best_model.pt \
        --waveform /path/to/ecg.npy \
        --question "Is the rhythm regular or irregular?"
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Tuple
from pathlib import Path
import sys
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from runners.llm_finetuning_runner import LLMFinetuningRunner
from utils.files_handler import load_yaml


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


def _is_lora_key(k: str) -> bool:
    """Check if a state dict key belongs to a LoRA adapter."""
    return (
        "lora_A" in k or "lora_B" in k or
        "lora_embedding_A" in k or "lora_embedding_B" in k or
        k.endswith("lora_scaling") or "lora_magnitude_vector" in k
    )


def _load_waveform(path: str, target_length: int, num_leads: int) -> torch.Tensor:
    """Load an ECG waveform from .npy and return tensor shaped [1, leads, length]."""
    waveform = np.load(path)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(-1)
    if waveform.shape[-1] == num_leads:
        # Stored as [length, leads]
        waveform_arr = waveform
    elif waveform.shape[0] == num_leads:
        # Stored as [leads, length]
        waveform_arr = waveform.T
    else:
        raise ValueError(f"Unexpected waveform shape {waveform.shape} for {path}")

    if waveform_arr.shape[0] != target_length:
        raise ValueError(
            f"Waveform length {waveform_arr.shape[0]} does not match expected {target_length}"
        )
    if waveform_arr.shape[1] != num_leads:
        raise ValueError(
            f"Waveform leads {waveform_arr.shape[1]} does not match expected {num_leads}"
        )

    tensor = torch.from_numpy(waveform_arr.astype(np.float32)).T.unsqueeze(0)
    return tensor


def _prepare_tokenizer(config: Any) -> Tuple[Any, int]:
    """Recreate tokenizer state exactly as in finetuning; returns tokenizer and ecg_token_start_id."""
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if isinstance(tokenizer.pad_token_id, list):
        tokenizer.pad_token_id = tokenizer.pad_token_id[0]

    ecg_start_id = getattr(config, "ecg_token_start_id", None)
    
    # Check if this is a MedGemma model (uses <start_of_image> for ECG injection)
    is_medgemma = "medgemma" in getattr(config, "huggingface_model_name", "").lower() or \
                  "medgemma" in getattr(config, "decoder_name", "").lower()
    
    if is_medgemma:
        # MedGemma uses <start_of_image> token for ECG injection - no custom tokens needed
        # The decoder handles injection after this token
        return tokenizer, None
    
    # Legacy LLaMA-style ECG token injection
    num_ecg_tokens = int(getattr(config, "num_ecg_tokens", 0))
    if getattr(config, "instruct_mode", False) and num_ecg_tokens > 0:
        if getattr(config, "chat_template", None):
            tokenizer.chat_template = config.chat_template
        else:
            custom_template = (
                "<|begin_of_text|>"
                "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                        "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'user' %}"
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                        "<|start_ecg|>" + "".join([f"<|ecg_pos_{i}|>" for i in range(num_ecg_tokens)]) + "<|end_ecg|>\n"
                        "{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'assistant' %}"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                    "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
            )
            tokenizer.chat_template = custom_template

        tokenizer.add_special_tokens({"additional_special_tokens": ["<|start_ecg|>", "<|end_ecg|>"]})
        ecg_tokens = [f"<|ecg_pos_{i}|>" for i in range(num_ecg_tokens)]
        if ecg_tokens:
            existing_id = tokenizer.convert_tokens_to_ids(ecg_tokens[0])
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if existing_id is None or existing_id == -1 or (unk_id is not None and int(existing_id) == int(unk_id)):
                start_id = len(tokenizer)
                tokenizer.add_tokens(ecg_tokens, special_tokens=True)
                ecg_start_id = start_id
            else:
                ecg_start_id = int(existing_id)

    return tokenizer, ecg_start_id


def _build_prompt_tensors(
    question: str,
    tokenizer,
    config: Any,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Create prompt tensors (text-only) following training chat template."""
    
    # Check if this is a MedGemma model
    is_medgemma = "medgemma" in getattr(config, "huggingface_model_name", "").lower() or \
                  "medgemma" in getattr(config, "decoder_name", "").lower()
    
    if is_medgemma:
        # MedGemma-style prompt with <start_of_image> for ECG injection
        system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
        user_content = f"<start_of_image>\n\nQuestion: {question}\n\nRespond concisely with the key finding or answer."
        
        prompt_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        
        encoding = tokenizer(
            prompt_text,
            add_special_tokens=True,
            return_tensors="pt",
        )
    else:
        # Legacy LLaMA-style prompt
        system_message = "An electrocardiogram analysis and question answering tool"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        encoding = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
    
    prompt_ids = encoding.input_ids[0]
    prompt_mask = encoding.attention_mask[0]

    total_ctx = int(getattr(config, "max_token_length", 640))
    headroom = max(8, total_ctx - int(max_new_tokens) - 1)
    if prompt_ids.numel() > headroom:
        prompt_ids = prompt_ids[-headroom:]
        prompt_mask = prompt_mask[-headroom:]

    return prompt_ids, prompt_mask, prompt_text


def _instantiate_model(config: Any, tokenizer, ecg_token_start_id: int, yaml_config: Any = None) -> ECG_Tokenizer_Wrapper:
    """Recreate ECG_Tokenizer_Wrapper with config->model hyperparameters.
    
    Args:
        config: Config object from checkpoint
        tokenizer: HuggingFace tokenizer
        ecg_token_start_id: Start ID for ECG tokens (if any)
        yaml_config: Optional config.yaml dict from checkpoint folder for bridge configuration
    """
    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    
    # Get num_visual_tokens from various config keys (Q-Former uses num_query_tokens)
    num_visual_tokens = getattr(config, "bridge_num_visual_tokens", None)
    if num_visual_tokens is None:
        num_visual_tokens = getattr(config, "num_query_tokens", None)
    if num_visual_tokens is None:
        num_visual_tokens = getattr(config, "num_visual_tokens", None)
    
    num_quantizers = int(getattr(config, "num_quantizers", 8))
    
    # Get num_codebooks_kept from config (prefer yaml_config, fall back to checkpoint config)
    # This is critical for matching the bridge shape (e.g., 1CB vs 8CB models)
    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", None)
    if num_codebooks_kept is None:
        num_codebooks_kept = _cfg_get(config, "num_codebooks_kept", None)
    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))
    
    # Log the bridge configuration being used
    if num_codebooks_kept is not None:
        print(f"   Bridge config: num_codebooks_kept={num_codebooks_kept}, codebook_offset={codebook_offset}, num_quantizers={num_quantizers}")
    
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
        adapter_dropout=0.0,
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
        stage1_checkpoint_path=None,  # Don't reload stage1 during inference - weights come from finetuned checkpoint
        use_lora=bool(getattr(config, "use_lora", False)),
        lora_config=getattr(config, "lora_config", None),
        tokenizer=tokenizer,
        ecg_token_start_id=ecg_token_start_id,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
        enable_attention_visualization=bool(getattr(config, "enable_attention_visualization", False)),
        attention_log_frequency=int(getattr(config, "attention_log_frequency", 200)),
        num_codebooks_kept=num_codebooks_kept,
        codebook_offset=codebook_offset,
    )
    return model


def _locate_llm_host(model: ECG_Tokenizer_Wrapper):
    """
    Find the HF CausalLM inside the wrapper (the module that actually has .generate()).
    Training stored LoRA under 'decoder.llm_model.base_model...'; we must wrap that object.
    """
    host = getattr(model, "decoder", model)
    # Preferred nesting names
    for attr in ("llm_model", "language_model", "model", "transformer"):
        if hasattr(host, attr):
            cand = getattr(host, attr)
            if hasattr(cand, "generate"):
                return host, attr, cand
    # Fallback: if decoder itself can generate
    if hasattr(host, "generate"):
        return None, None, host
    raise RuntimeError("Could not locate HF CausalLM host for PEFT wrapping.")


def _split_state_dict(state_dict: dict):
    """Separate base model weights from LoRA adapter weights."""
    base_sd = {k: v for k, v in state_dict.items() if not _is_lora_key(k)}
    lora_sd = {k: v for k, v in state_dict.items() if _is_lora_key(k)}
    return base_sd, lora_sd


def _clean_lora_key(k: str) -> str:
    """
    Normalize LoRA keys so PEFT can map them:
    - drop any leading prefix up to and including 'base_model.' (keep base_model.* onward)
    - else, drop known wrapper prefixes (decoder., llm_model., model., language_model., transformer.)
    """
    if "base_model." in k:
        return k[k.index("base_model."):]
    for pref in ("decoder.", "llm_model.", "model.", "language_model.", "transformer."):
        if k.startswith(pref):
            k = k[len(pref):]
    return k


def _maybe_attach_or_merge_lora(model: ECG_Tokenizer_Wrapper, lora_sd: dict, config: Any) -> None:
    """Attach LoRA adapters to the *LLM host* and merge for inference."""
    if not lora_sd:
        return
    # Identify HF CausalLM host
    container, attr_name, llm_host = _locate_llm_host(model)
    try:
        from peft import PeftModel, get_peft_model, set_peft_model_state_dict, LoraConfig
    except Exception as e:
        raise RuntimeError("PEFT is required to load LoRA adapters (pip install peft).") from e

    if not getattr(config, "use_lora", False):
        setattr(config, "use_lora", True)

    # Convert lora_config dict to LoraConfig object if needed
    lora_config = getattr(config, "lora_config", None)
    if isinstance(lora_config, dict):
        # Remove custom params that aren't part of standard LoraConfig
        lora_config_filtered = {k: v for k, v in lora_config.items() if k not in ('top_k_layers', 'modules_to_save')}
        lora_config = LoraConfig(**lora_config_filtered)
    elif lora_config is None:
        # Create default LoRA config by inferring from state dict keys
        # Infer rank from first LoRA weight
        lora_r = 8  # default
        for k, v in lora_sd.items():
            if 'lora_A' in k and v.ndim == 2:
                lora_r = v.shape[0]
                break
        
        # Infer target modules from LoRA keys
        target_modules = set()
        for k in lora_sd.keys():
            if 'lora_A' in k or 'lora_B' in k:
                # Extract module name (e.g., "q_proj" from "...q_proj.lora_A.weight")
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p in ('lora_A', 'lora_B'):
                        if i > 0:
                            target_modules.add(parts[i-1])
                        break
        
        target_modules = list(target_modules) if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"[LoRA] Inferred config: r={lora_r}, target_modules={target_modules[:5]}...")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Wrap host with PEFT and load adapter weights (if not already wrapped)
    if isinstance(llm_host, PeftModel):
        print("[LoRA] Model already has PEFT adapters, loading state dict directly")
        peft_llm = llm_host
    else:
        peft_llm = get_peft_model(llm_host, lora_config)

    adapter_sd = {_clean_lora_key(k): v for k, v in lora_sd.items()}
    # Count for logging
    total_adapter_params = sum(int(v.numel()) for v in adapter_sd.values())
    set_peft_model_state_dict(peft_llm, adapter_sd, adapter_name="default")
    print(f"[LoRA] adapters applied: keys={len(adapter_sd)} params={total_adapter_params:,}")

    # Merge for plain inference
    try:
        merged = peft_llm.merge_and_unload()
        print("[LoRA] Successfully merged adapters into base model")
    except Exception as merge_exc:
        print(f"[LoRA] Adapters attached but not merged: {merge_exc}")
        merged = peft_llm  # best-effort

    # Put merged model back into the wrapper at the right spot
    if container is not None and attr_name is not None:
        setattr(container, attr_name, merged)
    else:
        model.decoder = merged


def _extract_answer(decoded: str, question: str) -> str:
    """Remove the chat prompt from decoded text and sanitize whitespace."""
    cleaned = LLMFinetuningRunner._sanitize_chat_text(decoded)
    if not cleaned:
        return ""
    lower_q = question.strip().lower()
    lower_c = cleaned.lower()
    idx = lower_c.find(lower_q)
    if idx != -1:
        cleaned = cleaned[idx + len(lower_q):].lstrip(" :\n")
    if cleaned.strip() == "":
        cleaned = re.sub(r"\s+", " ", decoded).strip()
    return cleaned


def generate_answer(checkpoint: str, waveform_path: str, question: str, device_str: str | None = None) -> str:
    """Core CLI entrypoint."""
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

    # Load config.yaml from checkpoint folder for bridge configuration
    checkpoint_dir = os.path.dirname(checkpoint)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = None
    if os.path.exists(config_yaml_path):
        print(f"Loading config.yaml from {config_yaml_path}...")
        yaml_config = load_yaml(config_yaml_path)

    # Reconstruct LoRA config when the checkpoint stores scalar LoRA fields (lora_r/lora_alpha/...)
    # instead of a full lora_config dict.
    if bool(getattr(config, "use_lora", False)) and getattr(config, "lora_config", None) is None:
        try:
            setattr(config, "lora_config", {
                "r": int(getattr(config, "lora_r", 16)),
                "lora_alpha": int(getattr(config, "lora_alpha", getattr(config, "lora_r", 16))),
                "lora_dropout": float(getattr(config, "lora_dropout", 0.0)),
                "target_modules": list(getattr(config, "lora_target_modules", None) or []),
                "bias": str(getattr(config, "lora_bias", "none")),
            })
        except Exception:
            pass

    tokenizer, ecg_token_start_id = _prepare_tokenizer(config)
    model = _instantiate_model(config, tokenizer, ecg_token_start_id, yaml_config)
    state_dict = checkpoint_data["model_state_dict"]
    # Load the full checkpoint with the wrapper's LoRA-aware loader to preserve correct scaling (alpha/r).
    model._load_state_dict(state_dict, strict=False)
    try:
        if bool(getattr(config, "use_lora", False)):
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(target_device)

    generation_kwargs = dict(getattr(config, "default_generation_kwargs", {}) or {})
    generation_kwargs.setdefault("max_new_tokens", 96)  # Match training config
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    generation_kwargs.setdefault("no_repeat_ngram_size", 5)  # Prevent repetition
    generation_kwargs.setdefault("repetition_penalty", 1.1)
    
    # MedGemma should stop at <end_of_turn> (106) or <eos> (1)
    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
        eos_ids.append(end_of_turn_id)
    generation_kwargs.setdefault("eos_token_id", eos_ids)

    prompt_ids, prompt_mask, prompt_text = _build_prompt_tensors(
        question,
        tokenizer,
        config,
        int(generation_kwargs["max_new_tokens"]),
    )
    # Enforce prompt_len + max_new <= total ctx
    total_ctx = int(getattr(config, "max_token_length", 640))
    prompt_len = int(prompt_ids.numel())
    if prompt_len + int(generation_kwargs["max_new_tokens"]) > total_ctx:
        generation_kwargs["max_new_tokens"] = max(1, total_ctx - prompt_len)
        print(f"[CTX] trimmed max_new_tokens to {generation_kwargs['max_new_tokens']} (prompt={prompt_len}, ctx={total_ctx})")

    ecg_tensor = _load_waveform(
        waveform_path,
        target_length=int(getattr(config, "ecg_waveform_length", 2500)),
        num_leads=int(getattr(config, "ecg_num_leads", 12)),
    ).to(target_device)

    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor,
            prompt_input_ids=prompt_ids.unsqueeze(0).to(target_device),
            prompt_attention_mask=prompt_mask.unsqueeze(0).to(target_device),
            max_token_length=int(getattr(config, "max_token_length", 640)),
            **generation_kwargs,
        )

    generated_ids = generated_ids[0].to("cpu")
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = _extract_answer(decoded, question)
    if not answer:
        print("Warning: sanitized answer was empty; returning raw decode.")
        answer = decoded.strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="Generate ECG answer from finetuned checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument("--waveform", required=True, help="Path to ECG .npy file.")
    parser.add_argument("--question", required=True, help="Question to pose to the model.")
    parser.add_argument("--device", default=None, help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'.")
    args = parser.parse_args()

    answer = generate_answer(args.checkpoint, args.waveform, args.question, args.device)
    print("\n=== Generation Result ===")
    print(answer)


if __name__ == "__main__":
    main()
