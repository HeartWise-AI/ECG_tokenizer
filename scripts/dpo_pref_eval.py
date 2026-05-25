#!/usr/bin/env python3
"""
Evaluate preference accuracy on DPO pairs for a checkpoint.

Usage:
  python scripts/dpo_pref_eval.py \
    --base_checkpoint checkpoints/BEST_LLM/.../best_model.pt \
    --pairs output/dpo_inference/dpo_5gen_dpo_pairs.jsonl \
    --dpo_checkpoint checkpoints/ECG_Tokenizer_DPO_Finetuning/.../dpo_epoch_1.pt \
    --max_samples 512
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data.dpo_pair_dataset import DPOPairDataset, dpo_collate_fn
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def _sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    shift_labels = shift_labels.clamp_min(0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1)


def _expand_labels_with_ecg(
    decoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    ecg_embeddings: torch.Tensor,
) -> torch.Tensor:
    merged = decoder._inject_ecg_after_image_token(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ecg_embeddings=ecg_embeddings,
        embed_layer=decoder.llm_model.get_input_embeddings(),
    )
    if merged is not None:
        labels_out = merged[3]
        if labels_out is None:
            raise ValueError("Failed to expand labels with ECG injection.")
        return labels_out

    prefix_len = int(ecg_embeddings.size(1))
    if prefix_len <= 0:
        return labels
    ignore_pad = torch.full(
        (labels.size(0), prefix_len),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([ignore_pad, labels], dim=1)


def _forward_logits_and_labels(
    model,
    ecg_signal: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    prompt_input_ids: Optional[torch.Tensor],
    prompt_attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ecg_signal = ecg_signal.to(dtype=torch.float32)
    features = model.encoder(ecg_signal)
    quantized, indices, _ = model.quantizer(features)
    quantized_codes = model._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)

    if not hasattr(model.decoder, "_compute_ecg_embeddings"):
        raise ValueError("Decoder does not support ECG embeddings for DPO evaluation.")

    ecg_embeddings, _ = model.decoder._compute_ecg_embeddings(
        quantized,
        quantized_codes,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )

    outputs = model.decoder(
        quantized_features=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        quantized_codes=None,
        ecg_embeddings=ecg_embeddings,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )
    logits = outputs["logits"]
    labels_expanded = _expand_labels_with_ecg(
        decoder=model.decoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ecg_embeddings=ecg_embeddings,
    )
    return logits, labels_expanded


def load_model(base_checkpoint: str, device: torch.device, dpo_checkpoint: Optional[str] = None):
    checkpoint_data = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

    checkpoint_dir = os.path.dirname(base_checkpoint)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = load_yaml(config_yaml_path) if os.path.exists(config_yaml_path) else None

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

    model._load_state_dict(checkpoint_state_dict, strict=False)

    if dpo_checkpoint:
        dpo_data = torch.load(dpo_checkpoint, map_location="cpu", weights_only=False)
        dpo_state = dpo_data.get("model_state_dict", dpo_data)
        model._load_state_dict(dpo_state, strict=False)

    try:
        if bool(getattr(config, "use_lora", False)):
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(device)
    return model, tokenizer, config


def eval_preference_accuracy(
    model,
    tokenizer,
    config,
    pairs_path: str,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
):
    max_length = int(getattr(config, "max_token_length", 640))
    dataset = DPOPairDataset(
        path=pairs_path,
        tokenizer=tokenizer,
        config=config,
        max_length=max_length,
        waveform_key=getattr(config, "waveform_key", "waveform_path"),
        prompt_key=getattr(config, "prompt_key", "prompt"),
        chosen_key=getattr(config, "chosen_key", "chosen"),
        rejected_key=getattr(config, "rejected_key", "rejected"),
        weight_key=getattr(config, "weight_key", "weight"),
    )
    if max_samples and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(int(max_samples))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dpo_collate_fn,
    )

    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_autocast else torch.float32

    correct = 0.0
    total = 0.0
    margin_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
            signal = batch["signal"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)

            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits_c, labels_c = _forward_logits_and_labels(
                    model,
                    signal,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                    prompt_input_ids,
                    prompt_attention_mask,
                )
                logits_r, labels_r = _forward_logits_and_labels(
                    model,
                    signal,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                    prompt_input_ids,
                    prompt_attention_mask,
                )
                logp_c = _sequence_logp(logits_c, labels_c)
                logp_r = _sequence_logp(logits_r, labels_r)

            diff = logp_c - logp_r
            correct += float((diff > 0).float().sum().item())
            margin_sum += float(diff.sum().item())
            total += float(diff.numel())

    pref_accuracy = correct / total if total > 0 else 0.0
    pref_margin = margin_sum / total if total > 0 else 0.0
    return {
        "pref_accuracy": pref_accuracy,
        "pref_margin": pref_margin,
        "pairs": int(total),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO preference accuracy on a pairs JSONL.")
    parser.add_argument("--base_checkpoint", required=True, help="Base (pre-DPO) checkpoint path.")
    parser.add_argument("--pairs", required=True, help="DPO pairs JSONL path.")
    parser.add_argument("--dpo_checkpoint", default=None, help="Optional DPO checkpoint to apply.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(args.base_checkpoint, device, args.dpo_checkpoint)

    metrics = eval_preference_accuracy(
        model=model,
        tokenizer=tokenizer,
        config=config,
        pairs_path=args.pairs,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    metrics_out = {
        "base_checkpoint": args.base_checkpoint,
        "dpo_checkpoint": args.dpo_checkpoint,
        **metrics,
    }
    print(json.dumps(metrics_out, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=metrics_out,
        )
        wandb.log({
            "eval/pref_accuracy": metrics["pref_accuracy"],
            "eval/pref_margin": metrics["pref_margin"],
            "eval/pairs": metrics["pairs"],
        })
        run.finish()


if __name__ == "__main__":
    main()
