#!/usr/bin/env python3
"""
Train a DPO policy on ECG QA pairs (chosen/rejected) using a frozen reference model.

Input JSONL rows (from rank_dpo_generations.py):
{
  "waveform_path": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "weight": 0.4
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper  # noqa: E402
from utils.enums import DecoderMode  # noqa: E402
from utils.files_handler import load_yaml  # noqa: E402


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
    num_quantizers = int(getattr(config, "num_quantizers", 8))

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
        model.set_lora_inference_mode(False)

    model.to(device)
    return model, tokenizer, config


def set_trainable(model: torch.nn.Module, mode: str, regex: Optional[str] = None) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    if regex:
        import re as _re
        pattern = _re.compile(regex)
        for name, param in model.named_parameters():
            if pattern.search(name):
                param.requires_grad = True
    elif mode == "all":
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif mode == "lora":
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    elif mode == "bridge":
        for name, param in model.named_parameters():
            if "bridge" in name or "qformer" in name:
                param.requires_grad = True
    elif mode == "projection":
        for name, param in model.named_parameters():
            if "projection" in name or "proj" in name:
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")

    trainable = sum(p.requires_grad for p in model.parameters())
    if trainable == 0:
        raise ValueError("No trainable parameters selected.")


def load_ecg_waveform(waveform_path: str, target_length: int, num_leads: int) -> np.ndarray:
    waveform = np.load(waveform_path)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(-1)
    if waveform.shape[-1] == num_leads:
        pass
    elif waveform.shape[0] == num_leads:
        waveform = waveform.T
    current_length = waveform.shape[0]
    if current_length >= target_length:
        start = (current_length - target_length) // 2
        waveform = waveform[start:start + target_length, :]
    else:
        pad_before = (target_length - current_length) // 2
        pad_after = target_length - current_length - pad_before
        waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")
    if waveform.shape[1] != num_leads:
        raise ValueError(f"Unexpected lead count: {waveform.shape}")
    return waveform.astype(np.float32)


@dataclass
class TokenizedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor


def build_tokens(
    prompt_text: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    num_ecg_tokens: int,
    ecg_token_start_id: Optional[int],
    prefix_tuning: bool,
    medgemma_prompt_style: bool,
) -> TokenizedSample:
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    if not prompt_text:
        prompt_text = "Analyze this ECG and list the clinical findings."
    if medgemma_prompt_style:
        user_content = (
            "<start_of_image>\n\n"
            f"Question: {prompt_text}\n\n"
            "Respond concisely with the key finding or answer."
        )
        prompt_template_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        full_template_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{answer_text}<end_of_turn>"
        )
    else:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support chat template.")
        messages_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"<image_1> {prompt_text}".strip()},
        ]
        messages_full = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"<image_1> {prompt_text}".strip()},
            {"role": "assistant", "content": answer_text},
        ]
        prompt_template_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_template_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

    prompt_encoding = tokenizer.encode_plus(prompt_template_text, add_special_tokens=True, return_tensors=None)
    full_encoding = tokenizer.encode_plus(full_template_text, add_special_tokens=True, return_tensors=None)
    prompt_ids = prompt_encoding.input_ids
    full_ids = full_encoding.input_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    if num_ecg_tokens > 0:
        if prefix_tuning:
            ecg_prefix = torch.full((num_ecg_tokens,), fill_value=pad_id, dtype=torch.long)
        else:
            start_id = int(ecg_token_start_id) if ecg_token_start_id is not None else 0
            ecg_prefix = torch.arange(start_id, start_id + num_ecg_tokens, dtype=torch.long)
    else:
        ecg_prefix = torch.zeros(0, dtype=torch.long)
    prefix_len = int(ecg_prefix.numel())

    max_text_len = max(0, max_length - prefix_len)
    full_ids_trunc = full_ids[:max_text_len]

    text_ids = torch.tensor(full_ids_trunc, dtype=torch.long)
    if medgemma_prompt_style and prefix_len > 0:
        insert_after_image = False
        image_pos = None
        try:
            image_token_id = tokenizer.convert_tokens_to_ids("<start_of_image>")
            if isinstance(image_token_id, (list, tuple)):
                image_token_id = image_token_id[0] if image_token_id else None
        except Exception:
            image_token_id = None
        if image_token_id is not None:
            try:
                image_pos = text_ids.tolist().index(int(image_token_id))
                insert_after_image = True
            except ValueError:
                insert_after_image = False
        if insert_after_image and image_pos is not None:
            input_ids = torch.cat([text_ids[:image_pos + 1], ecg_prefix, text_ids[image_pos + 1:]], dim=0)
        else:
            input_ids = torch.cat([ecg_prefix, text_ids], dim=0)
    else:
        input_ids = torch.cat([ecg_prefix, text_ids], dim=0)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    if input_ids.numel() < max_length:
        pad_len = max_length - input_ids.numel()
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

    prompt_input_ids = torch.tensor(prompt_ids[:max_length], dtype=torch.long)
    prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
    if prompt_input_ids.numel() < max_length:
        pad_len = max_length - prompt_input_ids.numel()
        prompt_input_ids = torch.nn.functional.pad(prompt_input_ids, (0, pad_len), value=pad_id)
        prompt_attention_mask = torch.nn.functional.pad(prompt_attention_mask, (0, pad_len), value=0)

    labels = input_ids.clone()
    full_prompt_len = min(max_length, prefix_len + len(prompt_ids))
    labels[:full_prompt_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)

    return TokenizedSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )


class DPOPairDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        config,
        max_length: int,
        waveform_key: str = "waveform_path",
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        weight_key: str = "weight",
    ) -> None:
        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.waveform_key = waveform_key
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.weight_key = weight_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        waveform_path = row[self.waveform_key]
        prompt = str(row[self.prompt_key])
        chosen = str(row[self.chosen_key])
        rejected = str(row[self.rejected_key])
        weight = float(row.get(self.weight_key, 1.0))

        waveform = load_ecg_waveform(
            waveform_path=waveform_path,
            target_length=int(getattr(self.config, "ecg_waveform_length", 2500)),
            num_leads=int(getattr(self.config, "ecg_num_leads", 12)),
        )
        signal = torch.from_numpy(waveform).T  # [12, length]

        num_ecg_tokens = int(getattr(self.config, "num_ecg_tokens", 0))
        ecg_token_start_id = getattr(self.config, "ecg_token_start_id", None)
        prefix_tuning = bool(getattr(self.config, "prefix_tuning", False))
        medgemma_prompt_style = bool(getattr(self.config, "medgemma_prompt_style", False))

        chosen_tokens = build_tokens(
            prompt_text=prompt,
            answer_text=chosen,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=ecg_token_start_id,
            prefix_tuning=prefix_tuning,
            medgemma_prompt_style=medgemma_prompt_style,
        )
        rejected_tokens = build_tokens(
            prompt_text=prompt,
            answer_text=rejected,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=ecg_token_start_id,
            prefix_tuning=prefix_tuning,
            medgemma_prompt_style=medgemma_prompt_style,
        )

        return {
            "signal": signal,
            "chosen_input_ids": chosen_tokens.input_ids,
            "chosen_attention_mask": chosen_tokens.attention_mask,
            "chosen_labels": chosen_tokens.labels,
            "rejected_input_ids": rejected_tokens.input_ids,
            "rejected_attention_mask": rejected_tokens.attention_mask,
            "rejected_labels": rejected_tokens.labels,
            "prompt_input_ids": chosen_tokens.prompt_input_ids,
            "prompt_attention_mask": chosen_tokens.prompt_attention_mask,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


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
        _, _, _, labels_out = merged
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
    model: ECG_Tokenizer_Wrapper,
    ecg_signal: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    prompt_input_ids: Optional[torch.Tensor],
    prompt_attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ecg_signal = ecg_signal.to(dtype=torch.float32)
    features = model.encoder(ecg_signal)
    quantized, indices, _ = model.quantizer(features)
    quantized_codes = model._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)

    if not hasattr(model.decoder, "_compute_ecg_embeddings"):
        raise ValueError("Decoder does not support ECG embeddings for DPO training.")

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


def _sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    shift_labels = shift_labels.clamp_min(0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO policy for ECG QA.")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path.")
    parser.add_argument("--pairs", required=True, help="DPO pairs JSONL path.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_weight", type=float, default=0.0,
                        help="Weight for SFT anchor loss on chosen responses (0.0-1.0). Higher values preserve generation quality.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--trainable", choices=["lora", "bridge", "projection", "all"], default="lora")
    parser.add_argument("--trainable_regex", default=None)
    parser.add_argument("--waveform_key", default="waveform_path")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--chosen_key", default="chosen")
    parser.add_argument("--rejected_key", default="rejected")
    parser.add_argument("--weight_key", default="weight")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    policy, tokenizer, config = load_model(args.checkpoint, device)
    ref_model, _, _ = load_model(args.checkpoint, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    set_trainable(policy, args.trainable, regex=args.trainable_regex)
    policy.train()

    max_length = args.max_length or int(getattr(config, "max_token_length", 640))

    dataset = DPOPairDataset(
        path=args.pairs,
        tokenizer=tokenizer,
        config=config,
        max_length=max_length,
        waveform_key=args.waveform_key,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        weight_key=args.weight_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
    )

    use_autocast = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_dpo_loss = 0.0
        running_sft_loss = 0.0
        for step, batch in enumerate(loader):
            signal = batch["signal"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            weight = batch["weight"].to(device)

            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits_c, labels_c = _forward_logits_and_labels(
                    policy,
                    signal,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                    prompt_input_ids,
                    prompt_attention_mask,
                )
                logits_r, labels_r = _forward_logits_and_labels(
                    policy,
                    signal,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                    prompt_input_ids,
                    prompt_attention_mask,
                )
                logp_c = _sequence_logp(logits_c, labels_c)
                logp_r = _sequence_logp(logits_r, labels_r)

                with torch.no_grad():
                    logits_c_ref, labels_c_ref = _forward_logits_and_labels(
                        ref_model,
                        signal,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logits_r_ref, labels_r_ref = _forward_logits_and_labels(
                        ref_model,
                        signal,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logp_c_ref = _sequence_logp(logits_c_ref, labels_c_ref)
                    logp_r_ref = _sequence_logp(logits_r_ref, labels_r_ref)

                pi_logratio = logp_c - logp_r
                ref_logratio = logp_c_ref - logp_r_ref
                logits_diff = args.beta * (pi_logratio - ref_logratio)
                dpo_loss = -torch.nn.functional.logsigmoid(logits_diff)
                dpo_loss = (dpo_loss * weight).mean()

                # SFT anchor loss: NLL on chosen responses
                if args.sft_weight > 0:
                    # Compute per-token NLL from log-probabilities
                    num_tokens_c = (labels_c != -100).sum(dim=-1).clamp_min(1).float()
                    sft_loss = -logp_c / num_tokens_c  # Mean NLL per sample
                    sft_loss = (sft_loss * weight).mean()
                    loss = (1.0 - args.sft_weight) * dpo_loss + args.sft_weight * sft_loss
                else:
                    sft_loss = torch.tensor(0.0, device=device)
                    loss = dpo_loss

            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().cpu())
            running_dpo_loss += float(dpo_loss.detach().cpu())
            running_sft_loss += float(sft_loss.detach().cpu())

            # Log progress every 50 steps
            if (step + 1) % 50 == 0:
                avg_loss_so_far = running_loss / (step + 1)
                avg_dpo = running_dpo_loss / (step + 1)
                avg_sft = running_sft_loss / (step + 1)
                print(f"[step {step+1}/{len(loader)}] loss={avg_loss_so_far:.4f} dpo={avg_dpo:.4f} sft={avg_sft:.4f}")

            if global_step > 0 and global_step % args.save_steps == 0:
                out_path = os.path.join(args.output_dir, f"dpo_step_{global_step}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": running_loss / max(1, step + 1),
                        "config": config,
                        "use_lora": bool(getattr(config, "use_lora", False)),
                    },
                    out_path,
                )
                print(f"[checkpoint] Saved {out_path}")

        avg_loss = running_loss / max(1, len(loader))
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}")
        out_path = os.path.join(args.output_dir, f"dpo_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
                "use_lora": bool(getattr(config, "use_lora", False)),
            },
            out_path,
        )


if __name__ == "__main__":
    main()
