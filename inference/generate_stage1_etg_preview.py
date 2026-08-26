#!/usr/bin/env python3
"""
Quick ETG preview for Stage-1 checkpoints.

Loads the Stage-1 bridge checkpoint and the pretrained ECG tokenizer
(encoder + quantizer), takes a few ECGs from the validation split,
runs ETG (teacher-forcing) on the reference report text, and prints
the reference vs. generated text pairs.

Usage:
  python scripts/generate_stage1_etg_preview.py \
      --ckpt checkpoints/ECG_Text_Stage1/ecg_text_stage1/…/checkpoints/stage1_last_epoch_006.pt \
      --config config/ecg_text_stage1/base_config.yaml \
      --n 5
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from utils.files_handler import load_yaml
import torch.serialization as serialization
from utils.config.tokenizer_config import ECGTokenizerTrainingConfig
from data.ecg_text_stage1_dataset import ECGTextStage1Dataset, stage1_collate
from utils.registry import ModelRegistry
from models.ecg_tokenizer_wrapper import Conv_Encoder, Residual_Conv_Encoder, ECG_Tokenizer_Quantizer
from models.bridge.bridge import ECGQFormerBridgeStage1


def _prepend_dec_token(input_ids: torch.Tensor, attention_mask: torch.Tensor, dec_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepend [DEC] token to input ids and adjust attention mask.

    Keeps total sequence length <= max_len by trimming the tail.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D")
    bsz, L = input_ids.shape
    device = input_ids.device
    pad_id = None if attention_mask is None else None
    # Build a [DEC] column
    dec_col = torch.full((bsz, 1), int(dec_id), dtype=input_ids.dtype, device=device)
    new_ids = torch.cat([dec_col, input_ids], dim=1)
    if new_ids.size(1) > max_len:
        new_ids = new_ids[:, :max_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, : (max_len - 1)]
    if attention_mask is None:
        attn = torch.ones_like(new_ids, dtype=torch.long, device=device)
    else:
        attn = torch.cat([torch.ones((bsz, 1), dtype=attention_mask.dtype, device=device), attention_mask], dim=1)
        if attn.size(1) < new_ids.size(1):
            pad = new_ids.size(1) - attn.size(1)
            attn = F.pad(attn, (0, pad), value=0)
        elif attn.size(1) > new_ids.size(1):
            attn = attn[:, : new_ids.size(1)]
    return new_ids.contiguous(), attn.contiguous()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Stage-1 bridge checkpoint .pt")
    parser.add_argument("--config", default="config/ecg_text_stage1/base_config.yaml", help="Config yaml path")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to preview")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # Load pretrained ECG tokenizer parts (encoder + quantizer)
    tok_ckpt = cfg.get("pretrained_encoder_checkpoint")
    if not tok_ckpt or not os.path.exists(tok_ckpt):
        raise FileNotFoundError(f"Pretrained tokenizer checkpoint not found: {tok_ckpt}")
    # Allowlist tokenizer training config used in the checkpoint metadata
    serialization.add_safe_globals([ECGTokenizerTrainingConfig])
    tok_blob = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    tok_sd = tok_blob.get("model_state_dict") or tok_blob.get("state_dict") or tok_blob
    ck_cfg = tok_blob.get("config", {})
    num_quantizers = int(cfg.get("num_quantizers") or ck_cfg.get("num_quantizers") or 8)
    codebook_size = int(cfg.get("codebook_size") or ck_cfg.get("codebook_size") or 512)

    # Instantiate modules
    # Resolve encoder/quantizer classes from checkpoint metadata when available
    def _cfg_get(container, key, fallback=None):
        if container is None:
            return fallback
        if isinstance(container, dict) and key in container:
            return container[key]
        if hasattr(container, key):
            val = getattr(container, key)
            return val if val is not None else fallback
        return fallback
    encoder_name = _cfg_get(ck_cfg, "encoder_name", _cfg_get(cfg, "encoder_name", "RESIDUAL_CONV_ENCODER"))
    quantizer_name = _cfg_get(ck_cfg, "quantizer_name", _cfg_get(cfg, "quantizer_name", "ECG_Tokenizer_Quantizer"))
    enc_cls = ModelRegistry.get(encoder_name)
    q_cls = ModelRegistry.get(quantizer_name)
    encoder = enc_cls().to(device).eval()
    quantizer = q_cls(num_quantizers=num_quantizers, codebook_size=codebook_size).to(device).eval()
    # Load weights
    enc_sd = {k[len("encoder.") :]: v for k, v in tok_sd.items() if k.startswith("encoder.")}
    q_sd = {k[len("quantizer.") :]: v for k, v in tok_sd.items() if k.startswith("quantizer.")}
    missing_enc = encoder.load_state_dict(enc_sd, strict=False)
    missing_q = quantizer.load_state_dict(q_sd, strict=False)
    if missing_enc.missing_keys or missing_enc.unexpected_keys:
        raise RuntimeError(f"Encoder keys mismatch: {missing_enc}")
    if missing_q.missing_keys or missing_q.unexpected_keys:
        raise RuntimeError(f"Quantizer keys mismatch: {missing_q}")

    # Tiny tokenizer just for IDs <-> text
    model_name = cfg.get("text_encoder_model_name", "google/medgemma-4b-it")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=int(cfg.get("max_text_length", 128)),
        padding_side="right",
        truncation_side="right",
        trust_remote_code=("gemma" in model_name.lower()),
    )
    # Guarantee we have PAD and add [DEC]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if "[DEC]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[DEC]"]})
    dec_id = int(tokenizer.convert_tokens_to_ids("[DEC]"))

    # Load Stage-1 checkpoint metadata for exact bridge hyperparameters
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Bridge (Stage-1) instantiation
    hidden = int(cfg.get("bridge_hidden_size", 768))
    max_seq = int(cfg.get("bridge_max_seq_len", 128))
    ckpt_cfg = ckpt.get("config")
    bridge = ECGQFormerBridgeStage1(
        vocab_size=int(codebook_size),
        num_codebooks=int(_cfg_get(ckpt_cfg, "num_codebooks_kept", _cfg_get(ckpt_cfg, "num_quantizers", num_quantizers))),
        d_mid=int(_cfg_get(ckpt_cfg, "bridge_hidden_size", hidden)),
        d_llm=int(_cfg_get(ckpt_cfg, "bridge_hidden_size", hidden)),
        d_txt=int(_cfg_get(ckpt_cfg, "bridge_hidden_size", hidden)),
        num_steps=int(_cfg_get(ckpt_cfg, "bridge_max_seq_len", max_seq)),
        num_query_tokens=int(_cfg_get(ckpt_cfg, "num_query_tokens", _cfg_get(cfg, "num_query_tokens", 32))),
        num_layers=int(_cfg_get(ckpt_cfg, "bridge_num_layers", _cfg_get(cfg, "bridge_num_layers", 6))),
        num_heads=int(_cfg_get(ckpt_cfg, "bridge_num_heads", _cfg_get(cfg, "bridge_num_heads", 12))),
        dropout=float(_cfg_get(ckpt_cfg, "bridge_dropout", _cfg_get(cfg, "bridge_dropout", 0.1))),
        num_special_tokens=int(_cfg_get(ckpt_cfg, "bridge_num_special_tokens", _cfg_get(cfg, "bridge_num_special_tokens", 4))),
        bias_last_codebook=float(_cfg_get(ckpt_cfg, "bridge_bias_last_codebook", _cfg_get(cfg, "bridge_bias_last_codebook", 0.5))),
        codebook_dropout=float(_cfg_get(ckpt_cfg, "bridge_codebook_dropout", _cfg_get(cfg, "bridge_codebook_dropout", 0.0))),
        mix_strategy=str(_cfg_get(ckpt_cfg, "bridge_mix_strategy", _cfg_get(cfg, "bridge_mix_strategy", "softmax")) or "softmax"),
        token_axis=str(_cfg_get(ckpt_cfg, "bridge_token_axis", _cfg_get(cfg, "bridge_token_axis", "channel")) or "channel"),
        txt_vocab_size=len(tokenizer),
        txt_pad_id=int(tokenizer.pad_token_id),
        txt_cls_id=int(tokenizer.cls_token_id) if tokenizer.cls_token_id is not None else None,
        cross_every=int(_cfg_get(ckpt_cfg, "cross_every", _cfg_get(cfg, "cross_every", 2))),
    ).to(device).eval()
    if getattr(bridge, "token_axis", "channel") == "time":
        rvq = getattr(quantizer, "quantizer", quantizer)
        bridge.attach_quantizer(rvq)

    # Load Stage-1 checkpoint into bridge
    bridge_sd = ckpt.get("model_state_dict") or ckpt
    missing = bridge.load_state_dict(bridge_sd, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(f"Bridge keys mismatch: {missing}")

    # Prepare validation sample(s)
    val_ds = ECGTextStage1Dataset(
        mapping_csv=str(cfg.get("validation_mapping_csv") or cfg.get("mapping_csv")),
        text_bank_csv=str(cfg.get("validation_text_bank_csv") or cfg.get("text_bank_csv")),
        split=str(cfg.get("validation_mapping_split") or cfg.get("mapping_split") or "val"),
        expected_waveform_length=int(cfg.get("waveform_length", 2500)),
        num_leads=int(cfg.get("num_leads", 12)),
        normalize_waveforms=bool(cfg.get("normalize_waveforms", False)),
        lead_stats=cfg.get("lead_stats"),
        seed=cfg.get("seed", 42),
        max_positives_per_ecg=int(cfg.get("max_positives_per_ecg", 1)),
    )

    n = min(args.n, len(val_ds))
    print(f"Loaded dataset with {len(val_ds)} ECGs; previewing {n} sample(s)\n")

    shown = 0
    for i in range(len(val_ds)):
        if shown >= n:
            break
        batch = stage1_collate([val_ds[i]])
        signal = batch["signal"].to(device)
        report: List[str] = batch.get("report", [""])
        report_text = report[0] if report else ""
        if not report_text:
            # fallback to any positive text
            pos_texts = batch.get("positive_texts_lists", [[""]])[0]
            report_text = pos_texts[0] if pos_texts else ""
        if not report_text:
            continue

        # Encode -> quantize -> codes
        enc = encoder(signal)
        q_out = quantizer(enc, return_all_codes=False)
        if isinstance(q_out, tuple):
            _, indices, *_ = q_out
        else:
            raise RuntimeError("Unexpected quantizer output format")
        # Ensure indices [batch, seq, depth]
        if indices.dim() == 4:
            indices = indices[0]
        # Keep as-is (bridge can handle multi-codebook depth)
        codes = indices.to(device).long()

        # Tokenize report and prepend [DEC]
        tok = tokenizer(
            [report_text],
            padding=False,
            truncation=True,
            max_length=int(cfg.get("max_text_length", 128)),
            return_tensors="pt",
        )
        rep_ids = tok["input_ids"].to(device)
        rep_mask = tok["attention_mask"].to(device)
        rep_ids_etg, rep_mask_etg = _prepend_dec_token(rep_ids, rep_mask, dec_id, int(cfg.get("max_text_length", 128)))

        # Run ETG (teacher-forcing logits) and greedy decode for preview
        lm_logits, _ = bridge.forward_stage1(codes, rep_ids_etg, rep_mask_etg, mode="ETG")
        argmax_ids = lm_logits.argmax(dim=-1)
        # Drop the first [DEC] position, and any PADs from decoding
        mask = rep_mask_etg.bool()
        if mask.size(1) > 0:
            mask[:, 0] = False
        keep = mask[0]
        pred_ids = argmax_ids[0][keep].tolist()
        gen_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        if not gen_text:
            gen_text = "(empty)"

        print("ECG:", batch.get("ecg_id", [""])[0])
        print("Report:", report_text)
        print("Generated:", gen_text)
        print("-" * 80)
        shown += 1


if __name__ == "__main__":
    main()
