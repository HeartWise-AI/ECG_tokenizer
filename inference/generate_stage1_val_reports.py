#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.serialization as serialization
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from utils.files_handler import load_yaml
from utils.registry import ModelRegistry
from data.ecg_text_stage1_dataset import ECGTextStage1Dataset, stage1_collate
from models.bridge.bridge import ECGQFormerBridgeStage1
from utils.config.tokenizer_config import ECGTokenizerTrainingConfig


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage-1 validation reports (auto-regressive) and save to CSV")
    parser.add_argument("--ckpt", required=True, help="Path to Stage-1 bridge checkpoint .pt")
    parser.add_argument("--config", default="config/ecg_text_stage1/base_config.yaml", help="Config yaml path")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults next to checkpoint)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for generation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--max_new", type=int, default=80, help="Maximum new tokens to generate (after [DEC])")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="Autocast precision for GPU compute")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
    device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Load base config
    cfg = load_yaml(args.config)

    # Load pretrained tokenizer encoder+quantizer checkpoint (allowlist the config class)
    serialization.add_safe_globals([ECGTokenizerTrainingConfig])
    tok_ckpt = cfg.get("pretrained_encoder_checkpoint")
    if not tok_ckpt or not os.path.exists(tok_ckpt):
        raise FileNotFoundError(f"Pretrained tokenizer checkpoint not found: {tok_ckpt}")
    tok_blob = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    tok_sd = tok_blob.get("model_state_dict") or tok_blob.get("state_dict") or tok_blob
    tok_cfg = tok_blob.get("config", {})

    # Resolve encoder/quantizer classes and sizes
    encoder_name = _cfg_get(tok_cfg, "encoder_name", _cfg_get(cfg, "encoder_name", "RESIDUAL_CONV_ENCODER"))
    quantizer_name = _cfg_get(tok_cfg, "quantizer_name", _cfg_get(cfg, "quantizer_name", "ECG_Tokenizer_Quantizer"))
    num_quantizers = int(_cfg_get(tok_cfg, "num_quantizers", _cfg_get(cfg, "num_quantizers", 8)))
    codebook_size = int(_cfg_get(tok_cfg, "codebook_size", _cfg_get(cfg, "codebook_size", 512)))

    enc_cls = ModelRegistry.get(encoder_name)
    q_cls = ModelRegistry.get(quantizer_name)
    encoder = enc_cls().to(device).eval()
    quantizer = q_cls(num_quantizers=num_quantizers, codebook_size=codebook_size).to(device).eval()

    enc_sd = {k[len("encoder.") :]: v for k, v in tok_sd.items() if k.startswith("encoder.")}
    q_sd = {k[len("quantizer.") :]: v for k, v in tok_sd.items() if k.startswith("quantizer.")}
    missing_enc = encoder.load_state_dict(enc_sd, strict=False)
    missing_q = quantizer.load_state_dict(q_sd, strict=False)
    if missing_enc.missing_keys or missing_enc.unexpected_keys:
        raise RuntimeError(f"Encoder keys mismatch: {missing_enc}")
    if missing_q.missing_keys or missing_q.unexpected_keys:
        raise RuntimeError(f"Quantizer keys mismatch: {missing_q}")

    # Tokenizer for decoding
    model_name = cfg.get("text_encoder_model_name", "google/medgemma-4b-it")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=int(cfg.get("max_text_length", 128)),
        padding_side="right",
        truncation_side="right",
        trust_remote_code=("gemma" in model_name.lower()),
    )
    # Ensure pad + [DEC]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if "[DEC]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[DEC]"]})
    dec_id = int(tokenizer.convert_tokens_to_ids("[DEC]"))
    eos_id = tokenizer.eos_token_id

    # Load Stage-1 bridge checkpoint and reconstruct exact architecture from its config
    ckpt = torch.load(args.ckpt, map_location="cpu")
    bridge_sd = ckpt.get("model_state_dict") or ckpt
    ckpt_cfg = ckpt.get("config")
    hidden = int(_cfg_get(ckpt_cfg, "bridge_hidden_size", _cfg_get(cfg, "bridge_hidden_size", 768)))
    max_steps = int(_cfg_get(ckpt_cfg, "bridge_max_seq_len", _cfg_get(cfg, "bridge_max_seq_len", 128)))

    bridge = ECGQFormerBridgeStage1(
        vocab_size=int(codebook_size),
        num_codebooks=int(_cfg_get(ckpt_cfg, "num_codebooks_kept", _cfg_get(ckpt_cfg, "num_quantizers", num_quantizers))),
        d_mid=hidden,
        d_llm=hidden,
        d_txt=hidden,
        num_steps=max_steps,
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
    missing = bridge.load_state_dict(bridge_sd, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(f"Bridge checkpoint keys mismatch: {missing}")

    # Validation dataset/dataloader
    mapping_csv = str(cfg.get("validation_mapping_csv") or cfg.get("mapping_csv"))
    text_bank_csv = str(cfg.get("validation_text_bank_csv") or cfg.get("text_bank_csv"))
    split = str(cfg.get("validation_mapping_split") or cfg.get("mapping_split") or "val")
    ds = ECGTextStage1Dataset(
        mapping_csv=mapping_csv,
        text_bank_csv=text_bank_csv,
        split=split,
        expected_waveform_length=int(cfg.get("waveform_length", 2500)),
        num_leads=int(cfg.get("num_leads", 12)),
        normalize_waveforms=bool(cfg.get("normalize_waveforms", False)),
        lead_stats=cfg.get("lead_stats"),
        seed=cfg.get("seed", 42),
        max_positives_per_ecg=int(cfg.get("max_positives_per_ecg", 1)),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stage1_collate,
    )

    # Output path
    out_path = args.out
    if not out_path:
        base_dir = os.path.dirname(args.ckpt)
        base_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        out_path = os.path.join(base_dir, f"{base_name}_generated_val_reports.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Resume support: collect already-generated ecg_ids
    done_ids: set[str] = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # Expect header present; if not, treat as corrupted and rewrite
                for row in reader:
                    if len(row) >= 1:
                        done_ids.add(row[0])
        except Exception:
            done_ids.clear()

    # Write header if new file or empty
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ecg_id", "report_text", "generated_report"])  # minimal fields

    total = len(done_ids)

    # Autocast dtype
    amp_enabled = device.type == "cuda"
    if args.precision == "auto":
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    elif args.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    for batch in loader:
        signals: torch.Tensor = batch["signal"].to(device, non_blocking=True)
        ecg_ids: List[str] = batch.get("ecg_id", [""] * signals.size(0))  # type: ignore
        reports_raw = batch.get("report", [""] * signals.size(0))  # type: ignore
        reports: List[str] = [str(r) if r is not None else "" for r in reports_raw]

        # Skip ECGs already generated (resume)
        keep_idx = [i for i, eid in enumerate(ecg_ids) if eid not in done_ids]
        if not keep_idx:
            continue
        signals = signals[keep_idx]
        ecg_ids = [ecg_ids[i] for i in keep_idx]
        reports = [reports[i] for i in keep_idx]

        # Encode -> quantize -> indices (codes)
        # Forward under autocast to reduce memory
        if amp_enabled and amp_dtype is not None:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            enc = encoder(signals)
            q_out = quantizer(enc, return_all_codes=False)
        if isinstance(q_out, tuple):
            _, indices, *_ = q_out
        else:
            raise RuntimeError("Unexpected quantizer output format")
        if indices.dim() == 4:
            indices = indices[0]
        codes = indices.to(device).long()

        # Start sequences with [DEC, DEC] to satisfy ETG's minimum length requirement
        bsz = codes.size(0)
        cur_ids = torch.full((bsz, 2), int(dec_id), dtype=torch.long, device=device)
        cur_mask = torch.ones_like(cur_ids, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        max_text_len = int(cfg.get("max_text_length", 128))
        max_steps_eff = max(1, min(args.max_new, max_text_len - 1))

        for _ in range(max_steps_eff):
            with autocast_ctx:
                lm_logits, _ = bridge.forward_stage1(codes, cur_ids, cur_mask, mode="ETG")
                next_logits = lm_logits[:, -1, :]
                next_id = next_logits.argmax(dim=-1)

            # Apply EOS stop if configured
            if eos_id is not None:
                next_id = torch.where(finished, torch.tensor(eos_id, device=next_id.device, dtype=next_id.dtype), next_id)
            cur_ids = torch.cat([cur_ids, next_id.unsqueeze(1)], dim=1)
            extend_mask = torch.ones((bsz, 1), dtype=cur_mask.dtype, device=cur_mask.device)
            cur_mask = torch.cat([cur_mask, extend_mask], dim=1)

            if eos_id is not None:
                finished = finished | (next_id == eos_id)
                if bool(finished.all().item()):
                    break

        # Decode sequences (drop the two leading [DEC] tokens)
        dec_strs: List[str] = []
        ids_to_decode = cur_ids[:, 2:]
        # Optionally trim at EOS per row for nicer output
        if eos_id is not None:
            decoded_rows: List[str] = []
            for i in range(ids_to_decode.size(0)):
                row = ids_to_decode[i]
                # cut at first EOS
                eos_pos = (row == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = int(eos_pos[0].item())
                    row = row[:cut]
                decoded_rows.append(tokenizer.decode(row.tolist(), skip_special_tokens=True).strip())
        else:
            decoded_rows = tokenizer.batch_decode(ids_to_decode, skip_special_tokens=True)
            decoded_rows = [s.strip() for s in decoded_rows]

        # Write rows
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for eid, ref, gen in zip(ecg_ids, reports, decoded_rows):
                writer.writerow([eid, ref, gen])
        total += bsz
        done_ids.update(ecg_ids)
        if total % max(args.batch_size * 10, 200) == 0:
            print(f"Generated {total}/{len(ds)} rows -> {out_path}")

    print(f"Done. Wrote {len(ds)} rows to: {out_path}")


if __name__ == "__main__":
    main()
