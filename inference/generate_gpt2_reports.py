#!/usr/bin/env python3
"""
Run standalone GPT-2 ECG report generation on a validation parquet.

This is the GPT-2/report-only counterpart to inference/generate_all_qa_pairs.py:
it loads a finetuned ECG_Tokenizer_Wrapper checkpoint, reads ECG waveforms from
the validation parquet, generates one clinical report per ECG, and writes CSV
and JSON outputs.

Example:
    CUDA_VISIBLE_DEVICES=0 python inference/generate_gpt2_reports.py \
        --checkpoint /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_LLM_finetuning_gpt_large/l040bcr8_20260411-032143/checkpoint_epoch_1.pt \
        --output_dir /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_LLM_finetuning_gpt_large/l040bcr8_20260411-032143/gpt2_report_inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import package side effects so ModelRegistry contains the legacy GPT-2 decoder
# and GPT2_SequenceAdapter used by the l040bcr8 checkpoint.
import models  # noqa: F401,E402

from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper  # noqa: E402
from utils.enums import DecoderMode  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/"
    "ECG_tokenizer_LLM_finetuning_gpt_large/l040bcr8_20260411-032143/checkpoint_epoch_1.pt"
)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


def _coerce_config(config: Any) -> Any:
    if isinstance(config, SimpleNamespace):
        return config
    if isinstance(config, dict):
        return SimpleNamespace(**{k: _coerce_config(v) for k, v in config.items()})
    if isinstance(config, list):
        return [_coerce_config(v) for v in config]
    if isinstance(config, tuple):
        return tuple(_coerce_config(v) for v in config)
    return config


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    print(f"Loading checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_model(checkpoint_path: str, device: torch.device, tokenizer_path: str | None = None):
    checkpoint = _load_checkpoint(checkpoint_path)
    config = _coerce_config(checkpoint["config"])

    tokenizer_name = tokenizer_path or str(_cfg_get(config, "tokenizer_name", "gpt2-large"))
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = _cfg_get(config, "decoder_mode", "llm")
    if not isinstance(decoder_mode, DecoderMode):
        decoder_mode = DecoderMode(str(decoder_mode))

    adapter_name = str(
        _cfg_get(
            config,
            "adapter_name",
            _cfg_get(config, "bridge_name", "GPT2_SequenceAdapter"),
        )
    )

    model = ECG_Tokenizer_Wrapper(
        encoder_name=str(_cfg_get(config, "encoder_name", "Residual_Conv_Encoder")),
        quantizer_name=str(_cfg_get(config, "quantizer_name", "ECG_Tokenizer_Quantizer")),
        decoder_name=str(_cfg_get(config, "decoder_name", "GPT2_Decoder")),
        num_quantizers=int(_cfg_get(config, "num_quantizers", 8)),
        codebook_size=int(_cfg_get(config, "codebook_size", 512)),
        decoder_mode=decoder_mode,
        huggingface_model_name=str(_cfg_get(config, "huggingface_model_name", tokenizer_name)),
        llm_input_embedding_size=int(_cfg_get(config, "llm_input_embedding_size", 1280)),
        adapter_name=adapter_name,
        adapter_dropout=float(_cfg_get(config, "adapter_dropout", 0.2)),
    )

    model._load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    print(f"Loaded model on {device} with adapter={adapter_name}")
    return model, tokenizer, config


def load_ecg_waveform(path: str, target_length: int, expected_leads: int) -> np.ndarray:
    waveform = np.load(path)

    if waveform.ndim == 3:
        waveform = waveform.squeeze(-1)
    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got shape {waveform.shape}")

    if waveform.shape[1] == expected_leads:
        pass
    elif waveform.shape[0] == expected_leads:
        waveform = waveform.T
    else:
        raise ValueError(f"Expected {expected_leads} leads, got shape {waveform.shape}")

    current_length = waveform.shape[0]
    if current_length > target_length:
        step = current_length // target_length
        waveform = waveform[::step, :]

    if waveform.shape != (target_length, expected_leads):
        raise ValueError(f"Unexpected waveform shape after training-style preprocessing: {waveform.shape}")

    if np.isnan(waveform).any():
        raise ValueError("Waveform contains NaNs")

    return waveform.astype(np.float32, copy=False)


def make_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield start, df.iloc[start : start + batch_size]


def decode_reports(tokenizer, generated_ids: torch.Tensor) -> list[str]:
    reports = []
    for row in generated_ids.detach().cpu():
        text = tokenizer.decode(row, skip_special_tokens=True)
        reports.append(" ".join(text.split()).strip())
    return reports


def build_generation_kwargs(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.do_sample is not None:
        kwargs["do_sample"] = args.do_sample
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.num_beams is not None:
        kwargs["num_beams"] = args.num_beams
    return kwargs


def save_outputs(records: list[dict[str, Any]], output_dir: Path, output_prefix: str, is_final: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if is_final else "_checkpoint"
    csv_path = output_dir / f"{output_prefix}{suffix}.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"{'Final' if is_final else 'Checkpoint'} CSV saved: {csv_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPT-2 ECG reports for a validation parquet.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to finetuned GPT-2 checkpoint.")
    parser.add_argument(
        "--validation_parquet",
        default=None,
        help="Validation parquet. Defaults to validation_dataset_path in the checkpoint config.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <checkpoint_dir>/gpt2_report_inference.",
    )
    parser.add_argument("--output_prefix", default="gpt2_report_generations")
    parser.add_argument("--waveform_column", default=None, help="Defaults to config.signal_path_column or waveform_path_psa.")
    parser.add_argument("--report_column", default="report", help="Reference report column.")
    parser.add_argument(
        "--prompt_category",
        default=None,
        help="Optional filter for parquets with a prompt_category column, e.g. interpretation.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_token_length", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or a GPU index like 0.")
    parser.add_argument("--tokenizer_path", default=None, help="Optional local tokenizer/model path override.")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--do_sample", type=lambda v: str(v).lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    return parser.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "gpt2_report_inference"
    device = _resolve_device(args.device)

    model, tokenizer, config = load_model(str(checkpoint_path), device, tokenizer_path=args.tokenizer_path)

    validation_parquet = args.validation_parquet or str(_cfg_get(config, "validation_dataset_path"))
    if not validation_parquet:
        raise ValueError("No validation parquet provided and checkpoint config has no validation_dataset_path.")

    waveform_column = args.waveform_column or str(_cfg_get(config, "signal_path_column", "waveform_path_psa"))
    target_length = int(_cfg_get(config, "ecg_waveform_length", 2500))
    expected_leads = int(_cfg_get(config, "ecg_num_leads", 12))
    max_token_length = int(args.max_token_length or _cfg_get(config, "max_token_length", 256))

    print(f"Loading validation parquet: {validation_parquet}")
    df = pd.read_parquet(validation_parquet)
    if args.prompt_category and "prompt_category" in df.columns:
        before = len(df)
        df = df[df["prompt_category"].astype(str) == args.prompt_category].copy()
        print(f"Filtered prompt_category={args.prompt_category}: {len(df)}/{before} rows")
    elif args.prompt_category:
        print("prompt_category filter requested, but column is not present; leaving data unfiltered.")

    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
        print(f"Limited to {len(df)} rows")

    if waveform_column not in df.columns:
        raise KeyError(f"Waveform column '{waveform_column}' not found. Available columns: {list(df.columns)}")
    if args.report_column not in df.columns:
        raise KeyError(f"Report column '{args.report_column}' not found. Available columns: {list(df.columns)}")
    if "waveform_name" not in df.columns:
        df["waveform_name"] = df[waveform_column].astype(str).map(lambda p: os.path.basename(p))

    generation_kwargs = build_generation_kwargs(args, tokenizer)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"Generating reports for {len(df)} ECGs on {device}...")
    pbar = tqdm(total=len(df), desc="GPT-2 report inference")
    for start, batch_df in make_batches(df, args.batch_size):
        waveforms = []
        batch_rows = []
        for row_offset, row in batch_df.iterrows():
            try:
                waveform = load_ecg_waveform(str(row[waveform_column]), target_length, expected_leads)
            except Exception as exc:
                errors.append(
                    {
                        "row_index": int(row_offset),
                        "waveform_path": str(row.get(waveform_column, "")),
                        "error": str(exc),
                    }
                )
                pbar.update(1)
                continue
            waveforms.append(waveform.T)
            batch_rows.append((int(row_offset), row))

        if not waveforms:
            continue

        ecg_tensor = torch.from_numpy(np.stack(waveforms, axis=0)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            generated_ids = model.generate_report(
                x=ecg_tensor,
                max_token_length=max_token_length,
                **generation_kwargs,
            )

        generations = decode_reports(tokenizer, generated_ids)
        for (row_index, row), generation in zip(batch_rows, generations):
            records.append(
                {
                    "row_index": row_index,
                    "waveform_name": row["waveform_name"],
                    "waveform_path": row[waveform_column],
                    "generation": generation,
                    "reference_report": row[args.report_column],
                    "prompt_category": row.get("prompt_category", "report"),
                    "dataset": row.get("dataset", ""),
                }
            )
            pbar.update(1)

        if args.save_interval > 0 and len(records) > 0 and len(records) % args.save_interval == 0:
            save_outputs(records, output_dir, args.output_prefix, is_final=False)

    pbar.close()

    csv_path = save_outputs(records, output_dir, args.output_prefix, is_final=True)
    json_path = output_dir / f"{args.output_prefix}.json"
    with json_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Final JSON saved: {json_path}")

    if errors:
        errors_path = output_dir / f"{args.output_prefix}_errors.json"
        with errors_path.open("w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved: {errors_path} ({len(errors)} rows)")

    print(f"Generated {len(records)} reports. CSV: {csv_path}")
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
