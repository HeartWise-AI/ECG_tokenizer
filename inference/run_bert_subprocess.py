#!/usr/bin/env python3
"""
Run BERT 77-class classification as a standalone subprocess stage.

This script expects a preprocessed parquet containing:
  - ecg_path
  - reports

It creates a temporary CSV with predicted_report/reference_report (both from reports),
runs batched BERT inference using the existing dataloader utilities, and writes
class columns (ECG_PATTERNS) back into the parquet.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import BertTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config.heartwise_config import HeartWiseConfig
from utils.huggingface_wrapper import HuggingFaceWrapper
from utils.files_handler import load_api_keys
from utils.registry import ModelRegistry
from utils.constants import ECG_PATTERNS, BERT_THRESHOLDS
from data.bert_clinical_report_dataset import get_distributed_clinical_report_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT 77-class subprocess")
    parser.add_argument("--input-parquet", required=True, help="Preprocessed parquet with ecg_path and reports")
    parser.add_argument("--output-parquet", default=None, help="Output parquet path (default: overwrite input)")
    parser.add_argument("--base-config", default="config/bert_classifier/base_config.yaml", help="BERT base config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--device", default=None, help="Device (e.g., cuda:0 or cpu)")
    return parser.parse_args()


def _build_bert_input_csv(df: pd.DataFrame, csv_path: Path) -> None:
    bert_input = pd.DataFrame({
        "predicted_report": df["reports"],
        "reference_report": df["reports"],
    })
    bert_input.to_csv(csv_path, index=False)


def run_bert_on_parquet(
    input_parquet: str,
    output_parquet: str | None,
    base_config: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
) -> None:

    # Ensure single-process defaults if not running under torchrun
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    input_path = Path(input_parquet)
    output_path = Path(output_parquet) if output_parquet else input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if "reports" not in df.columns:
        raise ValueError("Input parquet must contain a 'reports' column")

    # Build temporary CSV for the BERT dataloader
    bert_csv_path = output_path.with_suffix(".bert_input.csv")
    _build_bert_input_csv(df, bert_csv_path)

    # Load base config and override paths
    config = HeartWiseConfig.from_yaml(base_config)
    config.predictions_reports_path = str(bert_csv_path)
    config.output_dir = str(output_path.parent)
    config.run_mode = "inference"
    config.use_wandb = False

    if batch_size is not None:
        config.batch_size = batch_size
    if num_workers is not None:
        config.num_workers = num_workers

    # Override device if requested
    if device is not None:
        if device == "cpu":
            config.device = "cpu"
        else:
            config.device = device

    # Load model + tokenizer (mirrors BertReportClassifierProject)
    huggingface_token = load_api_keys(config.api_keys_path)["HUGGING_FACE_TOKEN"]
    model_path = HuggingFaceWrapper.get_model(
        repo_id=config.huggingface_model_name,
        local_dir=config.store_model_path,
        hugging_face_api_key=huggingface_token,
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = ModelRegistry.get(config.model_name)(
        model_path=model_path,
        num_classes=config.num_classes,
    )
    model.to(config.device)
    model.eval()

    dataloader = get_distributed_clinical_report_dataloader(
        predicted_reports_path=config.predictions_reports_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_replicas=1,
        rank=0,
        shuffle=False,
        pin_memory=True,
    )

    class_rows: List[List[int]] = []
    prob_rows: List[List[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            reference_logits = model(
                input_ids=batch["encoded_reference_report"]["input_ids"].to(config.device),
                attention_mask=batch["encoded_reference_report"]["attention_mask"].to(config.device),
                token_type_ids=batch["encoded_reference_report"]["token_type_ids"].to(config.device),
            )
            reference_probs = torch.sigmoid(reference_logits["logits"])

            current_batch_size = reference_probs.shape[0]
            bert_thresholds_tensor = torch.zeros(
                (current_batch_size, config.num_classes), device=reference_probs.device
            )
            for i, pattern in enumerate(ECG_PATTERNS):
                bert_thresholds_tensor[:, i] = BERT_THRESHOLDS[pattern]["threshold"]

            reference_classes = torch.where(reference_probs >= bert_thresholds_tensor, 1, 0)
            class_rows.extend(reference_classes.detach().cpu().int().tolist())
            prob_rows.extend(reference_probs.detach().cpu().float().tolist())

    if len(class_rows) != len(df) or len(prob_rows) != len(df):
        raise ValueError(
            f"Classification count mismatch: {len(class_rows)} / {len(prob_rows)} vs {len(df)}"
        )

    class_df = pd.DataFrame(class_rows, columns=ECG_PATTERNS)
    out_df = pd.concat([df.reset_index(drop=True), class_df], axis=1)
    out_df.to_parquet(output_path, index=False)

    print(f"Wrote BERT class columns to {output_path}")

    # Save probabilities CSV (same columns as parquet, but class columns are probabilities)
    prob_df = pd.DataFrame(prob_rows, columns=ECG_PATTERNS)
    prob_out_df = pd.concat([df.reset_index(drop=True), prob_df], axis=1)
    prob_csv_path = output_path.with_suffix(".bert_probabilities.csv")
    prob_out_df.to_csv(prob_csv_path, index=False)
    print(f"Wrote BERT probability CSV to {prob_csv_path}")


def main() -> None:
    args = parse_args()
    run_bert_on_parquet(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        base_config=args.base_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
