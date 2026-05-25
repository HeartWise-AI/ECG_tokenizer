#!/usr/bin/env python3
"""
Sanity checks for schema SFT renderer/collator.

Checks implemented:
  1) Deterministic rendering
  2) Rendered JSON parses
  3) All supervised_paths resolve to spans
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ecg_clinical_report_dataset import ECGClinicalReportDataset
from utils.schema_sft import DeterministicJSONRenderer, OUTPUT_FIELD_ORDER, SchemaSFTBatchCollator


def _parse_target_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    raise TypeError(f"Unsupported target_json type: {type(raw)}")


def run_checks(df: pd.DataFrame, strict: bool, fail_fast: bool) -> None:
    renderer = DeterministicJSONRenderer(OUTPUT_FIELD_ORDER, strict=strict)
    total = 0
    failures = 0
    for idx, row in df.iterrows():
        total += 1
        target_raw = row.get("target_json")
        supervised_paths = row.get("supervised_paths", [])
        try:
            target_obj = _parse_target_json(target_raw)
            rr1 = renderer.render(target_obj)
            rr2 = renderer.render(target_obj)

            # Check 1: deterministic rendering
            if rr1.text != rr2.text:
                raise AssertionError("Deterministic render mismatch")

            # Check 2: JSON parses
            json.loads(rr1.text)

            # Check 3: supervised_paths resolve
            paths: List[str]
            if isinstance(supervised_paths, str):
                try:
                    paths = json.loads(supervised_paths)
                except json.JSONDecodeError:
                    paths = [p.strip() for p in supervised_paths.split(",") if p.strip()]
            elif isinstance(supervised_paths, (list, tuple)):
                paths = list(supervised_paths)
            else:
                paths = []
            missing = [p for p in paths if p not in rr1.spans]
            if missing:
                raise AssertionError(f"Missing spans for: {missing}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] row={idx}: {exc}")
            if fail_fast:
                raise
    if failures:
        raise SystemExit(f"Schema checks failed for {failures}/{total} rows.")
    print(f"[OK] Checks 1-3 passed for {total} rows.")


def run_smoke(
    dataset_path: str,
    tokenizer_name: str,
    max_length: int,
    signal_path_column: str,
    prompt_column: str,
    answer_column: str,
    category_column: str,
    ecg_waveform_length: int,
    ecg_num_leads: int,
    num_ecg_tokens: int,
    ecg_token_start_id: int | None,
    prefix_tuning: bool,
    medgemma_prompt_style: bool,
    strict: bool,
    smoke_rows: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = ECGClinicalReportDataset(
        dataset_path=dataset_path,
        signal_path_column=signal_path_column,
        ecg_waveform_length=ecg_waveform_length,
        ecg_num_leads=ecg_num_leads,
        tokenizer=tokenizer,
        max_length=max_length,
        instruct_mode=True,
        num_ecg_tokens=num_ecg_tokens,
        ecg_token_start_id=ecg_token_start_id,
        prompt_column=prompt_column,
        answer_column=answer_column,
        category_column=category_column,
        prefix_tuning=prefix_tuning,
        medgemma_prompt_style=medgemma_prompt_style,
        use_schema_collator=True,
    )
    collator = SchemaSFTBatchCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        num_ecg_tokens=num_ecg_tokens,
        ecg_token_start_id=ecg_token_start_id,
        prefix_tuning=prefix_tuning,
        medgemma_prompt_style=medgemma_prompt_style,
        renderer_strict=strict,
    )
    sample_count = min(smoke_rows, len(dataset))
    samples = [dataset[i] for i in range(sample_count)]
    batch = collator(samples)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    kept = (labels != -100).sum(dim=1).tolist()
    print("[SMOKE] batch shapes:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
    print("[SMOKE] kept label tokens:", kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema SFT sanity checks")
    parser.add_argument("--dataset", required=True, help="Path to schema parquet")
    parser.add_argument("--rows", type=int, default=200, help="Rows to sample for checks")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--strict", action="store_true", help="Fail on missing schema keys")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failure")

    parser.add_argument("--smoke", action="store_true", help="Run collator smoke test")
    parser.add_argument("--tokenizer_name", default="google/medgemma-4b-it", help="HF tokenizer name/path")
    parser.add_argument("--max_length", type=int, default=640, help="Max sequence length")
    parser.add_argument("--signal_path_column", default="waveform_path_psa")
    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--answer_column", default="target_json")
    parser.add_argument("--category_column", default="prompt_category")
    parser.add_argument("--ecg_waveform_length", type=int, default=2500)
    parser.add_argument("--ecg_num_leads", type=int, default=12)
    parser.add_argument("--num_ecg_tokens", type=int, default=0)
    parser.add_argument("--ecg_token_start_id", type=int, default=None)
    parser.add_argument("--prefix_tuning", action="store_true")
    parser.add_argument("--medgemma_prompt_style", action="store_true")
    parser.add_argument("--smoke_rows", type=int, default=2)

    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    if args.rows > 0 and len(df) > args.rows:
        df = df.sample(n=args.rows, random_state=args.seed)

    run_checks(df, strict=args.strict, fail_fast=args.fail_fast)

    if args.smoke:
        run_smoke(
            dataset_path=args.dataset,
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
            signal_path_column=args.signal_path_column,
            prompt_column=args.prompt_column,
            answer_column=args.answer_column,
            category_column=args.category_column,
            ecg_waveform_length=args.ecg_waveform_length,
            ecg_num_leads=args.ecg_num_leads,
            num_ecg_tokens=args.num_ecg_tokens,
            ecg_token_start_id=args.ecg_token_start_id,
            prefix_tuning=args.prefix_tuning,
            medgemma_prompt_style=args.medgemma_prompt_style,
            strict=args.strict,
            smoke_rows=args.smoke_rows,
        )


if __name__ == "__main__":
    main()
