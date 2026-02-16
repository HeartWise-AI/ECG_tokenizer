#!/usr/bin/env python3
"""
Create a stratified sample parquet for DPO candidate generation.

Why:
- DPO pairs for rare / hard categories can get wiped out by strict margin filtering.
- Sampling more of those categories upfront yields more candidates/pairs downstream.

Output:
- A parquet suitable for `inference/generate_dpo_multi.py` (expects:
  `waveform_path_psa`, `prompt`, `prompt_category`, `generated_answer`)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT = "output/combined_train_qa_m5000k_h5000k_weighted.parquet"
DEFAULT_OUTPUT = "output/dpo_train_sample_50k.parquet"

DEFAULT_CATEGORY_SAMPLING = {
    # Rare / hard (take all or most)
    "urgency_assessment": 70,
    "culprit_artery": 919,

    # Weak categories (LLM-judge)
    "acs_severity": 5000,
    "classification": 10000,
    "interpretation": 8000,
    "json_interpretation": 5000,

    # Medium
    "structural_heart_disease": 3000,
    "afib_risk": 3000,
    "category_conduction": 2000,
    "category_infarct_ischemia": 2000,
    "category_rhythm": 3000,
    "lvef": 2000,

    # Small coverage
    "category_other": 1000,
    "category_chamber_enlargement": 1000,
    "ecg_interval": 1000,
    "localization_t_wave": 600,
    "random_finding_question": 500,
}


@dataclass(frozen=True)
class BucketSpec:
    name: str
    prompt_re: re.Pattern
    n_samples: int
    category_in: Optional[set[str]]
    set_prompt_category: bool


def _load_bucket_specs(path: Optional[str]) -> list[BucketSpec]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("--bucket_sampling_json must be a JSON list.")
    out: list[BucketSpec] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Bucket entry #{i} must be an object.")
        name = entry.get("name")
        prompt_regex = entry.get("prompt_regex") or entry.get("regex")
        n_samples = entry.get("n_samples")
        if not name or not prompt_regex or n_samples is None:
            raise ValueError(f"Bucket entry #{i} requires name, prompt_regex, n_samples.")
        ignore_case = bool(entry.get("ignore_case", True))
        flags = re.IGNORECASE if ignore_case else 0
        prompt_re = re.compile(str(prompt_regex), flags=flags)
        category_in = entry.get("category_in")
        category_in_set = set(category_in) if isinstance(category_in, list) else None
        set_prompt_category = bool(entry.get("set_prompt_category", True))
        out.append(
            BucketSpec(
                name=str(name),
                prompt_re=prompt_re,
                n_samples=int(n_samples),
                category_in=category_in_set,
                set_prompt_category=set_prompt_category,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified DPO sample parquet (category + regex buckets).")
    parser.add_argument("--input_parquet", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_parquet", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling_config_json",
        type=str,
        default=None,
        help="Optional JSON mapping: prompt_category -> n_samples (overrides built-in defaults).",
    )
    parser.add_argument(
        "--bucket_sampling_json",
        type=str,
        default=None,
        help="Optional JSON list of bucket specs (regex-based sampling).",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle final sample before writing.")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    sampling_config = dict(DEFAULT_CATEGORY_SAMPLING)
    if args.sampling_config_json:
        with open(args.sampling_config_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--sampling_config_json must be a JSON object mapping category -> int.")
        sampling_config = {str(k): int(v) for k, v in loaded.items()}

    bucket_specs = _load_bucket_specs(args.bucket_sampling_json)
    if not bucket_specs:
        # Default: carve out a dedicated ST elevation bucket aligned with judge category naming.
        bucket_specs = [
            BucketSpec(
                name="localization_st_elevation",
                prompt_re=re.compile(r"\\bst\\s*elevation\\b", flags=re.IGNORECASE),
                n_samples=3000,
                category_in={"category_infarct_ischemia", "category_pericarditis"},
                set_prompt_category=True,
            )
        ]

    print(f"Loading: {args.input_parquet}")
    cols = ["waveform_path_psa", "prompt", "prompt_category", "generated_answer"]
    df = pd.read_parquet(args.input_parquet, columns=cols)
    df = df.dropna(subset=["waveform_path_psa", "prompt", "prompt_category", "generated_answer"])
    print(f"Total rows: {len(df):,}")

    # Bucket sampling first (so we can exclude these rows from base-category sampling).
    exclude_idx: set[int] = set()
    samples: list[pd.DataFrame] = []

    print("\nBucket sampling:")
    for spec in bucket_specs:
        sub = df
        if spec.category_in is not None:
            sub = sub[sub["prompt_category"].isin(spec.category_in)]
        mask = sub["prompt"].astype(str).str.contains(spec.prompt_re, na=False, regex=True)
        sub = sub[mask]
        if len(sub) == 0:
            print(f"  {spec.name}: 0 rows (skipping)")
            continue
        n = min(int(spec.n_samples), len(sub))
        sampled = sub.sample(n=n, random_state=rng.randint(0, 2**32 - 1))
        exclude_idx.update(sampled.index.tolist())
        if spec.set_prompt_category:
            sampled = sampled.copy()
            sampled["prompt_category"] = spec.name
        samples.append(sampled)
        print(f"  {spec.name}: {n:,} / {len(sub):,}")

    print("\nCategory sampling:")
    for cat, n_target in sampling_config.items():
        sub = df[df["prompt_category"] == cat]
        if exclude_idx:
            sub = sub[~sub.index.isin(exclude_idx)]
        if len(sub) == 0:
            print(f"  {cat}: 0 rows (skipping)")
            continue
        n = min(int(n_target), len(sub))
        sampled = sub.sample(n=n, random_state=rng.randint(0, 2**32 - 1))
        samples.append(sampled)
        print(f"  {cat}: {n:,} / {len(sub):,}")

    if not samples:
        raise RuntimeError("No samples selected; check sampling config and parquet schema.")

    result = pd.concat(samples, ignore_index=True)
    if args.shuffle:
        result = result.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"\nTotal sampled: {len(result):,}")
    print("Sampled prompt_category distribution (top 20):")
    print(result["prompt_category"].value_counts().head(20).to_string())

    print(f"\nWriting: {args.output_parquet}")
    result.to_parquet(args.output_parquet, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
