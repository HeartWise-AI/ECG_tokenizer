#!/usr/bin/env python3
"""
Generate per-diagnosis binary Yes/No QA pairs for training.

For each of the ~50 non-OTHER diagnoses, creates balanced binary QA pairs:
  Q: "Is {diagnosis} present in this ECG? Answer Yes or No."
  A: "Yes - {diagnosis} is present" / "No - {diagnosis} is not present"

Outputs a parquet with the same schema as the main training data.
"""

import sys
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import ECG_PATTERNS, DEEPECG_CATEGORIES

# Diagnoses in the OTHER category + normal markers — excluded from binary QA
OTHER_LABELS = set(DEEPECG_CATEGORIES.get("OTHER", []))
NORMAL_LABELS = {"Sinusal", "Regular", "Monomorph"}
EXCLUDED = OTHER_LABELS | NORMAL_LABELS

# Question templates (varied to avoid overfitting to one phrasing)
QUESTION_TEMPLATES = [
    "Is {diag} present in this ECG? Answer Yes or No.",
    "Does this ECG show {diag}? Answer Yes or No.",
    "Is there evidence of {diag}? Answer Yes or No.",
    "Can you identify {diag} in this tracing? Answer Yes or No.",
    "Is {diag} seen on this ECG? Answer Yes or No.",
]

YES_TEMPLATES = [
    "Yes - {diag} is present",
    "Yes - evidence of {diag}",
    "Yes - {diag} identified",
]

NO_TEMPLATES = [
    "No - {diag} is not present",
    "No - no evidence of {diag}",
    "No - {diag} not identified",
]


def generate_binary_qa(
    source_parquet: str,
    output_path: str,
    max_pos_per_diag: int = 2000,
    max_neg_per_diag: int = 2000,
    seed: int = 42,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    df = pd.read_parquet(source_parquet)
    # Deduplicate to unique ECGs
    df = df.drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    print(f"Source: {len(df)} unique ECGs")

    # Find available diagnosis columns
    diagnoses = [d for d in ECG_PATTERNS if d in df.columns and d not in EXCLUDED]
    print(f"Diagnoses for binary QA: {len(diagnoses)}")

    records = []
    for diag in diagnoses:
        col = df[diag].fillna(0).astype(float)
        pos_idx = df.index[col >= 1].tolist()
        neg_idx = df.index[col < 1].tolist()

        # Sample balanced: equal yes and no
        n_pos = min(len(pos_idx), max_pos_per_diag)
        n_neg = min(len(neg_idx), n_pos)  # match neg to pos for balance
        n_pos = min(n_pos, n_neg)          # also cap pos to neg if neg is fewer

        if n_pos == 0:
            continue

        sampled_pos = rng.sample(pos_idx, n_pos) if len(pos_idx) > n_pos else pos_idx[:n_pos]
        sampled_neg = rng.sample(neg_idx, n_neg) if len(neg_idx) > n_neg else neg_idx[:n_neg]

        for idx in sampled_pos:
            row = df.iloc[idx]
            q = rng.choice(QUESTION_TEMPLATES).format(diag=diag)
            a = rng.choice(YES_TEMPLATES).format(diag=diag)
            rec = {
                "waveform_path_psa": row["waveform_path_psa"],
                "prompt": q,
                "generated_answer": a,
                "prompt_category": "binary_diagnosis",
            }
            # Preserve pattern columns
            for d in diagnoses:
                v = row.get(d, 0)
                try:
                    rec[d] = int(float(v) >= 1)
                except (ValueError, TypeError):
                    rec[d] = 0
            records.append(rec)

        for idx in sampled_neg:
            row = df.iloc[idx]
            q = rng.choice(QUESTION_TEMPLATES).format(diag=diag)
            a = rng.choice(NO_TEMPLATES).format(diag=diag)
            rec = {
                "waveform_path_psa": row["waveform_path_psa"],
                "prompt": q,
                "generated_answer": a,
                "prompt_category": "binary_diagnosis",
            }
            for d in diagnoses:
                v = row.get(d, 0)
                try:
                    rec[d] = int(float(v) >= 1)
                except (ValueError, TypeError):
                    rec[d] = 0
            records.append(rec)

        print(f"  {diag}: {n_pos} pos + {n_neg} neg = {n_pos + n_neg}")

    out_df = pd.DataFrame.from_records(records)
    # Shuffle
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(out_df)} binary QA pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.parquet")
    parser.add_argument("--output", default="/volume/ECG_tokenizer/output/binary_diagnosis_qa_train.parquet")
    parser.add_argument("--max_pos", type=int, default=2000)
    parser.add_argument("--max_neg", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_binary_qa(args.source, args.output, args.max_pos, args.max_neg, args.seed)
