#!/usr/bin/env python3
"""
Generate train and validation parquet datasets with category-aware questions
derived from ECG pattern labels (77 constants in utils/constants.py).

Goal:
  - Train: N examples (e.g., 400k), half MIMIC, half MHI
  - Val:   M examples (e.g., 25k), half MIMIC, half MHI

Each example = one ECG row -> one question/answer pair chosen from the
positive labels present on that ECG. If multiple positives are present,
one is chosen at random. If none are positive, we fall back to a general
question with answer "No abnormality".

Schema (per row):
  - waveform_path_psa (string)
  - prompt (string)
  - generated_answer (string)
  - prompt_category (string)
  - [77 pattern columns] (0/1 ints if available on source)
  - optional identifiers if present: waveform_name, ecg_id

Usage example:
python dataset_generation/generate_train_val_parquets.py \
  --mimic_path /path/to/mimic.parquet \
  --mhi_path /path/to/mhi.parquet \
  --out_dir output/cf_train_val \
  --train_size 400000 \
  --val_size 25000

This script is intentionally lightweight and uses only simple column
presence checks for label detection (exact label names). It preserves all
found pattern columns if present to maximize compatibility with downstream
auxiliary heads.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.constants import ECG_PATTERNS, DEEPECG_CATEGORIES

import json as _json


def _load_category_questions() -> Dict[str, List[str]]:
    """Load expanded prompt variations if available, else use defaults."""
    variations_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompt_variations.json"
    )
    _cat_var_map = {
        "RHYTHM": "category_rhythm",
        "CONDUCTION": "category_conduction",
        "CHAMBER ENLARGEMENT": "category_chamber_enlargement",
        "INFARCT, ISCHEMIA": "category_infarct_ischemia",
        "PERICARDITIS": "category_pericarditis",
        "OTHER": "category_other",
    }
    defaults = {
        "RHYTHM": ["What is the cardiac rhythm on this ECG?"],
        "CONDUCTION": ["What is the main conduction abnormality on this ECG?"],
        "CHAMBER ENLARGEMENT": ["Which chamber enlargement is present?"],
        "INFARCT, ISCHEMIA": ["What ischemic or infarct pattern best describes this ECG?"],
        "PERICARDITIS": ["Are there signs of pericarditis?"],
        "OTHER": ["What other ECG finding is present?"],
    }
    if os.path.exists(variations_path):
        with open(variations_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        for cat, var_key in _cat_var_map.items():
            if var_key in data and data[var_key]:
                defaults[cat] = data[var_key]
        print(f"Loaded prompt variations: {', '.join(f'{k}={len(v)}' for k, v in defaults.items())}")
    return defaults


# Category-specific question templates (expanded from prompt_variations.json if available)
CATEGORY_QUESTION_POOLS: Dict[str, List[str]] = _load_category_questions()

# Legacy single-question dict (kept for backward compat but unused in new code)
CATEGORY_QUESTIONS: Dict[str, str] = {
    k: v[0] for k, v in CATEGORY_QUESTION_POOLS.items()
}


def _reverse_category_map() -> Dict[str, str]:
    rev: Dict[str, str] = {}
    for cat, labels in DEEPECG_CATEGORIES.items():
        for l in labels:
            rev[l] = cat
    return rev


REV_CAT = _reverse_category_map()


def _existing_pattern_columns(df: pd.DataFrame) -> List[str]:
    # Only allow labels among defined categories; we'll skip OTHER later at selection stage
    allowed = set(l for labels in DEEPECG_CATEGORIES.values() for l in labels)
    return [c for c in ECG_PATTERNS if c in df.columns and c in allowed]


def _is_positive(value: Any) -> bool:
    try:
        if value is None:
            return False
        if isinstance(value, (int, bool)):
            return int(value) == 1
        if isinstance(value, float):
            return float(value) > 0.5
        s = str(value).strip().lower()
        return s in {"1", "true", "yes", "y"}
    except Exception:
        return False


def _pick_positive_label(row: pd.Series, pattern_cols: List[str]) -> Tuple[str | None, str]:
    """Return (label, category) if any positive; else (None, 'OTHER')."""
    positives = [lab for lab in pattern_cols if _is_positive(row.get(lab))]
    # Skip OTHER category positives; keep only allowed (non-OTHER)
    non_other = [lab for lab in positives if REV_CAT.get(lab, "OTHER") != "OTHER"]
    positives = non_other
    if not positives:
        return None, "RHYTHM"
    label = random.choice(positives)
    category = REV_CAT.get(label, "RHYTHM")
    return label, category


def _map_signal_path(df: pd.DataFrame) -> pd.Series:
    # Prefer existing 'waveform_path_psa'; otherwise try common alternatives
    if 'waveform_path_psa' in df.columns:
        return df['waveform_path_psa'].astype(str)
    for alt in ('waveform_path_original', 'filename_hr', 'signal_path'):
        if alt in df.columns:
            return df[alt].astype(str)
    # As last resort, return an empty string column to avoid key errors
    return pd.Series([""] * len(df), index=df.index)


def _optional_col(df: pd.DataFrame, names: List[str]) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _build_split(df: pd.DataFrame, n_samples: int, rng: random.Random) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df.sample(frac=1.0, random_state=rng.randint(0, 10**9)).reset_index(drop=True)
    return df.sample(n=n_samples, random_state=rng.randint(0, 10**9)).reset_index(drop=True)


def _examples_from_df(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    pattern_cols = _existing_pattern_columns(df)
    if not pattern_cols:
        raise ValueError("No ECG pattern columns found in source parquet; expected 77 constants to be present.")

    # Select rows
    sel = _build_split(df, n, rng)

    # Build examples
    records: List[Dict[str, Any]] = []
    path_series = _map_signal_path(sel)
    name_series = _optional_col(sel, ["waveform_name", "ecg_id"])  # keep if available

    for idx, row in sel.iterrows():
        label, category = _pick_positive_label(row, pattern_cols)
        if label is None:
            question = "What is the primary ECG finding?"
            answer = "No abnormality"
            category = "RHYTHM"
        else:
            pool = CATEGORY_QUESTION_POOLS.get(category, ["What is the primary ECG finding?"])
            question = random.choice(pool)
            answer = label

        rec: Dict[str, Any] = {
            "waveform_path_psa": path_series.loc[idx],
            "prompt": question,
            "generated_answer": answer,
            "prompt_category": category,
        }

        # Keep optional identifiers
        if name_series is not None:
            rec["waveform_name"] = str(name_series.loc[idx])

        # Preserve available pattern columns (binarized)
        for lab in pattern_cols:
            v = row.get(lab)
            rec[lab] = int(1 if _is_positive(v) else 0)

        records.append(rec)

    return pd.DataFrame.from_records(records)


def _cf_examples_from_df(df: pd.DataFrame, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Build CF examples (records per ECG × category). Keep all positives per category.
    OTHER category is skipped. If no positives in a category, ground_truth_answers=["No abnormality"].
    """
    pattern_cols = _existing_pattern_columns(df)
    if not pattern_cols:
        raise ValueError("No ECG pattern columns found in source parquet; expected 77 constants to be present.")

    sel = _build_split(df, n, rng)
    out: List[Dict[str, Any]] = []
    allowed_categories = [c for c in DEEPECG_CATEGORIES.keys() if c != "OTHER"]

    def cat_candidates(cat: str) -> List[str]:
        pool = list(dict.fromkeys(DEEPECG_CATEGORIES.get(cat, [])))
        pool.append("No abnormality")
        return pool

    # map columns once for speed
    for idx, row in sel.iterrows():
        signal_path = _map_signal_path(sel).loc[idx]
        _ecg_series = _optional_col(sel, ["waveform_name", "ecg_id"])  # may be None
        if _ecg_series is not None and not pd.isna(_ecg_series.loc[idx]):
            ecg_id = str(_ecg_series.loc[idx])
        else:
            ecg_id = os.path.basename(str(signal_path))

        for category in allowed_categories:
            candidates = cat_candidates(category)
            # collect all positives for this category
            cat_labels = [lab for lab in candidates if lab != "No abnormality"]
            positives = [lab for lab in cat_labels if _is_positive(row.get(lab))]
            gt_answers = positives if positives else ["No abnormality"]
            gt_indices = [candidates.index(a) if a in candidates else len(candidates)-1 for a in gt_answers]
            pool = CATEGORY_QUESTION_POOLS.get(category, ["What is the primary ECG finding?"])
            question = random.choice(pool)
            out.append({
                "ecg_id": ecg_id,
                "signal_path": str(signal_path),
                "category": category,
                "question": question,
                "candidate_answers": candidates,
                "ground_truth_answers": gt_answers,
                "ground_truth_indices": gt_indices,
                # single-field back-compat
                "ground_truth_answer": gt_answers[0],
                "ground_truth_index": int(gt_indices[0]),
            })

    return out


def generate_train_val(
    mimic_path: str,
    mhi_path: str,
    out_dir: str,
    train_size: int = 400_000,
    val_size: int = 25_000,
    seed: int = 42,
    mode: str = "qa",
    fmt: str = "parquet",
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    print(f"Loading MIMIC: {mimic_path}")
    df_mimic = pd.read_parquet(mimic_path)
    print(f"  rows: {len(df_mimic)}")
    print(f"Loading MHI:   {mhi_path}")
    df_mhi = pd.read_parquet(mhi_path)
    print(f"  rows: {len(df_mhi)}")

    # If inputs are the same (e.g., a combined parquet), split by 'dataset' column if present
    if mimic_path == mhi_path and 'dataset' in df_mimic.columns:
        print("Detected combined parquet with 'dataset' column; splitting into MIMIC and MHI subsets.")
        df_mimic = df_mimic[df_mimic['dataset'].astype(str).str.lower() == 'mimic'].reset_index(drop=True)
        df_mhi = df_mimic if False else pd.read_parquet(mhi_path)  # reload to keep independent filtering
        df_mhi = df_mhi[df_mhi['dataset'].astype(str).str.lower() == 'mhi'].reset_index(drop=True)
        print(f"  MIMIC subset: {len(df_mimic)} rows")
        print(f"  MHI subset:   {len(df_mhi)} rows")

    # Half-half target sizes
    t_half = int(train_size // 2)
    v_half = int(val_size // 2)

    print(f"Building TRAIN examples: total={train_size} (MIMIC={t_half}, MHI={t_half}) in mode={mode}")
    if mode == "qa":
        train_mimic = _examples_from_df(df_mimic, t_half, rng)
        train_mhi = _examples_from_df(df_mhi, t_half, rng)
        train_df = pd.concat([train_mimic, train_mhi], axis=0, ignore_index=True)
        train_out = os.path.join(out_dir, f"train_qa_m{t_half}_h{t_half}.{ 'parquet' if fmt=='parquet' else 'json' }")
        if fmt == 'parquet':
            train_df.to_parquet(train_out, index=False)
        else:
            train_df.to_json(train_out, orient='records', indent=2)
        print(f"Saved train {fmt}: {train_out} (rows={len(train_df)})")
    elif mode == "cf":
        train_mimic = _cf_examples_from_df(df_mimic, t_half, rng)
        train_mhi = _cf_examples_from_df(df_mhi, t_half, rng)
        train_cf = train_mimic + train_mhi
        train_out = os.path.join(out_dir, f"train_cf_m{t_half}_h{t_half}.{ 'parquet' if fmt=='parquet' else 'json' }")
        if fmt == 'parquet':
            pd.DataFrame(train_cf).to_parquet(train_out, index=False)
        else:
            import json
            with open(train_out, 'w', encoding='utf-8') as f:
                json.dump(train_cf, f, indent=2)
        print(f"Saved train CF {fmt}: {train_out} (rows={len(train_cf)})")
    else:
        raise ValueError("mode must be 'qa' or 'cf'")

    print(f"Building VAL examples: total={val_size} (MIMIC={v_half}, MHI={v_half}) in mode={mode}")
    if mode == "qa":
        val_mimic = _examples_from_df(df_mimic, v_half, rng)
        val_mhi = _examples_from_df(df_mhi, v_half, rng)
        val_df = pd.concat([val_mimic, val_mhi], axis=0, ignore_index=True)
        val_out = os.path.join(out_dir, f"val_qa_m{v_half}_h{v_half}.{ 'parquet' if fmt=='parquet' else 'json' }")
        if fmt == 'parquet':
            val_df.to_parquet(val_out, index=False)
        else:
            val_df.to_json(val_out, orient='records', indent=2)
        print(f"Saved val {fmt}:   {val_out} (rows={len(val_df)})")
    else:
        val_mimic = _cf_examples_from_df(df_mimic, v_half, rng)
        val_mhi = _cf_examples_from_df(df_mhi, v_half, rng)
        val_cf = val_mimic + val_mhi
        val_out = os.path.join(out_dir, f"val_cf_m{v_half}_h{v_half}.{ 'parquet' if fmt=='parquet' else 'json' }")
        if fmt == 'parquet':
            pd.DataFrame(val_cf).to_parquet(val_out, index=False)
        else:
            import json
            with open(val_out, 'w', encoding='utf-8') as f:
                json.dump(val_cf, f, indent=2)
        print(f"Saved val CF {fmt}:   {val_out} (rows={len(val_cf)})")

    # Show a few examples for inspection
    print("\nSample examples (first 3 rows):")
    with pd.option_context('display.max_colwidth', 140):
        if mode == 'qa':
            print(train_df.loc[:2, ["waveform_path_psa", "prompt", "generated_answer", "prompt_category"]])
        else:
            import json
            preview = (train_cf[:3] if isinstance(train_cf, list) else [])
            print(json.dumps(preview, indent=2) if preview else "<no preview>")

    return train_out, val_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate train/val datasets from ECG label columns (QA or CF modes)")
    p.add_argument("--mimic_path", type=str, required=True, help="Path to MIMIC parquet")
    p.add_argument("--mhi_path", type=str, required=True, help="Path to MHI parquet")
    p.add_argument("--out_dir", type=str, default="output/cf_train_val", help="Output directory")
    p.add_argument("--train_size", type=int, default=400_000)
    p.add_argument("--val_size", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, choices=["qa", "cf"], default="qa", help="Output mode: qa (Q&A) or cf (Choice-Free)")
    p.add_argument("--format", type=str, choices=["parquet", "json"], default="parquet", help="Output format")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_train_val(
        mimic_path=args.mimic_path,
        mhi_path=args.mhi_path,
        out_dir=args.out_dir,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        seed=int(args.seed),
        mode=str(args.mode),
        fmt=str(args.format),
    )
