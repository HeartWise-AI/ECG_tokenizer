#!/usr/bin/env python3
"""
Choice-Free (CF) evaluation dataset generator for ECG interpretation.

This tool converts an ECG test/validation parquet into a CF dataset used
only for evaluation (never training). Each ECG yields exactly one record per
high-level category (6 total):
  - RHYTHM
  - CONDUCTION
  - CHAMBER ENLARGEMENT
  - INFARCT, ISCHEMIA
  - PERICARDITIS
  - OTHER

Each record contains a question, a fixed candidate answer list, and the
single ground-truth answer for that category. The ground-truth is selected
by priority among all positive findings within the category; if none are
present, the answer is "No abnormality".

Output format (JSON list of dicts):
{
  "ecg_id": "ecg_12345",
  "signal_path": "/path/to/waveform.npy",
  "category": "RHYTHM",
  "question": "What is the rhythm?",
  "candidate_answers": ["Afib", "Sinus rhythm", "Regular", ...],
  "ground_truth_answer": "Afib",
  "ground_truth_index": 0
}

Typical usage:
python utils/generate_ecg_cf_eval_dataset.py \
  --test_path /media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet \
  --output_path ecg_cf_eval \
  --normal_pct 0.05 \
  --save_format json

Notes:
- Works only with test/val data. Do not use for training.
- Designed to be robust across MIMIC/MHI/PTB-XL style parquets. When exact
  columns are not found, it falls back to label/slug heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

import pandas as pd

# Local imports
from utils.constants import DEEPECG_CATEGORIES

TRAIN_COMBINED="/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet"
TEST_COMBINED="/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet"
# -----------------------------
# Category questions and options
# -----------------------------

CATEGORY_QUESTIONS: Dict[str, str] = {
    "RHYTHM": "What is the cardiac rhythm on this ECG?",
    "CONDUCTION": "What is the main conduction abnormality on this ECG?",
    "CHAMBER ENLARGEMENT": "Which chamber enlargement is present?",
    "INFARCT, ISCHEMIA": "What ischemic or infarct pattern best describes this ECG?",
    "PERICARDITIS": "Are there signs of pericarditis?",
    "OTHER": "What other ECG finding is present?",
}


def _slug(text: str) -> str:
    import re
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _resolve_label_columns(df: pd.DataFrame, label: str) -> Tuple[str | None, str | None]:
    """
    Resolve diagnosis and BERT columns for a given label.

    Heuristics:
      1) Exact column name match (e.g., "Afib")
      2) Any column whose slug matches the label slug
      3) Prefer GT_* variants over BERT_* when multiple candidates exist
    Returns (diag_col, bert_col) — both optional.
    """
    label_slug = _slug(label)
    candidates = []
    for col in df.columns:
        if col == label:
            candidates.append(col)
            continue
        col_slug = _slug(col)
        if col_slug == label_slug or col_slug.endswith("_" + label_slug) or col_slug.startswith(label_slug + "_"):
            candidates.append(col)

    if not candidates:
        return None, None

    # Prefer GT over BERT if both exist
    gt_like = [c for c in candidates if _slug(c).startswith("gt_") or _slug(c).endswith("_gt")]
    bert_like = [c for c in candidates if "bert" in _slug(c)]
    exact = [c for c in candidates if c == label]

    diag_col = None
    bert_col = None

    if exact:
        # If the exact label column exists, treat it as diagnosis ground-truth style
        diag_col = exact[0]
    elif gt_like:
        diag_col = gt_like[0]
    else:
        # Fallback to any candidate (prefer non-bert)
        non_bert = [c for c in candidates if c not in bert_like]
        diag_col = non_bert[0] if non_bert else candidates[0]

    if bert_like:
        bert_col = bert_like[0]

    return diag_col, bert_col


def _is_positive(value: Any) -> bool:
    """Interpret a variety of data types as positive labels (1/True/>0.5)."""
    try:
        if value is None:
            return False
        if isinstance(value, (int, bool)):
            return bool(int(value) == 1)
        if isinstance(value, float):
            return float(value) > 0.5
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
    except Exception:
        return False
    return False


def _ground_truth_labels_for_category(row: pd.Series, category: str, candidates: List[str]) -> List[str]:
    """Return all positive labels for the given category among `candidates`.
    If none are positive, return ["No abnormality"].
    """
    positives: List[str] = []
    row_df = row.to_frame().T
    for label in candidates:
        diag_col, bert_col = _resolve_label_columns(row_df, label)
        if diag_col and _is_positive(row.get(diag_col)):
            positives.append(label)
        elif not diag_col and bert_col and _is_positive(row.get(bert_col)):
            positives.append(label)
    if not positives:
        return ["No abnormality"]
    # Stable-unique in candidate order
    uniq: List[str] = []
    seen = set()
    for l in positives:
        if l not in seen:
            uniq.append(l)
            seen.add(l)
    return uniq


def _build_resolved_cols(df: pd.DataFrame) -> Dict[str, Tuple[str | None, str | None]]:
    """Pre-resolve (diag_col, bert_col) for each label across all categories."""
    all_labels: List[str] = []
    for labels in DEEPECG_CATEGORIES.values():
        all_labels.extend(list(labels))
    all_labels = sorted(set(all_labels))
    return {label: _resolve_label_columns(df, label) for label in all_labels}


def _ground_truth_from_mapping(row_dict: Dict[str, Any], candidates: List[str], resolved: Dict[str, Tuple[str | None, str | None]]) -> List[str]:
    """Faster GT extraction using pre-resolved column mapping and a row dict."""
    positives: List[str] = []
    for label in candidates:
        diag_col, bert_col = resolved.get(label, (None, None))
        if diag_col is not None and _is_positive(row_dict.get(diag_col)):
            positives.append(label)
        elif diag_col is None and bert_col is not None and _is_positive(row_dict.get(bert_col)):
            positives.append(label)
    if not positives:
        return ["No abnormality"]
    seen = set()
    uniq: List[str] = []
    for l in positives:
        if l not in seen:
            uniq.append(l)
            seen.add(l)
    return uniq


def _build_records_chunk(
    rows: List[Dict[str, Any]],
    categories: List[str],
    candidates_by_cat: Dict[str, List[str]],
    questions_by_cat: Dict[str, str],
    resolved_cols: Dict[str, Tuple[str | None, str | None]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        ecg_id, signal_path = _infer_id_and_path(pd.Series(row))
        for cat in categories:
            candidates = candidates_by_cat[cat]
            question = questions_by_cat.get(cat) or f"What is the finding for {cat}?"
            gt_answers = _ground_truth_from_mapping(row, candidates, resolved_cols)
            gt_indices = [candidates.index(a) if a in candidates else len(candidates) - 1 for a in gt_answers]
            first_ans = gt_answers[0]
            first_idx = int(gt_indices[0])
            out.append({
                "ecg_id": ecg_id,
                "signal_path": signal_path,
                "category": cat,
                "question": question,
                "candidate_answers": candidates,
                "ground_truth_answers": gt_answers,
                "ground_truth_indices": gt_indices,
                "ground_truth_answer": first_ans,
                "ground_truth_index": first_idx,
            })
    return out


def _infer_id_and_path(row: pd.Series) -> Tuple[str, str]:
    """
    Infer `ecg_id` and `signal_path` from common columns.
    Prefers ('waveform_path_psa', 'waveform_name', 'ecg_id', 'filename_hr').
    """
    # Signal path
    signal_path = None
    for col in ("waveform_path_psa", "waveform_path_original", "filename_hr", "signal_path"):
        if col in row.index and pd.notna(row.get(col)):
            signal_path = str(row.get(col))
            break
    if signal_path is None:
        # Last resort: any column that looks like a path to .npy
        for col in row.index:
            val = row.get(col)
            if isinstance(val, str) and val.endswith(".npy"):
                signal_path = val
                break
    if signal_path is None:
        signal_path = ""

    # ECG id
    ecg_id = None
    for col in ("waveform_name", "ecg_id", "waveform_id"):
        if col in row.index and pd.notna(row.get(col)):
            ecg_id = str(row.get(col))
            break
    if ecg_id is None:
        # Use filename as id
        ecg_id = os.path.basename(signal_path) if signal_path else "unknown"

    return ecg_id, signal_path


def filter_and_sample_normal(df: pd.DataFrame, normal_pct: float) -> pd.DataFrame:
    """
    Keep all abnormal ECGs and sample a fraction of normal ones.
    Normal ECG = none of the category labels are positive.
    """
    if not 0.0 <= normal_pct <= 1.0:
        raise ValueError("normal_pct must be between 0.0 and 1.0")

    # Build a pool of all labels across categories
    all_labels: List[str] = []
    for labels in DEEPECG_CATEGORIES.values():
        all_labels.extend(list(labels))
    all_labels = sorted(set(all_labels))

    # Pre-resolve usable columns so we don't loop per-row too much
    resolved_cols: Dict[str, Tuple[str | None, str | None]] = {
        label: _resolve_label_columns(df, label) for label in all_labels
    }

    def row_is_normal(row: pd.Series) -> bool:
        for label, (diag_col, bert_col) in resolved_cols.items():
            if diag_col is not None and _is_positive(row.get(diag_col)):
                return False
            if diag_col is None and bert_col is not None and _is_positive(row.get(bert_col)):
                return False
        return True

    normal_mask = df.apply(row_is_normal, axis=1)
    normals = df[normal_mask]
    abnormals = df[~normal_mask]

    if len(normals) == 0:
        return abnormals.reset_index(drop=True)

    # Sample fraction of normals
    sample_size = max(0, int(round(len(normals) * float(normal_pct))))
    normals_sampled = normals.sample(n=sample_size, random_state=42) if sample_size > 0 else normals.iloc[0:0]

    filtered = pd.concat([abnormals, normals_sampled], axis=0).sample(frac=1.0, random_state=42)
    return filtered.reset_index(drop=True)


def _category_candidates_with_noabn(category: str) -> List[str]:
    base = list(DEEPECG_CATEGORIES.get(category, []))
    # Ensure deterministic order and add "No abnormality" sentinel
    base = list(dict.fromkeys(base))  # stable unique
    base.append("No abnormality")
    return base


def create_cf_multicategory_dataset(df: pd.DataFrame, split_name: str, workers: int = 1, chunk_size: int = 5000) -> List[Dict[str, Any]]:
    """
    Create CF records for every ECG × category pair (excluding OTHER).
    Supports parallel building with multiple workers.
    """
    # Use all defined categories but skip OTHER for CF generation
    categories = [c for c in DEEPECG_CATEGORIES.keys() if c != "OTHER"]
    candidates_by_cat = {c: _category_candidates_with_noabn(c) for c in categories}
    questions_by_cat = dict(CATEGORY_QUESTIONS)
    resolved_cols = _build_resolved_cols(df)

    if workers is None or workers <= 1:
        # Fallback to single-process with tqdm
        out: List[Dict[str, Any]] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CF[{split_name}]"):
            ecg_id, signal_path = _infer_id_and_path(row)
            for cat in categories:
                candidates = candidates_by_cat[cat]
                question = questions_by_cat.get(cat, f"What is the finding for {cat}?")
                gt_answers = _ground_truth_from_mapping(row.to_dict(), candidates, resolved_cols)
                gt_indices = [candidates.index(a) if a in candidates else len(candidates) - 1 for a in gt_answers]
                first_ans = gt_answers[0]
                first_idx = int(gt_indices[0])
                out.append({
                    "ecg_id": ecg_id,
                    "signal_path": signal_path,
                    "category": cat,
                    "question": question,
                    "candidate_answers": candidates,
                    "ground_truth_answers": gt_answers,
                    "ground_truth_indices": gt_indices,
                    "ground_truth_answer": first_ans,
                    "ground_truth_index": first_idx,
                })
        return out

    # Multi-process build: chunk rows, fan out, and aggregate
    total = len(df)
    n_chunks = max(1, math.ceil(total / max(1, int(chunk_size))))
    indices = df.index.to_list()
    out_all: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futures = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(total, (i + 1) * chunk_size)
            if start >= end:
                continue
            rows = [df.loc[idx].to_dict() for idx in indices[start:end]]
            fut = ex.submit(
                _build_records_chunk,
                rows,
                categories,
                candidates_by_cat,
                questions_by_cat,
                resolved_cols,
            )
            futures.append(fut)

        with tqdm(total=total, desc=f"CF[{split_name}][{workers}w]", unit="rows") as pbar:
            for fut in as_completed(futures):
                chunk_records = fut.result()
                out_all.extend(chunk_records)
                # Progress by chunk size (approximate)
                pbar.update(min(chunk_size, total - pbar.n))

    return out_all


def generate_cf_eval_dataset(
    test_path: str,
    output_path: str = "ecg_cf_eval",
    normal_pct: float = 0.05,
    save_format: str = "json",
    split_name: str = "test",
    max_ecgs: int | None = None,
    workers: int = 1,
    chunk_size: int = 5000,
) -> str:
    """
    Generate CF evaluation dataset from a test/val parquet file.

    Args:
        test_path: Path to parquet file (NOT train)
        output_path: Output directory (will be created if missing)
        normal_pct: Fraction of normal ECGs to include
        save_format: Currently only "json" is supported
        split_name: Name used for output filename prefix ("test" or "val")

    Returns:
        Path to the saved JSON file
    """
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading {split_name} data: {test_path}")
    df = pd.read_parquet(test_path)
    if max_ecgs is not None and int(max_ecgs) > 0 and len(df) > int(max_ecgs):
        print(f"Limiting to first {int(max_ecgs)} ECG rows for quick generation …")
        df = df.head(int(max_ecgs))
    print(f"Loaded {len(df)} ECGs")

    print(f"Applying filtering with {normal_pct*100:.1f}% normal sampling …")
    df_filt = filter_and_sample_normal(df, normal_pct)
    print(f"After filtering: {len(df_filt)} ECGs")

    print(f"Building CF records with workers={workers}, chunk_size={chunk_size} …")
    cf_records = create_cf_multicategory_dataset(df_filt, split_name, workers=workers, chunk_size=chunk_size)
    print(f"Generated CF records: {len(cf_records)}")

    if save_format == "json":
        out_path = os.path.join(output_path, f"{split_name}_cf.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cf_records, f, indent=2)
    elif save_format == "parquet":
        out_path = os.path.join(output_path, f"{split_name}_cf.parquet")
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            table = pa.Table.from_pylist(cf_records)
            pq.write_table(table, out_path)
        except Exception:
            # Fallback: Pandas to_parquet
            pd.DataFrame(cf_records).to_parquet(out_path, index=False)
    else:
        raise ValueError("Unsupported save_format; use 'json' or 'parquet'")

    print(f"Saved CF evaluation dataset: {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ECG CF evaluation dataset (test/val only)")
    p.add_argument("--test_path", type=str, required=True, help="Path to test parquet (or val parquet)")
    p.add_argument("--output_path", type=str, default="ecg_cf_eval", help="Output directory")
    p.add_argument("--normal_pct", type=float, default=0.05, help="Fraction of normal ECGs to include")
    p.add_argument("--save_format", type=str, default="json", choices=["json", "parquet"], help="Save format")
    p.add_argument("--split_name", type=str, default="test", choices=["test", "val"], help="Split name for output file")
    p.add_argument("--max_ecgs", type=int, default=0, help="Limit number of ECG rows (0 = no limit)")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers (>=1)")
    p.add_argument("--chunk_size", type=int, default=5000, help="Rows per worker task (tune for throughput)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_cf_eval_dataset(
        test_path=args.test_path,
        output_path=args.output_path,
        normal_pct=args.normal_pct,
        save_format=args.save_format,
        split_name=args.split_name,
        max_ecgs=(args.max_ecgs if args.max_ecgs and args.max_ecgs > 0 else None),
        workers=max(1, int(args.workers or 1)),
        chunk_size=max(1000, int(args.chunk_size or 5000)),
    )
