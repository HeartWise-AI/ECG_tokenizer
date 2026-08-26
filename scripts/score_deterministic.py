#!/usr/bin/env python3
"""Deterministic scoring of generated ECG QA answers without an LLM judge.

Parses the model's generated text for the endpoints that have a machine-checkable answer and
scores them against the parsed ground truth in the same CSV:
  - LVEF:   parse "ejection fraction is N%"  -> AUROC(EF<=40, score=-EF) + MAE + acc
  - AFib / SHD / ACS: parse binary Yes/No    -> balanced-acc (=AUROC for a binary rule), sens, spec, F1, n

Works on a glob of shard checkpoint CSVs (incremental) or the merged CSV. Columns expected:
  generation, ground_truth, prompt_category.

  python scripts/score_deterministic.py "<glob-or-path>" [--tag NEW]
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.endpoint_readout import has_binary_negation, parse_ef as parse_endpoint_ef


BINARY_TASKS = {
    "afib_risk": "AFib-5y",
    "structural_heart_disease": "SHD",
    "acs_severity": "ACS-acute",
}
SUPPORTED_CATEGORIES = ("lvef", *BINARY_TASKS)


def parse_ef(text):
    value = parse_endpoint_ef(str(text))
    return value if value is not None else np.nan


def _has_negated_term(text, terms):
    return has_binary_negation(text, terms)


def parse_bin(text, task):
    t = str(text).strip().lower()
    if task == "afib_risk":
        if _has_negated_term(t, ("high risk", "elevated risk", "atrial fibrillation", "present")):
            return 0
        if (
            re.match(r"^yes\b", t)
            or "high risk" in t
            or "elevated risk" in t
            or "atrial fibrillation" in t
        ):
            return 1
    elif task == "structural_heart_disease":
        if _has_negated_term(t, ("structural heart disease", "present")):
            return 0
        if "present" in t or re.match(r"^yes\b", t):
            return 1
    elif task == "acs_severity":
        if _has_negated_term(
            t,
            ("acute coronary occlusion", "acute coronary artery occlusion", "present"),
        ):
            return 0
        if (
            re.match(r"^yes\b", t)
            or "acute coronary occlusion" in t
            or "acute coronary artery occlusion" in t
        ):
            return 1
    return np.nan


def boot_ci(fn, *arrs, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrs[0])
    v = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            val = fn(*[a[idx] for a in arrs])
            if val == val:
                v.append(val)
        except Exception:
            pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def _fail(message):
    raise SystemExit(f"[score] failed: {message}")


def _require_finite(label, *values):
    if not all(np.isfinite(float(value)) for value in values):
        _fail(f"{label} produced a nonfinite metric or confidence interval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", help="glob or path to generated CSV(s)")
    ap.add_argument("--tag", default="model")
    ap.add_argument(
        "--expected-category",
        action="append",
        choices=SUPPORTED_CATEGORIES,
        default=[],
        help="deterministic category that must yield one valid metric",
    )
    a = ap.parse_args()
    files = glob.glob(a.paths) if any(c in a.paths for c in "*?[") else [a.paths]
    if not files or any(not Path(file_name).is_file() for file_name in files):
        _fail("no input CSV files matched")
    try:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    except Exception as exc:
        _fail(f"could not read input CSV files: {exc}")
    required_columns = {"generation", "ground_truth", "prompt_category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        _fail(f"input is missing required columns: {sorted(missing_columns)}")
    dedup_cols = [c for c in ("waveform_name", "question", "json_key", "prompt") if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)
    print(f"[{a.tag}] {len(df):,} rows from {len(files)} file(s)")

    configured_categories = list(dict.fromkeys(a.expected_category))
    observed_categories = set(df["prompt_category"].astype(str))
    required_categories = configured_categories or [
        category for category in SUPPORTED_CATEGORIES if category in observed_categories
    ]
    if not required_categories:
        _fail("no supported deterministic score categories were present")
    missing_categories = [
        category for category in required_categories if category not in observed_categories
    ]
    if missing_categories:
        _fail(f"expected score categories are missing: {missing_categories}")
    valid_metric_count = 0

    # ---- LVEF: proper AUROC from parsed EF ----
    if "lvef" in required_categories:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            _fail(f"lvef metric dependency import failed: {exc}")
        s = df[df["prompt_category"] == "lvef"].copy()
        s["gen_ef"] = s["generation"].map(parse_ef)
        s["gt_ef"] = s["ground_truth"].map(parse_ef)
        ok = s["gen_ef"].notna() & s["gt_ef"].notna()
        s = s[ok]
        y = (s["gt_ef"].values <= 40).astype(int)
        positive_count = int(y.sum())
        negative_count = int(len(y) - positive_count)
        if positive_count < 5 or negative_count < 5:
            _fail(
                "lvef requires at least 5 parsed examples from each ground-truth "
                f"class, got n={len(y)}, pos={positive_count}, neg={negative_count}"
            )
        score = -s["gen_ef"].values
        au = roc_auc_score(y, score)
        lo, hi = boot_ci(lambda yy, ss: roc_auc_score(yy, ss), y, score)
        mae = float(np.abs(s["gen_ef"].values - s["gt_ef"].values).mean())
        acc = float(((score > np.median(score)).astype(int) == y).mean())
        _require_finite("lvef", au, lo, hi, mae, acc)
        print(
            f"  LVEF<=40   AUROC {au:.2f} (95% CI {lo:.2f}-{hi:.2f}) | "
            f"MAE {mae:.1f} EF-pts | n={len(y)} pos={positive_count}"
        )
        valid_metric_count += 1

    # ---- AFib / SHD / ACS: binary rule -> balanced accuracy ----
    for task, label in BINARY_TASKS.items():
        if task not in required_categories:
            continue
        s = df[df["prompt_category"] == task].copy()
        s["p"] = s["generation"].map(lambda t: parse_bin(t, task))
        s["g"] = s["ground_truth"].map(lambda t: parse_bin(t, task))
        s = s[s["p"].notna() & s["g"].notna()]
        if len(s) < 20:
            _fail(f"{task} requires at least 20 parsed examples, got {len(s)}")
        p = s["p"].values.astype(int)
        g = s["g"].values.astype(int)
        positive_count = int((g == 1).sum())
        negative_count = int((g == 0).sum())
        if positive_count == 0 or negative_count == 0:
            _fail(
                f"{task} requires both ground-truth classes, got "
                f"pos={positive_count}, neg={negative_count}"
            )
        tp = int(((p == 1) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        bacc = 0.5 * (sens + spec)
        acc = (tp + tn) / len(s)
        f1 = (2 * tp) / (2 * tp + fp + fn)
        lo, hi = boot_ci(
            lambda pp, gg: 0.5
            * (pp[gg == 1].mean() + (1 - pp[gg == 0]).mean()),
            p,
            g,
        )
        _require_finite(task, bacc, lo, hi, sens, spec, f1, acc)
        print(
            f"  {label:9s} bal-acc {bacc:.2f} "
            f"(95% CI {lo:.2f}-{hi:.2f}) | sens {sens:.2f} "
            f"spec {spec:.2f} F1 {f1:.2f} acc {acc:.2f} | "
            f"n={len(s)} pos={positive_count}"
        )
        valid_metric_count += 1

    if valid_metric_count < 1:
        _fail("no valid required deterministic metric was produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
