#!/usr/bin/env python3
"""Deterministic scoring of generated ECG QA answers — no LLM judge needed.

Parses the model's generated text for the endpoints that have a machine-checkable answer and
scores them against the parsed ground truth in the same CSV:
  - LVEF:   parse "ejection fraction is N%"  -> AUROC(EF<=40, score=-EF) + MAE + acc
  - AFib / SHD / ACS: parse binary Yes/No    -> balanced-acc (=AUROC for a binary rule), sens, spec, F1, n

Works on a glob of shard checkpoint CSVs (incremental) or the merged CSV. Columns expected:
  generation, ground_truth, prompt_category.

  python scripts/score_deterministic.py "<glob-or-path>" [--tag NEW]
"""
import argparse, glob, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

EF_RE = re.compile(r"ejection fraction is\s*(\d+(?:\.\d+)?)\s*%", re.I)


def parse_ef(text):
    m = EF_RE.search(str(text))
    return float(m.group(1)) if m else np.nan


def parse_bin(text, task):
    t = str(text).strip().lower()
    if task == "afib_risk":
        if t.startswith("yes") or "high risk" in t: return 1
        if t.startswith("low") or "unlikely" in t or t.startswith("no"): return 0
    elif task == "structural_heart_disease":
        if "present" in t or t.startswith("yes"): return 1
        if "no structural" in t or "not present" in t or t.startswith("no"): return 0
    elif task == "acs_severity":
        if t.startswith("yes") or "acute coronary occlusion" in t: return 1
        if t.startswith("no"): return 0
    return np.nan


def boot_ci(fn, *arrs, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(arrs[0]); v = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            val = fn(*[a[idx] for a in arrs])
            if val == val: v.append(val)
        except Exception:
            pass
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", help="glob or path to generated CSV(s)")
    ap.add_argument("--tag", default="model")
    a = ap.parse_args()
    files = glob.glob(a.paths) if any(c in a.paths for c in "*?[") else [a.paths]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(
        subset=[c for c in ("waveform_name", "question") if c in pd.read_csv(files[0], nrows=1).columns])
    print(f"[{a.tag}] {len(df):,} rows from {len(files)} file(s)")

    # ---- LVEF: proper AUROC from parsed EF ----
    s = df[df["prompt_category"] == "lvef"].copy()
    if len(s):
        s["gen_ef"] = s["generation"].map(parse_ef)
        s["gt_ef"] = s["ground_truth"].map(parse_ef)
        ok = s["gen_ef"].notna() & s["gt_ef"].notna()
        s = s[ok]
        y = (s["gt_ef"].values <= 40).astype(int)
        score = -s["gen_ef"].values                      # lower EF -> higher predicted risk
        if y.sum() >= 5 and (len(y) - y.sum()) >= 5:
            au = roc_auc_score(y, score)
            lo, hi = boot_ci(lambda yy, ss: roc_auc_score(yy, ss), y, score)
            mae = float(np.abs(s["gen_ef"].values - s["gt_ef"].values).mean())
            acc = float(((score > np.median(score)).astype(int) == y).mean())
            print(f"  LVEF<=40   AUROC {au:.2f} (95% CI {lo:.2f}-{hi:.2f}) | MAE {mae:.1f} EF-pts | n={len(y)} pos={int(y.sum())}")
        else:
            print(f"  LVEF: too few pos/neg yet (n={len(y)}, pos={int(y.sum())})")

    # ---- AFib / SHD / ACS: binary rule -> balanced accuracy ----
    for task, label in [("afib_risk", "AFib-5y"), ("structural_heart_disease", "SHD"), ("acs_severity", "ACS-acute")]:
        s = df[df["prompt_category"] == task].copy()
        if not len(s): continue
        s["p"] = s["generation"].map(lambda t: parse_bin(t, task))
        s["g"] = s["ground_truth"].map(lambda t: parse_bin(t, task))
        s = s[s["p"].notna() & s["g"].notna()]
        if len(s) < 20:
            print(f"  {label}: n={len(s)} (too few)"); continue
        p = s["p"].values.astype(int); g = s["g"].values.astype(int)
        tp = int(((p == 1) & (g == 1)).sum()); tn = int(((p == 0) & (g == 0)).sum())
        fp = int(((p == 1) & (g == 0)).sum()); fn = int(((p == 0) & (g == 1)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        bacc = 0.5 * (sens + spec); acc = (tp + tn) / len(s)
        f1 = tp / max(tp + 0.5 * (fp + fn), 1e-9)
        lo, hi = boot_ci(lambda pp, gg: 0.5 * (pp[gg == 1].mean() + (1 - pp[gg == 0]).mean()) if (gg == 1).any() and (gg == 0).any() else np.nan, p, g)
        print(f"  {label:9s} bal-acc {bacc:.2f} (95% CI {lo:.2f}-{hi:.2f}) | sens {sens:.2f} spec {spec:.2f} F1 {f1:.2f} acc {acc:.2f} | n={len(s)} pos={int(g.sum())}")


if __name__ == "__main__":
    main()
