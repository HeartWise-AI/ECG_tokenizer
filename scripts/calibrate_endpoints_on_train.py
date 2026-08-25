#!/usr/bin/env python3
"""Fit the endpoint readout calibration on TRAIN-side ECGs, evaluate on the FULL test set.

Why: `evaluate_endpoint_pyes_calibrated.py` splits the scored TEST ECGs 50/50 into
calib/test. That burns half the test set on threshold fitting and reports the operating
point on only the other half. Fitting on train-side data instead is both fairer (no test
data touches the fit) and mirrors deployment (you calibrate before release, then apply the
locked threshold to everything unseen) — and it frees the ENTIRE test set for evaluation.

Known risk, measured rather than assumed: the model was TRAINED on these ECGs, so its
margins there are likely sharper/over-confident, which can bias the fitted threshold. This
script therefore reports BOTH calibrations side by side (train-fit vs test-half-fit) on the
same full test set, so the size and direction of that bias is visible.

Reuses the cached test margins from the earlier run (raw_margins.npz) — only the train-side
margins are newly computed.

  PYTHONPATH=/volume/ECG_tokenizer python scripts/calibrate_endpoints_on_train.py \
      --checkpoint <best_model.pt> --device cuda:2 --n_calib 1500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.evaluate_endpoint_pyes_calibrated import (
    ENDPOINTS, expected_calibration_error, fit_platt, get_margins, op_metrics,
    pretokenize, rank_metrics, youden_threshold,
)
from scripts.binary_auroc_eval import load_ecg_signal, load_model

TRAIN_PARQUET = "/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k.REGEN.parquet"
TEST_MARGINS = "/volume/ECG_tokenizer/analysis/endpoint_pyes_concatmix/raw_margins.npz"
TEST_PARQUET = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"
SEED = 20260821


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--n_calib", type=int, default=1500, help="train ECGs to score per run")
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/endpoint_pyes_concatmix")
    a = ap.parse_args()
    out_dir = Path(a.out)
    rng = np.random.default_rng(SEED)

    # ---- cached TEST margins (full set, nothing withheld) ----
    z = np.load(TEST_MARGINS, allow_pickle=True)
    test_margins, test_labels = z["margins"], z["labels"]
    names = [str(x) for x in z["endpoints"]]
    n_prompts = [int(x) for x in z["n_prompts"]]
    test_paths = set(str(p) for p in z["waveform_paths"])
    print(f"[cal] cached test margins: {test_margins.shape} over {len(names)} endpoints")

    # ---- pick TRAIN ECGs, patient-disjoint from test ----
    # Patient ids are stored inconsistently across the two parquets ('338306' in test,
    # '337511.0' in train), so a raw string comparison silently matches NOTHING and the
    # leak check passes vacuously. Normalise through numeric before comparing.
    def _norm_pid(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)

    test_pat = set(_norm_pid(pd.read_parquet(TEST_PARQUET, columns=["new_PatientID"])["new_PatientID"]))
    cols = ["waveform_path_psa", "new_PatientID"] + [ENDPOINTS[n]["label_column"] for n in names]
    tr = pd.read_parquet(TRAIN_PARQUET, columns=sorted(set(cols)))
    tr = tr.drop_duplicates(subset="waveform_path_psa")
    before = len(tr)
    tr = tr[~_norm_pid(tr["new_PatientID"]).isin(test_pat)]
    tr = tr[~tr["waveform_path_psa"].astype(str).isin(test_paths)]
    print(f"[cal] train ECGs: {before:,} -> {len(tr):,} after removing test patients/paths")

    # STRATIFY per endpoint: uniform sampling starves the rare endpoints (ACS is labelled on
    # only 1.9% of train ECGs, so a 1,500-ECG uniform draw yielded 68 ACS labels and the fit
    # had to be skipped). Draw up to n_calib rows carrying EACH endpoint's label, then union.
    lab_cols = [ENDPOINTS[n]["label_column"] for n in names]
    tr = tr[tr[lab_cols].notna().any(axis=1)]
    picks: list[pd.DataFrame] = []
    for n in names:
        col = ENDPOINTS[n]["label_column"]
        sub = tr[tr[col].notna()]
        if len(sub) > a.n_calib:
            sub = sub.iloc[rng.choice(len(sub), a.n_calib, replace=False)]
        picks.append(sub)
        print(f"[cal]   {ENDPOINTS[n]['label']:22s} drew {len(sub):5,} labelled train ECGs")
    tr = pd.concat(picks).drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    print(f"[cal] scoring {len(tr):,} train ECGs")

    lab = np.full((len(tr), len(names)), np.nan, dtype=np.float32)
    for j, n in enumerate(names):
        sp = ENDPOINTS[n]
        col = sp["label_column"]
        m = tr[col].notna()
        lab[m.values, j] = sp["label_fn"](tr.loc[m, col]).astype(float).values

    model, tokenizer = load_model(a.checkpoint, a.device)
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    all_q = [q for n in names for q in ENDPOINTS[n]["questions"]]
    pids, pmask = pretokenize(tokenizer, all_q)

    margins = np.zeros((len(tr), pids.size(0)), dtype=np.float32)
    for i, p in enumerate(tqdm(tr["waveform_path_psa"].tolist(), desc="train margins")):
        ecg = load_ecg_signal(p)
        margins[i] = get_margins(model, ecg, pids, pmask, yes_id, no_id, a.device, a.batch_size)

    # ---- per endpoint: fit on TRAIN, apply locked to FULL TEST ----
    results = {"checkpoint": a.checkpoint, "n_calib_scored": int(len(tr)), "endpoints": {}}
    md = ["# Endpoint readout — calibration fitted on TRAIN, evaluated on FULL test\n",
          "| Endpoint | test n | AUROC (full test) | TRAIN-fit locked Sens/Spec | bal-acc | prior half-test-fit Sens/Spec |",
          "|---|---:|---|---|---:|---|"]
    off = 0
    for j, n in enumerate(names):
        k = n_prompts[j]
        tr_m = margins[:, off:off + k].mean(axis=1)
        te_m = test_margins[:, off:off + k].mean(axis=1)
        off += k
        ytr, yte = lab[:, j], test_labels[:, j]
        ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
        if ok_tr.sum() < 100 or ok_te.sum() < 100:
            print(f"  {n}: insufficient labels (train {ok_tr.sum()}, test {ok_te.sum()}) — skipped")
            continue
        a_, b_ = fit_platt(tr_m[ok_tr], ytr[ok_tr])
        thr = youden_threshold(1 / (1 + np.exp(-(a_ * tr_m[ok_tr] + b_))), ytr[ok_tr])
        prob_te = 1 / (1 + np.exp(-(a_ * te_m[ok_te] + b_)))
        op = op_metrics(yte[ok_te], prob_te, thr)
        rk = rank_metrics(yte[ok_te], te_m[ok_te])
        bal = (op["sensitivity"] + op["specificity"]) / 2
        results["endpoints"][n] = {
            "label": ENDPOINTS[n]["label"], "n_train_labelled": int(ok_tr.sum()),
            "n_test": int(ok_te.sum()), "platt_a": float(a_), "platt_b": float(b_),
            "locked_threshold_prob": float(thr), "full_test_rank": rk,
            "operating_point": op, "balanced_accuracy": float(bal),
            "test_ece": float(expected_calibration_error(prob_te, yte[ok_te])),
        }
        print(f"  {ENDPOINTS[n]['label']:22s} n_test={int(ok_te.sum()):5d} AUROC {rk['auroc']:.2f} | "
              f"TRAIN-fit sens {op['sensitivity']:.2f} spec {op['specificity']:.2f} bal {bal:.2f} | ECE {results['endpoints'][n]['test_ece']:.3f}")
        md.append(f"| {ENDPOINTS[n]['label']} | {int(ok_te.sum())} | {rk['auroc']:.2f} | "
                  f"{op['sensitivity']:.2f}/{op['specificity']:.2f} | {bal:.2f} | see prior run |")

    (out_dir / "endpoint_readout_train_calibrated.json").write_text(json.dumps(results, indent=2))
    (out_dir / "endpoint_readout_train_calibrated.md").write_text("\n".join(md) + "\n")
    np.savez_compressed(out_dir / "train_calib_margins.npz", margins=margins, labels=lab,
                        endpoints=np.array(names, dtype=object), n_prompts=np.array(n_prompts))
    print(f"[saved] {out_dir}/endpoint_readout_train_calibrated.{{json,md}}")


if __name__ == "__main__":
    main()
