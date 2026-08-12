#!/usr/bin/env python3
"""Build the auxiliary-target table for tokenizer training.

Joins physiologic targets (HR, intervals) onto the tokenizer training/val parquets by
waveform_name, cleans the uint16 sentinels, and emits one compact parquet keyed by
waveform_path_psa (the key the tokenizer dataset uses).

Targets are deliberately left as NaN where unavailable — the training loss is masked
per-sample per-head, so partial coverage is fine and is NOT imputed.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, sys, os

TOK = {
    "train": "/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_code15_train_updated.parquet",
    "val":   "/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet",
}
QA = ["/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.REGEN.parquet",
      "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"]
OUT = "/volume/ECG_tokenizer/output/tokenizer_aux_targets_{split}.parquet"

RAW = {  # source column -> canonical target name
    "rr_interval": "rr",
    "RestingECG_OriginalRestingECGMeasurements_VentricularRate": "vrate",
    "RestingECG_OriginalRestingECGMeasurements_PRInterval": "pr",
    "RestingECG_OriginalRestingECGMeasurements_QRSDuration": "qrs_dur",
    "RestingECG_OriginalRestingECGMeasurements_QTInterval": "qt",
    "RestingECG_OriginalRestingECGMeasurements_QTCorrected": "qtc",
}
# physiologic plausibility windows — anything outside becomes NaN (kills sentinels 65534/65535/29999/0)
VALID = {"rr": (200, 3000), "vrate": (20, 300), "pr": (60, 500),
         "qrs_dur": (40, 300), "qt": (200, 800), "qtc": (250, 700), "hr": (20, 300)}


def clean(s, lo, hi):
    v = pd.to_numeric(s, errors="coerce")           # interval cols are strings
    return v.where((v >= lo) & (v <= hi))            # out-of-range -> NaN


def main():
    qa = []
    for p in QA:
        cols = [c for c in ["waveform_name"] + list(RAW) if c in pq.read_schema(p).names]
        qa.append(pd.read_parquet(p, columns=cols))
    qa = pd.concat(qa, ignore_index=True).drop_duplicates("waveform_name")
    qa = qa.rename(columns=RAW)
    for t in [v for v in RAW.values() if v in qa.columns]:
        qa[t] = clean(qa[t], *VALID[t])
    # heart rate: prefer the device's ventricular rate, else derive from RR
    hr_rr = 60000.0 / qa["rr"] if "rr" in qa.columns else np.nan
    qa["hr"] = qa["vrate"] if "vrate" in qa.columns else np.nan
    qa["hr"] = qa["hr"].where(qa["hr"].notna(), hr_rr)
    qa["hr"] = clean(qa["hr"], *VALID["hr"])

    for split, path in TOK.items():
        tk = pd.read_parquet(path, columns=["waveform_name", "waveform_path_psa"]).drop_duplicates("waveform_path_psa")
        keep = ["waveform_name", "hr", "qrs_dur", "qtc", "pr", "qt"]
        m = tk.merge(qa[[c for c in keep if c in qa.columns]], on="waveform_name", how="left")
        out = OUT.format(split=split)
        m.to_parquet(out, index=False)
        n = len(m)
        print(f"[{split}] {n:,} rows -> {out}")
        for c in keep[1:]:
            if c in m.columns:
                v = m[c].notna()
                print(f"    {c:9s} {int(v.sum()):>9,}  ({v.mean()*100:5.1f}%)"
                      f"  mean {m.loc[v, c].mean():7.1f}" if v.any() else f"    {c}: none")


if __name__ == "__main__":
    main()
