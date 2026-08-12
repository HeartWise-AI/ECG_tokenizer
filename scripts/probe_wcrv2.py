#!/usr/bin/env python3
"""Frozen linear probe of the DeepECG-SSL v2 WCRv2 encoder (fairseq_signals wav2vec2_cmsc).

Runs the SAME harness as scripts/probe_tokenizer.py — same test parquet, same head(n) row
selection, same amplitude-preserved mV signals (waveform_path_psa -> adjusted_signals), same
labels (77 diagnoses + LVEF<=40 / SHD / AFib-5y / ACS / axis), same patient-grouped split
(GroupShuffleSplit on new_PatientID), same LogisticRegression/Ridge + bootstrap CIs — so the
WCRv2 row is directly comparable to the tokenizer table. Only the encoder differs: WCRv2's
768-d transformer output, mean-pooled over time.

MUST run in the conda env that has fairseq_signals:
  cd /volume/DeepECG-SSL-finetune && \
  /opt/conda/bin/python /volume/ECG_tokenizer/scripts/probe_wcrv2.py --device 0 --tag WCRV2_PT
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/volume/DeepECG-SSL-finetune")  # fairseq_signals (resolves via repo dir, not egg-link)
sys.path.insert(0, "/volume/ECG_tokenizer")
import numpy as np, pandas as pd, torch
from utils.constants import ECG_PATTERNS
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

QA_TEST = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"
AUX_TR = "/volume/ECG_tokenizer/output/tokenizer_aux_targets_train.parquet"
WCRV2 = "/volume/DeepECG-SSL-finetune/data/ssl-amp-preserved/checkpoints-all/checkpoint250.pt"
PHYS = ["hr", "qrs_dur", "qtc", "pr", "qt"]


def load_wcrv2(dev):
    from fairseq_signals.utils import checkpoint_utils
    from fairseq_signals import tasks
    state = checkpoint_utils.load_checkpoint_to_cpu(WCRV2)
    cfg = state["cfg"]
    task = tasks.setup_task(cfg["task"])
    model = task.build_model(cfg["model"])
    model.load_state_dict(state["model"], strict=True)
    if hasattr(model, "remove_pretraining_modules"):
        model.remove_pretraining_modules()   # drop quantizer/project_q/final_proj — keep encoder
    return model.to(dev).eval()


@torch.no_grad()
def extract(model, paths, dev, bs=64, length=2500):
    """WCRv2 encoder output, mean-pooled over time -> (N, 768). Signals are already mV
    (adjusted_signals), 250 Hz, 2500 samples; feed as (B, 12, 2500)."""
    out = []
    for i in range(0, len(paths), bs):
        sigs = []
        for p in paths[i:i + bs]:
            try:
                s = np.load(p).astype(np.float32)
            except Exception:
                s = np.zeros((length, 12), np.float32)
            if s.ndim == 3:
                s = s.squeeze(-1)
            if s.shape[0] > length:
                s = s[:: max(1, s.shape[0] // length), :]
            if s.shape[0] < length:
                s = np.pad(s, ((0, length - s.shape[0]), (0, 0)))
            sigs.append(np.transpose(s[:length], (1, 0)))            # (12, 2500)
        x = torch.from_numpy(np.stack(sigs)).to(dev)                 # (B, 12, 2500)
        res = model(source=x, padding_mask=None, mask=False, features_only=True)
        feats = res["x"]                                             # (B, T, 768)
        out.append(feats.mean(dim=1).float().cpu().numpy())          # mean-pool over time
        if i % (bs * 50) == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


def auroc_ci(y, s, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="WCRV2_PT")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--group_col", default="new_PatientID")
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/tokenizer_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", a.device)

    qa = pd.read_parquet(QA_TEST, columns=(
        ["waveform_name", "waveform_path_psa", "dataset", "deepecho_Visually_Estimated_EF",
         "echonext_shd_binary", "afib_label_5y", "acs_condition_is_acute", "new_PatientID"] + ECG_PATTERNS
    )).drop_duplicates("waveform_name")
    aux = pd.read_parquet(AUX_TR)
    key = "waveform_name" if "waveform_name" in aux.columns else "waveform_path_psa"
    qa = qa.merge(aux[[key] + [p for p in PHYS if p in aux.columns]], on=key, how="left")
    qa = qa.head(a.n).reset_index(drop=True)
    print(f"[{a.tag}] probing WCRv2 on {len(qa):,} ECGs | mix: {qa['dataset'].value_counts().to_dict()}")

    model = load_wcrv2(dev)
    print(f"  WCRv2 params: {sum(p.numel() for p in model.parameters()):,}")
    t = time.time()
    X = extract(model, qa["waveform_path_psa"].tolist(), dev)
    print(f"  features {X.shape} in {(time.time()-t)/60:.1f} min")
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)

    g = qa[a.group_col].astype(str).values
    tr, te = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(Xs, groups=g))
    print(f"  PATIENT-GROUPED split ({a.group_col}): {len(set(g[tr]) & set(g[te]))} patient overlap (should be 0)")

    res = {"tag": a.tag, "encoder": "WCRv2_wav2vec2_cmsc", "pool": "time_mean_768",
           "n": int(n), "diagnostic": {}, "physiologic": {}}

    # ---- diagnostic: 77 binary labels ----
    aucs = []
    for lab in ECG_PATTERNS:
        y = pd.to_numeric(qa[lab], errors="coerce").fillna(0).ge(1).astype(int).values
        if y[tr].sum() < 20 or y[te].sum() < 10:
            continue
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        s = m.predict_proba(Xs[te])[:, 1]
        au = roc_auc_score(y[te], s); aucs.append(au)
        res["diagnostic"][lab] = {"auroc": round(float(au), 4), "n_pos": int(y.sum())}
    res["diagnostic_macro_auroc"] = round(float(np.mean(aucs)), 4)
    print(f"  DIAGNOSTIC macro AUROC = {np.mean(aucs):.4f}  over {len(aucs)} labels")

    # ---- physiologic: regression (MAE) ----
    for p in [c for c in PHYS if c in qa.columns]:
        y = pd.to_numeric(qa[p], errors="coerce").values
        ok = np.isfinite(y)
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(tr2) < 200 or len(te2) < 100:
            continue
        m = Ridge(alpha=1.0).fit(Xs[tr2], y[tr2])
        pr = m.predict(Xs[te2])
        mae = float(np.abs(pr - y[te2]).mean())
        base = float(np.abs(y[tr2].mean() - y[te2]).mean())
        r = float(np.corrcoef(pr, y[te2])[0, 1])
        res["physiologic"][p] = {"mae": round(mae, 3), "mae_predict_mean": round(base, 3),
                                 "r": round(r, 3), "n_test": int(len(te2))}
        print(f"  PHYS {p:8s} MAE {mae:7.2f} (mean-baseline {base:7.2f})  r={r:.3f}  n={len(te2)}")

    # ---- clinical binaries: axis, LVEF<=40, SHD, AFib, ACS ----
    bins = {"axis_left": ("Left axis deviation", None), "axis_right": ("Right axis deviation", None),
            "lvef_le40": ("deepecho_Visually_Estimated_EF", lambda v: (v <= 40).astype(int)),
            "shd": ("echonext_shd_binary", lambda v: (v >= 1).astype(int)),
            "afib_5y": ("afib_label_5y", lambda v: (v >= 1).astype(int)),
            "acs_acute": ("acs_condition_is_acute", lambda v: (v >= 1).astype(int))}
    for name, (col, fn) in bins.items():
        if col not in qa.columns:
            continue
        v = pd.to_numeric(qa[col], errors="coerce").values
        ok = np.isfinite(v)
        y = (fn(v) if fn else (v >= 1).astype(int))
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(tr2) < 100 or y[te2].sum() < 10:
            continue
        m = LogisticRegression(max_iter=2000).fit(Xs[tr2], y[tr2])
        s = m.predict_proba(Xs[te2])[:, 1]
        au = roc_auc_score(y[te2], s); lo, hi = auroc_ci(y[te2], s)
        res["physiologic"][name] = {"auroc": round(float(au), 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "n_test": int(len(te2)), "n_pos": int(y[te2].sum())}
        print(f"  {name:10s} AUROC {au:.2f} (95% CI {lo:.2f}-{hi:.2f})  n={len(te2)} pos={int(y[te2].sum())}")

    json.dump(res, open(f"{a.out}/probe_{a.tag}.json", "w"), indent=2)
    print(f"[saved] {a.out}/probe_{a.tag}.json")


if __name__ == "__main__":
    main()
