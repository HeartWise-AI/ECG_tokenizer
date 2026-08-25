#!/usr/bin/env python3
"""Step 3.1 falsification probe — TIME vs CHANNEL as the token axis.

The encoder emits (B, 128 channels, 82 timesteps) and RVQ(dim=82) quantises the
last dim, so the bridge's kv positions are CHANNELS and no query can address a
temporal window. Hypothesis: that is why `json_interpretation` (enumeration of
localised findings) fails while binary endpoints work. Before any retrain, ask
the cheap question: is localised (ST/Q/T-wave) information more addressable
along the time axis than the channel axis of the SAME frozen post-quant tensor?

Probes on the frozen x1_split winner, patient-grouped split, identical data
path to scripts/probe_tokenizer.py:

  flat        logistic on all 128x82 dims — information ceiling, identical
              under transposition (axis choice cannot change it)
  marg_time   logistic on the 82-d time profile   (mean over channel tokens)
  marg_chan   logistic on the 128-d channel profile (mean over time tokens)
  mil_chan_*  linear scorer per CHANNEL token (82-d feature) + LSE/max pool —
              "is there a channel whose time profile flags the label?"
  mil_time_*  linear scorer per TIME token (128-d feature) + LSE/max pool —
              "is there a time slice whose channel state flags the label?"

MIL probes are the attention analog: a Q-Former query soft-selects kv
positions the same way LSE/max selects tokens. If mil_time > mil_chan on the
localised battery, time-indexed kv positions (Step 3.2) should help; if not,
the hypothesis is weakened and the expensive rebuild is deprioritised.

  python scripts/probe_axis_views.py --ckpt checkpoints/x1_split/tokenizer_aux_final.pt \
      --tag X1_SPLIT --device 0 --n 10000
"""
import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, "/volume/ECG_tokenizer")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

spec = importlib.util.spec_from_file_location(
    "probe_tokenizer", "/volume/ECG_tokenizer/scripts/probe_tokenizer.py")
probe_tokenizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_tokenizer)

from utils.constants import ECG_PATTERNS  # noqa: E402

QA_TEST = probe_tokenizer.QA_TEST

LOCALISED_KEYS = ("ST elevation", "ST depression", "ST upslopping", "ST downslopping",
                  "Q wave", "T wave inversion")

PROBES = [
    ("flat", "flat", "linear"),
    ("marg_time", "marg_time", "linear"),
    ("marg_chan", "marg_chan", "linear"),
    ("mil_chan_lse", "chan_tok", "lse"),
    ("mil_chan_max", "chan_tok", "max"),
    ("mil_time_lse", "time_tok", "lse"),
    ("mil_time_max", "time_tok", "max"),
]


@torch.no_grad()
def extract_full(enc, q, paths, dev, bs=64, length=2500):
    """Full post-quantization tensor (B, 128, 82) — no pooling."""
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
            sigs.append(np.transpose(s[:length], (1, 0)))
        x = torch.from_numpy(np.stack(sigs)).to(dev)
        z, _, _ = q(enc(x))
        out.append(z.detach().to(torch.float16).cpu().numpy())
        if i % (bs * 50) == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


def _std_gpu(train, test, dev):
    """Standardize over rows (and tokens, if 3D) of the train split; return GPU tensors."""
    flat_dims = (0,) if train.ndim == 2 else (0, 1)
    mu = train.mean(axis=flat_dims, keepdims=True)
    sd = train.std(axis=flat_dims, keepdims=True) + 1e-6
    return (torch.as_tensor((train - mu) / sd, dtype=torch.float32, device=dev),
            torch.as_tensor((test - mu) / sd, dtype=torch.float32, device=dev))


def _fit(Xtr, ytr, Xte, kind, dev, steps=400, lr=0.05, wd=1e-3, seed=0):
    """Logistic (2D input) or MIL linear-scorer probe (3D input, lse/max pooling)."""
    torch.manual_seed(seed)
    y = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    d = Xtr.shape[-1]
    w = torch.zeros(d, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    pos = max(float(y.sum().item()), 1.0)
    pos_weight = torch.tensor(min((len(y) - pos) / pos, 100.0), device=dev)

    def scores(X):
        if kind == "linear":
            return X @ w + b
        s = X @ w  # (B, T)
        if kind == "max":
            return s.amax(dim=1) + b
        return torch.logsumexp(s, dim=1) - np.log(X.shape[1]) + b

    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores(Xtr), y, pos_weight=pos_weight) + wd * (w ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return scores(Xte).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/x1_split/tokenizer_aux_final.pt")
    ap.add_argument("--tag", default="X1_SPLIT")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--group_col", default="new_PatientID")
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/tokenizer_axis_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", a.device)

    qa = pd.read_parquet(QA_TEST, columns=(
        ["waveform_name", "waveform_path_psa", "dataset", "deepecho_Visually_Estimated_EF",
         "echonext_shd_binary", "afib_label_5y", "acs_condition_is_acute", "new_PatientID"]
        + ECG_PATTERNS)).drop_duplicates("waveform_name")
    qa = qa.head(a.n).reset_index(drop=True)
    print(f"[{a.tag}] axis-view probe on {len(qa):,} ECGs")

    cache = f"{a.out}/z_cache_{a.tag}_n{len(qa)}.npy"
    if os.path.exists(cache):
        Z = np.load(cache)
        print(f"  cache hit: {cache} {Z.shape}")
    else:
        enc, q = probe_tokenizer.load_models(a.ckpt, dev)
        t = time.time()
        Z = extract_full(enc, q, qa["waveform_path_psa"].tolist(), dev)
        print(f"  extracted {Z.shape} in {(time.time() - t) / 60:.1f} min")
        np.save(cache, Z)
        del enc, q
        torch.cuda.empty_cache()
    assert Z.shape[1:] == (128, 82), f"unexpected geometry {Z.shape}"
    Z = Z.astype(np.float32)

    g = qa[a.group_col].astype(str).values
    tr, te = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(Z, groups=g))
    print(f"  patient-grouped split: overlap={len(set(g[tr]) & set(g[te]))} (must be 0)")

    flat = Z.reshape(len(Z), -1)
    time_tok = np.ascontiguousarray(Z.transpose(0, 2, 1))  # tokens=timesteps, feature=channels
    feats = {
        "flat": _std_gpu(flat[tr], flat[te], dev),
        "marg_time": _std_gpu(Z.mean(axis=1)[tr], Z.mean(axis=1)[te], dev),   # (B, 82)
        "marg_chan": _std_gpu(Z.mean(axis=2)[tr], Z.mean(axis=2)[te], dev),   # (B, 128)
        "chan_tok": _std_gpu(Z[tr], Z[te], dev),                              # (B, 128, 82)
        "time_tok": _std_gpu(time_tok[tr], time_tok[te], dev),                # (B, 82, 128)
    }

    labels = {}
    for lab in ECG_PATTERNS:
        y = pd.to_numeric(qa[lab], errors="coerce").fillna(0).ge(1).astype(int).values
        if y[tr].sum() >= 20 and y[te].sum() >= 10:
            labels[lab] = y
    bins = {"lvef_le40": ("deepecho_Visually_Estimated_EF", lambda v: (v <= 40).astype(int)),
            "shd": ("echonext_shd_binary", lambda v: (v >= 1).astype(int)),
            "afib_5y": ("afib_label_5y", lambda v: (v >= 1).astype(int)),
            "acs_acute": ("acs_condition_is_acute", lambda v: (v >= 1).astype(int))}
    clinical = {}
    for name, (col, fn) in bins.items():
        v = pd.to_numeric(qa[col], errors="coerce").values
        ok = np.isfinite(v)
        y = np.where(ok, fn(np.nan_to_num(v)), 0).astype(int)
        if y[tr[ok[tr]]].sum() >= 20 and y[te[ok[te]]].sum() >= 10:
            clinical[name] = (y, ok)

    probe_names = [p[0] for p in PROBES]
    res = {"tag": a.tag, "n": int(len(qa)), "views": probe_names,
           "diagnostic": {}, "clinical": {}}
    t0 = time.time()
    for li, (lab, y) in enumerate(labels.items()):
        row = {}
        for pname, fkey, kind in PROBES:
            Xtr, Xte = feats[fkey]
            s = _fit(Xtr, y[tr], Xte, kind, dev, seed=li)
            row[pname] = round(float(roc_auc_score(y[te], s)), 4)
        row["n_pos"] = int(y.sum())
        res["diagnostic"][lab] = row
        if li % 10 == 0:
            print(f"  [{li + 1}/{len(labels)}] {lab[:40]:42s} " +
                  " ".join(f"{k}={row[k]:.3f}" for k in probe_names), flush=True)

    for name, (y, ok) in clinical.items():
        tr_sub = np.flatnonzero(ok[tr])   # row positions within the (already-split) train features
        te_sub = np.flatnonzero(ok[te])
        y_tr = y[tr][ok[tr]]
        y_te = y[te][ok[te]]
        row = {}
        for pname, fkey, kind in PROBES:
            Xtr, Xte = feats[fkey]
            s = _fit(Xtr[tr_sub], y_tr, Xte[te_sub], kind, dev, seed=99)
            row[pname] = round(float(roc_auc_score(y_te, s)), 4)
        row["n_pos"] = int(y_te.sum())
        res["clinical"][name] = row

    def _macro(subset):
        return {p: round(float(np.mean([res["diagnostic"][l][p] for l in subset])), 4)
                for p in probe_names}

    localised = [l for l in labels if any(k in l for k in LOCALISED_KEYS)]
    global_labels = [l for l in labels if l not in localised]
    res["macro_all"] = _macro(list(labels))
    res["macro_localised"] = _macro(localised)
    res["macro_global"] = _macro(global_labels)
    res["localised_labels"] = localised
    print(f"\n  fits done in {(time.time() - t0) / 60:.1f} min")
    print(f"  MACRO all ({len(labels)}):        ", res["macro_all"])
    print(f"  MACRO localised ({len(localised)}):  ", res["macro_localised"])
    print(f"  MACRO global ({len(global_labels)}):     ", res["macro_global"])
    for name, row in res["clinical"].items():
        print(f"  CLINICAL {name:10s}", {k: v for k, v in row.items() if k != 'n_pos'})

    path = f"{a.out}/axis_probe_{a.tag}.json"
    json.dump(res, open(path, "w"), indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
