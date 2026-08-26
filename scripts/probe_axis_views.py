#!/usr/bin/env python3
"""Step 3.1 falsification probe - TIME vs CHANNEL as the token axis.

The encoder emits (B, 128 channels, 82 timesteps) and RVQ(dim=82) quantises the
last dim, so the bridge's kv positions are CHANNELS and no query can address a
temporal window. Hypothesis: that is why `json_interpretation` (enumeration of
localised findings) fails while binary endpoints work. Before any retrain, ask
the cheap question: is localised (ST/Q/T-wave) information more addressable
along the time axis than the channel axis of the SAME frozen post-quant tensor?

Probes on the frozen x1_split winner, patient-grouped split, identical data
path to scripts/probe_tokenizer.py:

  flat        logistic on all 128x82 dims - information ceiling, identical
              under transposition (axis choice cannot change it)
  marg_time   logistic on the 82-d time profile   (mean over channel tokens)
  marg_chan   logistic on the 128-d channel profile (mean over time tokens)
  mil_chan_*  linear scorer per CHANNEL token (82-d feature) + LSE/max pool -
              "is there a channel whose time profile flags the label?"
  mil_time_*  linear scorer per TIME token (128-d feature) + LSE/max pool -
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
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_TOKENIZER_FILE = Path(
    os.environ.get("PROBE_TOKENIZER_PATH", str(ROOT / "scripts" / "probe_tokenizer.py"))
).expanduser().resolve()
spec = importlib.util.spec_from_file_location("probe_tokenizer", PROBE_TOKENIZER_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load probe_tokenizer module")
probe_tokenizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_tokenizer)

from utils.constants import ECG_PATTERNS  # noqa: E402
from utils.artifact_provenance import (  # noqa: E402
    atomic_savez_compressed,
    atomic_write_json,
    canonical_sha256,
    encode_manifest,
    file_identity,
    load_npz_if_current,
    ordered_files_identity,
    python_implementation_identity,
    require_finite_numeric_array,
    require_matching_provenance,
)
from utils.ecg_waveform import load_ecg_waveform  # noqa: E402
from utils.endpoint_labels import (  # noqa: E402
    encode_binary_labels,
    encode_endpoint_labels,
    require_binary_class_support,
    require_nonempty_label_battery,
)
from utils.patient_identity import normalize_patient_ids  # noqa: E402

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


def probe_implementation_identity():
    return python_implementation_identity(
        (
            Path(__file__),
            PROBE_TOKENIZER_FILE,
            ROOT / "scripts" / "train_tokenizer_aux.py",
            ROOT / "models",
            ROOT / "utils",
        ),
        runtime_packages=(
            "numpy",
            "pandas",
            "scikit-learn",
            "torch",
            "vector-quantize-pytorch",
        ),
    )


def probe_static_input_identity(checkpoint):
    return {
        "checkpoint": file_identity(checkpoint),
        "qa_parquet": file_identity(QA_TEST),
        "script": file_identity(__file__),
        "probe_tokenizer": file_identity(PROBE_TOKENIZER_FILE),
        "implementation": probe_implementation_identity(),
    }


def build_probe_cache_provenance(checkpoint, qa, paths, n, static_inputs=None):
    captured = static_inputs or probe_static_input_identity(checkpoint)
    return {
        "schema_version": 1,
        **captured,
        "ordered_cohort_sha256": canonical_sha256(
            qa[["waveform_name", "waveform_path_psa"]].astype(str).to_dict(orient="records")
        ),
        "n": int(n),
        "row_count": len(qa),
        "ordered_waveforms": ordered_files_identity(paths),
        "target_length": 2500,
        "num_leads": 12,
    }


def has_probe_class_support(y, valid, train_rows, test_rows):
    train_labels = y[train_rows][valid[train_rows]]
    test_labels = y[test_rows][valid[test_rows]]
    try:
        require_binary_class_support(
            train_labels,
            endpoint="probe label",
            cohort="probe train split",
        )
        require_binary_class_support(
            test_labels,
            endpoint="probe label",
            cohort="probe test split",
        )
    except ValueError:
        return False
    return int(train_labels.sum()) >= 20 and int(test_labels.sum()) >= 10


def build_probe_metrics_provenance(cache_provenance, group_col, groups, train_rows, test_rows):
    return {
        "schema_version": 1,
        "kind": "axis_probe_metrics",
        "feature_cache_provenance": cache_provenance,
        "group_col": str(group_col),
        "ordered_groups_sha256": canonical_sha256(groups.tolist()),
        "split": {
            "strategy": "GroupShuffleSplit",
            "train_size": 0.7,
            "random_state": 42,
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "row_indices_sha256": canonical_sha256(
                {"train": train_rows.tolist(), "test": test_rows.tolist()}
            ),
        },
    }


def macro_probe_scores(results, labels, probe_names):
    return {
        probe: round(
            float(np.mean([results["diagnostic"][label][probe] for label in labels])),
            4,
        )
        for probe in probe_names
    }


@torch.no_grad()
def extract_full(enc, q, paths, dev, bs=64, length=2500):
    """Full post-quantization tensor (B, 128, 82) - no pooling."""
    out = []
    for i in range(0, len(paths), bs):
        sigs = []
        for p in paths[i:i + bs]:
            sigs.append(load_ecg_waveform(p, target_length=length))
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

    captured_static_inputs = probe_static_input_identity(a.ckpt)
    qa_columns = list(dict.fromkeys(
        [
            "waveform_name",
            "waveform_path_psa",
            "dataset",
            "deepecho_Visually_Estimated_EF",
            "echonext_shd_binary",
            "afib_label_5y",
            "acs_condition_is_acute",
            a.group_col,
            *ECG_PATTERNS,
        ]
    ))
    qa = pd.read_parquet(QA_TEST, columns=qa_columns).drop_duplicates("waveform_name")
    qa = qa.head(a.n).reset_index(drop=True)
    print(f"[{a.tag}] axis-view probe on {len(qa):,} ECGs")

    out_dir = Path(a.out)
    cache = out_dir / f"z_cache_{a.tag}_n{len(qa)}.npz"
    paths = qa["waveform_path_psa"].astype(str).tolist()
    require_matching_provenance(
        captured_static_inputs,
        probe_static_input_identity(a.ckpt),
        artifact="axis probe cohort",
    )
    cache_provenance = build_probe_cache_provenance(
        a.ckpt,
        qa,
        paths,
        a.n,
        static_inputs=captured_static_inputs,
    )
    cached = load_npz_if_current(cache, cache_provenance)
    if cached is not None:
        cached_handle = cached
        try:
            Z = cached["features"].copy()
            require_finite_numeric_array(
                Z,
                artifact="axis probe feature cache",
                expected_shape=(len(qa), 128, 82),
            )
            print(f"  cache hit: {cache} {Z.shape}")
        except (KeyError, ValueError) as exc:
            print(f"  ignoring invalid cache: {exc}")
            cached = None
        finally:
            cached_handle.close()
    if cached is None:
        enc, q = probe_tokenizer.load_models(a.ckpt, dev)
        t = time.time()
        Z = extract_full(enc, q, paths, dev)
        print(f"  extracted {Z.shape} in {(time.time() - t) / 60:.1f} min")
        require_finite_numeric_array(
            Z,
            artifact="extracted axis probe features",
            expected_shape=(len(qa), 128, 82),
        )
        require_matching_provenance(
            cache_provenance,
            build_probe_cache_provenance(a.ckpt, qa, paths, a.n),
            artifact="axis probe feature cache",
        )
        atomic_savez_compressed(
            cache,
            features=Z,
            provenance=encode_manifest(cache_provenance),
        )
        del enc, q
        torch.cuda.empty_cache()
    assert Z.shape[1:] == (128, 82), f"unexpected geometry {Z.shape}"
    Z = Z.astype(np.float32)

    g = normalize_patient_ids(
        qa[a.group_col],
        source=f"axis probe group column {a.group_col}",
    ).to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(Z, groups=g))
    overlap = set(g[tr]).intersection(g[te])
    if overlap:
        raise RuntimeError("axis probe patient-grouped split contains patient overlap")
    print("  patient-grouped split: overlap=0")
    metrics_provenance = build_probe_metrics_provenance(
        cache_provenance,
        a.group_col,
        g,
        tr,
        te,
    )

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
        encoded = encode_binary_labels(qa[lab], label_name=f"diagnostic {lab}")
        ok = np.isfinite(encoded)
        y = np.nan_to_num(encoded).astype(int)
        if has_probe_class_support(y, ok, tr, te):
            labels[lab] = (y, ok)
    bins = {
        "lvef_le40": ("deepecho_Visually_Estimated_EF", "lvef_lte_40"),
        "shd": ("echonext_shd_binary", "shd"),
        "afib_5y": ("afib_label_5y", "incident_afib_5y"),
        "acs_acute": ("acs_condition_is_acute", "acute_coronary_occlusion"),
    }
    clinical = {}
    for name, (col, endpoint) in bins.items():
        encoded = encode_endpoint_labels(qa[col], endpoint)
        ok = np.isfinite(encoded)
        y = np.nan_to_num(encoded).astype(int)
        if has_probe_class_support(y, ok, tr, te):
            clinical[name] = (y, ok)

    probe_names = [p[0] for p in PROBES]
    res = {"tag": a.tag, "n": int(len(qa)), "views": probe_names,
           "provenance": metrics_provenance,
           "diagnostic": {}, "clinical": {}}
    t0 = time.time()
    for li, (lab, (y, ok)) in enumerate(labels.items()):
        tr_sub = np.flatnonzero(ok[tr])
        te_sub = np.flatnonzero(ok[te])
        y_tr = y[tr][ok[tr]]
        y_te = y[te][ok[te]]
        row = {}
        for pname, fkey, kind in PROBES:
            Xtr, Xte = feats[fkey]
            s = _fit(Xtr[tr_sub], y_tr, Xte[te_sub], kind, dev, seed=li)
            row[pname] = round(float(roc_auc_score(y_te, s)), 4)
        row["n_pos"] = int(y_te.sum())
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

    localised = [label for label in labels if any(key in label for key in LOCALISED_KEYS)]
    global_labels = [label for label in labels if label not in localised]
    require_nonempty_label_battery(list(labels), battery="axis probe diagnostic battery")
    require_nonempty_label_battery(localised, battery="axis probe localised battery")
    require_nonempty_label_battery(global_labels, battery="axis probe global battery")
    res["macro_all"] = macro_probe_scores(res, list(labels), probe_names)
    res["macro_localised"] = macro_probe_scores(res, localised, probe_names)
    res["macro_global"] = macro_probe_scores(res, global_labels, probe_names)
    res["localised_labels"] = localised
    print(f"\n  fits done in {(time.time() - t0) / 60:.1f} min")
    print(f"  MACRO all ({len(labels)}):        ", res["macro_all"])
    print(f"  MACRO localised ({len(localised)}):  ", res["macro_localised"])
    print(f"  MACRO global ({len(global_labels)}):     ", res["macro_global"])
    for name, row in res["clinical"].items():
        print(f"  CLINICAL {name:10s}", {k: v for k, v in row.items() if k != 'n_pos'})

    path = out_dir / f"axis_probe_{a.tag}.json"
    require_matching_provenance(
        cache_provenance,
        build_probe_cache_provenance(a.ckpt, qa, paths, a.n),
        artifact="axis probe metrics",
    )
    require_matching_provenance(
        metrics_provenance,
        build_probe_metrics_provenance(cache_provenance, a.group_col, g, tr, te),
        artifact="axis probe grouping",
    )
    atomic_write_json(path, res)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
