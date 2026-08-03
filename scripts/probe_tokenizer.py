#!/usr/bin/env python3
"""Linear-probe evaluation of a FROZEN ECG tokenizer — diagnostic + physiologic.

This is the acceptance harness from goal.md. It answers: how much task-relevant
information survives into the tokenizer's representation? Run it on the production
reconstruction tokenizer (baseline) and on the aux-trained tokenizer, and compare.

Probes are fit on the pooled POST-QUANTIZATION representation — i.e. what the LLM can
actually see through the code path — with the encoder/quantizer frozen.

  python scripts/probe_tokenizer.py --ckpt <path|baseline> --tag baseline --device 0
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/volume/ECG_tokenizer")
import numpy as np, pandas as pd, torch, torch.nn as nn
import models  # noqa: F401 — populates registry
from utils.registry import ModelRegistry
from utils.constants import ECG_PATTERNS
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

BASELINE = "/media/data1/models/ECG_Tokenizer/ECG_Tokenizer_Reconstruction/tfq5q94l_20250622-004552/best_model_epoch_10.pt"
QA_TEST = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"
AUX_TR = "/volume/ECG_tokenizer/output/tokenizer_aux_targets_train.parquet"
PHYS = ["hr", "qrs_dur", "qtc", "pr", "qt"]


def _build_encoder(sd):
    """Rebuild whichever encoder the checkpoint was trained with.
    Aux checkpoints record their argv; enc_width>0 means the ScalableEncoder."""
    w = int((sd.get("args") or {}).get("enc_width", 0) or 0)
    if w > 0:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tta", "/volume/ECG_tokenizer/scripts/train_tokenizer_aux.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        print(f"  ScalableEncoder(width={w})")
        return m.ScalableEncoder(width=w)
    return ModelRegistry.get("Residual_Conv_Encoder")()


def load_models(ckpt, dev, nq=8, cb=512):
    enc = ModelRegistry.get("Residual_Conv_Encoder")()
    q = ModelRegistry.get("ECG_Tokenizer_Quantizer")(num_quantizers=nq, codebook_size=cb)
    if ckpt == "baseline":
        sd = torch.load(BASELINE, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        e = {k.split("encoder.", 1)[1]: v for k, v in sd.items() if ".encoder." in k or k.startswith("encoder.")}
        qq = {k.split("quantizer.", 1)[1]: v for k, v in sd.items() if ".quantizer." in k or k.startswith("quantizer.")}
        print(f"  baseline keys matched: encoder {len(e)}, quantizer {len(qq)}")
        if e: enc.load_state_dict(e, strict=False)
        if qq: q.load_state_dict(qq, strict=False)
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        enc = _build_encoder(sd)
        _nq = int((sd.get("args") or {}).get("num_quantizers", nq) or nq)
        if _nq != nq:
            q = ModelRegistry.get("ECG_Tokenizer_Quantizer")(num_quantizers=_nq, codebook_size=cb)
            print(f"  quantizer rebuilt: num_quantizers={_nq}")
        enc.load_state_dict(sd["encoder"]); q.load_state_dict(sd["quantizer"])
        print(f"  encoder params: {sum(p.numel() for p in enc.parameters()):,}")
        print(f"  loaded aux ckpt @ step {sd.get('step')}")
    return enc.to(dev).eval(), q.to(dev).eval()


def pool(z, mode):
    """z: (B, 128, 82). RVQ(dim=82) treats 128 as the token axis and 82 as the feature axis,
    so mean(dim=-1) collapses the FEATURES and is a poor probe input. Default concatenates
    token-axis mean+max (keeping the 82-d feature space) with the per-token profile."""
    if mode == "feat_mean":  return z.mean(dim=1)                                   # (B, 82)
    if mode == "tok_mean":   return z.mean(dim=-1)                                  # (B, 128)
    if mode == "flat":       return z.flatten(1)                                    # (B, 10496)
    return torch.cat([z.mean(dim=1), z.amax(dim=1), z.mean(dim=-1)], dim=1)         # (B, 292)


@torch.no_grad()
def extract(enc, q, paths, dev, bs=64, length=2500, mode="concat"):
    """Pooled post-quantization features — what the code path preserves."""
    out = []
    for i in range(0, len(paths), bs):
        sigs = []
        for p in paths[i:i + bs]:
            try:
                s = np.load(p).astype(np.float32)
            except Exception:
                s = np.zeros((length, 12), np.float32)
            if s.ndim == 3: s = s.squeeze(-1)
            if s.shape[0] > length: s = s[:: max(1, s.shape[0] // length), :]
            if s.shape[0] < length: s = np.pad(s, ((0, length - s.shape[0]), (0, 0)))
            sigs.append(np.transpose(s[:length], (1, 0)))
        x = torch.from_numpy(np.stack(sigs)).to(dev)
        z, _, _ = q(enc(x))
        out.append(pool(z, mode).float().cpu().numpy())
        if i % (bs * 50) == 0: print(f"    {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


def auroc_ci(y, s, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        v.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="baseline")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--group_col", default=None, help="patient-id column for grouped split (no leakage)")
    ap.add_argument("--pool", default="concat", choices=["concat","feat_mean","tok_mean","flat"])
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/tokenizer_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", a.device)

    qa = pd.read_parquet(QA_TEST, columns=(
        ["waveform_name", "waveform_path_psa", "dataset", "deepecho_Visually_Estimated_EF",
         "echonext_shd_binary", "afib_label_5y", "acs_condition_is_acute", "new_PatientID"] + ECG_PATTERNS)).drop_duplicates("waveform_name")
    aux = pd.read_parquet(AUX_TR)
    key = "waveform_name" if "waveform_name" in aux.columns else "waveform_path_psa"
    qa = qa.merge(aux[[key] + [p for p in PHYS if p in aux.columns]], on=key, how="left")
    qa = qa.head(a.n).reset_index(drop=True)
    print(f"[{a.tag}] probing on {len(qa):,} ECGs")

    enc, q = load_models(a.ckpt, dev)
    t = time.time()
    X = extract(enc, q, qa["waveform_path_psa"].tolist(), dev, mode=a.pool)
    print(f"  features {X.shape} in {(time.time()-t)/60:.1f} min")
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    if a.group_col and a.group_col in qa.columns:
        g = qa[a.group_col].astype(str).values
        tr, te = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(Xs, groups=g))
        print(f"  PATIENT-GROUPED split ({a.group_col}): {len(set(g[tr]) & set(g[te]))} patient overlap (should be 0)")
    else:
        idx = np.arange(n); rng = np.random.default_rng(42); rng.shuffle(idx)
        tr, te = idx[: int(.7 * n)], idx[int(.7 * n):]

    res = {"tag": a.tag, "pool": a.pool, "n": int(n), "diagnostic": {}, "physiologic": {}}

    # ---- diagnostic: 77 binary labels ----
    aucs = []
    for lab in ECG_PATTERNS:
        y = pd.to_numeric(qa[lab], errors="coerce").fillna(0).ge(1).astype(int).values
        if y[tr].sum() < 20 or y[te].sum() < 10: continue
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
        if len(tr2) < 200 or len(te2) < 100: continue
        m = Ridge(alpha=1.0).fit(Xs[tr2], y[tr2])
        pr = m.predict(Xs[te2])
        mae = float(np.abs(pr - y[te2]).mean())
        base = float(np.abs(y[tr2].mean() - y[te2]).mean())
        r = float(np.corrcoef(pr, y[te2])[0, 1])
        res["physiologic"][p] = {"mae": round(mae, 3), "mae_predict_mean": round(base, 3),
                                 "r": round(r, 3), "n_test": int(len(te2))}
        print(f"  PHYS {p:8s} MAE {mae:7.2f} (mean-baseline {base:7.2f})  r={r:.3f}  n={len(te2)}")

    # ---- clinical binaries: axis, LVEF<=40, SHD, AFib ----
    bins = {"axis_left": ("Left axis deviation", None), "axis_right": ("Right axis deviation", None),
            "lvef_le40": ("deepecho_Visually_Estimated_EF", lambda v: (v <= 40).astype(int)),
            "shd": ("echonext_shd_binary", lambda v: (v >= 1).astype(int)),
            "afib_5y": ("afib_label_5y", lambda v: (v >= 1).astype(int)),
            "acs_acute": ("acs_condition_is_acute", lambda v: (v >= 1).astype(int))}
    for name, (col, fn) in bins.items():
        if col not in qa.columns: continue
        v = pd.to_numeric(qa[col], errors="coerce").values
        ok = np.isfinite(v)
        y = (fn(v) if fn else (v >= 1).astype(int))
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(tr2) < 100 or y[te2].sum() < 10: continue
        m = LogisticRegression(max_iter=2000).fit(Xs[tr2], y[tr2])
        s = m.predict_proba(Xs[te2])[:, 1]
        au = roc_auc_score(y[te2], s); lo, hi = auroc_ci(y[te2], s)
        res["physiologic"][name] = {"auroc": round(float(au), 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "n_test": int(len(te2)), "n_pos": int(y[te2].sum())}
        print(f"  {name:10s} AUROC {au:.2f} (95% CI {lo:.2f}–{hi:.2f})  n={len(te2)} pos={int(y[te2].sum())}")

    json.dump(res, open(f"{a.out}/probe_{a.tag}.json", "w"), indent=2)
    print(f"[saved] {a.out}/probe_{a.tag}.json")


if __name__ == "__main__":
    main()
