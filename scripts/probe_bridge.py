#!/usr/bin/env python3
"""Frozen linear probe of the TOKENIZER+BRIDGE representation — the ECG embedding the LLM sees.

Same harness as probe_tokenizer.py (same test parquet, head(n), amplitude-preserved mV signals,
same 77-diag + LVEF/SHD/AFib/ACS/axis labels, same GroupShuffleSplit on new_PatientID), but the
representation is the Q-Former bridge's pooled ECG embedding (mode="ETC"), not the raw codes.

Flow replicates the Stage-1 runner exactly:
  signal -> encoder -> quantizer(return_all_codes) -> indices[..., offset:offset+keep]  (= _compute_codes)
         -> bridge.forward_stage1(codes, None, None, mode="ETC") -> pooled ecg_vec

Tests whether the bridge preserves the tokenizer's clinical information (vs the historical
"32-query bridge caps it" concern). Run in the ECG_tokenizer venv.
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, "/volume/ECG_tokenizer")
import numpy as np, pandas as pd, torch
import models  # noqa: F401 — populates registry (incl. ScalableEncoder, ECGQFormerBridgeStage1)
from utils.registry import ModelRegistry
from utils.constants import ECG_PATTERNS
from utils.ecg_waveform import load_ecg_waveform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

TOK = "/volume/ECG_tokenizer/checkpoints/x1_split_adapted/tokenizer_prod_format.pt"
QA_TEST = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"
AUX_TR = "/volume/ECG_tokenizer/output/tokenizer_aux_targets_train.parquet"
PHYS = ["hr", "qrs_dur", "qtc", "pr", "qt"]


def latest_bridge_ckpt():
    fs = sorted(glob.glob("/volume/ECG_tokenizer/checkpoints/ECG_Text_Stage1/ecg_text_stage1_x1split_fulletg/*/checkpoints/stage1_last_epoch_*.pt"))
    return fs[-1] if fs else None


def load_tokenizer(dev, tok_path=TOK):
    sd = torch.load(tok_path, map_location="cpu", weights_only=False)
    cfg, msd = sd["config"], sd["model_state_dict"]
    c = lambda k, d=None: (cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d))
    enc = ModelRegistry.get(c("encoder_name"))()
    q = ModelRegistry.get(c("quantizer_name"))(num_quantizers=int(c("num_quantizers")), codebook_size=int(c("codebook_size")))
    enc.load_state_dict({k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")})
    q.load_state_dict({k[len("quantizer."):]: v for k, v in msd.items() if k.startswith("quantizer.")})
    return enc.to(dev).eval(), q.to(dev).eval()


def load_bridge(ckpt_path, dev, q):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, msd = sd["config"], sd["model_state_dict"]
    g = lambda k, d=None: (cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d))
    txt_vocab = int(next((msd[k].shape[0] for k in msd if k.endswith("txt_embed.weight")), 262146))
    keep = int(g("num_codebooks_kept", 2)); off = int(g("codebook_offset", 0) or 0)
    bridge = ModelRegistry.get(str(g("bridge_name", "ECGQFormerBridgeStage1")))(
        vocab_size=int(g("codebook_size", 512)), num_codebooks=keep,
        d_mid=int(g("bridge_hidden_size", 768)), d_llm=int(g("bridge_hidden_size", 768)),
        d_txt=int(g("bridge_hidden_size", 768)), num_steps=int(g("bridge_max_seq_len", 128)),
        num_query_tokens=int(g("num_query_tokens", 32)), num_layers=int(g("bridge_num_layers", 6)),
        num_heads=int(g("bridge_num_heads", 12)), dropout=float(g("bridge_dropout", 0.1)),
        num_special_tokens=int(g("bridge_num_special_tokens", 4)),
        bias_last_codebook=float(g("bridge_bias_last_codebook", 0.5)),
        codebook_dropout=float(g("bridge_codebook_dropout", 0.0)),
        mix_strategy=str(g("bridge_mix_strategy", "softmax") or "softmax"),
        token_axis=str(g("bridge_token_axis", "channel") or "channel"),
        txt_vocab_size=txt_vocab, txt_pad_id=0, txt_cls_id=1,
        cross_every=int(g("cross_every", 2)), bert_layers=g("bert_layers"),
    )
    if getattr(bridge, "token_axis", "channel") == "time":
        rvq = getattr(q, "quantizer", q)
        bridge.attach_quantizer(rvq)
    res = bridge.load_state_dict(msd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"Bridge checkpoint keys mismatch: {res}")
    print(f"  bridge loaded @ epoch {sd.get('epoch')}: {len(msd)} keys")
    return bridge.to(dev).eval(), keep, off


@torch.no_grad()
def extract(enc, q, bridge, paths, dev, keep, off, repr_mode="etc", bs=24, length=2500):
    # For query32* modes, capture the pre-pool 32-query tensor (what the LLM consumes)
    cap = {}
    if repr_mode.startswith("query32"):
        _orig_pool = bridge._pool_queries
        def _hook(qq):
            cap["q"] = qq.detach()
            return _orig_pool(qq)
        bridge._pool_queries = _hook
    out = []
    for i in range(0, len(paths), bs):
        sigs = []
        for p in paths[i:i + bs]:
            sigs.append(load_ecg_waveform(p, target_length=length, num_leads=12))
        x = torch.from_numpy(np.stack(sigs)).to(dev)
        feats = enc(x)
        qo = q(feats, return_all_codes=True)
        indices = qo[1]                                   # (B, 128, 8)
        codes = indices[..., off:off + keep].long()       # _compute_codes slice
        if codes.size(-1) == 1: codes = codes.squeeze(-1)
        ecg_vec, _ = bridge.forward_stage1(codes, None, None, mode="ETC")   # (B, dim); also triggers _hook
        if repr_mode == "etc":
            rep = ecg_vec
        else:
            qout = cap["q"]                               # (B, 32, hidden) — the injected query tokens
            if repr_mode == "query32_mean":
                rep = qout.mean(dim=1)
            elif repr_mode == "query32_meanmax":
                rep = torch.cat([qout.mean(dim=1), qout.amax(dim=1)], dim=-1)
            else:  # query32_flat
                rep = qout.flatten(1)
        out.append(rep.float().cpu().numpy())
        if i % (bs * 40) == 0: print(f"    {i}/{len(paths)}", flush=True)
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
    ap.add_argument("--bridge_ckpt", default=None)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--tag", default="BRIDGE_PT")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--group_col", default="new_PatientID")
    ap.add_argument("--repr", dest="repr_mode", default="etc",
                    choices=["etc", "query32_mean", "query32_meanmax", "query32_flat"])
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/tokenizer_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda", a.device)
    bridge_ckpt = a.bridge_ckpt or latest_bridge_ckpt()
    print(f"[{a.tag}] bridge ckpt = {bridge_ckpt}")

    qa = pd.read_parquet(QA_TEST, columns=(
        ["waveform_name", "waveform_path_psa", "dataset", "deepecho_Visually_Estimated_EF",
         "echonext_shd_binary", "afib_label_5y", "acs_condition_is_acute", "new_PatientID"] + ECG_PATTERNS)).drop_duplicates("waveform_name")
    aux = pd.read_parquet(AUX_TR)
    key = "waveform_name" if "waveform_name" in aux.columns else "waveform_path_psa"
    qa = qa.merge(aux[[key] + [p for p in PHYS if p in aux.columns]], on=key, how="left")
    qa = qa.head(a.n).reset_index(drop=True)
    print(f"[{a.tag}] probing tokenizer+bridge on {len(qa):,} ECGs | mix {qa['dataset'].value_counts().to_dict()}")

    enc, q = load_tokenizer(dev, a.tok)
    bridge, keep, off = load_bridge(bridge_ckpt, dev, q)
    print(f"  codebook slice: indices[..., {off}:{off+keep}]")
    t = time.time()
    X = extract(enc, q, bridge, qa["waveform_path_psa"].tolist(), dev, keep, off, repr_mode=a.repr_mode)
    print(f"  bridge features [{a.repr_mode}] {X.shape} in {(time.time()-t)/60:.1f} min")
    Xs = StandardScaler().fit_transform(X)
    n = len(Xs)
    g = qa[a.group_col].astype(str).values
    tr, te = next(GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42).split(Xs, groups=g))
    print(f"  PATIENT-GROUPED split: {len(set(g[tr]) & set(g[te]))} overlap (should be 0)")

    res = {"tag": a.tag, "repr": a.repr_mode, "bridge_ckpt": bridge_ckpt, "dim": int(X.shape[1]),
           "n": int(n), "diagnostic": {}, "physiologic": {}}
    aucs = []
    for lab in ECG_PATTERNS:
        y = pd.to_numeric(qa[lab], errors="coerce").fillna(0).ge(1).astype(int).values
        if y[tr].sum() < 20 or y[te].sum() < 10: continue
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        au = roc_auc_score(y[te], m.predict_proba(Xs[te])[:, 1]); aucs.append(au)
        res["diagnostic"][lab] = {"auroc": round(float(au), 4), "n_pos": int(y.sum())}
    res["diagnostic_macro_auroc"] = round(float(np.mean(aucs)), 4)
    print(f"  DIAGNOSTIC macro AUROC = {np.mean(aucs):.4f}  over {len(aucs)} labels")

    for p in [c for c in PHYS if c in qa.columns]:
        y = pd.to_numeric(qa[p], errors="coerce").values
        ok = np.isfinite(y); tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(tr2) < 200 or len(te2) < 100: continue
        m = Ridge(alpha=1.0).fit(Xs[tr2], y[tr2]); pr = m.predict(Xs[te2])
        res["physiologic"][p] = {"mae": round(float(np.abs(pr - y[te2]).mean()), 3),
                                 "r": round(float(np.corrcoef(pr, y[te2])[0, 1]), 3), "n_test": int(len(te2))}
        print(f"  PHYS {p:8s} MAE {res['physiologic'][p]['mae']:7.2f}  r={res['physiologic'][p]['r']:.3f}")

    bins = {"axis_left": ("Left axis deviation", None), "axis_right": ("Right axis deviation", None),
            "lvef_le40": ("deepecho_Visually_Estimated_EF", lambda v: (v <= 40).astype(int)),
            "shd": ("echonext_shd_binary", lambda v: (v >= 1).astype(int)),
            "afib_5y": ("afib_label_5y", lambda v: (v >= 1).astype(int)),
            "acs_acute": ("acs_condition_is_acute", lambda v: (v >= 1).astype(int))}
    for name, (col, fn) in bins.items():
        if col not in qa.columns: continue
        v = pd.to_numeric(qa[col], errors="coerce").values
        ok = np.isfinite(v); y = (fn(v) if fn else (v >= 1).astype(int))
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(tr2) < 100 or y[te2].sum() < 10: continue
        m = LogisticRegression(max_iter=2000).fit(Xs[tr2], y[tr2]); s = m.predict_proba(Xs[te2])[:, 1]
        au = roc_auc_score(y[te2], s); lo, hi = auroc_ci(y[te2], s)
        res["physiologic"][name] = {"auroc": round(float(au), 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "n_test": int(len(te2)), "n_pos": int(y[te2].sum())}
        print(f"  {name:10s} AUROC {au:.2f} (95% CI {lo:.2f}-{hi:.2f})  n={len(te2)} pos={int(y[te2].sum())}")

    json.dump(res, open(f"{a.out}/probe_{a.tag}.json", "w"), indent=2)
    print(f"[saved] {a.out}/probe_{a.tag}.json")


if __name__ == "__main__":
    main()
