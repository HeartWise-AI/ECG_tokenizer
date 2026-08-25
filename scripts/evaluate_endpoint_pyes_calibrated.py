#!/usr/bin/env python3
"""Calibrated, prompt-ensembled endpoint readout for e4d (action item #2).

Fixes the P(Yes) readout without retraining:
  1. Log-prob MARGIN readout  m = logit(Yes) - logit(No)  (natural boundary at 0).
     NOTE: the old "raw P(Yes)" was already a 2-way softmax over {Yes,No}; its logit
     equals this margin, so single-prompt margin is RANK-equivalent to the old score
     (AUROC unchanged). We report it to prove that and as the base for the two real fixes.
  2. Prompt ensemble: average the margin over K paraphrased Yes/No questions per endpoint.
     This is the only lever here that can move AUROC.
  3. Locked calibration: split the scored ECGs into calib/test. Fit temperature scaling
     + a Youden threshold on CALIB ONLY, lock them, and report operating-point metrics
     on TEST. Fixes the circular Youden-on-test and the degenerate AFib operating point.

AUROC/AUPRC (rank metrics) are reported on the FULL scored set (comparable to the
Notion e4d table). Operating-point metrics (sens/spec/PPV/NPV, Brier, ECE) are reported
on the held-out TEST split at the CALIB-locked threshold.

Usage:
  PYTHONPATH=/volume/ECG_tokenizer CUDA_VISIBLE_DEVICES=0 \
    python scripts/evaluate_endpoint_pyes_calibrated.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from tqdm import tqdm

from scripts.binary_auroc_eval import load_ecg_signal, load_model

CHECKPOINT = "/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt"
TEST_PARQUET = "/media/data1/datasets/ECG_Tokenizer/combined_test_qa_m25k_h25k.parquet"
OUT_DIR = Path("/volume/ECG_tokenizer/analysis/endpoint_pyes_e4d_calibrated")
BOOTSTRAPS = 2000
SEED = 20260701
CALIB_FRAC = 0.5

SYSTEM_MSG = ("You are an expert cardiologist. You interpret ECGs and answer "
             "in a concise, structured way.")

# Each endpoint carries K paraphrased Yes/No questions (clinically equivalent).
ENDPOINTS = {
    "lvef_lte_40": {
        "label": "LVEF <=40%",
        "label_column": "deepecho_Visually_Estimated_EF",
        "label_fn": lambda s: (s <= 40).astype(float),
        "questions": [
            "Is this patient's LVEF less than or equal to 40%? Answer Yes or No.",
            "Does this ECG indicate a left ventricular ejection fraction of 40% or lower? Answer Yes or No.",
            "Is the left ventricular ejection fraction reduced to 40% or below? Answer Yes or No.",
            "Based on this ECG, is LVEF 40% or less? Answer Yes or No.",
        ],
    },
    "lvef_lt_50": {
        "label": "LVEF <50%",
        "label_column": "deepecho_Visually_Estimated_EF",
        "label_fn": lambda s: (s < 50).astype(float),
        "questions": [
            "Is this patient's LVEF less than 50%? Answer Yes or No.",
            "Does this ECG indicate a left ventricular ejection fraction below 50%? Answer Yes or No.",
            "Is the left ventricular ejection fraction under 50%? Answer Yes or No.",
            "Based on this ECG, is LVEF below 50%? Answer Yes or No.",
        ],
    },
    "incident_afib_5y": {
        "label": "AFIB 5 years",
        "label_column": "afib_label_5y",
        "label_fn": lambda s: s.astype(float),
        "questions": [
            "Is this patient at risk for incident atrial fibrillation within 5 years? Answer Yes or No.",
            "Will this patient likely develop atrial fibrillation within the next 5 years? Answer Yes or No.",
            "Does this ECG suggest elevated risk of new-onset atrial fibrillation over 5 years? Answer Yes or No.",
            "Is future atrial fibrillation within 5 years likely for this patient? Answer Yes or No.",
        ],
    },
    "acute_coronary_occlusion": {
        "label": "ACS acute occlusion",
        "label_column": "acs_condition_is_acute",
        "label_fn": lambda s: s.astype(float),
        "questions": [
            "Does this patient have an acute coronary occlusion? Answer Yes or No.",
            "Is there evidence of an acute coronary artery occlusion on this ECG? Answer Yes or No.",
            "Does this ECG indicate acute coronary occlusion? Answer Yes or No.",
            "Is an acute coronary occlusion present? Answer Yes or No.",
        ],
    },
    "shd": {
        "label": "SHD",
        "label_column": "echonext_shd_binary",
        "label_fn": lambda s: s.astype(float),
        "questions": [
            "Does this patient have structural heart disease? Answer Yes or No.",
            "Is there structural heart disease indicated by this ECG? Answer Yes or No.",
            "Does this ECG suggest the presence of structural heart disease? Answer Yes or No.",
            "Is structural heart disease present in this patient? Answer Yes or No.",
        ],
    },
}


def build_prompt_text(q: str) -> str:
    return (f"<start_of_turn>system\n{SYSTEM_MSG}<end_of_turn>\n<start_of_turn>user\n"
            f"<start_of_image>\n\nQuestion: {q}\n\nRespond concisely with the key finding or answer."
            f"<end_of_turn>\n<start_of_turn>model\n")


def pretokenize(tokenizer, questions: List[str]):
    encs = [tokenizer(build_prompt_text(q), add_special_tokens=True, return_tensors="pt") for q in questions]
    max_len = max(e.input_ids.size(1) for e in encs)
    pad_id = tokenizer.pad_token_id or 0
    ids, masks = [], []
    for e in encs:
        ci, cm = e.input_ids[0], e.attention_mask[0]
        pad = max_len - ci.size(0)
        ids.append(torch.cat([ci, torch.full((pad,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([cm, torch.zeros(pad, dtype=torch.long)]))
    return torch.stack(ids), torch.stack(masks)


@torch.no_grad()
def get_margins(model, ecg, prompt_ids, prompt_mask, yes_id, no_id, device, batch_size):
    """Return margin = logit(Yes) - logit(No) for each of the P prompts (shape [P])."""
    ecg = ecg.to(device=device, dtype=torch.float32)
    features = model.encoder(ecg)
    quantized, indices, _ = model.quantizer(features)
    codes = model._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)
    margins = np.zeros(prompt_ids.size(0), dtype=np.float32)
    dec = model.decoder
    for start in range(0, prompt_ids.size(0), batch_size):
        end = min(start + batch_size, prompt_ids.size(0)); bsz = end - start
        q_feat = quantized.expand(bsz, -1, -1)
        q_codes = codes.expand(bsz, -1) if codes.dim() == 2 else codes.expand(bsz, -1, -1)
        b_ids = prompt_ids[start:end].to(device); b_mask = prompt_mask[start:end].to(device)
        inputs_embeds, attn_mask, _ = dec._prepare_inputs_for_generation(
            b_ids, b_mask, q_feat, q_codes, detach_soft_prompts=True)
        md = dec.llm_model.get_input_embeddings().weight.dtype
        out = dec.llm_model(inputs_embeds=inputs_embeds.to(md), attention_mask=attn_mask, return_dict=True)
        for b in range(bsz):
            sl = int(attn_mask[b].sum().item())
            ll = out.logits[b, sl - 1, :].float()
            margins[start + b] = float(ll[yes_id] - ll[no_id])
    return margins


# ----------------------------- calibration helpers -----------------------------

def fit_platt(margins: np.ndarray, y: np.ndarray):
    """Platt scaling: minimise BCE of sigmoid(a*margin + b). Returns (a, b).

    Uses scale AND bias. Temperature-only (b fixed at 0) degenerates on Yes-biased
    endpoints (LVEF/AFib) where the margin distribution is offset from 0 — the optimiser
    drives the scale to infinity to collapse everything to 0.5. The bias term recentres it.
    """
    m = torch.tensor(margins, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.05, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * m + b, t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.item()), float(b.item())


def youden_threshold(prob: np.ndarray, y: np.ndarray) -> float:
    """Threshold on probability maximising sensitivity+specificity-1 (Youden J)."""
    order = np.argsort(-prob)
    p_sorted = prob[order]
    # candidate thresholds = unique probs; evaluate midpoints
    cands = np.unique(p_sorted)
    best_thr, best_j = 0.5, -1.0
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return 0.5
    for thr in cands:
        pred = prob >= thr
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        sens = tp / P; spec = 1 - fp / N
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if m.sum() == 0:
            continue
        conf = prob[m].mean(); acc = y[m].mean()
        ece += (m.sum() / len(prob)) * abs(conf - acc)
    return float(ece)


def rank_metrics(y, s):
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def bootstrap_ci(y, s, rng, keys=("auroc", "auprc")):
    vals = {k: [] for k in keys}
    n = len(y)
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = rank_metrics(y[idx], s[idx])
        for k in keys:
            vals[k].append(m[k])
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2)
            for k, v in vals.items()}


def op_metrics(y, prob, thr):
    pred = prob >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main():
    global CHECKPOINT, TEST_PARQUET, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #ECGs")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--parquet", default=TEST_PARQUET,
                    help="test parquet; use the REGEN file for anything compared against "
                         "post-Jul-2026 results (GT was regenerated — see hazard R1b)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    CHECKPOINT, TEST_PARQUET = args.checkpoint, args.parquet
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pyes] checkpoint={CHECKPOINT}\n[pyes] parquet={TEST_PARQUET}\n[pyes] out={OUT_DIR}")

    df = pd.read_parquet(TEST_PARQUET).drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    names = list(ENDPOINTS)
    n_prompts = [len(ENDPOINTS[n]["questions"]) for n in names]

    lab = np.full((len(df), len(names)), np.nan, dtype=np.float32)
    for j, n in enumerate(names):
        sp = ENDPOINTS[n]
        if sp["label_column"] not in df.columns:
            continue
        mask = df[sp["label_column"]].notna()
        lab[mask.to_numpy(), j] = sp["label_fn"](df.loc[mask, sp["label_column"]]).to_numpy()
    keep = np.isfinite(lab).any(axis=1)
    df = df.loc[keep].reset_index(drop=True); lab = lab[keep]
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True); lab = lab[:args.limit]
    print(f"[calib] {len(df)} unique ECGs; endpoints={names}")

    raw_path = OUT_DIR / "raw_margins.npz"
    if raw_path.exists():
        raw = np.load(raw_path, allow_pickle=True)
        margins = raw["margins"]; lab = raw["labels"]
        names = raw["endpoints"].tolist(); n_prompts = raw["n_prompts"].tolist()
        wp = raw["waveform_paths"]
        print(f"[calib] loaded cached margins {margins.shape}")
    else:
        model, tokenizer = load_model(CHECKPOINT, args.device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]
        all_questions = [q for n in names for q in ENDPOINTS[n]["questions"]]
        pids, pmask = pretokenize(tokenizer, all_questions)
        total_p = pids.size(0)
        margins = np.full((len(df), total_p), np.nan, dtype=np.float32)
        for i, row in tqdm(df.iterrows(), total=len(df), desc="margins"):
            ecg = load_ecg_signal(row["waveform_path_psa"])
            margins[i] = get_margins(model, ecg, pids, pmask, yes_id, no_id, args.device, args.batch_size)
        wp = df["waveform_path_psa"].to_numpy(dtype=object)
        np.savez_compressed(raw_path, margins=margins, labels=lab,
                            endpoints=np.array(names, dtype=object),
                            n_prompts=np.array(n_prompts), waveform_paths=wp)
        print(f"[calib] saved margins {margins.shape}")

    # deterministic calib/test split over ECGs
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(lab))
    n_calib = int(len(lab) * CALIB_FRAC)
    calib_idx = np.zeros(len(lab), dtype=bool); calib_idx[perm[:n_calib]] = True
    test_idx = ~calib_idx

    # column offsets per endpoint in the margins matrix
    offsets = np.cumsum([0] + n_prompts)

    boot = np.random.default_rng(SEED)
    results = {"checkpoint": CHECKPOINT, "parquet": TEST_PARQUET, "seed": SEED,
               "calib_frac": CALIB_FRAC, "n_prompts_per_endpoint": dict(zip(names, n_prompts)),
               "endpoints": {}}
    rows_full, rows_op = [], []
    for j, n in enumerate(names):
        c0, c1 = offsets[j], offsets[j + 1]
        ep_margins = margins[:, c0:c1]                       # [N, P]
        single = ep_margins[:, 0]                            # prompt-0 margin (== old rank score)
        ensemble = np.nanmean(ep_margins, axis=1)            # mean margin across paraphrases
        gt = lab[:, j]
        valid = np.isfinite(gt) & np.isfinite(ensemble)
        y = gt[valid].astype(int)
        s_single = single[valid]; s_ens = ensemble[valid]
        v_calib = calib_idx[valid]; v_test = test_idx[valid]

        # rank metrics on FULL scored set (comparable to Notion)
        m_single = rank_metrics(y, s_single); m_single["ci_95"] = bootstrap_ci(y, s_single, boot)
        m_ens = rank_metrics(y, s_ens); m_ens["ci_95"] = bootstrap_ci(y, s_ens, boot)

        # locked calibration on ENSEMBLE margin: fit T + threshold on calib, apply to test
        yc, sc = y[v_calib], s_ens[v_calib]
        yt, st = y[v_test], s_ens[v_test]
        cal = {"note": "insufficient calib class balance"}
        if len(np.unique(yc)) == 2 and len(np.unique(yt)) == 2:
            a, b = fit_platt(sc, yc)
            prob_c = 1 / (1 + np.exp(-(a * sc + b)))
            prob_t = 1 / (1 + np.exp(-(a * st + b)))
            thr = youden_threshold(prob_c, yc)
            op_test = op_metrics(yt, prob_t, thr)
            # naive/circular baseline: raw P(Yes) >= 0.5 on test (== margin >= 0)
            op_naive = op_metrics(yt, 1 / (1 + np.exp(-st)), 0.5)
            cal = {
                "platt_a": a,
                "platt_b": b,
                "locked_threshold_prob": thr,
                "test_auroc": float(roc_auc_score(yt, st)),
                "test_brier": float(brier_score_loss(yt, prob_t)),
                "test_ece": expected_calibration_error(prob_t, yt),
                "n_calib": int(len(yc)), "n_test": int(len(yt)),
                "n_calib_pos": int(yc.sum()), "n_test_pos": int(yt.sum()),
                "operating_point_locked": op_test,
                "operating_point_naive_pyes0.5": op_naive,
            }
        results["endpoints"][n] = {
            "label": ENDPOINTS[n]["label"], "n": int(valid.sum()),
            "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
            "rank_single_prompt_margin": m_single,
            "rank_prompt_ensemble_margin": m_ens,
            "calibration_locked": cal,
        }

        def cell(m):
            return f"{m['auroc']:.2f} ({m['ci_95']['auroc'][0]:.2f}-{m['ci_95']['auroc'][1]:.2f})"
        rows_full.append(f"| {ENDPOINTS[n]['label']} | {int(y.sum())}/{len(y)} | "
                         f"{cell(m_single)} | {cell(m_ens)} |")
        if "operating_point_locked" in cal:
            ol, on = cal["operating_point_locked"], cal["operating_point_naive_pyes0.5"]
            rows_op.append(f"| {ENDPOINTS[n]['label']} | {cal['n_test']} | "
                           f"{on['sensitivity']:.2f}/{on['specificity']:.2f} | "
                           f"{ol['sensitivity']:.2f}/{ol['specificity']:.2f} | "
                           f"{ol['ppv']:.2f} | {ol['npv']:.2f} | {cal['test_ece']:.3f} |")

    (OUT_DIR / f"endpoint_pyes_{OUT_DIR.name.replace('endpoint_pyes_', '')}_calibrated_metrics.json").write_text(json.dumps(results, indent=2))
    md = ["# calibrated / prompt-ensembled endpoint readout", "",
          "## AUROC on full scored set (rank metric — comparable to Notion e4d table)", "",
          "| Endpoint | Pos/n | AUROC single-prompt margin (95% CI) | AUROC 4-prompt ensemble (95% CI) |",
          "|---|---:|---|---|", *rows_full, "",
          "## Operating point on held-out TEST split (threshold LOCKED from calib split)", "",
          "| Endpoint | n_test | naive P(Yes)>=0.5 Sens/Spec | locked Sens/Spec | PPV | NPV | ECE |",
          "|---|---:|---|---|---:|---:|---:|", *rows_op, ""]
    (OUT_DIR / f"endpoint_pyes_{OUT_DIR.name.replace('endpoint_pyes_', '')}_calibrated_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
