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
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from tqdm import tqdm

from scripts.binary_auroc_eval import load_ecg_signal, load_model
from utils.artifact_provenance import (
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_text,
    encode_manifest,
    exclusive_artifact_lock,
    file_identity,
    load_npz_if_current,
    ordered_files_identity,
    publish_artifact_bundle,
    require_finite_numeric_array,
    require_matching_provenance,
)
from utils.endpoint_contract import (
    ENDPOINT_PROMPT_CONTRACT,
    endpoint_contract_sha256,
    endpoint_implementation_identity,
)
from utils.endpoint_labels import encode_endpoint_labels, require_binary_class_support
from utils.patient_identity import patient_group_split_masks

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
        "label_column": ENDPOINT_PROMPT_CONTRACT["lvef_lte_40"]["label_column"],
        "questions": list(ENDPOINT_PROMPT_CONTRACT["lvef_lte_40"]["questions"]),
    },
    "lvef_lt_50": {
        "label": "LVEF <50%",
        "label_column": ENDPOINT_PROMPT_CONTRACT["lvef_lt_50"]["label_column"],
        "questions": list(ENDPOINT_PROMPT_CONTRACT["lvef_lt_50"]["questions"]),
    },
    "incident_afib_5y": {
        "label": "AFIB 5 years",
        "label_column": ENDPOINT_PROMPT_CONTRACT["incident_afib_5y"]["label_column"],
        "questions": list(ENDPOINT_PROMPT_CONTRACT["incident_afib_5y"]["questions"]),
    },
    "acute_coronary_occlusion": {
        "label": "ACS acute occlusion",
        "label_column": ENDPOINT_PROMPT_CONTRACT["acute_coronary_occlusion"]["label_column"],
        "questions": list(ENDPOINT_PROMPT_CONTRACT["acute_coronary_occlusion"]["questions"]),
    },
    "shd": {
        "label": "SHD",
        "label_column": ENDPOINT_PROMPT_CONTRACT["shd"]["label_column"],
        "questions": list(ENDPOINT_PROMPT_CONTRACT["shd"]["questions"]),
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
        end = min(start + batch_size, prompt_ids.size(0))
        bsz = end - start
        q_feat = quantized.expand(bsz, -1, -1)
        q_codes = codes.expand(bsz, -1) if codes.dim() == 2 else codes.expand(bsz, -1, -1)
        b_ids = prompt_ids[start:end].to(device)
        b_mask = prompt_mask[start:end].to(device)
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
    endpoints (LVEF/AFib) where the margin distribution is offset from 0 - the optimiser
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
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return 0.5
    for thr in cands:
        pred = prob >= thr
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        sens = tp / P
        spec = 1 - fp / N
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
        conf = prob[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / len(prob)) * abs(conf - acc)
    return float(ece)


def rank_metrics(y, s):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def bootstrap_ci(y, s, rng, keys=("auroc", "auprc")):
    vals = {k: [] for k in keys}
    n = len(y)
    if n == 0:
        return {key: [float("nan"), float("nan")] for key in keys}
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


def build_margin_cache_provenance(
    checkpoint: str,
    parquet: str,
    names: List[str],
    n_prompts: List[int],
    limit: int,
    waveform_paths: List[str],
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "endpoint_margin_cache",
        "semantic": {
            "checkpoint": file_identity(checkpoint),
            "parquet": file_identity(parquet),
            "script": file_identity(__file__),
            "implementation": endpoint_implementation_identity(),
            "endpoint_contract_sha256": endpoint_contract_sha256(names),
            "endpoint_names": names,
            "n_prompts": n_prompts,
            "limit": int(limit),
            "row_count": len(waveform_paths),
            "ordered_waveforms": ordered_files_identity(waveform_paths),
        },
        "execution": {
            "batch_size": int(batch_size),
            "device": str(device),
        },
    }


def build_static_input_provenance(checkpoint: str, parquet: str) -> Dict[str, object]:
    """Capture inputs that must remain stable while a margin artifact is computed."""
    return {
        "checkpoint": file_identity(checkpoint),
        "parquet": file_identity(parquet),
        "script": file_identity(__file__),
        "implementation": endpoint_implementation_identity(),
    }


def _evaluate_and_publish_metrics(
    args: argparse.Namespace,
    df: pd.DataFrame,
    labels: np.ndarray,
    margins: np.ndarray,
    names: List[str],
    n_prompts: List[int],
    waveform_paths: List[str],
    margin_provenance: Dict[str, object],
) -> None:
    if "new_PatientID" not in df.columns:
        raise ValueError("endpoint cohort is missing required new_PatientID values")
    calib_idx, test_idx = patient_group_split_masks(
        df["new_PatientID"],
        first_fraction=CALIB_FRAC,
        seed=SEED,
        source="endpoint calibration cohort",
    )
    offsets = np.cumsum([0] + n_prompts)
    bootstrap_rng = np.random.default_rng(SEED)
    metrics_provenance = {
        "schema_version": 1,
        "kind": "endpoint_calibrated_metrics",
        "margin_cache_provenance": margin_provenance,
        "seed": SEED,
        "calib_frac": CALIB_FRAC,
        "split_strategy": "patient_grouped",
    }
    endpoint_results: dict[str, object] = {}
    results = {
        "checkpoint": CHECKPOINT,
        "parquet": TEST_PARQUET,
        "seed": SEED,
        "calib_frac": CALIB_FRAC,
        "n_prompts_per_endpoint": dict(zip(names, n_prompts)),
        "split_strategy": "patient_grouped",
        "provenance": metrics_provenance,
        "endpoints": endpoint_results,
    }
    rows_full: list[str] = []
    rows_op: list[str] = []

    def metric_cell(metric: dict[str, object]) -> str:
        interval = metric["ci_95"]["auroc"]
        return f"{metric['auroc']:.2f} ({interval[0]:.2f}-{interval[1]:.2f})"

    for j, name in enumerate(names):
        c0, c1 = offsets[j], offsets[j + 1]
        endpoint_margins = margins[:, c0:c1]
        single = endpoint_margins[:, 0]
        ensemble = endpoint_margins.mean(axis=1)
        ground_truth = labels[:, j]
        valid = np.isfinite(ground_truth) & np.isfinite(ensemble)
        y = ground_truth[valid].astype(int)
        single_scores = single[valid]
        ensemble_scores = ensemble[valid]
        valid_calib = calib_idx[valid]
        valid_test = test_idx[valid]
        require_binary_class_support(y, endpoint=name, cohort="full scored cohort")
        single_metrics = rank_metrics(y, single_scores)
        single_metrics["ci_95"] = bootstrap_ci(y, single_scores, bootstrap_rng)
        ensemble_metrics = rank_metrics(y, ensemble_scores)
        ensemble_metrics["ci_95"] = bootstrap_ci(y, ensemble_scores, bootstrap_rng)

        calibration_labels = y[valid_calib]
        calibration_scores = ensemble_scores[valid_calib]
        test_labels = y[valid_test]
        test_scores = ensemble_scores[valid_test]
        require_binary_class_support(
            calibration_labels,
            endpoint=name,
            cohort="calibration split",
        )
        require_binary_class_support(test_labels, endpoint=name, cohort="test split")
        platt_a, platt_b = fit_platt(calibration_scores, calibration_labels)
        calibration_probabilities = 1 / (
            1 + np.exp(-(platt_a * calibration_scores + platt_b))
        )
        test_probabilities = 1 / (1 + np.exp(-(platt_a * test_scores + platt_b)))
        threshold = youden_threshold(calibration_probabilities, calibration_labels)
        locked = op_metrics(test_labels, test_probabilities, threshold)
        naive = op_metrics(test_labels, 1 / (1 + np.exp(-test_scores)), 0.5)
        calibration = {
            "platt_a": platt_a,
            "platt_b": platt_b,
            "locked_threshold_prob": threshold,
            "test_auroc": float(roc_auc_score(test_labels, test_scores)),
            "test_brier": float(brier_score_loss(test_labels, test_probabilities)),
            "test_ece": expected_calibration_error(test_probabilities, test_labels),
            "n_calib": int(len(calibration_labels)),
            "n_test": int(len(test_labels)),
            "n_calib_pos": int(calibration_labels.sum()),
            "n_test_pos": int(test_labels.sum()),
            "operating_point_locked": locked,
            "operating_point_naive_pyes0.5": naive,
        }
        endpoint_results[name] = {
            "label": ENDPOINTS[name]["label"],
            "n": int(valid.sum()),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "rank_single_prompt_margin": single_metrics,
            "rank_prompt_ensemble_margin": ensemble_metrics,
            "calibration_locked": calibration,
        }
        rows_full.append(
            f"| {ENDPOINTS[name]['label']} | {int(y.sum())}/{len(y)} | "
            f"{metric_cell(single_metrics)} | {metric_cell(ensemble_metrics)} |"
        )
        rows_op.append(
            f"| {ENDPOINTS[name]['label']} | {calibration['n_test']} | "
            f"{naive['sensitivity']:.2f}/{naive['specificity']:.2f} | "
            f"{locked['sensitivity']:.2f}/{locked['specificity']:.2f} | "
            f"{locked['ppv']:.2f} | {locked['npv']:.2f} | "
            f"{calibration['test_ece']:.3f} |"
        )

    json_path = OUT_DIR / f"endpoint_pyes_{OUT_DIR.name.replace('endpoint_pyes_', '')}_calibrated_metrics.json"
    markdown = [
        "# calibrated / prompt-ensembled endpoint readout",
        "",
        "## AUROC on full scored set (rank metric, comparable to Notion e4d table)",
        "",
        "| Endpoint | Pos/n | AUROC single-prompt margin (95% CI) | AUROC 4-prompt ensemble (95% CI) |",
        "|---|---:|---|---|",
        *rows_full,
        "",
        "## Operating point on held-out TEST split (threshold LOCKED from calib split)",
        "",
        "| Endpoint | n_test | naive P(Yes)>=0.5 Sens/Spec | locked Sens/Spec | PPV | NPV | ECE |",
        "|---|---:|---|---|---:|---:|---:|",
        *rows_op,
        "",
    ]
    md_path = json_path.with_suffix(".md")

    def verify_current() -> None:
        require_matching_provenance(
            margin_provenance,
            build_margin_cache_provenance(
                CHECKPOINT,
                TEST_PARQUET,
                names,
                n_prompts,
                args.limit,
                waveform_paths,
                args.batch_size,
                args.device,
            ),
            artifact="endpoint calibrated metrics",
        )

    with tempfile.TemporaryDirectory(
        dir=OUT_DIR,
        prefix=".endpoint-metrics-staging-",
    ) as staging_dir:
        staging = Path(staging_dir)
        atomic_write_json(staging / json_path.name, results)
        atomic_write_text(staging / md_path.name, "\n".join(markdown) + "\n")
        publish_artifact_bundle(
            {
                json_path.name: staging / json_path.name,
                md_path.name: staging / md_path.name,
            },
            json_path.with_suffix(".bundle.json"),
            provenance=metrics_provenance,
            verify_current=verify_current,
        )
    print("\n".join(markdown))


def _run(args: argparse.Namespace) -> None:
    global CHECKPOINT, TEST_PARQUET, OUT_DIR
    CHECKPOINT, TEST_PARQUET = args.checkpoint, args.parquet
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pyes] checkpoint={CHECKPOINT}\n[pyes] parquet={TEST_PARQUET}\n[pyes] out={OUT_DIR}")

    captured_static_inputs = build_static_input_provenance(CHECKPOINT, TEST_PARQUET)
    df = pd.read_parquet(TEST_PARQUET).drop_duplicates(subset="waveform_path_psa").reset_index(drop=True)
    names = list(ENDPOINTS)
    n_prompts = [len(ENDPOINTS[n]["questions"]) for n in names]

    lab = np.full((len(df), len(names)), np.nan, dtype=np.float32)
    for j, n in enumerate(names):
        sp = ENDPOINTS[n]
        if sp["label_column"] not in df.columns:
            continue
        lab[:, j] = encode_endpoint_labels(df[sp["label_column"]], n)
    keep = np.isfinite(lab).any(axis=1)
    df = df.loc[keep].reset_index(drop=True)
    lab = lab[keep]
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
        lab = lab[:args.limit]
    print(f"[calib] {len(df)} unique ECGs; endpoints={names}")

    raw_path = OUT_DIR / "raw_margins.npz"
    waveform_paths = df["waveform_path_psa"].astype(str).tolist()
    provenance = build_margin_cache_provenance(
        CHECKPOINT,
        TEST_PARQUET,
        names,
        n_prompts,
        args.limit,
        waveform_paths,
        args.batch_size,
        args.device,
    )
    semantic = provenance["semantic"]
    if not isinstance(semantic, dict):
        raise RuntimeError("endpoint margin provenance is missing semantic identities")
    require_matching_provenance(
        captured_static_inputs,
        {key: semantic[key] for key in captured_static_inputs},
        artifact="endpoint margin cohort",
    )
    cached = load_npz_if_current(raw_path, provenance)
    if cached is not None:
        cached_handle = cached
        try:
            required = {"margins", "labels", "endpoints", "n_prompts", "waveform_paths"}
            missing = required - set(cached.files)
            if missing:
                raise ValueError(f"cache missing arrays: {sorted(missing)}")
            margins = cached["margins"].copy()
            cached_labels = cached["labels"].copy()
            cached_names = [str(value) for value in cached["endpoints"].tolist()]
            cached_n_prompts = [int(value) for value in cached["n_prompts"].tolist()]
            wp = cached["waveform_paths"].astype(str)
            require_finite_numeric_array(
                margins,
                artifact="endpoint margin cache",
                expected_shape=(len(df), sum(n_prompts)),
            )
            if cached_labels.shape != lab.shape:
                raise ValueError("cache label shape does not match current cohort")
            if cached_names != names or cached_n_prompts != n_prompts:
                raise ValueError("cache endpoint layout does not match current contract")
            if wp.tolist() != waveform_paths:
                raise ValueError("cache waveform order does not match current cohort")
            if not np.array_equal(cached_labels, lab, equal_nan=True):
                raise ValueError("cache labels do not match the validated source labels")
            print(f"[calib] loaded current cached margins {margins.shape}")
        except (KeyError, ValueError) as exc:
            print(f"[calib] ignoring invalid margin cache: {exc}")
            cached = None
        finally:
            cached_handle.close()
    if cached is None:
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
        require_finite_numeric_array(
            margins,
            artifact="generated endpoint margins",
            expected_shape=(len(df), total_p),
        )
        wp = np.asarray(waveform_paths, dtype=np.str_)
        require_matching_provenance(
            provenance,
            build_margin_cache_provenance(
                CHECKPOINT,
                TEST_PARQUET,
                names,
                n_prompts,
                args.limit,
                waveform_paths,
                args.batch_size,
                args.device,
            ),
            artifact="endpoint margin cache",
        )
        atomic_savez_compressed(
            raw_path,
            margins=margins,
            labels=lab,
            endpoints=np.asarray(names, dtype=np.str_),
            n_prompts=np.asarray(n_prompts, dtype=np.int64),
            waveform_paths=wp,
            provenance=encode_manifest(provenance),
        )
        print(f"[calib] saved margins {margins.shape}")

    _evaluate_and_publish_metrics(
        args,
        df,
        lab,
        margins,
        names,
        n_prompts,
        waveform_paths,
        provenance,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #ECGs")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--parquet", default=TEST_PARQUET,
                    help="test parquet; use the REGEN file for anything compared against "
                         "post-Jul-2026 results (GT was regenerated; see hazard R1b)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_artifact_lock(out_dir / ".endpoint-calibrated-metrics.lock"):
        _run(args)


if __name__ == "__main__":
    main()
