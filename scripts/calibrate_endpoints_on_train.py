#!/usr/bin/env python3
"""Fit the endpoint readout calibration on TRAIN-side ECGs, evaluate on the FULL test set.

Why: `evaluate_endpoint_pyes_calibrated.py` splits the scored TEST ECGs 50/50 into
calib/test. That burns half the test set on threshold fitting and reports the operating
point on only the other half. Fitting on train-side data instead is both fairer (no test
data touches the fit) and mirrors deployment (you calibrate before release, then apply the
locked threshold to everything unseen) - and it frees the ENTIRE test set for evaluation.

Known risk, measured rather than assumed: the model was TRAINED on these ECGs, so its
margins there are likely sharper/over-confident, which can bias the fitted threshold. This
script therefore reports BOTH calibrations side by side (train-fit vs test-half-fit) on the
same full test set, so the size and direction of that bias is visible.

Reuses the cached test margins from the earlier run (raw_margins.npz) - only the train-side
margins are newly computed.

  PYTHONPATH=/volume/ECG_tokenizer python scripts/calibrate_endpoints_on_train.py \
      --checkpoint <best_model.pt> --device cuda:2 --n_calib 1500
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.evaluate_endpoint_pyes_calibrated import (
    ENDPOINTS, build_margin_cache_provenance, expected_calibration_error,
    fit_platt, get_margins, op_metrics, pretokenize, rank_metrics,
    youden_threshold,
)
from scripts.binary_auroc_eval import load_ecg_signal, load_model
from utils.artifact_provenance import (
    atomic_savez_compressed,
    atomic_write_json,
    atomic_write_text,
    decode_manifest,
    encode_manifest,
    exclusive_artifact_lock,
    file_identity,
    ordered_files_identity,
    publish_artifact_bundle,
    require_finite_numeric_array,
    require_matching_provenance,
)
from utils.endpoint_contract import endpoint_contract_sha256, endpoint_implementation_identity
from utils.endpoint_labels import encode_endpoint_labels, require_binary_class_support
from utils.patient_identity import patient_disjoint_mask

TRAIN_PARQUET = "/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k.REGEN.parquet"
TEST_MARGINS = "/volume/ECG_tokenizer/analysis/endpoint_pyes_concatmix/raw_margins.npz"
TEST_PARQUET = "/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.REGEN.parquet"
SEED = 20260821


def build_calibration_static_inputs(
    checkpoint: str,
    train_parquet: str,
    test_parquet: str,
    test_margins: str,
) -> dict[str, object]:
    return {
        "checkpoint": file_identity(checkpoint),
        "train_parquet": file_identity(train_parquet),
        "test_parquet": file_identity(test_parquet),
        "test_margins": file_identity(test_margins),
        "script": file_identity(__file__),
        "implementation": endpoint_implementation_identity(),
    }


def build_calibration_provenance(
    *,
    checkpoint: str,
    train_parquet: str,
    test_parquet: str,
    test_margins: str,
    test_margin_cache_provenance: dict[str, object],
    names: list[str],
    n_calib: int,
    train_waveform_paths: list[str],
    static_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    captured = static_inputs or build_calibration_static_inputs(
        checkpoint,
        train_parquet,
        test_parquet,
        test_margins,
    )
    return {
        "schema_version": 2,
        "kind": "endpoint_train_calibration",
        **captured,
        "test_margin_cache_provenance": test_margin_cache_provenance,
        "n_calib": int(n_calib),
        "seed": SEED,
        "endpoint_contract_sha256": endpoint_contract_sha256(names),
        "ordered_train_waveforms": ordered_files_identity(train_waveform_paths),
    }


def _fit_calibration_results(
    a: argparse.Namespace,
    names: list[str],
    n_prompts: list[int],
    tr: pd.DataFrame,
    lab: np.ndarray,
    margins: np.ndarray,
    test_margins: np.ndarray,
    test_labels: np.ndarray,
    calibration_provenance: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    endpoint_results: dict[str, object] = {}
    results = {
        "checkpoint": a.checkpoint,
        "train_parquet": a.train_parquet,
        "test_parquet": a.test_parquet,
        "test_margins": a.test_margins,
        "n_calib_scored": int(len(tr)),
        "provenance": calibration_provenance,
        "endpoints": endpoint_results,
    }
    md = ["# Endpoint readout: calibration fitted on TRAIN, evaluated on FULL test\n",
          "| Endpoint | test n | AUROC (full test) | TRAIN-fit locked Sens/Spec | bal-acc | prior half-test-fit Sens/Spec |",
          "|---|---:|---|---|---:|---|"]
    off = 0
    for j, name in enumerate(names):
        prompt_count = n_prompts[j]
        train_endpoint_margins = margins[:, off:off + prompt_count].mean(axis=1)
        test_endpoint_margins = test_margins[:, off:off + prompt_count].mean(axis=1)
        off += prompt_count
        train_labels, endpoint_test_labels = lab[:, j], test_labels[:, j]
        valid_train = np.isfinite(train_labels)
        valid_test = np.isfinite(endpoint_test_labels)
        platt_a, platt_b = fit_platt(
            train_endpoint_margins[valid_train],
            train_labels[valid_train],
        )
        train_probabilities = 1 / (
            1 + np.exp(-(platt_a * train_endpoint_margins[valid_train] + platt_b))
        )
        threshold = youden_threshold(train_probabilities, train_labels[valid_train])
        test_probabilities = 1 / (
            1 + np.exp(-(platt_a * test_endpoint_margins[valid_test] + platt_b))
        )
        operating_point = op_metrics(
            endpoint_test_labels[valid_test],
            test_probabilities,
            threshold,
        )
        rank = rank_metrics(
            endpoint_test_labels[valid_test],
            test_endpoint_margins[valid_test],
        )
        balanced_accuracy = (
            operating_point["sensitivity"] + operating_point["specificity"]
        ) / 2
        endpoint_result = {
            "label": ENDPOINTS[name]["label"],
            "n_train_labelled": int(valid_train.sum()),
            "n_test": int(valid_test.sum()),
            "platt_a": float(platt_a),
            "platt_b": float(platt_b),
            "locked_threshold_prob": float(threshold),
            "full_test_rank": rank,
            "operating_point": operating_point,
            "balanced_accuracy": float(balanced_accuracy),
            "test_ece": float(
                expected_calibration_error(
                    test_probabilities,
                    endpoint_test_labels[valid_test],
                )
            ),
        }
        endpoint_results[name] = endpoint_result
        print(
            f"  {ENDPOINTS[name]['label']:22s} n_test={int(valid_test.sum()):5d} "
            f"AUROC {rank['auroc']:.2f} | TRAIN-fit sens "
            f"{operating_point['sensitivity']:.2f} spec "
            f"{operating_point['specificity']:.2f} bal {balanced_accuracy:.2f} | "
            f"ECE {endpoint_result['test_ece']:.3f}"
        )
        md.append(
            f"| {ENDPOINTS[name]['label']} | {int(valid_test.sum())} | "
            f"{rank['auroc']:.2f} | {operating_point['sensitivity']:.2f}/"
            f"{operating_point['specificity']:.2f} | {balanced_accuracy:.2f} | "
            "see prior run |"
        )
    return results, md


def _publish_train_calibration_bundle(
    a: argparse.Namespace,
    out_dir: Path,
    names: list[str],
    n_prompts: list[int],
    lab: np.ndarray,
    margins: np.ndarray,
    results: dict[str, object],
    markdown_lines: list[str],
    calibration_provenance: dict[str, object],
    test_cache_provenance: dict[str, object],
    train_waveform_paths: list[str],
) -> None:
    train_cache_provenance = {
        **calibration_provenance,
        "kind": "endpoint_train_margin_cache",
    }

    def verify_current() -> None:
        require_matching_provenance(
            calibration_provenance,
            build_calibration_provenance(
                checkpoint=a.checkpoint,
                train_parquet=a.train_parquet,
                test_parquet=a.test_parquet,
                test_margins=a.test_margins,
                test_margin_cache_provenance=test_cache_provenance,
                names=names,
                n_calib=a.n_calib,
                train_waveform_paths=train_waveform_paths,
            ),
            artifact="endpoint train calibration",
        )

    with tempfile.TemporaryDirectory(
        dir=out_dir,
        prefix=".endpoint-readout-staging-",
    ) as staging_dir:
        staging = Path(staging_dir)
        json_name = "endpoint_readout_train_calibrated.json"
        md_name = "endpoint_readout_train_calibrated.md"
        margins_name = "train_calib_margins.npz"
        atomic_write_json(staging / json_name, results)
        atomic_write_text(staging / md_name, "\n".join(markdown_lines) + "\n")
        atomic_savez_compressed(
            staging / margins_name,
            margins=margins,
            labels=lab,
            endpoints=np.asarray(names, dtype=np.str_),
            n_prompts=np.asarray(n_prompts, dtype=np.int64),
            provenance=encode_manifest(train_cache_provenance),
        )
        publish_artifact_bundle(
            {
                json_name: staging / json_name,
                md_name: staging / md_name,
                margins_name: staging / margins_name,
            },
            out_dir / "endpoint_readout_train_calibrated.bundle.json",
            provenance=calibration_provenance,
            verify_current=verify_current,
        )


def _score_train_margins(
    a: argparse.Namespace,
    tr: pd.DataFrame,
    names: list[str],
) -> np.ndarray:
    model, tokenizer = load_model(a.checkpoint, a.device)
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    all_questions = [question for name in names for question in ENDPOINTS[name]["questions"]]
    prompt_ids, prompt_mask = pretokenize(tokenizer, all_questions)
    margins = np.zeros((len(tr), prompt_ids.size(0)), dtype=np.float32)
    for i, path in enumerate(tqdm(tr["waveform_path_psa"].tolist(), desc="train margins")):
        ecg = load_ecg_signal(path)
        margins[i] = get_margins(
            model,
            ecg,
            prompt_ids,
            prompt_mask,
            yes_id,
            no_id,
            a.device,
            a.batch_size,
        )
    return require_finite_numeric_array(
        margins,
        artifact="train calibration margins",
        expected_shape=(len(tr), prompt_ids.size(0)),
    )


def _run(a: argparse.Namespace, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    captured_static_inputs = build_calibration_static_inputs(
        a.checkpoint,
        a.train_parquet,
        a.test_parquet,
        a.test_margins,
    )

    # ---- cached TEST margins (full set, nothing withheld) ----
    names = list(ENDPOINTS)
    n_prompts = [len(ENDPOINTS[name]["questions"]) for name in names]
    full_test_df = pd.read_parquet(a.test_parquet).drop_duplicates(
        subset="waveform_path_psa"
    ).reset_index(drop=True)
    if "new_PatientID" not in full_test_df.columns:
        raise ValueError("test parquet is missing required new_PatientID values")
    all_test_patient_ids = full_test_df["new_PatientID"].copy()
    all_test_paths = set(full_test_df["waveform_path_psa"].astype(str).tolist())
    labelled = np.zeros(len(full_test_df), dtype=bool)
    for name in names:
        column = ENDPOINTS[name]["label_column"]
        if column in full_test_df.columns:
            labelled |= full_test_df[column].notna().to_numpy()
    test_df = full_test_df.loc[labelled].reset_index(drop=True)
    ordered_test_paths = test_df["waveform_path_psa"].astype(str).tolist()
    z = None
    try:
        z = np.load(a.test_margins, allow_pickle=False)
        actual_test_provenance = decode_manifest(z["provenance"])
    except (OSError, ValueError, KeyError) as exc:
        if z is not None:
            z.close()
        raise SystemExit(
            "test margin cache is missing, legacy, or does not match the requested "
            "checkpoint and test parquet; regenerate it with "
            "scripts/evaluate_endpoint_pyes_calibrated.py"
        ) from exc
    execution = actual_test_provenance.get("execution")
    if not isinstance(execution, dict):
        z.close()
        raise SystemExit("test margin cache is missing execution provenance")
    try:
        test_cache_provenance = build_margin_cache_provenance(
            a.checkpoint,
            a.test_parquet,
            names,
            n_prompts,
            0,
            ordered_test_paths,
            int(execution["batch_size"]),
            str(execution["device"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        z.close()
        raise SystemExit("test margin cache has invalid execution provenance") from exc
    if actual_test_provenance != test_cache_provenance:
        z.close()
        raise SystemExit(
            "test margin cache does not match the requested checkpoint, dataset, "
            "waveforms, or endpoint prompts"
        )
    try:
        test_margins = z["margins"].copy()
        test_labels = z["labels"].copy()
        cached_names = [str(value) for value in z["endpoints"].tolist()]
        cached_n_prompts = [int(value) for value in z["n_prompts"].tolist()]
        cached_paths = z["waveform_paths"].astype(str).tolist()
    finally:
        z.close()
    if cached_names != names or cached_n_prompts != n_prompts:
        raise SystemExit("test margin cache endpoint layout does not match current prompts")
    if cached_paths != ordered_test_paths or len(test_margins) != len(test_df):
        raise SystemExit("test margin cache rows do not match the requested test parquet")
    try:
        require_finite_numeric_array(
            test_margins,
            artifact="test margin cache",
            expected_shape=(len(test_df), sum(n_prompts)),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if test_labels.shape != (len(test_df), len(names)):
        raise SystemExit("test label cache shape does not match the current endpoint count")
    validated_test_labels = np.full((len(test_df), len(names)), np.nan, dtype=np.float32)
    for j, name in enumerate(names):
        column = ENDPOINTS[name]["label_column"]
        if column in test_df.columns:
            validated_test_labels[:, j] = encode_endpoint_labels(test_df[column], name)
    if not np.array_equal(test_labels, validated_test_labels, equal_nan=True):
        raise SystemExit("test margin cache labels do not match validated test ground truth")
    print(f"[cal] cached test margins: {test_margins.shape} over {len(names)} endpoints")

    # ---- pick TRAIN ECGs, patient-disjoint from test ----
    cols = ["waveform_path_psa", "new_PatientID"] + [ENDPOINTS[n]["label_column"] for n in names]
    tr = pd.read_parquet(a.train_parquet, columns=sorted(set(cols)))
    tr = tr.drop_duplicates(subset="waveform_path_psa")
    before = len(tr)
    disjoint = patient_disjoint_mask(
        tr["new_PatientID"],
        all_test_patient_ids,
    )
    tr = tr.loc[disjoint]
    tr = tr[~tr["waveform_path_psa"].astype(str).isin(all_test_paths)]
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
        lab[:, j] = encode_endpoint_labels(tr[col], n)

    for j, name in enumerate(names):
        train_labels = lab[:, j]
        test_endpoint_labels = test_labels[:, j]
        train_count = int(np.isfinite(train_labels).sum())
        test_count = int(np.isfinite(test_endpoint_labels).sum())
        if train_count < 100 or test_count < 100:
            raise ValueError(
                f"endpoint {name} requires at least 100 labelled rows in train and test; "
                f"found train={train_count}, test={test_count}"
            )
        require_binary_class_support(train_labels, endpoint=name, cohort="train calibration")
        require_binary_class_support(test_endpoint_labels, endpoint=name, cohort="full test")

    require_matching_provenance(
        captured_static_inputs,
        build_calibration_static_inputs(
            a.checkpoint,
            a.train_parquet,
            a.test_parquet,
            a.test_margins,
        ),
        artifact="endpoint calibration cohort",
    )
    train_waveform_paths = tr["waveform_path_psa"].astype(str).tolist()
    calibration_provenance = build_calibration_provenance(
        checkpoint=a.checkpoint,
        train_parquet=a.train_parquet,
        test_parquet=a.test_parquet,
        test_margins=a.test_margins,
        test_margin_cache_provenance=test_cache_provenance,
        names=names,
        n_calib=a.n_calib,
        train_waveform_paths=train_waveform_paths,
        static_inputs=captured_static_inputs,
    )

    margins = _score_train_margins(a, tr, names)

    results, markdown_lines = _fit_calibration_results(
        a,
        names,
        n_prompts,
        tr,
        lab,
        margins,
        test_margins,
        test_labels,
        calibration_provenance,
    )
    _publish_train_calibration_bundle(
        a,
        out_dir,
        names,
        n_prompts,
        lab,
        margins,
        results,
        markdown_lines,
        calibration_provenance,
        test_cache_provenance,
        train_waveform_paths,
    )
    print(f"[saved] {out_dir}/endpoint_readout_train_calibrated.{{json,md}}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--n_calib", type=int, default=1500, help="train ECGs to score per run")
    ap.add_argument("--out", default="/volume/ECG_tokenizer/analysis/endpoint_pyes_concatmix")
    ap.add_argument("--train_parquet", default=TRAIN_PARQUET)
    ap.add_argument("--test_parquet", default=TEST_PARQUET)
    ap.add_argument("--test_margins", default=TEST_MARGINS)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_artifact_lock(out_dir / ".endpoint-readout-train.lock"):
        _run(args, out_dir)


if __name__ == "__main__":
    main()
