#!/usr/bin/env python3
"""CLI for exporting SigLIP text bank and ECG-text mapping from the MHI parquet."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on the path when invoked directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SigLIP alignment text bank and mapping directly from the MHI parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--parquet",
        type=str,
        default="/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet",
        help="Path to the wide MHI parquet (used as metadata when --labels-path is not provided)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet",
        help="Metadata parquet path (overrides --parquet when provided)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet",
        help="Separate labels parquet to merge with metadata (ground-truth and BERT scores)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ecg_text_alignment",
        help="Directory where the SigLIP text bank and mapping will be saved",
    )
    parser.add_argument(
        "--train-ecgs",
        type=int,
        default=None,
        help="Maximum train split ECGs to retain when sampling (None keeps all)",
    )
    parser.add_argument(
        "--val-ecgs",
        type=int,
        default=10000,
        help="Maximum validation split ECGs to retain when sampling",
    )
    parser.add_argument(
        "--val-balanced-limit",
        type=int,
        default=None,
        help=(
            "If provided, balance the validation split across DEEPECG labels with an equal per-label target "
            "up to this total size"
        ),
    )
    parser.add_argument(
        "--val-balanced-min-per-label",
        type=int,
        default=50,
        help="Minimum desired number of validation ECGs per DEEPECG label when balancing",
    )
    parser.add_argument(
        "--test-ecgs",
        type=int,
        default=None,
        help="Maximum test split ECGs to retain when sampling",
    )
    parser.add_argument(
        "--max-normal-percentage",
        type=float,
        default=0.05,
        help="Maximum fraction of normal ECGs to retain per split",
    )
    parser.add_argument(
        "--max-par-ailleurs-percentage",
        type=float,
        default=0.05,
        help="Maximum fraction of ECGs per split whose diagnosis contains 'ECG normal par ailleurs'",
    )
    parser.add_argument(
        "--max-positive-per-label",
        type=int,
        default=None,
        help="Cap on positive samples per text label (None keeps all)",
    )
    parser.add_argument(
        "--max-val-ecgs",
        type=int,
        default=10000,
        help="Maximum number of unique ECGs to keep in the validation split",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=1000,
        dest="sample_checks",
        help="Number of ECGs to sample when printing exclusivity diagnostics",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for sampling",
    )
    parser.add_argument(
        "--implicit-neg-sample-size",
        type=int,
        default=64,
        help="Implicit negative sample size per ECG",
    )
    parser.add_argument(
        "--max-hardneg-per-group",
        type=int,
        default=3,
        help="Maximum hard negatives to keep per exclusivity group",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="Positive label weight for SigLIP mapping",
    )
    parser.add_argument(
        "--hardneg-weight",
        type=float,
        default=3.0,
        help="Hard negative weight for SigLIP mapping",
    )
    parser.add_argument(
        "--implneg-weight",
        type=float,
        default=0.2,
        help="Implicit negative weight for SigLIP mapping",
    )
    parser.add_argument(
        "--sample-size-for-checks",
        type=int,
        dest="sample_checks",
        help="Alias for --sample-checks (legacy flag)",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Disable generation of QA text variants (defaults to include)",
    )
    parser.add_argument(
        "--diagnosis-column",
        type=str,
        default="translated_diagnosis",
        help=(
            "Column in the metadata/labels parquet that contains the free-text diagnosis/report. "
            "When the requested column is unavailable the script falls back to 'diagnosis'. "
            "Use an empty string or 'none' to skip adding diagnosis text to the mapping."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dataset_generation.generate_train_test_datasets import generate_siglip_alignment_dataset

    metadata_path = args.metadata_path or args.parquet

    diagnosis_column: str | None
    if args.diagnosis_column is None:
        diagnosis_column = None
    else:
        diag_value = args.diagnosis_column.strip()
        diagnosis_column = None if diag_value == "" or diag_value.lower() == "none" else diag_value

    text_bank_path, mapping_path = generate_siglip_alignment_dataset(
        parquet_path=metadata_path,
        output_dir=args.output_dir,
        include_qa=not args.no_qa,
        w_pos=args.pos_weight,
        w_hardneg=args.hardneg_weight,
        w_implneg=args.implneg_weight,
        sample_size_for_checks=args.sample_checks,
        random_state=args.random_seed,
        implicit_negative_sample_size=args.implicit_neg_sample_size,
        max_hardneg_per_group=args.max_hardneg_per_group,
        train_ecg_limit=args.train_ecgs,
        val_ecg_limit=args.val_ecgs,
        val_balanced_limit=args.val_balanced_limit,
        val_balanced_min_per_label=args.val_balanced_min_per_label,
        test_ecg_limit=args.test_ecgs,
        max_normal_percentage=args.max_normal_percentage,
        max_par_ailleurs_percentage=args.max_par_ailleurs_percentage,
        max_validation_ecgs=args.max_val_ecgs,
        max_positive_per_label=args.max_positive_per_label,
        labels_path=args.labels_path,
        diagnosis_column=diagnosis_column,
    )

    print("\nArtifacts saved:")
    print(f"  Text bank:   {text_bank_path}")
    print(f"  Mapping CSV: {mapping_path}")


if __name__ == "__main__":
    main()
