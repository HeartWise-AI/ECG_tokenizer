#!/usr/bin/env python3
"""
ECG Tokenizer Inference Pipeline - Main Entry Point.

This module follows the DeepECG_Docker pattern with clean entry point,
mode-based execution, and configuration-driven setup.

Usage:
    python inference/preprocessing.py --input data.parquet --output results.json
    python inference/preprocessing.py --config heartwise.config

Required Input Columns:
    - reports: Text reports for BERT classification
    - ecg_path: Absolute/relative path to ECG file (authoritative)
"""

import sys
import re
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.pipeline_args import PipelineArgs
from utils.files_handler import load_df, save_df
from utils.preprocessing.analysis_pipeline import AnalysisPipeline
from utils.constants import DIAGNOSIS_COLUMN


def setup_directories(args: PipelineArgs) -> None:
    """
    Set up necessary directories for output.
    """
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def validate_input(args: PipelineArgs) -> pd.DataFrame:
    """
    Validate and load input file (CSV or Parquet).
    
    Required columns (standardized like DeepECG_Docker):
        - reports: Text reports for BERT classification
        - ecg_path: Absolute/relative path to ECG file
    
    Args:
        args: Pipeline arguments containing input path.
        
    Returns:
        Loaded DataFrame with required columns validated.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(args.input_parquet)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading input DataFrame from {input_path}")
    df = load_df(str(input_path))
    print(f"Loaded {len(df)} samples")
    
    # Check for required columns (standardized)
    if DIAGNOSIS_COLUMN not in df.columns or "ecg_path" not in df.columns:
        available = list(df.columns)[:15]
        raise ValueError(
            f"Missing required columns. Expected '{DIAGNOSIS_COLUMN}' and 'ecg_path'. "
            f"Available columns (first 15): {available}"
        )
    
    # Remove rows with empty reports
    missing_reports_count = df[DIAGNOSIS_COLUMN].isna().sum()
    if missing_reports_count > 0:
        df = df.dropna(subset=[DIAGNOSIS_COLUMN]).reset_index(drop=True)
        print(f"  Removed {missing_reports_count} rows with empty '{DIAGNOSIS_COLUMN}' column")
    
    print(f"  Valid samples: {len(df)}")
    
    return df


def run_preprocessing(args: PipelineArgs, df: pd.DataFrame) -> Path:
    """
    Run preprocessing to save signals as .npy files.
    """
    print("\n" + "=" * 60)
    print("Running Preprocessing")
    print("=" * 60)
    print(f"ECG Signals Path: {args.ecg_signals_path}")
    dataset_tag = args.dataset_name or Path(args.input_parquet).stem
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dataset_tag)).strip("._-") or "dataset"
    preprocessing_folder = Path(args.preprocessing_folder) / f"{prefix}_preprocessing"
    args.preprocessing_folder = str(preprocessing_folder)

    print(f"Output folder: {args.preprocessing_folder}")
    print(f"Workers: {args.preprocessing_n_workers}")
    print("=" * 60 + "\n")

    processed_df = AnalysisPipeline.save_and_preprocess_data(
        df=df,
        output_folder=args.output_dir,
        preprocessing_folder=str(preprocessing_folder),
        preprocessing_n_workers=args.preprocessing_n_workers,
        path_column="ecg_path",
        include_optional_100hz_cluster=args.include_optional_100hz_cluster,
        scale_factor=args.ecg_scale,
        source=args.ecg_source or args.dataset_name,
    )

    if args.preprocessing_output:
        output_parquet = Path(args.preprocessing_output)
    else:
        output_parquet = Path(args.output_dir) / f"{prefix}_preprocessed_data.parquet"
    save_df(processed_df, str(output_parquet))
    print(f"Preprocessed DataFrame saved to {output_parquet}")
    return output_parquet


def run_bert_classification(args: PipelineArgs, input_parquet: str, output_parquet: Optional[str] = None) -> str:
    """Run BERT 77-class classification and write labels into the input parquet."""
    from inference.run_bert_subprocess import run_bert_on_parquet
    run_bert_on_parquet(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        base_config=args.bert_base_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    return output_parquet or input_parquet


def main(args: PipelineArgs) -> None:
    """
    Main entry point for the ECG Tokenizer inference pipeline.
    
    Args:
        args: Pipeline arguments parsed from config/CLI.
    """
    print("=" * 60)
    print("ECG Tokenizer Inference Pipeline")
    print("=" * 60)
    print(f"Input: {args.input_parquet}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"ECG Signals Path: {args.ecg_signals_path}")
    print("=" * 60 + "\n")
    
    # Setup directories
    setup_directories(args)
    
    # Resolve intermediate paths
    if not args.preprocessing_output:
        args.preprocessing_output = str(Path(args.output_dir) / "preprocessed.parquet")
    
    # Step 0/1: preprocessing
    current_file = args.input_parquet
    if args.use_preprocessing:
        df_input = validate_input(args)
        preprocessed_parquet = run_preprocessing(args, df_input)
        current_file = str(preprocessed_parquet)
    elif args.use_bert_classification:
        # BERT-only mode: use input file directly (CSV or parquet)
        current_file = args.input_parquet
    else:
        if not Path(args.preprocessing_output).exists():
            raise FileNotFoundError(
                f"Preprocessing skipped but {args.preprocessing_output} not found. "
                "Run with --run-step preprocess first."
            )
        current_file = args.preprocessing_output
    
    # Set default BERT output path based on input file
    if not args.bert_output:
        input_stem = Path(current_file).stem
        args.bert_output = str(Path(args.output_dir) / f"{input_stem}_bert_output.parquet")
    
    # Step 2: BERT classification (only if requested)
    if args.use_bert_classification:
        current_file = run_bert_classification(
            args,
            input_parquet=current_file,
            output_parquet=args.bert_output,
        )
    else:
        current_file = args.bert_output
    
    # Step 3: EfficientNet runs externally via scripts/runner.sh


if __name__ == "__main__":
    args = PipelineArgs.parse_arguments()
    
    print("\nConfiguration Summary:")
    print(f"  Input: {args.input_parquet}")
    print(f"  ECG Signals Path: {args.ecg_signals_path}")
    print(f"  Device: {args.device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Required Columns: {DIAGNOSIS_COLUMN}, ecg_path")
    
    main(args)
