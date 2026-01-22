#!/usr/bin/env python3
"""
ECG Tokenizer Inference Pipeline - Main Entry Point.

This module follows the DeepECG_Docker pattern with clean entry point,
mode-based execution, and configuration-driven setup.

Usage:
    python inference/main.py --input data.parquet --output results.json
    python inference/main.py --config heartwise.config

Required Input Columns:
    - diagnosis: Text report/diagnosis for BERT classification
    - ecg_path: Absolute/relative path to ECG file (authoritative)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.pipeline_config import PipelineConfig
from inference.pipeline_args import PipelineArgs
from inference.files_handler import load_df, save_df
from utils.constants import DIAGNOSIS_COLUMN, ECG_PATTERNS, Mode


def setup_directories(args: PipelineArgs) -> None:
    """
    Set up necessary directories for output.
    
    Args:
        args: Pipeline arguments containing output paths.
    """
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def validate_input(args: PipelineArgs) -> pd.DataFrame:
    """
    Validate and load input file (CSV or Parquet).
    
    Required columns (standardized like DeepECG_Docker):
        - diagnosis: Text report/diagnosis for BERT classification
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
            "Missing required columns. Expected 'diagnosis' and 'ecg_path'. "
            f"Available columns (first 15): {available}"
        )
    
    # Remove rows with empty diagnosis
    missing_diagnosis_count = df[DIAGNOSIS_COLUMN].isna().sum()
    if missing_diagnosis_count > 0:
        df = df.dropna(subset=[DIAGNOSIS_COLUMN]).reset_index(drop=True)
        print(f"  Removed {missing_diagnosis_count} rows with empty '{DIAGNOSIS_COLUMN}' column")
    
    print(f"  Valid samples: {len(df)}")
    
    return df


def run_analysis(args: PipelineArgs, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the full analysis pipeline.
    
    Args:
        args: Pipeline arguments.
        df: Input DataFrame.
        
    Returns:
        Dictionary containing all results and metrics.
    """
    from inference.ecg_pipeline import ECGInferencePipeline
    
    # Create pipeline config from args (using standardized column names)
    config = PipelineConfig(
        input_parquet=args.input_parquet,
        output_json=args.output_json,
        output_dir=args.output_dir,
        bert_checkpoint=args.bert_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        efficientnet_checkpoint=args.efficientnet_checkpoint,
        gpt2_checkpoint=args.gpt2_checkpoint,
        enable_report_generation=args.enable_report_generation,
        device=args.device,
        cpu_fallback=args.cpu_fallback,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_psa_normalization=args.apply_psa_normalization,
        psa_region=args.psa_region,
        ecg_signals_path=args.ecg_signals_path,
        num_classes=args.num_classes,
        classification_threshold=args.classification_threshold,
        max_report_length=args.max_report_length,
        max_prompts_per_ecg=args.max_prompts_per_ecg,
        compute_rouge=args.compute_text_metrics,
        compute_bleu=args.compute_text_metrics,
        compute_meteor=args.compute_text_metrics,
        run_llm_judge=args.run_llm_judge,
        verbose=args.verbose,
    )
    
    # Run pipeline
    pipeline = ECGInferencePipeline(config)
    results = pipeline.run_pipeline()
    
    return results


def run_preprocessing(args: PipelineArgs, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run preprocessing to save signals as .base64 files.
    
    Following DeepECG_Docker pattern for preprocessing mode.
    
    Args:
        args: Pipeline arguments.
        df: Input DataFrame.
        
    Returns:
        DataFrame with updated paths pointing to .base64 files.
    """
    from inference.ecg_pipeline import ECGInferencePipeline
    
    print("\n" + "=" * 60)
    print("Running Preprocessing Mode")
    print("=" * 60)
    print(f"ECG Signals Path: {args.ecg_signals_path}")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_id}_{args.dataset_name}" if args.dataset_name else run_id
    preprocessing_folder = Path(args.preprocessing_folder) / f"{prefix}_preprocessing"
    args.preprocessing_folder = str(preprocessing_folder)
    
    print(f"Output folder: {args.preprocessing_folder}")
    print(f"Workers: {args.preprocessing_n_workers}")
    print("=" * 60 + "\n")
    
    # Create preprocessing folder
    preprocessing_folder.mkdir(parents=True, exist_ok=True)
    
    # Create minimal config for preprocessing (using standardized column names)
    config = PipelineConfig(
        input_parquet=args.input_parquet,
        output_json=args.output_json,
        device=args.device,
        cpu_fallback=args.cpu_fallback,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_psa_normalization=args.apply_psa_normalization,
        psa_region=args.psa_region,
        ecg_signals_path=args.ecg_signals_path,
        bert_checkpoint=args.bert_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        verbose=args.verbose,
    )
    
    # Create pipeline and run preprocessing
    pipeline = ECGInferencePipeline(config)
    processed_df = pipeline.save_and_preprocess_data(
        df=df,
        preprocessing_folder=args.preprocessing_folder,
        preprocessing_n_workers=args.preprocessing_n_workers
    )

    # Add BERT 77-class predictions to the preprocessed parquet
    diagnoses = processed_df[DIAGNOSIS_COLUMN].tolist()
    bert_results = pipeline.run_bert_classification(diagnoses)
    bert_rows = []
    for result in bert_results:
        preds = result.get("predictions", [])
        if preds and len(preds) == len(ECG_PATTERNS):
            bert_rows.append(preds)
        else:
            bert_rows.append([0] * len(ECG_PATTERNS))
    bert_df = pd.DataFrame(bert_rows, columns=ECG_PATTERNS)
    processed_df = pd.concat([processed_df.reset_index(drop=True), bert_df], axis=1)
    
    # Save processed DataFrame with updated paths
    output_parquet = Path(args.output_dir) / f"{prefix}_preprocessed_data.parquet"
    save_df(processed_df, str(output_parquet))
    print(f"Preprocessed DataFrame saved to {output_parquet}")
    
    return processed_df


def update_paths_for_analysis(df: pd.DataFrame, args: PipelineArgs) -> pd.DataFrame:
    """
    Update DataFrame paths to point to preprocessed .base64 files.
    
    Args:
        df: Input DataFrame.
        args: Pipeline arguments.
        
    Returns:
        DataFrame with updated ecg_path column.
    """
    preprocessing_folder = args.preprocessing_folder
    
    def get_base64_path(ecg_path):
        if not ecg_path:
            return ecg_path
        file_id = Path(ecg_path).stem
        return str(Path(preprocessing_folder) / f"{file_id}.base64")
    
    df = df.copy()
    df["ecg_path"] = df["ecg_path"].apply(get_base64_path)
    return df


def save_results(
    results: Dict[str, Any],
    output_json: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Pipeline results dictionary.
        output_json: Path to output JSON file.
        output_dir: Optional directory for additional outputs.
    """
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Save summary CSV if output_dir specified
    if output_dir:
        summary_path = Path(output_dir) / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create summary DataFrame
        summary_data = []
        for result in results.get("results", []):
            row = {
                "waveform_name": result.get("waveform_name", ""),
                "has_bert_classification": "bert_classification" in result,
                "has_efficientnet_classification": "efficientnet_classification" in result,
                "has_generated_report": bool(result.get("generated_report")),
                "num_qa_pairs": len(result.get("qa_results", [])),
            }
            
            # Add BERT summary if available
            bert = result.get("bert_classification", {})
            if bert.get("predictions"):
                row["bert_positive_classes"] = sum(bert["predictions"])
            
            # Add EfficientNet summary if available
            eff = result.get("efficientnet_classification", {})
            if eff.get("predictions"):
                row["efficientnet_positive_classes"] = sum(eff["predictions"])
            
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the pipeline results."""
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    
    num_samples = results.get("metadata", {}).get("num_samples", 0)
    print(f"Total samples processed: {num_samples}")
    
    # Classification metrics summary
    class_metrics = results.get("aggregate_metrics", {}).get("classification_metrics", {})
    
    if class_metrics.get("signal_vs_text"):
        sv_metrics = class_metrics["signal_vs_text"]
        print(f"\nSignal vs Text Classification (EfficientNet vs BERT):")
        if "overall_macro_auc" in sv_metrics:
            print(f"  Overall Macro AUC: {sv_metrics['overall_macro_auc']:.4f}")
    
    if class_metrics.get("bert"):
        bert_metrics = class_metrics["bert"]
        if "overall_macro_auc" in bert_metrics:
            print(f"\nBERT Classification:")
            print(f"  Overall Macro AUC: {bert_metrics['overall_macro_auc']:.4f}")
    
    # Text metrics summary
    text_metrics = results.get("aggregate_metrics", {}).get("text_metrics", {})
    if text_metrics:
        print(f"\nText Generation Metrics:")
        for metric, value in text_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    print("=" * 60)


def main(args: PipelineArgs) -> None:
    """
    Main entry point for the ECG Tokenizer inference pipeline.
    
    Args:
        args: Pipeline arguments parsed from config/CLI.
    """
    mode = args.mode if args.mode else Mode.FULL_RUN
    
    print("=" * 60)
    print("ECG Tokenizer Inference Pipeline")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Input: {args.input_parquet}")
    print(f"Output: {args.output_json}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"ECG Signals Path: {args.ecg_signals_path}")
    print("=" * 60 + "\n")
    
    # Setup directories
    setup_directories(args)
    
    # Validate and load input
    df = validate_input(args)
    
    # Run based on mode
    if mode == Mode.PREPROCESSING:
        run_preprocessing(args, df)
        return
    
    if mode == Mode.FULL_RUN:
        # Full run: preprocessing + analysis
        run_preprocessing(args, df)
        # Update waveform paths to point to preprocessed files
        df = update_paths_for_analysis(df, args)
    
    if mode in (Mode.ANALYSIS, Mode.FULL_RUN):
        results = run_analysis(args, df)
        save_results(results, args.output_json, args.output_dir)
        print_summary(results)


if __name__ == "__main__":
    args = PipelineArgs.parse_arguments()
    
    print("\nConfiguration Summary:")
    print(f"  Mode: {args.mode}")
    print(f"  Input: {args.input_parquet}")
    print(f"  Output: {args.output_json}")
    print(f"  ECG Signals Path: {args.ecg_signals_path}")
    print(f"  Device: {args.device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  PSA Normalization: {args.apply_psa_normalization}")
    print(f"  Report Generation: {args.enable_report_generation}")
    print(f"  Required Columns: {DIAGNOSIS_COLUMN}, ecg_path")
    
    main(args)
