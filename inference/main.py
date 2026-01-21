#!/usr/bin/env python3
"""
ECG Tokenizer Inference Pipeline - Main Entry Point.

This module follows the DeepECG_Docker pattern with clean entry point,
mode-based execution, and configuration-driven setup.

Usage:
    python inference/main.py --input data.parquet --output results.json
    python inference/main.py --config heartwise.config
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.pipeline_config import PipelineConfig
from inference.pipeline_args import PipelineArgs


class Mode(Enum):
    """Pipeline execution modes."""
    PREPROCESSING = "preprocessing"
    ANALYSIS = "analysis"
    FULL_RUN = "full_run"


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
    Validate and load input parquet file.
    
    Args:
        args: Pipeline arguments containing input path.
        
    Returns:
        Loaded DataFrame with only required columns.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(args.input_parquet)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading input DataFrame from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Check for required columns
    required_columns = [args.waveform_path_column, args.report_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available = list(df.columns)[:10]
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns (first 10): {available}"
        )
    
    # Warn about optional columns
    if args.waveform_name_column not in df.columns:
        print(f"  Note: '{args.waveform_name_column}' column not found, will use index")
    
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
    
    # Create pipeline config from args
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
        waveform_path_column=args.waveform_path_column,
        report_column=args.report_column,
        waveform_name_column=args.waveform_name_column,
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
    mode = Mode(args.mode) if args.mode else Mode.FULL_RUN
    
    print("=" * 60)
    print("ECG Tokenizer Inference Pipeline")
    print("=" * 60)
    print(f"Mode: {mode.value}")
    print(f"Input: {args.input_parquet}")
    print(f"Output: {args.output_json}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60 + "\n")
    
    # Setup directories
    setup_directories(args)
    
    # Validate and load input
    df = validate_input(args)
    
    # Run based on mode
    if mode == Mode.PREPROCESSING:
        print("Preprocessing mode not yet implemented for ECG Tokenizer")
        print("Use full_run or analysis mode instead")
        return
    
    if mode in (Mode.ANALYSIS, Mode.FULL_RUN):
        results = run_analysis(args, df)
        save_results(results, args.output_json, args.output_dir)
        print_summary(results)


if __name__ == "__main__":
    args = PipelineArgs.parse_arguments()
    
    print("\nConfiguration Summary:")
    print(f"  Input Parquet: {args.input_parquet}")
    print(f"  Output JSON: {args.output_json}")
    print(f"  Device: {args.device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  PSA Normalization: {args.apply_psa_normalization}")
    print(f"  BERT as Ground Truth: {args.use_bert_as_ground_truth}")
    print(f"  Report Generation: {args.enable_report_generation}")
    print(f"  Run LLM Judge: {args.run_llm_judge}")
    
    main(args)
