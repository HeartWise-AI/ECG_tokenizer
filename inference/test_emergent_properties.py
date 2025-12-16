#!/usr/bin/env python3
"""
Test emergent properties of the ECG MedGemma model.

This script tests whether the model can answer questions about ECG measurements
(QT interval, heart rate, PR interval, etc.) that it was NOT trained on.

The model was trained primarily on rhythm-related questions; we test here if it
developed emergent understanding of interval measurements.
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Import the main generate_answer function which handles all model loading
from inference.generate_ecg_answer import generate_answer as _generate_answer_single


# Emergent property test questions - questions about data NOT in training
EMERGENT_PROPERTY_QUESTIONS = {
    "heart_rate": {
        "questions": [
            "What is the heart rate in this ECG?",
            "What is the ventricular rate?",
        ],
        "gt_column": "RestingECG_OriginalRestingECGMeasurements_VentricularRate",
        "unit": "bpm",
        "tolerance": 10,  # +/- 10 bpm considered correct
    },
    "pr_interval": {
        "questions": [
            "What is the PR interval in this ECG?",
            "What is the PR interval in milliseconds?",
        ],
        "gt_column": "RestingECG_OriginalRestingECGMeasurements_PRInterval",
        "unit": "ms",
        "tolerance": 20,  # +/- 20 ms
    },
    "qrs_duration": {
        "questions": [
            "What is the QRS duration in this ECG?",
            "What is the QRS complex duration in milliseconds?",
        ],
        "gt_column": "RestingECG_OriginalRestingECGMeasurements_QRSDuration",
        "unit": "ms",
        "tolerance": 15,
    },
    "qt_interval": {
        "questions": [
            "What is the QT interval in this ECG?",
            "What is the QT interval in milliseconds?",
        ],
        "gt_column": "RestingECG_OriginalRestingECGMeasurements_QTInterval",
        "unit": "ms",
        "tolerance": 30,
    },
    "qtc": {
        "questions": [
            "What is the corrected QT interval (QTc) in this ECG?",
            "What is the QTc?",
        ],
        "gt_column": "RestingECG_OriginalRestingECGMeasurements_QTCorrected",
        "unit": "ms",
        "tolerance": 30,
    },
}

# Questions about demographics/clinical that model might have seen indirectly
CLINICAL_QUESTIONS = {
    "age": {
        "questions": [
            "Based on the ECG, what is the estimated patient age?",
        ],
        "gt_column": "age_at_ecg",
        "unit": "years",
        "tolerance": 10,
    },
    "gender": {
        "questions": [
            "Based on the ECG, is this patient male or female?",
        ],
        "gt_column": "gender",
        "unit": None,
        "tolerance": None,
    },
}


def extract_numeric_from_answer(answer: str, unit: str = None) -> Optional[float]:
    """Try to extract a numeric value from the model's answer."""
    import re
    
    # Try to find numbers in the answer
    # Pattern: optional negative, digits, optional decimal
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    
    if not numbers:
        return None
    
    # Return the first number found
    try:
        return float(numbers[0])
    except ValueError:
        return None


def compare_values(predicted: Optional[float], ground_truth: float, tolerance: float) -> Dict[str, Any]:
    """Compare predicted value with ground truth."""
    if predicted is None:
        return {
            "match": False,
            "predicted": None,
            "ground_truth": ground_truth,
            "error": None,
            "within_tolerance": False,
            "reason": "Could not extract numeric value from answer"
        }
    
    error = abs(predicted - ground_truth)
    within_tolerance = error <= tolerance
    
    return {
        "match": within_tolerance,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "error": error,
        "within_tolerance": within_tolerance,
        "reason": None
    }


def load_test_samples(
    test_qa_path: str,
    n_samples: int = 10,
    require_all_measurements: bool = False,
) -> pd.DataFrame:
    """Load test samples that have ground truth measurements."""
    
    print(f"Loading test QA data from: {test_qa_path}")
    df = pd.read_parquet(test_qa_path)
    print(f"Total samples: {len(df)}")
    
    # Filter to samples that have measurement ground truth
    measurement_cols = [
        "RestingECG_OriginalRestingECGMeasurements_VentricularRate",
        "RestingECG_OriginalRestingECGMeasurements_PRInterval",
        "RestingECG_OriginalRestingECGMeasurements_QRSDuration",
        "RestingECG_OriginalRestingECGMeasurements_QTInterval",
        "RestingECG_OriginalRestingECGMeasurements_QTCorrected",
    ]
    
    if require_all_measurements:
        mask = df[measurement_cols].notna().all(axis=1)
    else:
        mask = df[measurement_cols].notna().any(axis=1)
    
    df_filtered = df[mask].copy()
    print(f"Samples with measurements: {len(df_filtered)}")
    
    # Filter to existing waveform files
    df_filtered = df_filtered[df_filtered['waveform_path_psa'].apply(os.path.exists)]
    print(f"Samples with existing waveforms: {len(df_filtered)}")
    
    # Sample random subset
    if n_samples < len(df_filtered):
        df_sampled = df_filtered.sample(n=n_samples, random_state=42)
    else:
        df_sampled = df_filtered
    
    print(f"Selected {len(df_sampled)} samples for testing")
    return df_sampled


def run_emergent_property_tests(
    checkpoint_path: str,
    test_df: pd.DataFrame,
    device_str: str,
    property_configs: Dict[str, Dict],
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run emergent property tests on all samples."""
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(test_df),
            "properties_tested": list(property_configs.keys()),
            "checkpoint": checkpoint_path,
        },
        "samples": [],
        "summary": {},
    }
    
    # Track per-property results
    property_results = {prop: {"correct": 0, "total": 0, "errors": []} for prop in property_configs}
    
    for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Testing samples")):
        sample_result = {
            "sample_idx": idx,
            "waveform_path": row['waveform_path_psa'],
            "properties": {},
        }
        
        for prop_name, prop_config in property_configs.items():
            gt_col = prop_config["gt_column"]
            gt_value = row.get(gt_col)
            
            # Skip if no ground truth
            if pd.isna(gt_value):
                continue
            
            try:
                gt_value = float(gt_value)
            except (ValueError, TypeError):
                if prop_config.get("unit") is None:
                    # Categorical comparison (e.g., gender)
                    gt_value = str(gt_value)
                else:
                    continue
            
            # Use first question for this property
            question = prop_config["questions"][0]
            
            # Generate answer using the main generate_answer function
            try:
                answer = _generate_answer_single(
                    checkpoint=checkpoint_path,
                    waveform_path=row['waveform_path_psa'],
                    question=question,
                    device_str=device_str,
                )
            except Exception as e:
                print(f"Error generating answer for sample {idx}, property {prop_name}: {e}")
                answer = f"Error: {e}"
            
            # Compare with ground truth
            if prop_config.get("tolerance") is not None:
                predicted = extract_numeric_from_answer(answer, prop_config.get("unit"))
                comparison = compare_values(predicted, gt_value, prop_config["tolerance"])
            else:
                # Categorical comparison
                comparison = {
                    "match": str(gt_value).lower() in answer.lower(),
                    "predicted": answer,
                    "ground_truth": gt_value,
                    "error": None,
                    "within_tolerance": None,
                }
            
            sample_result["properties"][prop_name] = {
                "question": question,
                "answer": answer,
                "ground_truth": gt_value,
                "unit": prop_config.get("unit"),
                **comparison,
            }
            
            # Update stats
            property_results[prop_name]["total"] += 1
            if comparison.get("match") or comparison.get("within_tolerance"):
                property_results[prop_name]["correct"] += 1
            if comparison.get("error") is not None:
                property_results[prop_name]["errors"].append(comparison["error"])
        
        results["samples"].append(sample_result)
    
    # Calculate summary statistics
    for prop_name, prop_stats in property_results.items():
        total = prop_stats["total"]
        correct = prop_stats["correct"]
        errors = prop_stats["errors"]
        
        results["summary"][prop_name] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "mean_error": np.mean(errors) if errors else None,
            "std_error": np.std(errors) if errors else None,
            "median_error": np.median(errors) if errors else None,
        }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"emergent_property_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a nice summary of the results."""
    print("\n" + "=" * 70)
    print("EMERGENT PROPERTY TEST RESULTS")
    print("=" * 70)
    
    for prop_name, stats in results["summary"].items():
        print(f"\n{prop_name.upper().replace('_', ' ')}")
        print("-" * 40)
        print(f"  Total samples tested: {stats['total']}")
        print(f"  Correct (within tolerance): {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.1%}")
        if stats.get('mean_error') is not None:
            print(f"  Mean error: {stats['mean_error']:.2f}")
            print(f"  Std error: {stats['std_error']:.2f}")
            print(f"  Median error: {stats['median_error']:.2f}")
    
    print("\n" + "=" * 70)
    
    # Print some example answers
    print("\nSAMPLE PREDICTIONS:")
    print("-" * 70)
    
    for i, sample in enumerate(results["samples"][:3]):
        print(f"\nSample {i+1}:")
        print(f"  Waveform: {sample['waveform_path']}")
        for prop_name, prop_data in sample["properties"].items():
            print(f"  [{prop_name}]")
            print(f"    Question: {prop_data['question']}")
            print(f"    Model answer: {prop_data['answer'][:100]}...")
            print(f"    Ground truth: {prop_data['ground_truth']} {prop_data.get('unit', '')}")
            if prop_data.get('predicted') is not None:
                print(f"    Extracted value: {prop_data['predicted']}")
            print(f"    Match: {prop_data.get('match') or prop_data.get('within_tolerance', False)}")


def main():
    parser = argparse.ArgumentParser(description="Test emergent properties of ECG MedGemma model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/rjjsarff_20251207-024755/checkpoint_step_10000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_parquet",
        type=str,
        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet",
        help="Path to test QA parquet file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/volume/ECG_tokenizer/output/emergent_property_tests",
        help="Directory to save results",
    )
    parser.add_argument(
        "--include_clinical",
        action="store_true",
        help="Also test clinical questions (age, gender)",
    )
    
    args = parser.parse_args()
    
    # Setup device string
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")
    
    # Load test samples
    test_df = load_test_samples(args.test_parquet, n_samples=args.n_samples)
    
    # Select property configs
    property_configs = EMERGENT_PROPERTY_QUESTIONS.copy()
    if args.include_clinical:
        property_configs.update(CLINICAL_QUESTIONS)
    
    # Run tests (model is loaded internally per question - inefficient but correct)
    print("\nNOTE: Loading model for each question. This is slow but ensures correct loading.")
    results = run_emergent_property_tests(
        checkpoint_path=args.checkpoint,
        test_df=test_df,
        device_str=device_str,
        property_configs=property_configs,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print_summary(results)
    
    return results


if __name__ == "__main__":
    main()

