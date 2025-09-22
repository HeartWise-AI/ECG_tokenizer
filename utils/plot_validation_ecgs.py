#!/usr/bin/env python3
"""
Plot ECGs from validation generation JSON files with Q&A annotations.
Supports both filename-only JSON (looks up path from parquet) and full-path JSON.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple

# Add DeepECG_Preprocess to path
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')

from ecg_plotter.core import NPYECGPlotter


def load_validation_json(json_path: str) -> Dict:
    """Load validation generations JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_parquet_mapping(parquet_path: str) -> pd.DataFrame:
    """Load parquet file with ECG path mappings"""
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Check which columns are available
    if 'waveform_name' not in df.columns:
        # Try to extract from waveform_path_psa
        if 'waveform_path_psa' in df.columns:
            df['waveform_name'] = df['waveform_path_psa'].apply(lambda x: os.path.basename(x))
    
    print(f"Loaded {len(df)} ECG records from parquet")
    return df


def resolve_ecg_path(ecg_name: str, parquet_df: Optional[pd.DataFrame] = None) -> str:
    """
    Resolve ECG path from name.
    If it's already a full path, return as-is.
    Otherwise, look up in parquet dataframe.
    """
    # Check if already a full path
    if ecg_name.startswith('/'):
        return ecg_name
    
    # Need to look up in parquet
    if parquet_df is None:
        raise ValueError(f"Need parquet dataframe to resolve path for {ecg_name}")
    
    # Look for matching waveform_name
    matches = parquet_df[parquet_df['waveform_name'] == ecg_name]
    
    if len(matches) == 0:
        raise ValueError(f"Could not find path for ECG: {ecg_name}")
    
    if 'waveform_path_psa' in matches.columns:
        return matches.iloc[0]['waveform_path_psa']
    elif 'waveform_path' in matches.columns:
        return matches.iloc[0]['waveform_path']
    else:
        raise ValueError(f"No path column found in parquet for {ecg_name}")


def format_qa_text(question: str, generated: str, ground_truth: str, max_len: int = 150) -> str:
    """Format Q&A text for display, with truncation if needed"""
    
    # Truncate long answers
    if len(generated) > max_len:
        generated = generated[:max_len] + "..."
    if len(ground_truth) > max_len:
        ground_truth = ground_truth[:max_len] + "..."
    
    # Format for display
    text = f"QUESTION: {question}\n\n"
    text += f"GENERATED: {generated}\n\n"
    text += f"GROUND TRUTH: {ground_truth}"
    
    return text


def plot_single_ecg(
    ecg_name: str,
    ecg_data: Dict,
    ecg_path: str,
    output_dir: str,
    epoch: Optional[int] = None,
    index: int = 0
) -> Tuple[bool, str]:
    """
    Plot a single ECG with Q&A annotations.
    
    Returns:
        (success, save_path or error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(ecg_path):
            return False, f"ECG file not found: {ecg_path}"
        
        # Create plotter
        plotter = NPYECGPlotter(
            npy_path=ecg_path,
            dataset="MIMICIV",
            out_dir=output_dir,
            width=2500,
            fft_normalized=True  # CRITICAL: Must be True for MIMIC-IV preprocessed data
        )
        
        # Format Q&A text for title
        qa_text = format_qa_text(
            question=ecg_data.get('Question', 'N/A'),
            generated=ecg_data.get('Generation', 'N/A'),
            ground_truth=ecg_data.get('Ground truth', 'N/A')
        )
        
        # Create title with ECG name and epoch
        if epoch is not None:
            title = f"Epoch {epoch} - ECG: {ecg_name}\n{qa_text}"
        else:
            title = f"ECG: {ecg_name}\n{qa_text}"
        
        # Plot without auto-saving to avoid duplicates
        img, _ = plotter.plot_ecg(
            title=title,
            save=False,  # Don't auto-save, we'll save with custom name
            anonymize=True,
            show_diagnosis=False  # We're showing Q&A instead
        )
        
        # Custom save path with meaningful name
        epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
        custom_save_path = os.path.join(output_dir, f"{epoch_str}{index:03d}_{ecg_name.replace('.npy', '')}_qa.png")
        
        # Save with custom name only
        if img:
            img.save(custom_save_path, dpi=(240, 240))
            print(f"✓ Saved plot: {custom_save_path}")
            return True, custom_save_path
        else:
            return False, "Failed to generate plot"
            
    except Exception as e:
        return False, f"Error plotting {ecg_name}: {str(e)}"


def plot_validation_ecgs(
    val_json_path: str,
    parquet_path: Optional[str] = None,
    output_dir: str = "validation_plots",
    num_ecgs: int = 10,
    epoch: Optional[int] = None
):
    """
    Plot ECGs from validation generation JSON with Q&A annotations.
    
    Args:
        val_json_path: Path to validation generations JSON
        parquet_path: Path to parquet file with ECG paths (optional if JSON has full paths)
        output_dir: Directory to save plots
        num_ecgs: Number of ECGs to plot
        epoch: Epoch number for labeling
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load validation JSON
    print(f"Loading validation JSON: {val_json_path}")
    val_data = load_validation_json(val_json_path)
    print(f"Found {len(val_data)} ECGs in validation JSON")
    
    # Load parquet if provided
    parquet_df = None
    if parquet_path:
        parquet_df = load_parquet_mapping(parquet_path)
    
    # Process ECGs
    ecg_names = list(val_data.keys())[:num_ecgs]
    print(f"\nPlotting {len(ecg_names)} ECGs...")
    
    success_count = 0
    failed_ecgs = []
    
    for i, ecg_name in enumerate(ecg_names):
        print(f"\n[{i+1}/{len(ecg_names)}] Processing {ecg_name}")
        ecg_data = val_data[ecg_name]
        
        # Check if JSON already has path
        if 'waveform_path' in ecg_data:
            ecg_path = ecg_data['waveform_path']
            print(f"  Using path from JSON: {ecg_path}")
        else:
            # Try to resolve path
            try:
                ecg_path = resolve_ecg_path(ecg_name, parquet_df)
                print(f"  Resolved path from parquet: {ecg_path}")
            except Exception as e:
                print(f"  ✗ Failed to resolve path: {e}")
                failed_ecgs.append((ecg_name, str(e)))
                continue
        
        # Plot ECG
        success, result = plot_single_ecg(
            ecg_name=ecg_name,
            ecg_data=ecg_data,
            ecg_path=ecg_path,
            output_dir=output_dir,
            epoch=epoch,
            index=i
        )
        
        if success:
            success_count += 1
        else:
            print(f"  ✗ Failed: {result}")
            failed_ecgs.append((ecg_name, result))
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: Successfully plotted {success_count}/{len(ecg_names)} ECGs")
    
    if failed_ecgs:
        print(f"\nFailed ECGs ({len(failed_ecgs)}):")
        for ecg_name, error in failed_ecgs:
            print(f"  - {ecg_name}: {error}")
    
    print(f"\nPlots saved to: {output_dir}")
    print("="*60)
    

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Plot validation ECGs with Q&A annotations")
    parser.add_argument("val_json", help="Path to validation generations JSON file")
    parser.add_argument("--parquet", help="Path to parquet file with ECG paths", 
                       default="/volume/ECG_tokenizer/output/mimic_train_qa_800k.parquet")
    parser.add_argument("--output-dir", help="Output directory for plots", 
                       default="validation_plots")
    parser.add_argument("--num-ecgs", type=int, help="Number of ECGs to plot", 
                       default=10)
    parser.add_argument("--epoch", type=int, help="Epoch number for labeling")
    
    args = parser.parse_args()
    
    plot_validation_ecgs(
        val_json_path=args.val_json,
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        num_ecgs=args.num_ecgs,
        epoch=args.epoch
    )


if __name__ == "__main__":
    main()