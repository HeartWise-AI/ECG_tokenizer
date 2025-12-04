#!/usr/bin/env python3
"""
Demo script to plot ECGs from validation JSON with reconstructed paths.
Handles cases where ECGs aren't in the parquet files by reconstructing paths.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')

from ecg_plotter.core import NPYECGPlotter


def reconstruct_ecg_path(ecg_name: str, dataset_type: str = 'test') -> str:
    """
    Reconstruct ECG path based on common patterns.
    Try multiple possible locations.
    """
    possible_paths = [
        f"/media/data1/datasets/MIMIC-IV/adjusted_signals/{dataset_type}/{ecg_name}",
        f"/media/data1/datasets/MIMIC-IV/adjusted_signals/train/{ecg_name}",
        f"/media/data1/datasets/MIMIC-IV/adjusted_signals/val/{ecg_name}",
        f"/media/data1/datasets/MIMIC-IV/adjusted_signals/{ecg_name}",
        f"/data/mimic-iv-ecg/processed/{ecg_name}",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Return most likely path even if it doesn't exist
    return possible_paths[0]


def plot_ecg_with_qa(ecg_name: str, ecg_data: dict, output_dir: str = "demo_plots"):
    """
    Plot a single ECG with Q&A information.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find or reconstruct the path
    ecg_path = reconstruct_ecg_path(ecg_name)
    
    print(f"\nPlotting: {ecg_name}")
    print(f"Path: {ecg_path}")
    print(f"File exists: {os.path.exists(ecg_path)}")
    
    if not os.path.exists(ecg_path):
        print(f"Warning: File not found, will try alternate path construction")
        # Try without dataset type subfolder
        alt_path = f"/media/data1/datasets/MIMIC-IV/adjusted_signals/{ecg_name}"
        if os.path.exists(alt_path):
            ecg_path = alt_path
            print(f"Found at: {ecg_path}")
        else:
            print(f"Could not find ECG file. Expected at: {ecg_path}")
            return None
    
    try:
        # Create plotter
        plotter = NPYECGPlotter(
            npy_path=ecg_path,
            dataset="MIMICIV",
            out_dir=output_dir,
            width=2500,
            fft_normalized=True  # CRITICAL: Must be True for MIMIC-IV preprocessed data
        )
        
        # Format Q&A for display
        question = ecg_data.get('Question', 'N/A')
        generated = ecg_data.get('Generation', 'N/A')
        ground_truth = ecg_data.get('Ground truth', 'N/A')
        
        # Create informative title
        title = f"ECG: {ecg_name}\n\n"
        title += f"QUESTION: {question}\n\n"
        
        # Truncate long answers
        max_len = 120
        if len(generated) > max_len:
            generated = generated[:max_len] + "..."
        if len(ground_truth) > max_len:
            ground_truth = ground_truth[:max_len] + "..."
            
        title += f"GENERATED: {generated}\n\n"
        title += f"GROUND TRUTH: {ground_truth}"
        
        # Plot without auto-saving to avoid duplicates
        img, _ = plotter.plot_ecg(
            title=title,
            save=False,  # Don't auto-save
            anonymize=True,
            show_diagnosis=False
        )
        
        # Save with meaningful name
        save_path = os.path.join(output_dir, f"{ecg_name.replace('.npy', '')}_qa.png")
        if img:
            img.save(save_path, dpi=(240, 240))
            print(f"✓ Saved plot to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"✗ Error plotting: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot ECG demos from validation JSON")
    parser.add_argument("--val-json", 
                       default="/volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/m7azzwgu_20250922-124603/val_generations_epoch_3.json",
                       help="Path to validation JSON")
    parser.add_argument("--ecg-name", help="Specific ECG to plot (e.g., 42584692.npy)")
    parser.add_argument("--num-ecgs", type=int, default=5, help="Number of ECGs to plot")
    parser.add_argument("--output-dir", default="demo_plots", help="Output directory")
    
    args = parser.parse_args()
    
    # Load validation JSON
    print(f"Loading: {args.val_json}")
    with open(args.val_json, 'r') as f:
        val_data = json.load(f)
    
    print(f"Total ECGs in JSON: {len(val_data)}")
    
    # Plot specific ECG or multiple
    if args.ecg_name:
        if args.ecg_name in val_data:
            plot_ecg_with_qa(args.ecg_name, val_data[args.ecg_name], args.output_dir)
        else:
            print(f"ECG {args.ecg_name} not found in validation JSON")
            print("Available ECGs (first 10):")
            for i, name in enumerate(list(val_data.keys())[:10]):
                print(f"  {i+1}. {name}")
    else:
        # Plot multiple ECGs
        ecg_names = list(val_data.keys())[:args.num_ecgs]
        success_count = 0
        
        for ecg_name in ecg_names:
            result = plot_ecg_with_qa(ecg_name, val_data[ecg_name], args.output_dir)
            if result:
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Successfully plotted {success_count}/{len(ecg_names)} ECGs")
        print(f"Plots saved to: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()