#!/usr/bin/env python3
"""
Plot comparison ECGs from MHI and MIMIC-IV with and without FFT normalization.
Shows the same ECG with both normalization settings to demonstrate the difference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append('/volume/ECG_tokenizer')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')

from ecg_plotter.core import NPYECGPlotter

def plot_ecg_comparison(ecg_path, ecg_path_psa, ecg_name, dataset_source, output_dir):
    """Plot the same ECG with and without FFT normalization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset-specific settings
    if dataset_source.lower() == 'mimic':
        dataset = "MIMICIV"
        lead_order = ["I", "II", "III", "aVR", "aVF", "aVL", 
                     "V1", "V2", "V3", "V4", "V5", "V6"]
    else:  # MHI
        dataset = "MHI"
        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # Common settings
    width = 2500
    
    # Plot WITH FFT normalization (using PSA path)
    print(f"\n  Plotting {ecg_name} WITH FFT normalization...")
    plotter_fft = NPYECGPlotter(
        npy_path=ecg_path_psa,
        out_dir=str(output_dir),
        dataset=dataset,
        width=width,
        fft_normalized=True  # FFT normalized
    )
    
    img_fft, _ = plotter_fft.plot_ecg(
        save=False,
        title=f"{Path(ecg_name).stem} - FFT Normalized"
    )
    
    # Save with descriptive name
    output_path_fft = output_dir / f"{Path(ecg_name).stem}_{dataset_source}_FFT_TRUE.png"
    img_fft.save(str(output_path_fft), dpi=(240, 240))
    print(f"    Saved: {output_path_fft}")
    
    # Plot WITHOUT FFT normalization (using original path if available)
    if ecg_path and Path(ecg_path).exists():
        print(f"  Plotting {ecg_name} WITHOUT FFT normalization...")
        plotter_no_fft = NPYECGPlotter(
            npy_path=ecg_path,
            out_dir=str(output_dir),
            dataset=dataset,
            width=width,
            fft_normalized=False  # No FFT normalization
        )
        
        img_no_fft, _ = plotter_no_fft.plot_ecg(
            save=False,
            title=f"{Path(ecg_name).stem} - No FFT Normalization"
        )
        
        # Save with descriptive name
        output_path_no_fft = output_dir / f"{Path(ecg_name).stem}_{dataset_source}_FFT_FALSE.png"
        img_no_fft.save(str(output_path_no_fft), dpi=(240, 240))
        print(f"    Saved: {output_path_no_fft}")
    else:
        print(f"    Original path not available for non-FFT plotting")
    
    return True

def main():
    # Load the dataset
    print("Loading combined dataset...")
    df = pd.read_parquet('/volume/ECG_tokenizer/output/combined_test_qa_m5k_h5k.parquet')
    
    # Output directory for plots
    output_dir = Path('/volume/ECG_tokenizer/ECG_Comparison_FFT_vs_NoFFT')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. MHI Example: 0314598_01-25-2006_15-11-26.npy
    print("\n=== MHI ECG Example ===")
    mhi_ecg = df[df['waveform_name'].str.contains('0314598', na=False)]
    if not mhi_ecg.empty:
        row = mhi_ecg.iloc[0]
        ecg_name = row['waveform_name']
        ecg_path_psa = row['waveform_path_psa']
        ecg_path_original = row.get('waveform_path_original', ecg_path_psa.replace('/adjusted_signals/', '/npy_files/'))
        
        print(f"Processing MHI ECG: {ecg_name}")
        print(f"  PSA Path (FFT): {ecg_path_psa}")
        print(f"  Original Path: {ecg_path_original}")
        
        plot_ecg_comparison(ecg_path_original, ecg_path_psa, ecg_name, 'MHI', output_dir)
    
    # 2. Get another MHI example
    print("\n=== Second MHI ECG Example ===")
    mhi_samples = df[df['dataset'] == 'mhi'].sample(1, random_state=42)
    if not mhi_samples.empty:
        row = mhi_samples.iloc[0]
        ecg_name = row['waveform_name']
        ecg_path_psa = row['waveform_path_psa']
        ecg_path_original = row.get('waveform_path_original', ecg_path_psa.replace('/adjusted_signals/', '/npy_files/'))
        
        print(f"Processing MHI ECG: {ecg_name}")
        print(f"  PSA Path (FFT): {ecg_path_psa}")
        print(f"  Original Path: {ecg_path_original}")
        
        plot_ecg_comparison(ecg_path_original, ecg_path_psa, ecg_name, 'MHI', output_dir)
    
    # 3. MIMIC-IV Examples
    print("\n=== MIMIC-IV ECG Examples ===")
    mimic_samples = df[df['dataset'] == 'mimic'].sample(2, random_state=42)
    for idx, row in mimic_samples.iterrows():
        ecg_name = row['waveform_name']
        ecg_path_psa = row['waveform_path_psa']
        # For MIMIC, try to construct original path
        ecg_path_original = row.get('waveform_path_original', 
                                   ecg_path_psa.replace('/adjusted_signals/', '/npy_files/'))
        
        print(f"\nProcessing MIMIC ECG: {ecg_name}")
        print(f"  PSA Path (FFT): {ecg_path_psa}")
        print(f"  Original Path: {ecg_path_original}")
        
        plot_ecg_comparison(ecg_path_original, ecg_path_psa, ecg_name, 'MIMIC', output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("\nSummary:")
    print("- 2 MHI ECGs (including 0314598_01-25-2006_15-11-26.npy)")
    print("- 2 MIMIC-IV ECGs")
    print("- Each ECG plotted with FFT normalization (from waveform_path_psa)")
    print("- Each ECG plotted without FFT normalization (from original path)")

if __name__ == "__main__":
    main()