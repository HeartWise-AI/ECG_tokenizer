#!/usr/bin/env python3
"""
Extract and display SAX strings from ECG signals

This script extracts SAX representations from ECG signals and saves them
for analysis and visualization.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from classical_baseline import ClassicalECGBaseline, ClassicalBaselineConfig, ClassicalFeatureExtractor
from data.ecg_tokenizer_classifier_dataset import ECGTokenizerClassifierDataset

def extract_sax_strings(parquet_file: str, max_samples: int = 10, output_file: str = None):
    """Extract SAX strings from ECG signals"""
    
    print(f"Extracting SAX strings from {max_samples} samples...")
    
    # Load configuration
    config = ClassicalBaselineConfig()
    
    # Create dataset
    dataset = ECGTokenizerClassifierDataset(
        parquet_file=parquet_file,
        expected_waveform_length=2500,
        num_leads=12,
        normalize_waveforms=False,
        lead_stats=None,
        signal_path_column='waveform_path_psa'
    )
    
    # Initialize feature extractor
    extractor = ClassicalFeatureExtractor(config)
    
    # Extract SAX strings
    sax_data = []
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
            signal = sample['signal']  # Shape: (12, 2500)
            
            sample_sax = {'sample_id': i}
            
            # Extract SAX for each lead
            for lead_idx, lead_name in enumerate(lead_names):
                lead_signal = signal[lead_idx, :]
                
                try:
                    # Extract SAX features (which includes the string)
                    sax_features = extractor.extract_sax_features(lead_signal, lead_name)
                    sax_string = sax_features.get(f'{lead_name}_sax_string', 'N/A')
                    sample_sax[f'{lead_name}_sax'] = sax_string
                    
                except Exception as e:
                    print(f"Warning: SAX extraction failed for sample {i}, lead {lead_name}: {e}")
                    sample_sax[f'{lead_name}_sax'] = 'ERROR'
            
            sax_data.append(sample_sax)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1} samples...")
                
        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue
    
    # Convert to DataFrame
    sax_df = pd.DataFrame(sax_data)
    
    # Display results
    print(f"\nSAX Strings extracted from {len(sax_df)} samples:")
    print("="*80)
    
    for idx, row in sax_df.iterrows():
        print(f"\nSample {row['sample_id']}:")
        for lead in lead_names:
            sax_col = f'{lead}_sax'
            if sax_col in row:
                sax_str = row[sax_col]
                print(f"  {lead:>3}: {sax_str}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sax_df.to_csv(output_path, index=False)
        print(f"\nSAX strings saved to: {output_path}")
    
    return sax_df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract SAX strings from ECG signals')
    parser.add_argument('--parquet_file', required=True, help='Path to parquet file')
    parser.add_argument('--max_samples', type=int, default=10, help='Maximum number of samples to process')
    parser.add_argument('--output_file', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Extract SAX strings
    sax_df = extract_sax_strings(
        parquet_file=args.parquet_file,
        max_samples=args.max_samples,
        output_file=args.output_file
    )
    
    print(f"\nExtraction completed! Processed {len(sax_df)} samples.")

if __name__ == "__main__":
    main()
