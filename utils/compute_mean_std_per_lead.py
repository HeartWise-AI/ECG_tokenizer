import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std per lead from ECG data')
    parser.add_argument('--input_path', type=str, 
                       default="parquets/mhi/test_trial_v1.1_with_report_subset.parquet",
                       help='Path to the input parquet file')
    parser.add_argument('--waveform_column', type=str,
                       default="waveform_path",
                       help='Column name containing the path to waveform files')
    return parser.parse_args()

def compute_stats(df, waveform_column):
    # Initialize arrays to store sum and sum of squares for each lead
    lead_count = 12
    time_samples = 2500
    lead_sum = np.zeros(lead_count)
    lead_sum_sq = np.zeros(lead_count)
    total_samples = 0
    
    # Iterate through all ECG files
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECGs"):
        ecg = np.load(row[waveform_column])
        # Need to squeeze ECGs for MHI dataset
        if ecg.ndim > 2:
            ecg = ecg.squeeze()
        
        # Ensure the ECG has the expected shape
        if ecg.shape != (time_samples, lead_count):
            print(f"Skipping ECG with unexpected shape: {ecg.shape}")
            continue
        
        # Calculate sum and sum of squares for each lead
        lead_sum += np.sum(ecg, axis=0)
        lead_sum_sq += np.sum(ecg**2, axis=0)
        total_samples += time_samples

    # Calculate mean and standard deviation for each lead
    lead_mean = lead_sum / total_samples
    lead_variance = (lead_sum_sq / total_samples) - (lead_mean**2)
    lead_std = np.sqrt(lead_variance)

    # Create a DataFrame to display the results
    stats_df = pd.DataFrame({
        'lead': [f'Lead {i+1}' for i in range(lead_count)],
        'mean': lead_mean,
        'std': lead_std
    })
    return stats_df

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    stats_df = compute_stats(df, args.waveform_column)
    print("ECG Lead Statistics:")
    print(stats_df)
    



