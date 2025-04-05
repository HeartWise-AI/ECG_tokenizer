import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import multiprocessing

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std per lead from ECG data')
    parser.add_argument('--input_path', type=str, 
                       default="parquets/mhi/test_trial_v1.1_with_report_subset.parquet",
                       help='Path to the input parquet file')
    parser.add_argument('--waveform_column', type=str,
                       default="waveform_path",
                       help='Column name containing the path to waveform files')
    parser.add_argument('--num_workers', type=int,
                       help='Number of worker threads to use. If not specified, will be calculated automatically')
    return parser.parse_args()

def get_optimal_workers():
    # For I/O bound tasks, we can use more workers than CPU cores
    # A good rule of thumb is: min(32, (CPU cores * 4))
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(32, cpu_count * 4)
    print(f"System has {cpu_count} CPU cores")
    print(f"Using {optimal_workers} worker threads")
    return optimal_workers

def process_ecg_file(args):
    row, waveform_column = args
    try:
        if not os.path.exists(row[waveform_column]):
            print(f"Warning: File not found: {row[waveform_column]}")
            return None
            
        ecg = np.load(row[waveform_column])
        # Need to squeeze ECGs for MHI dataset
        if ecg.ndim > 2:
            ecg = ecg.squeeze()
        
        # Ensure the ECG has the expected shape
        if ecg.shape != (2500, 12):
            print(f"Skipping ECG with unexpected shape: {ecg.shape}")
            return None
        
        return np.sum(ecg, axis=0), np.sum(ecg**2, axis=0), 2500
    except Exception as e:
        print(f"Error processing file {row[waveform_column]}: {str(e)}")
        return None

def compute_stats(df, waveform_column, num_workers=4):
    # Initialize arrays to store sum and sum of squares for each lead
    lead_count = 12
    lead_sum = np.zeros(lead_count)  # Initialize with zeros for 12 leads
    lead_sum_sq = np.zeros(lead_count)  # Initialize with zeros for 12 leads
    total_samples = 0    
    
    # Process ECGs in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(row, waveform_column) for _, row in df.iterrows()]
        results = list(tqdm(executor.map(process_ecg_file, args), 
                          total=len(df), 
                          desc="Processing ECGs"))
    
    for result in results:
        if result is not None:
            sum_values, sum_sq_values, samples = result
            lead_sum += sum_values
            lead_sum_sq += sum_sq_values
            total_samples += samples

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
    num_workers = args.num_workers if args.num_workers is not None else get_optimal_workers()
    stats_df = compute_stats(df, args.waveform_column, num_workers)
    print("ECG Lead Statistics:")
    print(stats_df)
    



