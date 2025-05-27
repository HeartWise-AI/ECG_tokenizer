import sys
import os
import pandas as pd
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing.analysis_pipeline import AnalysisPipeline
from utils.constants import lead_to_idx

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess ECG data from parquet files.')
    parser.add_argument('--config', type=str, default="/volume/ECG_tokenizer/config/vqvae_training/base_config.yaml",
                        help='Path to the configuration file')
    parser.add_argument('--output', type=str, default="/volume/ECG_tokenizer/output",
                        help='Base output folder path for both processed data and parquet files')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Process a specific dataset (e.g., MIMIC, MHI)')
    parser.add_argument('--file_type', type=str, default=None, choices=['train', 'test', None],
                        help='File type to process (train, test, or None for any)')
    parser.add_argument('--rows', type=int, default=10000,
                        help='Number of rows to process (use -1 for all)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of workers for preprocessing')
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def find_dataset_files(config, dataset_name=None, file_type=None):
    """Find dataset files in config based on optional filters."""
    dataset_files = {}
    
    for key, path in config.items():
        if not key.endswith('_file') or not isinstance(path, str):
            continue
            
        if dataset_name and dataset_name.upper() not in key.upper():
            continue
            
        if file_type and not key.startswith(f"{file_type}_"):
            continue
            
        # Extract dataset name from the key
        key_parts = key.replace('_file', '').split('_')
        # The dataset name is usually the last part before "_file"
        extracted_dataset = key_parts[-1]
        
        dataset_files[key] = {
            'path': path,
            'dataset_name': extracted_dataset,
            'key': key
        }
    
    return dataset_files

def swap_leads(signal, lead1, lead2):
    """Swap two leads in the ECG signal array."""
    lead1_idx = lead_to_idx[lead1]
    lead2_idx = lead_to_idx[lead2]
    signal_copy = signal.copy()
    signal_copy[:, [lead1_idx, lead2_idx]] = signal_copy[:, [lead2_idx, lead1_idx]]
    return signal_copy

def process_dataset(parquet_path, dataset_name, output_folder, row_limit, n_workers, file_type=None):
    dataset_output_folder = os.path.join(output_folder, dataset_name)
    
    # Create a preprocessing subfolder with file_type as suffix if available
    preprocessing_suffix = f"_{file_type}" if file_type else ""
    preprocessing_subfolder = os.path.join(dataset_output_folder, f"preprocessing{preprocessing_suffix}")

    os.makedirs(dataset_output_folder, exist_ok=True)
    os.makedirs(preprocessing_subfolder, exist_ok=True)
    
    print(f"Using output folder: {dataset_output_folder}")
    print(f"Using preprocessing subfolder: {preprocessing_subfolder}")
    
    df = pd.read_parquet(parquet_path)
    if row_limit > 0:
        df = df.iloc[:row_limit, :]
        print(f"Processing {row_limit} rows")
    else:
        print(f"Processing all {len(df)} rows")
    
    # Check if it's a MIMIC dataset to enable lead swapping
    needs_lead_swap = dataset_name.upper() == 'MIMIC'
    if needs_lead_swap:
        print("MIMIC dataset detected - aVL and aVF leads will be swapped during processing")
    
    processed_df = AnalysisPipeline.save_and_preprocess_data(
        df=df,
        output_folder=dataset_output_folder, 
        preprocessing_folder=preprocessing_subfolder,
        preprocessing_n_workers=n_workers,
        swap_leads_fn=swap_leads if needs_lead_swap else None,
        swap_lead1='aVL',
        swap_lead2='aVF'
    )

    output_filename = f"{dataset_name}_cleaned.parquet"
    if file_type:
        output_filename = f"{dataset_name}_{file_type}_cleaned.parquet"
    output_parquet_path = os.path.join(dataset_output_folder, output_filename)
    
    processed_df.to_parquet(output_parquet_path, index=False)
    print(f"Processed data saved to {output_parquet_path}")
    return processed_df

def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    dataset_files = find_dataset_files(config, args.dataset, args.file_type)
    
    if not dataset_files:
        if args.dataset:
            message = f"No matching files found for dataset '{args.dataset}'"
            if args.file_type:
                message += f" with file type '{args.file_type}'"
        else:
            message = "No dataset files found in configuration"
        
        print(f"Error: {message}")
        available_keys = [k for k in config.keys() if k.endswith("_file") and isinstance(config[k], str)]
        print(f"Available dataset files in config: {available_keys}")
        sys.exit(1)
    
    print(f"Found {len(dataset_files)} dataset files to process")
    
    for key, file_info in dataset_files.items():
        print(f"Processing {key}: {file_info['path']}")
        
        try:
            dataset_name = file_info['dataset_name']
            
            # Determine file_type from the key if not explicitly provided
            current_file_type = args.file_type
            if not current_file_type:
                # Extract file_type from key if present (e.g., "train_parquet_MIMIC_file" -> "train")
                key_parts = key.split('_')
                if len(key_parts) > 0 and key_parts[0] in ['train', 'test']:
                    current_file_type = key_parts[0]
            
            process_dataset(
                parquet_path=file_info['path'],
                dataset_name=dataset_name,
                output_folder=output_folder,
                row_limit=args.rows,
                n_workers=args.workers,
                file_type=current_file_type
            )
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

if __name__ == "__main__":
    main()