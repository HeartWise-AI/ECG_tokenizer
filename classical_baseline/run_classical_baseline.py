#!/usr/bin/env python3
"""
Classical ECG Baseline Runner

This script runs the classical baseline pipeline using the ECG_tokenizer infrastructure
to load preprocessed MIMIC data and apply traditional signal processing methods.

Usage:
    python run_classical_baseline.py --parquet_file /path/to/data.parquet [options]
"""

import argparse
import sys
import os
from pathlib import Path
import traceback
import time

# Add ECG_tokenizer to path
sys.path.append('/volume/ECG_tokenizer')

from classical_baseline import ClassicalECGBaseline, ClassicalBaselineConfig
from config import (
    load_config, save_config, get_dataset_paths, detect_lead_stats, 
    create_default_config_file, DEFAULT_LEAD_STATS
)
from utils.constants import ECG_PATTERNS, standard_lead_order


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Classical ECG Baseline Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with specific parquet file
    python run_classical_baseline.py --parquet_file /volume/ECG_tokenizer/data/train.parquet
    
    # Run with custom configuration
    python run_classical_baseline.py --config config/custom_config.yaml
    
    # Run with automatic dataset detection
    python run_classical_baseline.py --auto_detect
    
    # Test with small sample
    python run_classical_baseline.py --parquet_file data.parquet --max_samples 100
        """
    )
    
    parser.add_argument(
        '--parquet_file', 
        type=str,
        help='Path to the parquet file containing ECG data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--auto_detect',
        action='store_true',
        help='Automatically detect available datasets'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    
    parser.add_argument(
        '--no_emd',
        action='store_true',
        help='Skip EMD/VMD features (faster, only use wavelets and SAX)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/volume/ECG_tokenizer/classical_baseline/output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--lead_stats_file',
        type=str,
        default=None,
        help='Path to JSON file containing lead statistics'
    )
    
    parser.add_argument(
        '--create_config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    parser.add_argument(
        '--list_datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Perform dry run (setup only, no training)'
    )
    
    parser.add_argument(
        '--dataset_filter',
        type=str,
        default=None,
        help='Filter for specific dataset (e.g., MIMIC, MHI). Leave empty to use all data.'
    )
    
    return parser.parse_args()


def validate_parquet_file(parquet_file: str) -> bool:
    """Validate that parquet file exists and is readable"""
    if not os.path.exists(parquet_file):
        print(f"Error: Parquet file does not exist: {parquet_file}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        print(f"Parquet file validation:")
        print(f"  Path: {parquet_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for required column
        if 'waveform_path_psa' not in df.columns:
            print(f"Warning: Expected column 'waveform_path_psa' not found")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check for ECG patterns
        pattern_cols = [col for col in df.columns if col in ECG_PATTERNS]
        print(f"  ECG pattern columns found: {len(pattern_cols)}")
        
        return True
        
    except Exception as e:
        print(f"Error: Could not read parquet file: {e}")
        return False


def load_lead_stats(lead_stats_file: str = None, parquet_file: str = None):
    """Load or compute lead statistics"""
    if lead_stats_file and os.path.exists(lead_stats_file):
        print(f"Loading lead statistics from: {lead_stats_file}")
        import json
        with open(lead_stats_file, 'r') as f:
            return json.load(f)
    
    elif parquet_file:
        print(f"Computing lead statistics from dataset...")
        return detect_lead_stats(parquet_file)
    
    else:
        print("Using default lead statistics")
        return DEFAULT_LEAD_STATS


def create_limited_dataset(parquet_file: str, max_samples: int, output_dir: str):
    """Create a limited dataset for testing"""
    import pandas as pd
    
    print(f"Creating limited dataset with {max_samples} samples...")
    
    df = pd.read_parquet(parquet_file)
    df_limited = df.head(max_samples)
    
    output_path = Path(output_dir) / f"limited_{max_samples}_samples.parquet"
    df_limited.to_parquet(output_path)
    
    print(f"Limited dataset saved to: {output_path}")
    return str(output_path)


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Handle special commands
    if args.create_config:
        config_file = create_default_config_file()
        print(f"Default configuration created at: {config_file}")
        return
    
    if args.list_datasets:
        paths = get_dataset_paths()
        print("Available dataset paths:")
        for path in paths:
            print(f"  {path}")
        if not paths:
            print("  No datasets found in common locations")
        return
    
    # Determine parquet file
    parquet_file = None
    
    if args.parquet_file:
        parquet_file = args.parquet_file
    elif args.auto_detect:
        paths = get_dataset_paths()
        if paths:
            parquet_file = paths[0]
            print(f"Auto-detected dataset: {parquet_file}")
        else:
            print("Error: No datasets found for auto-detection")
            return
    else:
        print("Error: Must specify --parquet_file or use --auto_detect")
        return
    
    # Validate parquet file
    if not validate_parquet_file(parquet_file):
        return
    
    # Create output directory early (needed for limited dataset creation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle limited dataset for testing
    if args.max_samples:
        parquet_file = create_limited_dataset(parquet_file, args.max_samples, args.output_dir)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Create configuration object
    config = ClassicalBaselineConfig(
        expected_waveform_length=config_dict['data']['expected_waveform_length'],
        num_leads=config_dict['data']['num_leads'],
        normalize_waveforms=config_dict['data']['normalize_waveforms'],
        signal_path_column=config_dict['data']['signal_path_column'],
        wavelet_name=config_dict['signal_processing']['wavelet']['name'],
        wavelet_levels=config_dict['signal_processing']['wavelet']['levels'],
        emd_max_imf=config_dict['signal_processing']['emd']['max_imf'],
        vmd_alpha=config_dict['signal_processing']['vmd']['alpha'],
        vmd_tau=config_dict['signal_processing']['vmd']['tau'],
        vmd_K=config_dict['signal_processing']['vmd']['K'],
        vmd_DC=config_dict['signal_processing']['vmd']['DC'],
        vmd_init=config_dict['signal_processing']['vmd']['init'],
        vmd_tol=config_dict['signal_processing']['vmd']['tol'],
        sax_n_segments=config_dict['signal_processing']['sax']['n_segments'],
        sax_alphabet_size=config_dict['signal_processing']['sax']['alphabet_size'],
        bow_window_size=config_dict['signal_processing']['sax']['bow_window_size'],
        bow_word_size=config_dict['signal_processing']['sax']['bow_word_size'],
        bow_n_bins=config_dict['signal_processing']['sax']['bow_n_bins'],
        test_size=config_dict['data']['test_size'],
        validation_size=config_dict['data']['validation_size'],
        random_state=config_dict['data']['random_state'],
        cv_folds=config_dict['machine_learning']['cv_folds'],
        n_jobs=config_dict['machine_learning']['n_jobs'],
        save_features=config_dict['output']['save_features'],
        save_models=config_dict['output']['save_models']
    )
    
    # Load lead statistics
    lead_stats = load_lead_stats(args.lead_stats_file, parquet_file)
    
    print(f"Output directory: {output_dir}")
    
    if args.dry_run:
        print("Dry run completed successfully")
        return
    
    # Run the baseline
    start_time = time.time()
    
    try:
        print("\nStarting Classical ECG Baseline Pipeline...")
        
        baseline = ClassicalECGBaseline(config)
        # Only filter if dataset_filter is not empty
        dataset_filter = args.dataset_filter if args.dataset_filter and args.dataset_filter.strip() else None
        results = baseline.run_baseline(parquet_file, lead_stats, dataset_filter=dataset_filter, max_samples=args.max_samples)
        
        # Print summary
        baseline.print_summary()
        
        # Save additional outputs
        if config.save_features:
            # Save configuration used
            config_output = output_dir / 'run_config.yaml'
            save_config(config_dict, str(config_output))
            
            # Save lead statistics
            import json
            stats_output = output_dir / 'lead_statistics.json'
            with open(stats_output, 'w') as f:
                json.dump(lead_stats, f, indent=2)
        
        end_time = time.time()
        print(f"\nPipeline completed successfully in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error running baseline: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
