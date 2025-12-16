"""
Classical SAX Baseline: Extracts SAX codes for each lead from ECG signals using ECG_tokenizer infrastructure.
"""
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import warnings

from pyts.approximation import SymbolicAggregateApproximation

# ECG_tokenizer imports
sys.path.append('/volume/ECG_tokenizer')
from data.ecg_tokenizer_classifier_dataset import ECGTokenizerClassifierDataset
from utils.constants import standard_lead_order
from dataset_utils import FilteredECGDataset

warnings.filterwarnings('ignore')

@dataclass
class SAXConfig:
    expected_waveform_length: int = 5000
    num_leads: int = 12
    normalize_waveforms: bool = True
    signal_path_column: str = 'waveform_path_psa'
    sax_alphabet_size: int = 26
    save_results: bool = True

class SAXExtractor:
    def __init__(self, config: SAXConfig):
        self.config = config
    def extract_sax_string(self, signal: np.ndarray) -> str:
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        sax = SymbolicAggregateApproximation(n_bins=self.config.sax_alphabet_size, strategy='quantile')
        sax_str = ''.join(sax.fit_transform(signal_norm.reshape(1, -1))[0])
        return sax_str
    def extract_all_leads(self, signal: np.ndarray) -> Dict[str, str]:
        sax_codes = {}
        if signal.shape[0] != self.config.num_leads:
            if signal.shape[1] == self.config.num_leads:
                signal = signal.T
            else:
                raise ValueError(f"Signal shape {signal.shape} doesn't match expected leads {self.config.num_leads}")
        for idx, lead in enumerate(standard_lead_order):
            if idx < signal.shape[0]:
                lead_signal = signal[idx, :]
                if np.all(lead_signal == 0) or np.any(np.isnan(lead_signal)) or np.any(np.isinf(lead_signal)):
                    sax_codes[lead] = ''
                else:
                    sax_codes[lead] = self.extract_sax_string(lead_signal)
        return sax_codes

class SAXBaseline:
    def __init__(self, config: SAXConfig):
        self.config = config
        self.extractor = SAXExtractor(config)
        self.results = []
    def calculate_lead_stats(self, parquet_file: str, sample_size: int = 1000) -> Dict[str, Dict[str, float]]:
        """Calculate lead statistics from a sample of the dataset"""
        print(f"Calculating lead statistics from {sample_size} samples...")
        
        # Create dummy lead stats for initial loading
        dummy_stats = {lead: {"mean": 0.0, "std": 1.0} for lead in standard_lead_order}
        
        # Load a sample without normalization first to calculate stats
        temp_dataset = ECGTokenizerClassifierDataset(
            parquet_file=parquet_file,
            expected_waveform_length=self.config.expected_waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=False,  # Don't normalize yet
            signal_path_column=self.config.signal_path_column,
            lead_stats=dummy_stats  # Use dummy stats initially
        )
        
        # Sample a subset for statistics calculation
        actual_sample_size = min(sample_size, len(temp_dataset))
        lead_signals = {lead: [] for lead in standard_lead_order}
        
        print(f"Sampling {actual_sample_size} signals for statistics...")
        for i in range(actual_sample_size):
            if i % 100 == 0:
                print(f"  Processing sample {i}/{actual_sample_size}")
            
            try:
                sample = temp_dataset[i]
                signal = sample['signal']
                
                # Ensure correct shape
                if signal.shape[0] != self.config.num_leads:
                    if signal.shape[1] == self.config.num_leads:
                        signal = signal.T
                
                for idx, lead in enumerate(standard_lead_order):
                    if idx < signal.shape[0]:
                        lead_signal = signal[idx, :]
                        if not (np.all(lead_signal == 0) or np.any(np.isnan(lead_signal)) or np.any(np.isinf(lead_signal))):
                            lead_signals[lead].append(lead_signal)
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                continue
        
        # Calculate statistics
        lead_stats = {}
        for lead in standard_lead_order:
            if lead_signals[lead]:
                all_values = np.concatenate(lead_signals[lead])
                lead_stats[lead] = {
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values))
                }
                print(f"  {lead}: mean={lead_stats[lead]['mean']:.3f}, std={lead_stats[lead]['std']:.3f}")
            else:
                lead_stats[lead] = {"mean": 0.0, "std": 1.0}
                print(f"  {lead}: No valid data, using defaults")
        
        return lead_stats

    def run(self, parquet_file: str, dataset_filter: str = None, max_samples: int = None, lead_stats: Dict[str, Dict[str, float]] = None):
        """Run SAX extraction on dataset
        
        Args:
            parquet_file: Path to the parquet file containing ECG data
            dataset_filter: Optional filter string for the dataset
            max_samples: Optional limit on number of samples to process
            lead_stats: Optional pre-computed lead statistics. If not provided, will be calculated from the data.
        """
        print(f"Loading dataset from: {parquet_file}")
        
        # Use provided lead_stats or calculate from the data
        if lead_stats is None:
            lead_stats = self.calculate_lead_stats(parquet_file)
        else:
            print("Using provided lead statistics")
        
        if dataset_filter:
            dataset = FilteredECGDataset(
                parquet_file=parquet_file,
                expected_waveform_length=self.config.expected_waveform_length,
                num_leads=self.config.num_leads,
                normalize_waveforms=self.config.normalize_waveforms,
                signal_path_column=self.config.signal_path_column,
                dataset_filter=dataset_filter,
                lead_stats=lead_stats
            )
        else:
            dataset = ECGTokenizerClassifierDataset(
                parquet_file=parquet_file,
                expected_waveform_length=self.config.expected_waveform_length,
                num_leads=self.config.num_leads,
                normalize_waveforms=self.config.normalize_waveforms,
                signal_path_column=self.config.signal_path_column,
                lead_stats=lead_stats
            )
        
        # Limit samples for testing if requested
        total_samples = len(dataset)
        if max_samples:
            total_samples = min(max_samples, total_samples)
        
        print(f"Processing {total_samples} samples for SAX extraction...")
        
        for i in range(total_samples):
            if i % 100 == 0:
                print(f"  Processing sample {i}/{total_samples}")
            
            try:
                sample = dataset[i]
                signal = sample['signal']
                # Get the actual waveform path from the dataset row
                row = dataset.data.iloc[i]
                waveform_path = row[dataset.signal_path_column]
                sax_codes = self.extractor.extract_all_leads(signal)
                # Create concatenated final code from all leads in standard order
                final_code = ''.join([sax_codes[lead] for lead in standard_lead_order if lead in sax_codes])
                
                self.results.append({
                    'sample_id': i, 
                    'waveform_path_psa': waveform_path,
                    'sax_codes': sax_codes,
                    'final_code': final_code
                })
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                continue
        if self.config.save_results:
            self.save_results()
    def save_results(self):
        output_dir = Path('/volume/ECG_tokenizer/classical_baseline/output')
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'sax_codes_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        # CSV
        rows = []
        for r in self.results:
            row = {
                'sample_id': r['sample_id'],
                'waveform_path_psa': r.get('waveform_path_psa', ''),
                'final_code': r.get('final_code', '')
            }
            row.update({f'{lead}_sax': code for lead, code in r['sax_codes'].items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / 'sax_codes_results.csv', index=False)
    def print_summary(self):
        print(f"Total samples processed: {len(self.results)}")
        if self.results:
            print("Example SAX codes for first sample:")
            for lead, code in self.results[0]['sax_codes'].items():
                print(f"  {lead}: {code[:30]}{'...' if len(code) > 30 else ''}")

def main():
    config = SAXConfig()
    parquet_file = "/volume/ECG_tokenizer/output/MHI/mimic_mhi_psa_train_updated_stratified_20%.parquet"
    
    print("Running SAX baseline with subset for testing...")
    sax_baseline = SAXBaseline(config)
    
    # Use only 500 samples for testing
    sax_baseline.run(parquet_file, max_samples=500)
    sax_baseline.print_summary()

@dataclass
class ClassicalBaselineConfig:
    """Configuration for Classical ECG Baseline processing"""
    # Data parameters
    expected_waveform_length: int = 5000
    num_leads: int = 12
    normalize_waveforms: bool = True
    signal_path_column: str = 'waveform_path_psa'
    
    # Signal processing parameters
    # Wavelet
    wavelet_name: str = 'db4'
    wavelet_levels: int = 5
    
    # EMD/VMD
    emd_max_imf: int = 5
    vmd_alpha: float = 2000.0
    vmd_tau: float = 0.0
    vmd_K: int = 3
    vmd_DC: bool = False
    vmd_init: int = 1
    vmd_tol: float = 1e-7
    
    # SAX
    sax_n_segments: int = 5000
    sax_alphabet_size: int = 26
    bow_window_size: int = 20
    bow_word_size: int = 4
    bow_n_bins: int = 100
    
    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Machine learning
    cv_folds: int = 5
    n_jobs: int = -1
    
    # Output
    save_features: bool = True
    save_models: bool = True


class ClassicalECGBaseline:
    """Implements classical signal processing techniques for ECG analysis"""
    
    def __init__(self, config: ClassicalBaselineConfig):
        self.config = config
        self.results = {}
        
    def run_baseline(self, parquet_file: str, lead_stats: Dict, dataset_filter: str = None, max_samples: int = None):
        """Run the full baseline pipeline"""
        print(f"Loading data from {parquet_file}")
        
        # Load the dataframe
        df = pd.read_parquet(parquet_file)
        
        if dataset_filter:
            print(f"Filtering dataset for: {dataset_filter}")
            # Implement filtering logic based on your dataset structure
            # Example: df = df[df['source'].str.contains(dataset_filter)]
        
        # Process the data and run the pipeline
        # This is a placeholder - implement your actual pipeline logic
        print(f"Processing {len(df)} samples using classical techniques")
        
        # For now, let's run the SAX baseline that already exists
        sax_config = SAXConfig(
            expected_waveform_length=self.config.expected_waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=self.config.normalize_waveforms,
            signal_path_column=self.config.signal_path_column,
            sax_alphabet_size=self.config.sax_alphabet_size,
            save_results=self.config.save_features
        )
        
        # Run SAX baseline with provided lead_stats
        sax_baseline = SAXBaseline(sax_config)
        sax_baseline.run(parquet_file, dataset_filter=dataset_filter, max_samples=max_samples, lead_stats=lead_stats)
        
        # Extract results from SAX baseline
        # Note: This is a feature extraction baseline, not a classifier.
        # Classification metrics would require a downstream classifier trained on these features.
        num_samples = len(sax_baseline.results)
        avg_code_length = sum(len(r.get('final_code', '')) for r in sax_baseline.results) / num_samples if num_samples > 0 else 0
        
        self.results = {
            'summary': {
                'num_samples_processed': num_samples,
                'avg_sax_code_length': avg_code_length,
                'num_leads': self.config.num_leads,
                'sax_alphabet_size': self.config.sax_alphabet_size,
            },
            'lead_stats_used': lead_stats,
            'sax_results': sax_baseline.results
        }
        
        return self.results
    
    def print_summary(self):
        """Print a summary of the results"""
        if not self.results:
            print("No results available. Run the baseline first.")
            return
            
        print("\n===== CLASSICAL BASELINE RESULTS SUMMARY =====")
        
        # Print processing summary
        summary = self.results.get('summary', {})
        if summary:
            print("Processing Summary:")
            print(f"  Samples processed: {summary.get('num_samples_processed', 0)}")
            print(f"  Average SAX code length: {summary.get('avg_sax_code_length', 0):.0f}")
            print(f"  Number of leads: {summary.get('num_leads', 0)}")
            print(f"  SAX alphabet size: {summary.get('sax_alphabet_size', 0)}")
            
        # Print SAX results details
        if 'sax_results' in self.results:
            sax_results = self.results['sax_results']
            if sax_results:
                print(f"\nSAX Feature Extraction:")
                print(f"  Total samples with SAX codes: {len(sax_results)}")
                # Show example from first sample
                if sax_results[0].get('sax_codes'):
                    print("  Example SAX codes (first sample):")
                    for lead, code in list(sax_results[0]['sax_codes'].items())[:3]:
                        print(f"    {lead}: {code[:30]}{'...' if len(code) > 30 else ''}")


if __name__ == "__main__":
    main()
