#!/usr/bin/env python3
"""
Test script for Classical ECG Baseline

This script tests the classical baseline implementation with synthetic data
to verify that all components work correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add ECG_tokenizer to path
sys.path.append('/volume/ECG_tokenizer')

try:
    from classical_baseline import SAXExtractor, SAXConfig, ClassicalBaselineConfig, ClassicalECGBaseline
    from classical_baseline.config import DEFAULT_LEAD_STATS, create_default_config_file, load_config
    from utils.constants import ECG_PATTERNS, standard_lead_order
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def generate_synthetic_ecg(duration=10, sampling_rate=500, num_leads=12, heart_rate=70):
    """Generate synthetic ECG data for testing"""
    
    # Time vector
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # RR interval
    rr_interval = 60 / heart_rate  # seconds
    
    signals = []
    
    for lead in range(num_leads):
        # Generate synthetic ECG with P, QRS, T waves
        ecg = np.zeros_like(t)
        
        # Add multiple heartbeats
        for beat_start in np.arange(0, duration, rr_interval):
            beat_indices = (t >= beat_start) & (t < beat_start + 0.8)
            
            if np.any(beat_indices):
                beat_time = t[beat_indices] - beat_start
                
                # P wave (around 0.1s)
                p_wave = 0.1 * np.exp(-((beat_time - 0.1) / 0.02) ** 2)
                
                # QRS complex (around 0.3s)  
                qrs_real = 1.0 * np.exp(-((beat_time - 0.3) / 0.02) ** 2)
                
                # T wave (around 0.5s)
                t_wave = 0.3 * np.exp(-((beat_time - 0.5) / 0.05) ** 2)
                
                # Combine waves
                beat_signal = p_wave + qrs_real + t_wave
                ecg[beat_indices] += beat_signal
        
        # Add some noise and lead-specific variations
        noise = np.random.normal(0, 0.01, len(t))
        lead_factor = 0.8 + 0.4 * np.random.random()  # Lead variation
        
        signals.append((ecg + noise) * lead_factor)
    
    # Stack as (leads, time) to match expected format
    signal_array = np.array(signals)
    
    return signal_array


def create_test_dataset():
    """Create a small test dataset in parquet format"""
    print("Creating test dataset...")
    
    # Generate synthetic data
    num_samples = 50
    data = []
    
    for i in range(num_samples):
        # Generate signal
        signal = generate_synthetic_ecg(duration=10, sampling_rate=500, heart_rate=60 + i)
        
        # Save signal as numpy file
        signal_path = f"/tmp/test_ecg_{i:03d}.npy"
        np.save(signal_path, signal.T)  # Save as (time, leads) format
        
        # Create random labels
        labels = np.random.randint(0, 2, len(ECG_PATTERNS))
        
        # Create row
        row = {'waveform_path_psa': signal_path}
        for j, pattern in enumerate(ECG_PATTERNS):
            row[pattern] = labels[j]
        
        data.append(row)
    
    # Save as parquet
    df = pd.DataFrame(data)
    test_parquet = "/tmp/test_ecg_dataset.parquet"
    df.to_parquet(test_parquet)
    
    print(f"Test dataset created: {test_parquet}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    return test_parquet


def test_feature_extraction():
    """Test SAX feature extraction on synthetic data"""
    print("\nTesting SAX feature extraction...")
    
    config = SAXConfig()
    extractor = SAXExtractor(config)
    
    # Generate test signal
    signal = generate_synthetic_ecg(duration=10, sampling_rate=500)
    print(f"Test signal shape: {signal.shape}")
    
    # Test single lead SAX string extraction
    lead_signal = signal[1, :]  # Lead II
    
    try:
        sax_string = extractor.extract_sax_string(lead_signal)
        print(f"✅ SAX string extracted: length={len(sax_string)}, sample='{sax_string[:30]}...'")
    except Exception as e:
        print(f"❌ SAX string extraction failed: {e}")
        return
    
    # Test all leads extraction
    try:
        all_leads_sax = extractor.extract_all_leads(signal)
        print(f"✅ All leads SAX extracted: {len(all_leads_sax)} leads")
        
        # Show some example SAX codes
        for lead_name in list(all_leads_sax.keys())[:3]:
            code = all_leads_sax[lead_name]
            print(f"   {lead_name}: {code[:30]}{'...' if len(code) > 30 else ''}")
        
    except Exception as e:
        print(f"❌ All leads extraction failed: {e}")


def test_dataset_loading():
    """Test dataset loading with ECG_tokenizer infrastructure"""
    print("\nTesting dataset loading...")
    
    # Create test dataset
    test_parquet = create_test_dataset()
    
    try:
        from data.ecg_tokenizer_classifier_dataset import ECGTokenizerClassifierDataset
        
        dataset = ECGTokenizerClassifierDataset(
            parquet_file=test_parquet,
            expected_waveform_length=5000,
            num_leads=12,
            normalize_waveforms=False,
            lead_stats=None
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        signal = sample['signal']
        labels = sample['labels']
        
        print(f"✅ Sample loaded - Signal: {signal.shape}, Labels: {labels.shape}")
        
        return test_parquet
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None


def test_pipeline():
    """Test the complete pipeline with synthetic data"""
    print("\nTesting complete pipeline...")
    
    # Create test dataset
    test_parquet = create_test_dataset()
    
    if test_parquet is None:
        print("❌ Cannot test pipeline without dataset")
        return
    
    try:
        config = ClassicalBaselineConfig(
            expected_waveform_length=5000,
            num_leads=12,
            normalize_waveforms=False,
            test_size=0.3,
            random_state=42
        )
        
        baseline = ClassicalECGBaseline(config)
        
        # Run on small dataset
        results = baseline.run_baseline(test_parquet, DEFAULT_LEAD_STATS)
        
        print("✅ Pipeline completed successfully")
        
        # Show summary results
        # run_baseline returns a dict with keys: 'summary', 'lead_stats_used', 'sax_results'
        summary = results.get('summary', {})
        if summary:
            print(f"   Samples processed: {summary.get('num_samples_processed', 0)}")
            print(f"   Avg SAX code length: {summary.get('avg_sax_code_length', 0):.0f}")
            print(f"   Number of leads: {summary.get('num_leads', 0)}")
            print(f"   SAX alphabet size: {summary.get('sax_alphabet_size', 0)}")
        
        # Show SAX results summary
        sax_results = results.get('sax_results', [])
        if sax_results:
            print(f"   SAX codes extracted for {len(sax_results)} samples")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        # Create default config
        config_file = create_default_config_file()
        print(f"✅ Default configuration created: {config_file}")
        
        # Load config
        config = load_config(config_file)
        print(f"✅ Configuration loaded with {len(config)} sections")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


def main():
    """Run all tests"""
    print("🧪 TESTING CLASSICAL ECG BASELINE")
    print("=" * 50)
    
    # Test components
    test_configuration()
    test_feature_extraction()
    test_dataset_loading()
    test_pipeline()
    
    print("\n" + "=" * 50)
    print("🏁 ALL TESTS COMPLETED")
    
    # Cleanup
    try:
        import os
        import glob
        for f in glob.glob("/tmp/test_ecg_*"):
            os.remove(f)
        print("🧹 Temporary files cleaned up")
    except:
        pass


if __name__ == "__main__":
    main()
