# Classical ECG Baseline

This directory contains a classical signal processing baseline that integrates with the ECG_tokenizer infrastructure to process preprocessed MIMIC-IV ECG data.

## Overview

The classical baseline implements the following pipeline:
1. **Load signals** using ECG_tokenizer dataset infrastructure
2. **Feature extraction** using PyWavelets, PyEMD/VMD (optional), and pyts/SAX
3. **Classification** using traditional machine learning algorithms

## Architecture

```
ECG Signal (12 leads) 
    ↓
PyWavelets Decomposition → Wavelet Features
    ↓
PyEMD/VMD Decomposition → Mode Features (optional)
    ↓
pyts/SAX Transformation → Symbolic Features
    ↓
Traditional ML Classifiers → ECG Pattern Predictions
```

## Features Extracted

### Wavelet Features (PyWavelets)
- **Approximation coefficients**: Energy, mean, std, max
- **Detail coefficients**: Energy, mean, std, max for each decomposition level
- **Wavelet entropy**: Information content measure

### EMD/VMD Features (Optional)
- **EMD Intrinsic Mode Functions (IMFs)**: Energy, statistical features
- **VMD Mode decompositions**: Energy, frequency, statistical features

### SAX Features (pyts)
- **Symbolic representation**: Discrete word sequences
- **Symbol statistics**: Frequency, entropy, unique symbols
- **Bag of Words**: Pattern frequency features

### Per-Lead Analysis
All features are extracted for each of the 12 ECG leads independently.

## Installation

```bash
cd /volume/ECG_tokenizer/classical_baseline
pip install -r requirements.txt

# Optional EMD/VMD libraries (uncomment in requirements.txt if needed)
# pip install PyEMD vmdpy
```

## Usage

### Quick Start

```bash
# Test the installation
python test_baseline.py

# Run with automatic dataset detection
python run_classical_baseline.py --auto_detect

# Run with specific parquet file
python run_classical_baseline.py --parquet_file /path/to/your/data.parquet
```

### Configuration

Create a custom configuration:

```bash
# Create default config
python run_classical_baseline.py --create_config

# Edit config/classical_baseline_config.yaml as needed
# Run with custom config
python run_classical_baseline.py --config config/custom_config.yaml --parquet_file data.parquet
```

### Command Line Options

```bash
python run_classical_baseline.py [OPTIONS]

Options:
  --parquet_file PATH      Path to parquet file with ECG data
  --config PATH           Configuration YAML file
  --auto_detect           Automatically detect available datasets
  --max_samples N         Limit to N samples (for testing)
  --no_emd               Skip EMD/VMD features (faster)
  --output_dir PATH       Output directory for results
  --lead_stats_file PATH  JSON file with lead statistics
  --dry_run              Setup only, no training
  --list_datasets        List available datasets
```

### Examples

```bash
# Quick test with 100 samples
python run_classical_baseline.py --auto_detect --max_samples 100

# Full run with custom output directory
python run_classical_baseline.py --parquet_file data.parquet --output_dir results/

# Skip EMD features for faster processing
python run_classical_baseline.py --auto_detect --no_emd
```

## Data Format

### Expected Input
- **Parquet file** with columns:
  - `waveform_path_psa`: Path to .npy files containing ECG signals
  - ECG pattern columns (see `utils.constants.ECG_PATTERNS`)

### Signal Format
- **Shape**: `(time_points, leads)` or `(leads, time_points)`
- **Leads**: 12-lead ECG in standard order (I, II, III, aVR, aVL, aVF, V1-V6)
- **Sampling rate**: 500 Hz (configurable)
- **Duration**: ~10 seconds (configurable)

### Output
- **Results**: JSON file with classifier performance metrics
- **Models**: Pickled trained classifiers and scalers
- **Features**: Extracted feature matrices (optional)

## Configuration

### Signal Processing Parameters

```yaml
signal_processing:
  wavelet:
    name: 'db4'              # Wavelet type
    levels: 6                # Decomposition levels
  
  emd:
    max_imf: 8              # Maximum IMFs
  
  vmd:
    alpha: 2000             # Bandwidth constraint
    K: 8                    # Number of modes
  
  sax:
    n_segments: 20          # SAX segments
    alphabet_size: 8        # Symbol alphabet size
```

### Machine Learning Parameters

```yaml
machine_learning:
  cv_folds: 5               # Cross-validation folds
  n_jobs: -1               # Parallel jobs
  classifiers:
    random_forest:
      n_estimators: 100
    gradient_boosting:
      n_estimators: 100
    svm:
      kernel: 'rbf'
    logistic_regression:
      max_iter: 1000
```

## Results

### Performance Metrics
- **Exact Match Accuracy**: Percentage of samples with all patterns correctly predicted
- **Per-Pattern Metrics**: Precision, recall, F1-score, AUC for each ECG pattern
- **Average Metrics**: Mean performance across all patterns

### Output Files
- `classical_baseline_results.json`: Detailed performance metrics
- `classical_baseline_models.pkl`: Trained models and preprocessing
- `run_config.yaml`: Configuration used for the run
- `lead_statistics.json`: Dataset lead statistics

## Integration with ECG_tokenizer

This baseline leverages the existing ECG_tokenizer infrastructure:

- **Dataset loading**: Uses `ECGTokenizerClassifierDataset`
- **Constants**: Imports `ECG_PATTERNS` and `standard_lead_order`
- **Data paths**: Compatible with ECG_tokenizer data organization
- **Output format**: Consistent with project structure

## Performance Expectations

### Feature Counts
- **Wavelet features**: ~40 per lead (480 total for 12 leads)
- **EMD features**: ~32 per lead (384 total, if enabled)
- **SAX features**: ~15 per lead (180 total)
- **Total**: ~700-1000+ features per ECG sample

### Processing Speed
- **Feature extraction**: ~1-5 seconds per sample
- **Training**: Depends on dataset size and classifier
- **Memory usage**: ~1-2 GB for typical datasets

### Typical Accuracy
- **Random Forest**: 60-80% exact match accuracy
- **Gradient Boosting**: 65-85% exact match accuracy
- **SVM**: 55-75% exact match accuracy
- **Logistic Regression**: 50-70% exact match accuracy

*Note: Performance varies significantly based on dataset quality and class distribution*

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory errors**: Reduce `max_samples` or enable feature saving
3. **EMD/VMD failures**: Use `--no_emd` flag or install optional libraries
4. **Dataset format**: Verify parquet file structure and signal paths

### Debug Mode

```bash
# Run with limited samples for debugging
python run_classical_baseline.py --auto_detect --max_samples 10 --dry_run

# Test individual components
python test_baseline.py
```

## Dependencies

### Required
- `pywavelets>=1.4.1`: Wavelet transforms
- `pyts>=0.12.0`: Time series transformations (SAX)
- `scikit-learn>=1.3.0`: Machine learning algorithms
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing

### Optional
- `PyEMD>=0.3.0`: Empirical Mode Decomposition
- `vmdpy>=0.2.0`: Variational Mode Decomposition

### Development
- `matplotlib>=3.5.0`: Plotting
- `seaborn>=0.11.0`: Statistical visualization
- `tqdm>=4.64.0`: Progress bars

## Comparison with Deep Learning

This classical baseline provides:

- **Interpretable features**: Each feature has clear signal processing meaning
- **Fast training**: Traditional ML is much faster than deep learning
- **Low computational requirements**: Can run on CPU
- **Domain knowledge**: Incorporates established ECG analysis techniques
- **Baseline performance**: Reference point for deep learning improvements

Use this baseline to:
1. Validate dataset quality
2. Establish performance baselines
3. Compare with deep learning approaches
4. Understand feature importance
5. Debug preprocessing issues

## Citation

When using this classical baseline, please cite:

- PyWavelets: Lee, G.R., et al. (2019). PyWavelets
- pyts: Faouzi, J. (2020). pyts: A Python Package for Time Series Classification
- scikit-learn: Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python
