# ECG Plotting Documentation for Validation Generations

## Overview
This document describes the ECG plotting functionality created for visualizing model validation results with Q&A annotations. The system plots 12-lead ECGs from validation generation JSON files, displaying the model's predictions alongside ground truth answers.

## Key Features
- **12-lead ECG visualization** with proper lead ordering for MIMIC-IV dataset
- **Q&A overlay** showing question, generated answer, and ground truth
- **FFT normalization support** for preprocessed ECG signals
- **Path resolution** from parquet dataset files
- **Single file output** to avoid duplicates

## Implementation Details

### 1. Scripts Created

#### `/volume/ECG_tokenizer/utils/plot_validation_ecgs.py`
Main plotting script that:
- Loads validation generation JSON files
- Resolves ECG file paths from parquet datasets
- Plots ECGs with Q&A annotations
- Supports batch processing of multiple ECGs

#### `/volume/ECG_tokenizer/utils/plot_ecg_demo.py`
Demo script with path reconstruction capabilities for cases where ECGs aren't in the parquet files.

### 2. Key Configuration for MIMIC Dataset

```python
# MIMIC-IV specific settings
dataset = "MIMICIV"
fft_normalized = True  # CRITICAL: Must be True for FFT-preprocessed MIMIC data
width = 2500
amplitude_factor = 1.2 * 1000  # Base factor × 1000 for FFT normalized data
```

### 3. FFT Normalization
The MIMIC-IV ECG data has been preprocessed with FFT normalization, requiring:
- `fft_normalized=True` parameter in NPYECGPlotter
- Amplitude scaling factor of 1200 (base 1.2 × 1000)
- This ensures proper waveform visualization

### 4. Lead Ordering for MIMIC-IV
```python
lead_order = ["I", "II", "III", "aVR", "aVF", "aVL", 
              "V1", "V2", "V3", "V4", "V5", "V6"]
```

## Usage Examples

### Basic Usage
```bash
# Plot validation ECGs from epoch 3
python utils/plot_validation_ecgs.py \
    /volume/ECG_tokenizer/checkpoints/.../val_generations_epoch_3.json \
    --parquet /volume/ECG_tokenizer/output/mimic_test_qa_10k.parquet \
    --output-dir validation_plots_epoch3 \
    --num-ecgs 10 \
    --epoch 3
```

### Plotting Specific ECG
```bash
python utils/plot_ecg_demo.py \
    --ecg-name 42584692.npy \
    --output-dir demo_plots
```

## Example Case Study: ECG 42584692.npy

### Clinical Context
- **Question**: "Is urgent action needed for this ECG?"
- **Model Generated**: "*** ACUTE STEMI (anterior - V3-V4, septal - V1-V2, inferior - II, III, aVF) *** - Yes"
- **Ground Truth**: "No - prompt follow-up recommended; Previous inferior wall MI, Atrial tachycardia (>= 100 BPM), Right bundle branch block"

### Analysis
The model incorrectly identified an urgent STEMI when the actual findings were:
1. **Previous inferior wall MI** (old, not acute)
2. **Atrial tachycardia** (>= 100 BPM)
3. **Right bundle branch block** (RBBB)

This demonstrates a critical misclassification where the model confused old MI changes with acute STEMI.

## Technical Improvements Made

### 1. Single File Output
Modified plotting to avoid duplicate file generation:
```python
# Instead of save=True which auto-generates timestamped file
img, _ = plotter.plot_ecg(save=False, ...)
# Manual save with meaningful name
img.save(custom_path, dpi=(240, 240))
```

### 2. Path Resolution Strategy
Handles both current JSON structure (filename only) and future structure (full paths):
```python
if ecg_name.startswith('/'):
    ecg_path = ecg_name  # Already full path
else:
    # Look up in parquet dataset
    ecg_path = parquet_df[parquet_df['waveform_name'] == ecg_name]['waveform_path_psa'].iloc[0]
```

### 3. Q&A Formatting
Truncates long answers for better visualization:
```python
max_len = 150
if len(answer) > max_len:
    answer = answer[:max_len] + "..."
```

## Dataset Information

### Validation Dataset
- **Path**: `/volume/ECG_tokenizer/output/mimic_test_qa_10k.parquet`
- **Size**: 10,000 ECGs
- **Signal column**: `waveform_path_psa`
- **Format**: 12-lead ECG, 2500 samples per lead

### File Structure
```
/media/data1/datasets/MIMIC-IV/adjusted_signals/test/{ecg_id}.npy
```

## Visualization Output

Each plot contains:
1. **12-lead ECG waveform** displayed in standard clinical format
2. **Question** posed to the model
3. **Generated answer** from the model
4. **Ground truth answer** for comparison
5. **ECG identifier** and epoch number (if applicable)

## Common Issues and Solutions

### Issue 1: Incorrect Waveform Scaling
**Solution**: Ensure `fft_normalized=True` for MIMIC-IV preprocessed data

### Issue 2: ECG Not Found
**Solution**: Check if ECG exists in the validation dataset parquet file

### Issue 3: Duplicate Files
**Solution**: Use `save=False` in plot_ecg() and manually save

## Future Enhancements

1. **Add waveform_path to validation JSON** during model validation
2. **Support for other datasets** (PTB-XL, etc.)
3. **Batch comparison plots** showing multiple ECGs side-by-side
4. **Metrics overlay** (accuracy, F1, etc.) on plots
5. **Interactive HTML reports** with zoomable ECGs

## Important Update: FFT Normalization Fixed

**CRITICAL**: All plotting scripts have been updated to use `fft_normalized=True` for MIMIC-IV data. This is essential because the MIMIC-IV ECG signals have been preprocessed with FFT normalization, requiring:
- Amplitude scaling factor of 1200 (base 1.2 × 1000)
- Proper waveform visualization with correct voltage scaling

### Successfully Plotted 5 ECGs with FFT Normalization:
1. **42246715.npy** - Atrial flutter detection case
2. **46236501.npy** - Missed STEMI case  
3. **47884129.npy** - Critical miss of acute ST elevation MI
4. **40420459.npy** - Simplified findings case
5. **44709741.npy** - Missed atrial fibrillation case

All plots saved in `/volume/ECG_tokenizer/FINAL_5_ECGS_FFT_TRUE/` with proper FFT normalization applied.

## Conclusion

This ECG plotting system provides crucial visualization for understanding model performance on ECG interpretation tasks. By displaying the actual waveforms alongside Q&A pairs with correct FFT normalization, it helps identify specific failure modes and guides model improvement efforts.

The case of ECG 42584692.npy exemplifies the importance of visual validation - the model's confusion between old MI and acute STEMI is immediately apparent when viewing the ECG with the annotations.