# Adding New Datasets

This guide explains how to add support for new ECG datasets with different column naming conventions.

## Overview

The system uses a column mapping approach to handle different dataset structures. All dataset-specific column names are defined in `dataset_column_mappings.py`.

## Steps to Add a New Dataset

### 1. Define Column Mappings

Edit `dataset_column_mappings.py` and add your dataset's column mappings:

```python
# Your dataset column mappings
YOUR_DATASET_COLUMNS = {
    # Core identifiers and paths
    "waveform_id": "your_ecg_id_column",
    "signal_path": "your_signal_path_column",
    "report": "your_report_column",
    "ecg_type": "your_classification_column",
    
    # Demographics
    "gender": "your_gender_column",
    "age": "your_age_column",
    "rr_interval": "your_rr_interval_column",  # Or None if not available
    
    # Rhythm findings
    "sinus_rhythm": "your_sinus_column",
    "regular_rhythm": "your_regular_column",
    "atrial_fibrillation": "your_afib_column",
    # ... add all rhythm mappings
    
    # Conduction findings
    "left_bundle_branch": "your_lbbb_column",
    "right_bundle_branch": "your_rbbb_column",
    # ... add all conduction mappings
    
    # Axis findings
    "left_axis_deviation": "your_lad_column",
    "right_axis_deviation": "your_rad_column",
    "normal_axis": "your_normal_axis_column",
    
    # Localization patterns (for Q waves, ST changes, etc.)
    "q_wave_patterns": [
        "your_q_wave_inferior_column",
        "your_q_wave_anterior_column",
        # ... list all Q wave location columns
    ],
    "st_elevation_patterns": [
        # ... list all ST elevation location columns
    ],
    "st_depression_patterns": [
        # ... list all ST depression location columns
    ],
    "t_wave_patterns": [
        # ... list all T wave location columns
    ],
    
    # Other findings as needed
}
```

### 2. Register the Dataset

Add your dataset to the registry in `dataset_column_mappings.py`:

```python
# Dataset registry
DATASET_MAPPINGS = {
    "mimic": MIMIC_COLUMNS,
    "ptbxl": PTBXL_COLUMNS,
    "your_dataset": YOUR_DATASET_COLUMNS,  # Add your dataset here
}
```

### 3. Update the Generate Script

If needed, update the file paths in `generate_train_test_datasets.py`:

```python
if dataset_type == "your_dataset":
    test_input = '/path/to/your/test/data.parquet'
    train_input = '/path/to/your/train/data.parquet'
```

### 4. Run Dataset Generation

Generate the Q&A dataset with your mappings:

```bash
python generate_train_test_datasets.py --dataset your_dataset
```

## Column Mapping Rules

### Required Columns

These columns must be mapped for basic functionality:
- `waveform_id`: Unique identifier for each ECG
- `signal_path`: Path to the ECG signal file (.npy or similar)
- `report`: Text report or diagnosis

### Optional Columns

These enhance the Q&A generation but aren't required:
- `ecg_type`: Classification (normal/borderline/pathological)
- Demographics: `gender`, `age`, `rr_interval`
- Specific findings: Map as many as available

### Handling Missing Columns

If a column doesn't exist in your dataset, set it to `None`:

```python
"rr_interval": None,  # Not available in this dataset
```

### Pattern Columns

For localization findings (Q waves, ST changes, etc.), use lists:

```python
"q_wave_patterns": [
    "Q_wave_II_III_aVF",  # Inferior
    "Q_wave_V1_V4",       # Anterior
    "Q_wave_I_aVL_V5_V6", # Lateral
]
```

## Example: Adding PTB-XL Dataset

```python
PTBXL_COLUMNS = {
    # Core
    "waveform_id": "ecg_id",
    "signal_path": "filename_hr",
    "report": "report",
    "ecg_type": "diagnostic_superclass",
    
    # Demographics
    "gender": "sex",  # PTB-XL uses 0/1
    "age": "age",
    "rr_interval": None,  # Not directly available
    
    # Rhythm (using SCP codes)
    "sinus_rhythm": "NORM",
    "atrial_fibrillation": "AFIB",
    "atrial_flutter": "AFLT",
    
    # Map other findings...
}
```

## Testing Your Mappings

Test with a small sample first:

```python
from dataset_column_mappings import DatasetColumnMapper

# Test your mapper
mapper = DatasetColumnMapper("your_dataset")

# Check a mapping
print(mapper.get_column("signal_path"))  # Should print your column name

# Test with actual data
import pandas as pd
df = pd.read_parquet("your_data.parquet")
row = df.iloc[0]

# Get value using mapper
signal_path = mapper.get_value(row, "signal_path")
print(f"Signal path: {signal_path}")
```

## Troubleshooting

### Column Not Found
If you get "column not found" errors:
1. Check the exact column name in your dataset
2. Verify the mapping in `dataset_column_mappings.py`
3. Use `df.columns` to see all available columns

### Type Mismatches
The system expects:
- Binary columns: 0/1 or boolean values
- Text columns: strings
- Numeric columns: int or float

Convert as needed in your mappings or preprocessing.

### Performance
For large datasets, use sampling during development:
```bash
python generate_train_test_datasets.py --dataset your_dataset --sample_size 100
```

## Support

For questions or issues adding new datasets, check:
- `dataset_column_mappings.py` for examples
- `generate_train_test_datasets.py` for the processing logic
- The MIMIC mappings as a reference implementation