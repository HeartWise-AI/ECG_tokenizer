#!/usr/bin/env python3
"""
Dataset-specific column mappings for different ECG datasets.
This allows the system to work with different datasets that have different column naming conventions.
"""

# MIMIC-IV dataset column mappings
MIMIC_COLUMNS = {
    # Core identifiers and paths
    "waveform_id": "waveform_name",
    "signal_path": "waveform_path_psa",
    "report": "report",
    "ecg_type": "ecg_type",
    
    # Demographics
    "gender": "gender",
    "age": "age_at_ecg",
    "rr_interval": "rr_interval",
    
    # Rhythm findings
    "sinus_rhythm": "Sinusal",
    "regular_rhythm": "Regular",
    "atrial_fibrillation": "Atrial fibrillation",
    "atrial_flutter": "Atrial flutter",
    "bradycardia": "Bradycardia (<= 60 BPM)",
    "tachycardia": "Atrial tachycardia (>= 100 BPM)",
    "premature_ventricular": "Premature ventricular complex",
    "premature_atrial": "Premature atrial complex",
    
    # Conduction findings
    "left_bundle_branch": "Left bundle branch block",
    "right_bundle_branch": "Right bundle branch block",
    "first_degree_av_block": "First Degree AV Block",
    "second_degree_av_block": "Second Degree AV Block",
    "third_degree_av_block": "Third Degree AV Block",
    "wpw": "WPW",
    
    # Axis findings
    "left_axis_deviation": "Left axis deviation",
    "right_axis_deviation": "Right axis deviation",
    "normal_axis": "Normal axis",
    
    # Chamber enlargement
    "left_ventricular_hypertrophy": "LVH",
    "right_ventricular_hypertrophy": "RVH",
    "left_atrial_enlargement": "LAE",
    "right_atrial_enlargement": "RAE",
    
    # Ischemia/Infarction with locations
    "q_wave_patterns": [
        "Q wave (inferior - II, III, aVF)",
        "Q wave (anterior - V1-V4)",
        "Q wave (lateral - I, aVL, V5-V6)",
        "Q wave (septal - V1-V2)"
    ],
    "st_elevation_patterns": [
        "ST elevation (inferior - II, III, aVF)",
        "ST elevation (anterior - V3-V4)",
        "ST elevation (lateral - I, aVL, V5-V6)",
        "ST elevation (septal - V1-V2)"
    ],
    "st_depression_patterns": [
        "ST depression (inferior - II, III, aVF)",
        "ST depression (anterior - V3-V4)",
        "ST depression (lateral - I, aVL, V5-V6)",
        "ST depression (septal - V1-V2)"
    ],
    "t_wave_patterns": [
        "T wave inversion (inferior - II, III, aVF)",
        "T wave inversion (anterior - V3-V4)",
        "T wave inversion (lateral - I, aVL, V5-V6)",
        "T wave inversion (septal- V1-V2)"
    ],
    
    # ST morphology
    "st_downsloping": "ST downsloping",
    "st_upsloping": "ST upslopping",
    "early_repolarization": "Early repolarization",
    
    # Other findings
    "monomorph_qrs": "Monomorph",
    "pericarditis": "Pericarditis",
    
    # Pacing
    "atrial_paced": "Atrial paced",
    "ventricular_paced": "Ventricular paced",
}

# PTB-XL dataset column mappings (example for another dataset)
PTBXL_COLUMNS = {
    # Core identifiers and paths
    "waveform_id": "ecg_id",
    "signal_path": "filename_hr",
    "report": "report",
    "ecg_type": "diagnostic_superclass",
    
    # Demographics
    "gender": "sex",
    "age": "age",
    "rr_interval": None,  # Not available in PTB-XL
    
    # Rhythm findings (using SCP codes)
    "sinus_rhythm": "SR",
    "regular_rhythm": None,
    "atrial_fibrillation": "AFIB",
    "atrial_flutter": "AFLT",
    "bradycardia": "BRADY",
    "tachycardia": "TACHY",
    
    # Add more mappings as needed for PTB-XL
}


# MHI dataset columns - uses same column names as MIMIC since the data is preprocessed,
# but prefers the translated diagnosis text when available.
MHI_COLUMNS = MIMIC_COLUMNS.copy()
MHI_COLUMNS['report'] = ['translated_diagnosis', 'report']

# Dataset registry
DATASET_MAPPINGS = {
    "mimic": MIMIC_COLUMNS,
    "ptbxl": PTBXL_COLUMNS,
    "mhi": MHI_COLUMNS,
    # "cpsc": CPSC_COLUMNS,  # Uncomment when CPSC mappings are defined
}


class DatasetColumnMapper:
    """
    Helper class to map dataset-specific columns to standardized names.
    """
    
    def __init__(self, dataset_name: str = "mimic"):
        """
        Initialize with a specific dataset's column mappings.
        
        Args:
            dataset_name: Name of the dataset ("mimic", "ptbxl", "cpsc", etc.)
        """
        if dataset_name not in DATASET_MAPPINGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_MAPPINGS.keys())}")
        
        self.dataset_name = dataset_name
        self.column_map = DATASET_MAPPINGS[dataset_name]
    
    def get_column(self, standard_name: str, default=None):
        """
        Get the dataset-specific column name for a standardized name.
        
        Args:
            standard_name: The standardized column name
            default: Default value if column not found
            
        Returns:
            The dataset-specific column name or default
        """
        return self.column_map.get(standard_name, default)
    
    def get_pattern_columns(self, pattern_type: str):
        """
        Get list of pattern columns (for localization findings).
        
        Args:
            pattern_type: Type of pattern ("q_wave", "st_elevation", etc.)
            
        Returns:
            List of column names for that pattern type
        """
        pattern_key = f"{pattern_type}_patterns"
        patterns = self.column_map.get(pattern_key, [])
        
        # Handle both single string and list
        if isinstance(patterns, str):
            return [patterns]
        elif isinstance(patterns, list):
            return patterns
        else:
            return []
    
    def check_column_exists(self, df, standard_name: str):
        """
        Check if a standardized column exists in the dataframe.
        
        Args:
            df: Pandas dataframe
            standard_name: The standardized column name
            
        Returns:
            True if column exists, False otherwise
        """
        column_name = self.get_column(standard_name)
        if column_name is None:
            return False
        
        if isinstance(column_name, list):
            # Check if any of the columns exist
            return any(col in df.columns for col in column_name)
        else:
            return column_name in df.columns
    
    def get_value(self, row, standard_name: str, default=None):
        """
        Get value from a row using standardized column name.
        
        Args:
            row: Pandas Series (dataframe row)
            standard_name: The standardized column name
            default: Default value if column not found or value is null
            
        Returns:
            The value from the row or default
        """
        column_name = self.get_column(standard_name)
        if column_name is None:
            return default
        
        if isinstance(column_name, list):
            # Return first non-null value from list of columns
            import pandas as pd
            for col in column_name:
                if col in row.index and pd.notna(row[col]):
                    return row[col]
            return default
        else:
            import pandas as pd
            if column_name in row.index and pd.notna(row[column_name]):
                return row[column_name]
            return default


# Example usage
if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available for example")
    
    # Example with MIMIC dataset
    mapper = DatasetColumnMapper("mimic")
    
    print("MIMIC Column Mappings:")
    print(f"  Signal path: {mapper.get_column('signal_path')}")
    print(f"  Gender: {mapper.get_column('gender')}")
    print(f"  Sinus rhythm: {mapper.get_column('sinus_rhythm')}")
    
    # Get Q wave pattern columns
    q_wave_cols = mapper.get_pattern_columns("q_wave")
    print(f"\nQ wave columns: {q_wave_cols}")
    
    # Example with different dataset
    try:
        ptbxl_mapper = DatasetColumnMapper("ptbxl")
        print(f"\nPTB-XL Signal path: {ptbxl_mapper.get_column('signal_path')}")
    except ValueError as e:
        print(f"Error: {e}")
