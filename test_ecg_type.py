#!/usr/bin/env python3
"""
Test script to verify ecg_type column generation
"""

import pandas as pd
import json
import numpy as np

# Create sample data
data = {
    'waveform_name': ['ecg1.npy', 'ecg2.npy', 'ecg3.npy', 'ecg4.npy'],
    'Sinusal': [1, 1, 1, 0],
    'Regular': [1, 1, 0, 0],
    # Pathological conditions
    'Afib': [0, 0, 1, 0],  # Pathological
    'Acute MI': [0, 0, 0, 1],  # Pathological
    # Limit conditions  
    '1st degree AV block': [0, 1, 0, 0],  # Borderline
    'Left axis deviation': [0, 0, 0, 0],  # Borderline
}

df = pd.DataFrame(data)
print("Sample ECG Data:")
print(df)
print("\n" + "="*50)

# Load deepecg dictionary
with open('/volume/ECG_tokenizer/dictionary/deepecg.json', 'r') as f:
    deepecg = json.load(f)['deepecg']

pathological_cols = deepecg['pathological']
limit_cols = deepecg['limit']

print(f"\nPathological conditions to check: {len(pathological_cols)}")
print(f"Limit conditions to check: {len(limit_cols)}")

# Initialize ecg_type as normal
df['ecg_type'] = 'normal'

# Check for pathological conditions
for col in pathological_cols:
    if col in df.columns:
        # Mark as pathological if any pathological column >= 1
        mask = df[col] >= 1
        df.loc[mask, 'ecg_type'] = 'pathological'
        if mask.any():
            print(f"  Found pathological: {col}")

# Check for borderline conditions (only if not already pathological)
for col in limit_cols:
    if col in df.columns:
        # Mark as borderline if any limit column >= 1 and not already pathological
        mask = (df[col] >= 1) & (df['ecg_type'] == 'normal')
        df.loc[mask, 'ecg_type'] = 'borderline'
        if mask.any():
            print(f"  Found borderline: {col}")

print("\n" + "="*50)
print("Results with ecg_type column:")
print(df[['waveform_name', 'ecg_type']])

print("\nECG Type Distribution:")
print(df['ecg_type'].value_counts())