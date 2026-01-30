#!/usr/bin/env python3
"""Create stratified sample for DPO training - oversample weak categories"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("Loading training data...")
df = pd.read_parquet('/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_balanced.parquet')
print(f"Total rows: {len(df)}")

# Sampling strategy: oversample weak categories
sampling_config = {
    # CRITICAL - take all or most
    'urgency_assessment': 70,
    'culprit_artery': 919,
    'localization_st_depression': 18,
    'localization_q_wave': 329,
    'localization_t_wave': 489,
    'localization_qrs_axis': 1000,
    
    # HIGH PRIORITY - weak categories
    'acs_severity': 5000,
    'classification': 10000,
    'interpretation': 8000,
    
    # MEDIUM
    'structural_heart_disease': 3000,
    'afib_risk': 3000,
    'category_conduction': 2000,
    'category_infarct_ischemia': 2000,
    
    # LOW PRIORITY - already good
    'json_interpretation': 5000,
    'category_rhythm': 3000,
    'lvef': 2000,
    'category_other': 1000,
    'category_chamber_enlargement': 1000,
    'ecg_interval': 1000,
    'category_pericarditis': 500,
    'random_finding_question': 500,
}

samples = []
for cat, n in sampling_config.items():
    cat_df = df[df['prompt_category'] == cat]
    if len(cat_df) == 0:
        print(f"  {cat}: 0 rows (skipping)")
        continue
    n_sample = min(n, len(cat_df))
    sampled = cat_df.sample(n=n_sample, random_state=42)
    samples.append(sampled)
    print(f"  {cat}: {n_sample} / {len(cat_df)}")

result = pd.concat(samples, ignore_index=True)
print(f"\nTotal sampled: {len(result)}")

output_path = '/volume/ECG_tokenizer/output/dpo_train_sample_50k.parquet'
result.to_parquet(output_path, index=False)
print(f"Saved to {output_path}")
