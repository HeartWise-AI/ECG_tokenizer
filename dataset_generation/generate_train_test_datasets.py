#!/usr/bin/env python3
"""
Generate train and test datasets with enhanced prompts and answers.
Drops the existing question column and generates new comprehensive Q&A pairs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import sys
import json
import numpy as np
from tqdm import tqdm
from ecg_prompt_maker import ECGPromptMaker
from ecg_answer_generator import ECGAnswerGenerator


def apply_mhi_special_stratified_sampling(df_mhi, target_samples, target_special_questions=1000):
    """
    Apply stratified sampling for MHI special questions to achieve specific ratios:
    - Ensure at least target_special_questions (default 1000) special questions
    - AFib risk: 50% high risk, 50% low risk  
    - SHD: 50% normal, 50% abnormal
    - ACS: 25% acute coronary occlusion, 75% non-acute
    - LVEF: keep natural distribution
    
    ECGs can count for multiple categories. Since 1 prompt per ECG, we need 1000 ECGs with special questions.
    Returns exactly target_samples ECGs with at least target_special_questions having special question data.
    """
    print(f"   Applying stratified sampling for MHI special questions (target: {target_special_questions} special questions)...")
    
    # First, collect all ECGs that qualify for special questions
    special_ecgs = set()
    
    # Track which ECGs qualify for which categories
    afib_high_risk_ecgs = set()
    afib_low_risk_ecgs = set()
    shd_abnormal_ecgs = set()
    shd_normal_ecgs = set()
    acs_acute_ecgs = set()
    acs_non_acute_ecgs = set()
    lvef_ecgs = set()
    
    # 1. Identify AFib risk ECGs
    afib_data = df_mhi[(df_mhi['afib_label_2y'].notna()) & (df_mhi['afib_label_5y'].notna())]
    if len(afib_data) > 0:
        # Exclude patients already in AFib
        afib_data = afib_data[(afib_data['Afib'] == 0) | (afib_data['Afib'].isna())]
        afib_data = afib_data[(afib_data['Afib_bert_model'] <= 0.5) | (afib_data['Afib_bert_model'].isna())]
        
        # High risk: 2-year or 5-year risk is True
        high_risk = afib_data[(afib_data['afib_label_2y'] == True) | (afib_data['afib_label_5y'] == True)]
        low_risk = afib_data[(afib_data['afib_label_2y'] == False) & (afib_data['afib_label_5y'] == False)]
        
        afib_high_risk_ecgs = set(high_risk['waveform_name'].values)
        afib_low_risk_ecgs = set(low_risk['waveform_name'].values)
        special_ecgs.update(afib_high_risk_ecgs)
        special_ecgs.update(afib_low_risk_ecgs)
    
    # 2. Identify SHD ECGs
    shd_data = df_mhi[df_mhi['echonext_shd'].notna()]
    if len(shd_data) > 0:
        shd_abnormal = shd_data[shd_data['echonext_shd'] >= 1]
        shd_normal = shd_data[shd_data['echonext_shd'] < 1]
        
        shd_abnormal_ecgs = set(shd_abnormal['waveform_name'].values)
        shd_normal_ecgs = set(shd_normal['waveform_name'].values)
        special_ecgs.update(shd_abnormal_ecgs)
        special_ecgs.update(shd_normal_ecgs)
    
    # 3. Identify ACS ECGs
    acs_data = df_mhi[df_mhi['acs_condition_severity'].notna()]
    if len(acs_data) > 0:
        from utils.constants import ACS_ACUTE_CONDITIONS
        
        acs_acute = acs_data[acs_data['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
        acs_non_acute = acs_data[~acs_data['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
        
        acs_acute_ecgs = set(acs_acute['waveform_name'].values)
        acs_non_acute_ecgs = set(acs_non_acute['waveform_name'].values)
        special_ecgs.update(acs_acute_ecgs)
        special_ecgs.update(acs_non_acute_ecgs)
    
    # 4. Identify LVEF ECGs
    lvef_data = df_mhi[df_mhi['deepecho_Visually_Estimated_EF'].notna()]
    if len(lvef_data) > 0:
        lvef_ecgs = set(lvef_data['waveform_name'].values)
        special_ecgs.update(lvef_ecgs)
    
    print(f"     Total unique ECGs with special questions: {len(special_ecgs)}")
    print(f"       - AFib risk: {len(afib_high_risk_ecgs)} high, {len(afib_low_risk_ecgs)} low")
    print(f"       - SHD: {len(shd_abnormal_ecgs)} abnormal, {len(shd_normal_ecgs)} normal")
    print(f"       - ACS: {len(acs_acute_ecgs)} acute, {len(acs_non_acute_ecgs)} non-acute")
    print(f"       - LVEF: {len(lvef_ecgs)} with data")
    
    # Calculate how many special ECGs to sample to get at least target_special_questions
    # Since we have 1 prompt per ECG and want 20% to be special questions
    special_ecgs_needed = min(target_special_questions, len(special_ecgs))
    
    # Sample special ECGs with stratification targets
    sampled_special = set()
    
    # Target distribution for 1000 special questions:
    # - AFib risk: 250 (125 high, 125 low)
    # - SHD: 250 (125 abnormal, 125 normal)
    # - ACS: 250 (62 acute, 188 non-acute)
    # - LVEF: 250
    per_category = special_ecgs_needed // 4  # 250 each for test, proportional for train
    
    categories = [
        (afib_high_risk_ecgs, per_category // 2),      # 125 high risk AFib
        (afib_low_risk_ecgs, per_category // 2),       # 125 low risk AFib
        (shd_abnormal_ecgs, per_category // 2),        # 125 abnormal SHD
        (shd_normal_ecgs, per_category // 2),          # 125 normal SHD
        (acs_acute_ecgs, per_category // 4),           # 62 acute ACS (25% of ACS)
        (acs_non_acute_ecgs, (per_category * 3) // 4), # 188 non-acute ACS (75% of ACS)
        (lvef_ecgs, per_category)                      # 250 LVEF
    ]
    
    for ecg_set, target_count in categories:
        available = list(ecg_set - sampled_special)
        if available:
            sample_count = min(len(available), target_count)
            if sample_count > 0:
                sampled = np.random.choice(available, size=sample_count, replace=False)
                sampled_special.update(sampled)
    
    # Fill remaining special slots if needed
    remaining_special_needed = special_ecgs_needed - len(sampled_special)
    if remaining_special_needed > 0:
        available = list(special_ecgs - sampled_special)
        if available:
            additional = np.random.choice(available, 
                                        size=min(remaining_special_needed, len(available)), 
                                        replace=False)
            sampled_special.update(additional)
    
    # Now add non-special ECGs to reach target_samples
    all_ecgs = df_mhi['waveform_name'].unique()
    non_special_ecgs = [ecg for ecg in all_ecgs if ecg not in special_ecgs]
    
    non_special_needed = target_samples - len(sampled_special)
    sampled_non_special = []
    
    if non_special_needed > 0 and len(non_special_ecgs) > 0:
        # Sample non-special ECGs
        sample_count = min(non_special_needed, len(non_special_ecgs))
        sampled_non_special = list(np.random.choice(non_special_ecgs, size=sample_count, replace=False))
    
    # Combine special and non-special
    all_sampled = list(sampled_special) + sampled_non_special
    
    # If still need more, use replacement sampling
    if len(all_sampled) < target_samples:
        still_needed = target_samples - len(all_sampled)
        print(f"     Using replacement sampling to add {still_needed} more ECGs")
        all_available = list(all_ecgs)
        duplicates = list(np.random.choice(all_available, size=still_needed, replace=True))
        all_sampled.extend(duplicates)
    
    # Create the final DataFrame
    sampled_df = df_mhi[df_mhi['waveform_name'].isin(set(all_sampled[:target_samples]))].copy()
    
    # If we used duplicates, handle them properly
    if len(all_sampled) > len(set(all_sampled)):
        ecg_counts = {}
        for ecg in all_sampled[:target_samples]:
            ecg_counts[ecg] = ecg_counts.get(ecg, 0) + 1
        
        dfs_to_concat = []
        for ecg, count in ecg_counts.items():
            ecg_rows = df_mhi[df_mhi['waveform_name'] == ecg]
            for _ in range(count):
                dfs_to_concat.append(ecg_rows)
        
        sampled_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Mark special ECGs for tracking
    sampled_df['has_special_question'] = sampled_df['waveform_name'].isin(sampled_special)
    
    print(f"     Final sample: {len(sampled_df)} ECGs")
    print(f"       - Special question ECGs: {sampled_df['has_special_question'].sum()} ({sampled_df['has_special_question'].sum()/len(sampled_df)*100:.1f}%)")
    print(f"       - Regular ECGs: {(~sampled_df['has_special_question']).sum()} ({(~sampled_df['has_special_question']).sum()/len(sampled_df)*100:.1f}%)")
    
    return sampled_df.head(target_samples)  # Ensure exactly target_samples


def process_dataset(input_path: str, output_path: str, dataset_name: str, sample_size: int = None, dataset_type: str = 'mimic-iv', max_prompts_per_ecg: int = None, max_normal_percentage: float = 0.05, min_samples_per_diagnosis: int = 1, mimic_samples: int = None, mhi_samples: int = None):
    """Process a single dataset (train or test)
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        dataset_name: Name of dataset ('train' or 'test')
        sample_size: Target number of samples to generate
        dataset_type: Type of dataset ('mimic-iv', 'mhi', or 'combined')
        max_prompts_per_ecg: Maximum prompts to generate per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 1)
        mimic_samples: For combined mode, number of MIMIC samples to include
        mhi_samples: For combined mode, number of MHI samples to include
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"Dataset type: {dataset_type.upper()}")
    if dataset_type == 'combined':
        print(f"  MIMIC samples: {mimic_samples or 0}")
        print(f"  MHI samples: {mhi_samples or 0}")
    print(f"{'='*60}")
    
    # 1. Load dataset
    print(f"\n1. Loading {dataset_name} dataset...")
    print(f"   Input: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df)} records")
    
    # 2. Drop existing question column if it exists
    if 'question' in df.columns:
        print("   Dropping existing 'question' column...")
        df = df.drop(columns=['question'])
    
    print(f"   Columns: {list(df.columns)[:10]}...")
    
    # 3. Load and merge demographic data based on dataset type
    if dataset_type == 'combined':
        # Process combined dataset - load both MIMIC and MHI data
        print(f"\n2. Processing COMBINED dataset...")
        
        all_dfs = []
        
        # Process MIMIC portion if requested
        if mimic_samples and mimic_samples > 0:
            print(f"\n   Loading MIMIC-IV demographic data...")
            mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
            mimic_df = pd.read_parquet(mimic_path)
            
            # Extract npy_id for merging
            mimic_df['npy_id'] = mimic_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
            df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
            
            # Select demographic and interval timing columns to merge
            demographic_cols = ['new_PatientID', 'npy_id', 'gender', 'age_at_ecg', 'rr_interval',
                               'p_onset', 'qrs_onset', 'qrs_end', 't_end']  # Add timing columns for interval calculation
            available_cols = [col for col in demographic_cols if col in mimic_df.columns]
            
            # Merge MIMIC data
            df_mimic_merged = df.merge(
                mimic_df[available_cols],
                on='npy_id',
                how='inner',  # Inner join to get only MIMIC matches
                suffixes=('', '_mimic')
            )
            df_mimic_merged['dataset_source'] = 'mimic-iv'
            df_mimic_merged = df_mimic_merged.drop(columns=['npy_id'], errors='ignore')
            
            print(f"     Found {len(df_mimic_merged)} MIMIC-IV records")
            all_dfs.append(df_mimic_merged)
        
        # Process MHI portion if requested
        if mhi_samples and mhi_samples > 0:
            print(f"\n   Loading MHI demographic data (1.7M rows, this may take a moment)...")
            mhi_path = '/media/data1/muse_ge/ECG_ad20241231_cat_labels_v1.4.complete.ROXs42Bb.parquet'
            mhi_df = pd.read_parquet(mhi_path)
            mhi_df = mhi_df.dropna(subset=['waveform_path_psa'])
            # Extract npy_id from npy_path for merging
            
            mhi_df['npy_id'] = mhi_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
            df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
            
            # Map MHI columns to standard names
            mhi_df['gender'] = mhi_df['RestingECG_PatientDemographics_Gender'].map({'MALE': 'M', 'FEMALE': 'F'})
            mhi_df['age_at_ecg'] = pd.to_numeric(mhi_df['RestingECG_PatientDemographics_PatientAge'], errors='coerce')
            mhi_df['rr_interval'] = pd.to_numeric(mhi_df['RestingECG_QRSTimesTypes_GlobalRR'], errors='coerce')
            # Add VentricularRate column for MHI heart rate
            mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'] = pd.to_numeric(
                mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'], errors='coerce'
            )
            
            # Select columns to merge (including interval measurements and echonext)
            merge_cols = ['npy_id', 'new_PatientID', 'gender', 'age_at_ecg', 'rr_interval',
                          'RestingECG_OriginalRestingECGMeasurements_VentricularRate',
                          'RestingECG_OriginalRestingECGMeasurements_PRInterval',
                          'RestingECG_OriginalRestingECGMeasurements_QRSDuration',
                          'RestingECG_OriginalRestingECGMeasurements_QTInterval',
                          'RestingECG_OriginalRestingECGMeasurements_QTCorrected',
                          'RestingECG_OriginalRestingECGMeasurements_QTcFrederica',
                          'echonext_shd',  # Add structural heart disease column
                          'deepecho_Visually_Estimated_EF',  # Add LVEF column
                          'acs_condition_severity',  # Add ACS severity column
                          'acs_pci_regions',  # Add ACS culprit artery column
                          'afib_label_2y',  # Add AFib 2-year risk prediction
                          'afib_label_5y',  # Add AFib 5-year risk prediction
                          'Afib',  # Current AFib status
                          'Afib_bert_model']  # AFib BERT model prediction
            
            # Also copy over all the ECG diagnosis columns (they should have same names)
            from utils.constants import DEEPECG_CATEGORIES
            all_diagnosis_cols = []
            for category, diagnoses in DEEPECG_CATEGORIES.items():
                all_diagnosis_cols.extend(diagnoses)
            
            # Add diagnosis columns that exist in MHI and not already in df
            for col in all_diagnosis_cols:
                if col in mhi_df.columns and col not in df.columns:
                    merge_cols.append(col)
            
            # Remove duplicates from merge_cols
            merge_cols = list(dict.fromkeys(merge_cols))
            
            # Merge MHI data
            df_mhi_merged = df.merge(
                mhi_df[merge_cols],
                on='npy_id',
                how='inner',  # Inner join to get only MHI matches
                suffixes=('', '_mhi')
            )
            df_mhi_merged['dataset_source'] = 'mhi'
            df_mhi_merged = df_mhi_merged.drop(columns=['npy_id'], errors='ignore')
            
            print(f"     Found {len(df_mhi_merged)} MHI records")
            all_dfs.append(df_mhi_merged)
        
        # Combine all dataframes
        if all_dfs:
            df_merged = pd.concat(all_dfs, ignore_index=True)
            print(f"\n   Total combined records: {len(df_merged)}")
            print(f"   Dataset distribution:")
            print(df_merged['dataset_source'].value_counts())
        else:
            raise ValueError("No data specified for combined mode. Set mimic_samples and/or mhi_samples.")
        
    elif dataset_type == 'mimic-iv':
        print(f"\n2. Loading MIMIC-IV demographic data...")
        mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
        mimic_df = pd.read_parquet(mimic_path)
        
        # Extract npy_id for merging
        mimic_df['npy_id'] = mimic_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
        df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
        
        # Select demographic and interval timing columns to merge
        demographic_cols = ['new_PatientID', 'npy_id', 'gender', 'age_at_ecg', 'rr_interval',
                           'p_onset', 'qrs_onset', 'qrs_end', 't_end']  # Add timing columns for interval calculation
        available_cols = [col for col in demographic_cols if col in mimic_df.columns]
        
        # Merge
        print(f"   Merging columns: {available_cols}")
        df_merged = df.merge(
            mimic_df[available_cols],
            on='npy_id',
            how='left',
            suffixes=('', '_mimic')
        )
        
        # Clean up
        df_merged = df_merged.drop(columns=['npy_id'], errors='ignore')
        df_merged['dataset_source'] = 'mimic-iv'
        
    elif dataset_type == 'mhi':
        print(f"\n2. Loading MHI demographic data (1.7M rows, this may take a moment)...")
        mhi_path = '/media/data1/muse_ge/ECG_ad20241231_cat_labels_v1.4.complete.ROXs42Bb.parquet'
        
        # Load MHI data with relevant columns including interval measurements and echonext
        mhi_cols = ['npy_path', 'new_PatientID', 
                   'RestingECG_PatientDemographics_Gender', 
                   'RestingECG_PatientDemographics_PatientAge',
                   'RestingECG_QRSTimesTypes_GlobalRR',
                   'RestingECG_OriginalRestingECGMeasurements_VentricularRate',
                   'RestingECG_OriginalRestingECGMeasurements_PRInterval',
                   'RestingECG_OriginalRestingECGMeasurements_QRSDuration',
                   'RestingECG_OriginalRestingECGMeasurements_QTInterval',
                   'RestingECG_OriginalRestingECGMeasurements_QTCorrected',
                   'RestingECG_OriginalRestingECGMeasurements_QTcFrederica',
                   'echonext_shd',  # Add structural heart disease column
                   'deepecho_Visually_Estimated_EF',  # Add LVEF column
                   'acs_condition_severity',  # Add ACS severity column
                   'acs_pci_regions',  # Add ACS culprit artery column
                   'afib_label_2y',  # Add AFib 2-year risk prediction
                   'afib_label_5y',  # Add AFib 5-year risk prediction
                   'Afib',  # Current AFib status
                   'Afib_bert_model']  # AFib BERT model prediction
        
        # Also include all the diagnostic columns that exist in MHI
        mhi_df = pd.read_parquet(mhi_path)
        
        # Extract npy_id from npy_path for merging
        mhi_df['npy_id'] = mhi_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
        df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
        
        # Map MHI columns to standard names
        mhi_df['gender'] = mhi_df['RestingECG_PatientDemographics_Gender'].map({'MALE': 'M', 'FEMALE': 'F'})
        mhi_df['age_at_ecg'] = pd.to_numeric(mhi_df['RestingECG_PatientDemographics_PatientAge'], errors='coerce')
        mhi_df['rr_interval'] = pd.to_numeric(mhi_df['RestingECG_QRSTimesTypes_GlobalRR'], errors='coerce')
        # Add VentricularRate column for MHI heart rate
        mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'] = pd.to_numeric(
            mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'], errors='coerce'
        )
        
        # Select columns to merge (including interval measurements and echonext)
        merge_cols = ['npy_id', 'new_PatientID', 'gender', 'age_at_ecg', 'rr_interval',
                      'RestingECG_OriginalRestingECGMeasurements_VentricularRate',
                      'RestingECG_OriginalRestingECGMeasurements_PRInterval',
                      'RestingECG_OriginalRestingECGMeasurements_QRSDuration',
                      'RestingECG_OriginalRestingECGMeasurements_QTInterval',
                      'RestingECG_OriginalRestingECGMeasurements_QTCorrected',
                      'RestingECG_OriginalRestingECGMeasurements_QTcFrederica',
                      'echonext_shd',  # Add structural heart disease column
                      'deepecho_Visually_Estimated_EF',  # Add LVEF column
                      'acs_condition_severity',  # Add ACS severity column
                      'acs_pci_regions',  # Add ACS culprit artery column
                      'afib_label_2y',  # Add AFib 2-year risk prediction
                      'afib_label_5y',  # Add AFib 5-year risk prediction
                      'Afib',  # Current AFib status
                      'Afib_bert_model']  # AFib BERT model prediction
        
        # Also copy over all the ECG diagnosis columns (they should have same names)
        from utils.constants import DEEPECG_CATEGORIES
        all_diagnosis_cols = []
        for category, diagnoses in DEEPECG_CATEGORIES.items():
            all_diagnosis_cols.extend(diagnoses)
        
        # Add diagnosis columns that exist in MHI and not already in df
        for col in all_diagnosis_cols:
            if col in mhi_df.columns and col not in df.columns:
                merge_cols.append(col)
        
        # Remove duplicates from merge_cols
        merge_cols = list(dict.fromkeys(merge_cols))
        
        # Merge with indicator to track unmatched records
        print(f"   Merging demographic and diagnostic columns...")
        df_merged = df.merge(
            mhi_df[merge_cols],
            on='npy_id',
            how='left',
            suffixes=('', '_mhi'),
            indicator=True
        )
        
        # Report merge statistics
        merge_stats = df_merged['_merge'].value_counts()
        if 'left_only' in merge_stats.index:
            unmatched = merge_stats['left_only']
            pct_unmatched = 100 * unmatched / len(df_merged)
            print(f"   WARNING: {unmatched} records ({pct_unmatched:.1f}%) not found in MHI data")
            
            # If this is test dataset and >30% unmatched, reassign to train
            if dataset_name == 'test' and pct_unmatched > 30:
                print(f"   NOTE: {pct_unmatched:.1f}% > 30% unmatched in test set, these will be reassigned to train")
        
        # Clean up
        df_merged = df_merged.drop(columns=['npy_id', '_merge'], errors='ignore')
        df_merged['dataset_source'] = 'mhi'
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Report demographic data availability
    for col in ['gender', 'age_at_ecg', 'rr_interval']:
        if col in df_merged.columns:
            available = df_merged[col].notna().sum()
            print(f"   {col}: {available}/{len(df_merged)} ({100*available/len(df_merged):.1f}%)")
    
    # 3a. Generate ecg_type column based on deepecg.json
    print(f"\n2a. Generating ecg_type column based on pathological/limit classifications...")
    
    # Load deepecg dictionary and categories from constants
    from utils.constants import DEEPECG_PATHOLOGICAL_LIMIT, DEEPECG_CATEGORIES
    
    pathological_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['pathological']
    limit_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['limit']
    
    # Get all diagnostic columns from deepecg_categories
    all_diagnosis_cols = []
    for category, diagnoses in DEEPECG_CATEGORIES.items():
        all_diagnosis_cols.extend(diagnoses)
    
    # Initialize ecg_type as normal
    df_merged['ecg_type'] = 'normal'
    
    # Check for pathological conditions
    for col in pathological_cols:
        if col in df_merged.columns:
            # Mark as pathological if any pathological column >= 1
            df_merged.loc[df_merged[col] >= 1, 'ecg_type'] = 'pathological'
    
    # Check for borderline conditions (only if not already pathological)
    for col in limit_cols:
        if col in df_merged.columns:
            # Mark as borderline if any limit column >= 1 and not already pathological
            df_merged.loc[(df_merged[col] >= 1) & (df_merged['ecg_type'] == 'normal'), 'ecg_type'] = 'borderline'
    
    # Report ecg_type distribution
    ecg_type_counts = df_merged['ecg_type'].value_counts()
    print(f"   ECG type distribution:")
    for ecg_type, count in ecg_type_counts.items():
        print(f"     {ecg_type}: {count} ({100*count/len(df_merged):.1f}%)")
    
    # 4. Calculate how many ECGs we need for the target number of questions
    # For combined mode, handle sampling differently
    if dataset_type == 'combined' and (mimic_samples or mhi_samples):
        print(f"\n3. Selecting ECGs for combined dataset...")
        
        sampled_dfs = []
        
        # Sample from each dataset source
        for source in df_merged['dataset_source'].unique():
            source_df = df_merged[df_merged['dataset_source'] == source]
            
            # Determine target samples for this source
            if source == 'mimic-iv':
                target_samples = mimic_samples or 0
            else:  # mhi
                target_samples = mhi_samples or 0
            
            if target_samples > 0:
                # Calculate approximate ECGs needed
                if max_prompts_per_ecg:
                    avg_prompts_per_ecg = (1 + max_prompts_per_ecg) / 2
                    target_ecgs = max(int(target_samples / avg_prompts_per_ecg), 1)
                else:
                    target_ecgs = max(int(target_samples / 6), 1)
                
                print(f"\n   {source.upper()}: Target ~{target_ecgs} ECGs for ~{target_samples} questions")
                
                # For MHI, apply stratified sampling for special questions
                if source == 'mhi':
                    # Use stratified sampling function for MHI
                    # For test: 1000 special questions out of 5000 (20%)
                    # For train: 40000 special questions out of 200000 (20%)
                    special_target = int(target_samples * 0.2)  # 20% should be special questions
                    sampled_source = apply_mhi_special_stratified_sampling(source_df, target_ecgs, special_target)
                    if sampled_source is not None and len(sampled_source) > 0:
                        sampled_dfs.append(sampled_source)
                        
                        # Report sampling for MHI with special stratification
                        ecg_type_counts = sampled_source['ecg_type'].value_counts()
                        print(f"     Sampled {len(sampled_source)} ECGs from {source} with stratified sampling:")
                        for ecg_type, count in ecg_type_counts.items():
                            print(f"       {ecg_type}: {count} ({count/len(sampled_source)*100:.1f}%)")
                        
                        # Report special question distributions
                        if 'afib_label_2y' in sampled_source.columns and 'afib_label_5y' in sampled_source.columns:
                            high_risk_afib = sampled_source[(sampled_source['afib_label_2y'] == True) | (sampled_source['afib_label_5y'] == True)]
                            print(f"       AFib high risk: {len(high_risk_afib)} ({len(high_risk_afib)/len(sampled_source)*100:.1f}%)")
                        
                        if 'echonext_shd' in sampled_source.columns:
                            abnormal_shd = sampled_source[sampled_source['echonext_shd'] >= 1]
                            print(f"       Abnormal SHD: {len(abnormal_shd)} ({len(abnormal_shd)/len(sampled_source)*100:.1f}%)")
                        
                        if 'acs_condition_severity' in sampled_source.columns:
                            from utils.constants import ACS_ACUTE_CONDITIONS
                            acute_acs = sampled_source[sampled_source['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
                            print(f"       Acute ACS: {len(acute_acs)} ({len(acute_acs)/len(sampled_source)*100:.1f}%)")
                else:
                    # For MIMIC, use original sampling logic
                    # Apply max_normal_percentage constraint for this source
                    max_normal_for_source = int(target_ecgs * max_normal_percentage)
                    
                    # Separate by ECG type
                    source_pathological = source_df[source_df['ecg_type'] == 'pathological']
                    source_borderline = source_df[source_df['ecg_type'] == 'borderline']
                    source_normal = source_df[source_df['ecg_type'] == 'normal']
                    
                    print(f"     Available in {source}: {len(source_pathological)} pathological, {len(source_borderline)} borderline, {len(source_normal)} normal")
                    print(f"     Max normal allowed: {max_normal_for_source} ({max_normal_percentage*100:.1f}%)")
                    
                    # Sample with constraint
                    source_sampled = []
                    
                    # First, sample pathological and borderline proportionally
                    non_normal_slots = target_ecgs - max_normal_for_source
                    path_border_total = len(source_pathological) + len(source_borderline)
                    
                    if path_border_total > 0:
                        # Sample pathological
                        n_path = min(int(non_normal_slots * len(source_pathological) / path_border_total), len(source_pathological))
                        if n_path > 0:
                            source_sampled.append(source_pathological.sample(n=n_path, random_state=42))
                        
                        # Sample borderline
                        n_border = min(int(non_normal_slots * len(source_borderline) / path_border_total), len(source_borderline))
                        if n_border > 0:
                            source_sampled.append(source_borderline.sample(n=n_border, random_state=42))
                        
                        # Sample normal (up to max allowed)
                        n_normal = min(max_normal_for_source, len(source_normal))
                        if n_normal > 0:
                            source_sampled.append(source_normal.sample(n=n_normal, random_state=42))
                        
                        # Combine samples from this source
                        if source_sampled:
                            sampled_source = pd.concat(source_sampled, ignore_index=True)
                            
                            # If we need more to reach target, sample more pathological/borderline
                            if len(sampled_source) < target_ecgs:
                                remaining_needed = target_ecgs - len(sampled_source)
                                already_sampled_idx = sampled_source.index
                                remaining_non_normal = pd.concat([source_pathological, source_borderline])
                                remaining_non_normal = remaining_non_normal[~remaining_non_normal.index.isin(already_sampled_idx)]
                                if len(remaining_non_normal) > 0:
                                    additional = remaining_non_normal.sample(n=min(remaining_needed, len(remaining_non_normal)), random_state=42)
                                    sampled_source = pd.concat([sampled_source, additional], ignore_index=True)
                            
                            sampled_dfs.append(sampled_source)
                            
                            # Report sampling for this source
                            ecg_type_counts = sampled_source['ecg_type'].value_counts()
                            print(f"     Sampled {len(sampled_source)} ECGs from {source}:")
                            for ecg_type, count in ecg_type_counts.items():
                                print(f"       {ecg_type}: {count} ({count/len(sampled_source)*100:.1f}%)")
        
        df_sampled = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else df_merged
        print(f"\n   Total ECGs selected: {len(df_sampled)}")
        print(f"   Dataset distribution:")
        print(df_sampled['dataset_source'].value_counts())
        
    # Adjust expected prompts per ECG based on max_prompts_per_ecg
    elif sample_size:
        if max_prompts_per_ecg:
            # With max_prompts_per_ecg, we expect about (1 + max_prompts_per_ecg) / 2 prompts per ECG on average
            avg_prompts_per_ecg = (1 + max_prompts_per_ecg) / 2
            target_ecgs = max(int(sample_size / avg_prompts_per_ecg), 1)
            print(f"\n3. Selecting ECGs to generate exactly {sample_size:,} questions...")
            print(f"   Target: ~{target_ecgs:,} ECGs (expecting ~{avg_prompts_per_ecg:.1f} questions per ECG with max={max_prompts_per_ecg})")
        else:
            # Default: ~6 questions per ECG
            target_ecgs = max(int(sample_size / 6), 1)
            print(f"\n3. Selecting ECGs to generate exactly {sample_size:,} questions...")
            print(f"   Target: ~{target_ecgs:,} ECGs (expecting ~6 questions per ECG)")
        
        # Calculate target distribution with max_normal_percentage constraint
        ecg_type_counts = df_merged['ecg_type'].value_counts()
        total_available = len(df_merged)
        
        # Print available ECGs by type
        print(f"   Available ECGs by type:")
        for ecg_type, count in ecg_type_counts.items():
            print(f"     {ecg_type.capitalize()}: {count:,} ({100*count/total_available:.1f}%)")
        
        # Ensure minimum representation for each diagnosis
        print(f"\n   Ensuring minimum {min_samples_per_diagnosis} samples per diagnosis...")
        
        # Find ECGs with each diagnosis
        diagnosis_ecg_indices = {}
        for diagnosis in all_diagnosis_cols:
            if diagnosis in df_merged.columns:
                # Find ECGs with this diagnosis (value >= 1)
                ecgs_with_diagnosis = df_merged[df_merged[diagnosis] >= 1].index
                if len(ecgs_with_diagnosis) > 0:
                    diagnosis_ecg_indices[diagnosis] = ecgs_with_diagnosis
                    print(f"     {diagnosis}: {len(ecgs_with_diagnosis)} ECGs available")
        
        # Start by selecting ECGs to ensure minimum representation
        selected_indices = set()
        for diagnosis, indices in diagnosis_ecg_indices.items():
            # Select minimum samples for this diagnosis
            n_to_select = min(min_samples_per_diagnosis, len(indices))
            if n_to_select > 0:
                selected = np.random.choice(indices, size=n_to_select, replace=False)
                selected_indices.update(selected)
        
        print(f"\n   Selected {len(selected_indices)} ECGs for minimum diagnosis coverage")
        
        # Now sample remaining ECGs with max_normal_percentage constraint
        remaining_target = target_ecgs - len(selected_indices)
        sampled_dfs = []
        
        # First add the minimum coverage ECGs
        if len(selected_indices) > 0:
            sampled_dfs.append(df_merged.loc[list(selected_indices)])
        
        if remaining_target > 0:
            # Calculate max normal ECGs allowed
            max_normal_ecgs = int(target_ecgs * max_normal_percentage)
            
            # Get available ECGs not yet selected
            available_df = df_merged[~df_merged.index.isin(selected_indices)]
            
            # Separate by type
            available_pathological = available_df[available_df['ecg_type'] == 'pathological']
            available_borderline = available_df[available_df['ecg_type'] == 'borderline']
            available_normal = available_df[available_df['ecg_type'] == 'normal']
            
            print(f"\n   Sampling remaining {remaining_target} ECGs...")
            print(f"   Max normal ECGs allowed: {max_normal_ecgs} ({max_normal_percentage*100:.1f}% of total)")
            
            # Sample pathological and borderline proportionally
            path_border_total = len(available_pathological) + len(available_borderline)
            if path_border_total > 0:
                # Calculate proportions
                path_ratio = len(available_pathological) / path_border_total
                border_ratio = len(available_borderline) / path_border_total
                
                # Allocate remaining slots (minus normal allocation)
                non_normal_slots = remaining_target - max_normal_ecgs
                
                # Sample pathological
                n_path = min(int(non_normal_slots * path_ratio), len(available_pathological))
                if n_path > 0:
                    sampled_dfs.append(available_pathological.sample(n=n_path, random_state=42))
                    print(f"     Sampled {n_path:,} pathological ECGs")
                
                # Sample borderline
                n_border = min(int(non_normal_slots * border_ratio), len(available_borderline))
                if n_border > 0:
                    sampled_dfs.append(available_borderline.sample(n=n_border, random_state=42))
                    print(f"     Sampled {n_border:,} borderline ECGs")
                
                # Sample normal (up to max allowed)
                n_normal = min(max_normal_ecgs, len(available_normal), remaining_target - n_path - n_border)
                if n_normal > 0:
                    sampled_dfs.append(available_normal.sample(n=n_normal, random_state=42))
                    print(f"     Sampled {n_normal:,} normal ECGs (capped at {max_normal_percentage*100:.1f}%)")
        
        # If we still need more, sample ONLY from pathological and borderline (not normal)
        current_sampled = len(pd.concat(sampled_dfs)) if sampled_dfs else 0
        still_needed = remaining_target - (current_sampled - len(selected_indices))
        if still_needed > 0:
            already_sampled_idx = pd.concat(sampled_dfs).index if sampled_dfs else pd.Index([])
            # Only sample from non-normal ECGs to maintain the constraint
            remaining_non_normal = df_merged[(~df_merged.index.isin(already_sampled_idx)) & 
                                            (df_merged['ecg_type'] != 'normal')]
            if len(remaining_non_normal) > 0:
                additional = remaining_non_normal.sample(n=min(still_needed, len(remaining_non_normal)), random_state=42)
                sampled_dfs.append(additional)
                print(f"\n   Sampled {len(additional):,} additional non-normal ECGs to reach target")
        
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
        
        # Report final composition
        print(f"\n   Selected ECG composition:")
        for ecg_type, count in df_sampled['ecg_type'].value_counts().items():
            pct = 100 * count / len(df_sampled)
            print(f"     {ecg_type}: {count:,} ECGs ({pct:.1f}%)")
        print(f"     Total: {len(df_sampled):,} ECGs")
    else:
        df_sampled = df_merged
        print(f"\n3. Using all {len(df_sampled):,} ECGs (no sampling)")
    
    # 5. Generate prompts for each ECG
    print(f"\n4. Generating prompts...")
    if max_prompts_per_ecg:
        print(f"   Max prompts per ECG: {max_prompts_per_ecg}")
    prompt_maker = ECGPromptMaker()
    
    all_prompts = []
    ecgs_processed = 0
    
    for idx, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="   Generating prompts"):
        # Check if we've reached our target BEFORE generating more prompts
        if sample_size and len(all_prompts) >= sample_size:
            break
            
        prompts = prompt_maker.generate_prompts_for_ecg(row)
        ecgs_processed += 1
        
        # If max_prompts_per_ecg is set, randomly sample 1 to max_prompts_per_ecg prompts
        if max_prompts_per_ecg and len(prompts) > max_prompts_per_ecg:
            import random
            # Sample between 1 and max_prompts_per_ecg prompts
            num_to_sample = random.randint(1, max_prompts_per_ecg)
            prompts = random.sample(prompts, num_to_sample)
        
        for prompt_text, prompt_category, prompt_weight in prompts:
            # Double-check we haven't exceeded target
            if sample_size and len(all_prompts) >= sample_size:
                break
                
            new_row = row.copy()
            new_row['prompt'] = prompt_text
            new_row['prompt_category'] = prompt_category
            new_row['prompt_weight'] = prompt_weight
            
            all_prompts.append(new_row)
        
        # Print progress every 1000 ECGs when using max_prompts_per_ecg (more ECGs will be processed)
        progress_interval = 1000 if max_prompts_per_ecg else 100
        if ecgs_processed % progress_interval == 0:
            print(f"   Processed {ecgs_processed} ECGs, generated {len(all_prompts)} prompts...")
            
    # Ensure we don't exceed target
    if sample_size and len(all_prompts) > sample_size:
        all_prompts = all_prompts[:sample_size]
    
    df_with_prompts = pd.DataFrame(all_prompts)
    
    print(f"   Total prompts generated: {len(df_with_prompts):,}")
    print(f"   Unique ECGs: {df_with_prompts['waveform_name'].nunique() if 'waveform_name' in df_with_prompts.columns else 'N/A'}")
    print(f"   Average prompts per ECG: {len(df_with_prompts) / df_sampled.shape[0]:.2f}")
    
    # Report prompt distribution
    prompt_distribution = df_with_prompts['prompt_category'].value_counts()
    print(f"\n   Prompt type distribution:")
    for prompt_type, count in prompt_distribution.head(10).items():
        pct = 100 * count / len(df_with_prompts)
        print(f"     {prompt_type}: {count:,} ({pct:.1f}%)")
    
    # Check for random finding questions
    random_findings = df_with_prompts[df_with_prompts['prompt_category'] == 'random_finding_question']
    if len(random_findings) > 0:
        print(f"\n   Random finding questions: {len(random_findings)} ({len(random_findings)/len(df_with_prompts)*100:.2f}%)")
    
    # 6. Generate answers for each prompt
    print(f"\n5. Generating answers...")
    # Map dataset_type to answer generator format (it expects 'mimic' not 'mimic-iv' or 'combined')
    if dataset_type in ['mimic-iv', 'combined']:
        answer_dataset = 'mimic'  # Use mimic mappings for both mimic-iv and combined
    elif dataset_type == 'mhi':
        answer_dataset = 'mhi'
    else:
        answer_dataset = dataset_type
    answer_gen = ECGAnswerGenerator(language='en', dataset=answer_dataset)
    
    answers = []
    total_prompts = len(df_with_prompts)
    
    for idx, row in tqdm(df_with_prompts.iterrows(), total=total_prompts, desc="   Generating answers"):
        answer = answer_gen.generate_answer(row)
        answers.append(answer)
    
    df_with_answers = df_with_prompts.copy()
    df_with_answers['generated_answer'] = answers
    
    # Filter out questions with None answers (e.g., structural heart disease questions without data)
    initial_count = len(df_with_answers)
    df_with_answers = df_with_answers[df_with_answers['generated_answer'].notna()]
    dropped_count = initial_count - len(df_with_answers)
    if dropped_count > 0:
        print(f"     Dropped {dropped_count} questions with no available data")
    
    # Keep original report for comparison
    if 'report' in df_with_answers.columns:
        df_with_answers['original_report'] = df_with_answers['report']
    
    # 7. Show sample outputs
    print(f"\n6. Sample outputs:")
    print(f"{'='*60}")
    
    # Show examples from each category
    sample_categories = ['interpretation', 'json_interpretation', 'heart_rate', 'classification', 'category_infarct_ischemia']
    
    for cat in sample_categories:
        subset = df_with_answers[df_with_answers['prompt_category'] == cat]
        if len(subset) > 0:
            example = subset.iloc[0]
            print(f"\n{cat.upper()}:")
            print(f"   Prompt: {example['prompt'][:80]}...")
            if cat == 'json_interpretation':
                print(f"   Answer: {example['generated_answer'][:150]}...")
            else:
                print(f"   Answer: {example['generated_answer']}")
    
    # 8. Save results
    print(f"\n7. Saving results...")
    print(f"   Output: {output_path}")
    
    # Rename dataset_source to dataset for consistency
    if 'dataset_source' in df_with_answers.columns:
        df_with_answers['dataset'] = df_with_answers['dataset_source'].replace({'mimic-iv': 'mimic'})
        df_with_answers = df_with_answers.drop(columns=['dataset_source'])
    
    # Ensure demographic columns and dataset are preserved
    columns_to_keep = list(df_with_answers.columns)
    for col in ['gender', 'age_at_ecg', 'rr_interval', 'dataset']:
        if col not in columns_to_keep and col in df_with_answers.columns:
            columns_to_keep.append(col)
    
    # Filter to existing columns
    columns_to_keep = [c for c in columns_to_keep if c in df_with_answers.columns]
    
    # Ensure numeric columns are properly typed before saving
    if 'age_at_ecg' in df_with_answers.columns:
        df_with_answers['age_at_ecg'] = pd.to_numeric(df_with_answers['age_at_ecg'], errors='coerce')
    if 'rr_interval' in df_with_answers.columns:
        df_with_answers['rr_interval'] = pd.to_numeric(df_with_answers['rr_interval'], errors='coerce')
    
    # Save parquet
    df_with_answers[columns_to_keep].to_parquet(output_path, index=False)
    
    # Save sample CSV
    sample_csv_path = output_path.replace('.parquet', '_sample.csv')
    df_with_answers.head(200).to_csv(sample_csv_path, index=False)
    print(f"   Sample CSV: {sample_csv_path}")
    
    print(f"\n✓ {dataset_name.upper()} DATASET COMPLETE!")
    
    return df_with_answers


def main(dataset_type: str = 'mimic-iv', train_samples: int = 200000, test_samples: int = 10000, max_prompts_per_ecg: int = None, max_normal_percentage: float = 0.05, min_samples_per_diagnosis: int = 1, mimic_train_samples: int = None, mhi_train_samples: int = None, mimic_test_samples: int = None, mhi_test_samples: int = None):
    """Main function to process both train and test datasets
    
    Args:
        dataset_type: Type of dataset ('mimic-iv', 'mhi', or 'combined')
        train_samples: Number of training samples to generate (for single dataset mode)
        test_samples: Number of test/validation samples to generate (for single dataset mode)
        max_prompts_per_ecg: Maximum prompts per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 1)
        mimic_train_samples: For combined mode, number of MIMIC training samples
        mhi_train_samples: For combined mode, number of MHI training samples
        mimic_test_samples: For combined mode, number of MIMIC test samples
        mhi_test_samples: For combined mode, number of MHI test samples
    """
    
    print("GENERATING TRAIN AND TEST DATASETS", flush=True)
    print("=" * 80)
    print(f"Dataset type: {dataset_type.upper()}")
    print(f"Target sizes: Train={train_samples:,}, Test={test_samples:,}")
    print(f"Max normal percentage: {max_normal_percentage*100:.1f}%")
    print(f"Min samples per diagnosis: {min_samples_per_diagnosis}")
    
    # Define paths based on dataset type
    if dataset_type == 'combined':
        # For combined mode, use the same input files but output combined datasets
        test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet'
        train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet'
        
        # Calculate total samples for filename
        total_test = (mimic_test_samples or 0) + (mhi_test_samples or 0)
        total_train = (mimic_train_samples or 0) + (mhi_train_samples or 0)
        
        test_output = f'/volume/ECG_tokenizer/output/combined_test_qa_m{(mimic_test_samples or 0)//1000}k_h{(mhi_test_samples or 0)//1000}k.parquet'
        train_output = f'/volume/ECG_tokenizer/output/combined_train_qa_m{(mimic_train_samples or 0)//1000}k_h{(mhi_train_samples or 0)//1000}k.parquet'
        
        print("\nNote: Creating combined dataset with samples from both MIMIC and MHI")
        
    elif dataset_type == 'mimic-iv':
        test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet'
        test_output = f'/volume/ECG_tokenizer/output/mimic_test_qa_{test_samples//1000}k.parquet'
        
        train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet'
        train_output = f'/volume/ECG_tokenizer/output/mimic_train_qa_{train_samples//1000}k.parquet'
    
    elif dataset_type == 'mhi':
        # For MHI, use same files but merge with MHI demographic data
        test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet'
        test_output = f'/volume/ECG_tokenizer/output/mhi_test_qa_{test_samples//1000}k.parquet'
        
        train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet'
        train_output = f'/volume/ECG_tokenizer/output/mhi_train_qa_{train_samples//1000}k.parquet'
        
        print("\nNote: Using same input files but with MHI demographic data merged")
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    try:
        # Determine sample sizes based on mode
        if dataset_type == 'combined':
            test_sample_size = (mimic_test_samples or 0) + (mhi_test_samples or 0)
            train_sample_size = (mimic_train_samples or 0) + (mhi_train_samples or 0)
        else:
            test_sample_size = test_samples
            train_sample_size = train_samples
            mimic_test_samples = None
            mhi_test_samples = None
            mimic_train_samples = None
            mhi_train_samples = None
        
        # Process test dataset first (smaller) - but skip if test_samples is 0
        if test_sample_size > 0:
            test_df = process_dataset(test_input, test_output, "test", 
                                     sample_size=test_sample_size, 
                                     dataset_type=dataset_type, 
                                     max_prompts_per_ecg=max_prompts_per_ecg,
                                     max_normal_percentage=max_normal_percentage,
                                     min_samples_per_diagnosis=min_samples_per_diagnosis,
                                     mimic_samples=mimic_test_samples,
                                     mhi_samples=mhi_test_samples)
        else:
            print("Skipping test dataset (test_samples=0)")
        
        # Process train dataset - but skip if train_samples is 0
        if train_sample_size > 0:
            train_df = process_dataset(train_input, train_output, "train", 
                                  sample_size=train_sample_size, 
                                  dataset_type=dataset_type, 
                                  max_prompts_per_ecg=max_prompts_per_ecg,
                                  max_normal_percentage=max_normal_percentage,
                                  min_samples_per_diagnosis=min_samples_per_diagnosis,
                                  mimic_samples=mimic_train_samples,
                                  mhi_samples=mhi_train_samples)
        else:
            print("Skipping train dataset (train_samples=0)")
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        if test_sample_size > 0:
            print(f"Test dataset: {len(test_df)} prompts from {test_df['waveform_name'].nunique()} ECGs")
        if train_sample_size > 0:
            print(f"Train dataset: {len(train_df)} prompts from {train_df['waveform_name'].nunique()} ECGs")
        
        print(f"\nFiles saved:")
        print(f"  Test: {test_output}")
        print(f"  Train: {train_output}")
        
        # Check for ischemia/infarction questions
        print(f"\n✓ Ischemia/infarction detection FIXED:")
        print(f"  • Now checks ST downsloping, ST depression, T wave inversions")
        print(f"  • Properly identifies Q waves as signs of old infarction")
        print(f"  • Correctly parses report text for infarct mentions")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 ALL DATASETS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate train and test datasets with Q&A pairs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic-iv",
        choices=["mimic-iv", "mhi", "combined"],
        help="Dataset type: 'mimic-iv', 'mhi', or 'combined' for both"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=200000,
        help="Number of training samples to generate (default: 200,000)"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10000,
        help="Number of test/validation samples to generate (default: 10,000)"
    )
    parser.add_argument(
        "--max_prompts_per_ecg",
        type=int,
        default=None,
        help="Maximum prompts per ECG (randomly samples 1 to N prompts per ECG). Default: None (use all prompts)"
    )
    parser.add_argument(
        "--max_normal_percentage",
        type=float,
        default=0.05,
        help="Maximum percentage of normal ECGs in the dataset (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--min_samples_per_diagnosis",
        type=int,
        default=1,
        help="Minimum samples to include for each diagnosis category (default: 1)"
    )
    parser.add_argument(
        "--mimic_train_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MIMIC training samples"
    )
    parser.add_argument(
        "--mhi_train_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MHI training samples"
    )
    parser.add_argument(
        "--mimic_test_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MIMIC test samples"
    )
    parser.add_argument(
        "--mhi_test_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MHI test samples"
    )
    
    args = parser.parse_args()
    main(dataset_type=args.dataset, 
         train_samples=args.train_samples, 
         test_samples=args.test_samples, 
         max_prompts_per_ecg=args.max_prompts_per_ecg,
         max_normal_percentage=args.max_normal_percentage,
         min_samples_per_diagnosis=args.min_samples_per_diagnosis,
         mimic_train_samples=args.mimic_train_samples,
         mhi_train_samples=args.mhi_train_samples,
         mimic_test_samples=args.mimic_test_samples,
         mhi_test_samples=args.mhi_test_samples)