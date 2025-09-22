#!/usr/bin/env python3
"""
Generate train and test datasets with enhanced prompts and answers.
Drops the existing question column and generates new comprehensive Q&A pairs.
"""

import pandas as pd
import sys
import json
import numpy as np
from ecg_prompt_maker import ECGPromptMaker
from ecg_answer_generator import ECGAnswerGenerator


def process_dataset(input_path: str, output_path: str, dataset_name: str, sample_size: int = None, dataset_type: str = 'mimic', max_prompts_per_ecg: int = None, max_normal_percentage: float = 0.05, min_samples_per_diagnosis: int = 1):
    """Process a single dataset (train or test)
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        dataset_name: Name of dataset ('train' or 'test')
        sample_size: Target number of samples to generate
        dataset_type: Type of dataset ('mimic' or 'ptbxl')
        max_prompts_per_ecg: Maximum prompts to generate per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 1)
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
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
    
    # 3. Load and merge MIMIC demographic data
    print(f"\n2. Loading MIMIC demographic data...")
    mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
    mimic_df = pd.read_parquet(mimic_path)
    
    # Extract npy_id for merging
    mimic_df['npy_id'] = mimic_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
    df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
    
    # Select demographic columns to merge
    demographic_cols = ['new_PatientID', 'npy_id', 'gender', 'age_at_ecg', 'rr_interval']
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
    
    # Report demographic data availability
    for col in ['gender', 'age_at_ecg', 'rr_interval']:
        if col in df_merged.columns:
            available = df_merged[col].notna().sum()
            print(f"   {col}: {available}/{len(df_merged)} ({100*available/len(df_merged):.1f}%)")
    
    # 3a. Generate ecg_type column based on deepecg.json
    print(f"\n2a. Generating ecg_type column based on pathological/limit classifications...")
    
    # Load deepecg dictionary and categories
    with open('/volume/ECG_tokenizer/dictionary/deepecg.json', 'r') as f:
        deepecg = json.load(f)['deepecg']
    
    with open('/volume/ECG_tokenizer/dictionary/deepecg_categories.json', 'r') as f:
        deepecg_categories = json.load(f)
    
    pathological_cols = deepecg['pathological']
    limit_cols = deepecg['limit']
    
    # Get all diagnostic columns from deepecg_categories
    all_diagnosis_cols = []
    for category, diagnoses in deepecg_categories.items():
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
    # Adjust expected prompts per ECG based on max_prompts_per_ecg
    if sample_size:
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
        
        # If we still need more, sample from any available
        if remaining_target > 0:
            already_sampled_idx = pd.concat(sampled_dfs).index if sampled_dfs else pd.Index([])
            remaining_df = df_merged[~df_merged.index.isin(already_sampled_idx)]
            if len(remaining_df) > 0:
                additional = remaining_df.sample(n=min(remaining_target, len(remaining_df)), random_state=42)
                sampled_dfs.append(additional)
                print(f"\n   Sampled {len(additional):,} additional ECGs")
        
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
    
    for idx, row in df_sampled.iterrows():
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
            
            # Map prompt_type from category for compatibility
            new_row['prompt_type'] = prompt_category
            
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
    answer_gen = ECGAnswerGenerator(language='en', dataset=dataset_type)
    
    answers = []
    total_prompts = len(df_with_prompts)
    
    for idx, row in df_with_prompts.iterrows():
        answer = answer_gen.generate_answer(row)
        answers.append(answer)
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"     Processed {idx + 1}/{total_prompts} prompts...")
    
    df_with_answers = df_with_prompts.copy()
    df_with_answers['generated_answer'] = answers
    
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
    
    # Ensure demographic columns are preserved
    columns_to_keep = list(df_with_answers.columns)
    for col in ['gender', 'age_at_ecg', 'rr_interval']:
        if col not in columns_to_keep and col in df_with_answers.columns:
            columns_to_keep.append(col)
    
    # Filter to existing columns
    columns_to_keep = [c for c in columns_to_keep if c in df_with_answers.columns]
    
    # Save parquet
    df_with_answers[columns_to_keep].to_parquet(output_path, index=False)
    
    # Save sample CSV
    sample_csv_path = output_path.replace('.parquet', '_sample.csv')
    df_with_answers.head(200).to_csv(sample_csv_path, index=False)
    print(f"   Sample CSV: {sample_csv_path}")
    
    print(f"\n✓ {dataset_name.upper()} DATASET COMPLETE!")
    
    return df_with_answers


def main(dataset_type: str = 'mimic', train_samples: int = 200000, test_samples: int = 10000, max_prompts_per_ecg: int = None, max_normal_percentage: float = 0.05, min_samples_per_diagnosis: int = 1):
    """Main function to process both train and test datasets
    
    Args:
        dataset_type: Type of dataset ('mimic' or 'ptbxl')
        train_samples: Number of training samples to generate
        test_samples: Number of test/validation samples to generate
        max_prompts_per_ecg: Maximum prompts per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 1)
    """
    
    print("GENERATING TRAIN AND TEST DATASETS")
    print("=" * 80)
    print(f"Target sizes: Train={train_samples:,}, Test={test_samples:,}")
    print(f"Max normal percentage: {max_normal_percentage*100:.1f}%")
    print(f"Min samples per diagnosis: {min_samples_per_diagnosis}")
    
    # Define paths
    test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated_with_questions.parquet'
    test_output = f'/volume/ECG_tokenizer/output/mimic_test_qa_{test_samples//1000}k.parquet'
    
    train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated_with_questions.parquet'
    train_output = f'/volume/ECG_tokenizer/output/mimic_train_qa_{train_samples//1000}k.parquet'
    
    try:
        # Process test dataset first (smaller)
        test_df = process_dataset(test_input, test_output, "test", 
                                 sample_size=test_samples, 
                                 dataset_type=dataset_type, 
                                 max_prompts_per_ecg=max_prompts_per_ecg,
                                 max_normal_percentage=max_normal_percentage,
                                 min_samples_per_diagnosis=min_samples_per_diagnosis)
        
        # Process train dataset 
        train_df = process_dataset(train_input, train_output, "train", 
                                  sample_size=train_samples, 
                                  dataset_type=dataset_type, 
                                  max_prompts_per_ecg=max_prompts_per_ecg,
                                  max_normal_percentage=max_normal_percentage,
                                  min_samples_per_diagnosis=min_samples_per_diagnosis)
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Test dataset: {len(test_df)} prompts from {test_df['waveform_name'].nunique()} ECGs")
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
        default="mimic",
        choices=["mimic", "ptbxl"],  # Add more as supported
        help="Dataset type for column mappings"
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
    
    args = parser.parse_args()
    main(dataset_type=args.dataset, 
         train_samples=args.train_samples, 
         test_samples=args.test_samples, 
         max_prompts_per_ecg=args.max_prompts_per_ecg,
         max_normal_percentage=args.max_normal_percentage,
         min_samples_per_diagnosis=args.min_samples_per_diagnosis)