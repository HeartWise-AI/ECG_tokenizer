#!/usr/bin/env python3
"""
Generate train and test datasets with enhanced prompts and answers.
Drops the existing question column and generates new comprehensive Q&A pairs.
"""

import pandas as pd
import sys
import json
from ecg_prompt_maker import ECGPromptMaker
from ecg_answer_generator import ECGAnswerGenerator


def process_dataset(input_path: str, output_path: str, dataset_name: str, sample_size: int = None, dataset_type: str = 'mimic'):
    """Process a single dataset (train or test)"""
    
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
    demographic_cols = ['npy_id', 'gender', 'age_at_ecg', 'rr_interval']
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
    
    # Load deepecg dictionary
    with open('/volume/ECG_tokenizer/dictionary/deepecg.json', 'r') as f:
        deepecg = json.load(f)['deepecg']
    
    pathological_cols = deepecg['pathological']
    limit_cols = deepecg['limit']
    
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
    # Average ~6 questions per ECG, so we need approximately sample_size/6 ECGs
    if sample_size:
        target_ecgs = max(int(sample_size / 6), 1)  # Assuming ~6 questions per ECG
        print(f"\n3. Selecting ECGs to generate exactly {sample_size:,} questions...")
        print(f"   Target: ~{target_ecgs:,} ECGs (expecting ~6 questions per ECG)")
        
        # Separate by ecg_type
        df_normal = df_merged[df_merged['ecg_type'] == 'normal'].copy()
        df_borderline = df_merged[df_merged['ecg_type'] == 'borderline'].copy()
        df_pathological = df_merged[df_merged['ecg_type'] == 'pathological'].copy()
        
        print(f"   Available ECGs by type:")
        print(f"     Normal: {len(df_normal):,}")
        print(f"     Borderline: {len(df_borderline):,}")
        print(f"     Pathological: {len(df_pathological):,}")
        
        # Calculate ECG targets based on 3% normal distribution
        target_normal = int(target_ecgs * 0.03)  # 3% of ECGs should be normal
        remaining = target_ecgs - target_normal
        
        # Prioritize pathological over borderline (70% pathological, 30% borderline of non-normal)
        target_pathological = min(len(df_pathological), int(remaining * 0.7))
        target_borderline = min(len(df_borderline), remaining - target_pathological)
        
        # Adjust if not enough pathological/borderline
        if target_pathological + target_borderline < remaining:
            target_pathological = len(df_pathological)
            target_borderline = len(df_borderline)
            target_normal = min(len(df_normal), sample_size - target_pathological - target_borderline)
        
        # Sample from each category
        samples = []
        
        if target_normal > 0 and len(df_normal) > 0:
            normal_sample = df_normal.sample(n=min(target_normal, len(df_normal)), random_state=42)
            samples.append(normal_sample)
            print(f"\n   Sampled {len(normal_sample):,} normal ECGs")
        
        if target_borderline > 0 and len(df_borderline) > 0:
            borderline_sample = df_borderline.sample(n=min(target_borderline, len(df_borderline)), random_state=42)
            samples.append(borderline_sample)
            print(f"   Sampled {len(borderline_sample):,} borderline ECGs")
        
        if target_pathological > 0 and len(df_pathological) > 0:
            pathological_sample = df_pathological.sample(n=min(target_pathological, len(df_pathological)), random_state=42)
            samples.append(pathological_sample)
            print(f"   Sampled {len(pathological_sample):,} pathological ECGs")
        
        # Combine and shuffle
        df_sample = pd.concat(samples, ignore_index=True)
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"\n   Selected ECG composition:")
        final_counts = df_sample['ecg_type'].value_counts()
        for ecg_type, count in final_counts.items():
            print(f"     {ecg_type}: {count:,} ECGs ({100*count/len(df_sample):.1f}%)")
        print(f"     Total: {len(df_sample):,} ECGs")
    else:
        print(f"\n   Processing full dataset ({len(df_merged):,} records)...")
        df_sample = df_merged.copy()
    
    # 5. Generate prompts
    print(f"\n4. Generating prompts...")
    prompt_maker = ECGPromptMaker(dataset=dataset_type)
    df_with_prompts = prompt_maker.process_dataframe(df_sample, max_prompts_per_ecg=6)
    
    # If we have a target sample size, trim to exactly that many questions
    if sample_size and len(df_with_prompts) > sample_size:
        print(f"   Trimming from {len(df_with_prompts):,} to exactly {sample_size:,} questions...")
        df_with_prompts = df_with_prompts.sample(n=sample_size, random_state=42)
        df_with_prompts = df_with_prompts.reset_index(drop=True)
    
    # Get statistics
    stats = prompt_maker.get_prompt_statistics(df_with_prompts)
    print(f"   Total prompts generated: {stats['total_prompts']}")
    print(f"   Unique ECGs: {stats['unique_ecgs']}")
    print(f"   Average prompts per ECG: {stats['prompts_per_ecg']:.2f}")
    
    # Show prompt type distribution
    print(f"\n   Prompt type distribution:")
    for ptype, count in stats['prompt_type_distribution'].items():
        percentage = (count / stats['total_prompts']) * 100
        print(f"     {ptype}: {count} ({percentage:.1f}%)")
    
    # 6. Generate answers
    print(f"\n5. Generating answers...")
    answer_gen = ECGAnswerGenerator(dataset=dataset_type)
    
    # Calculate heart rate if not already present
    if 'heart_rate' not in df_with_prompts.columns and 'rr_interval' in df_with_prompts.columns:
        df_with_prompts['heart_rate'] = df_with_prompts['rr_interval'].apply(
            lambda x: round(60000.0 / x, 1) if pd.notna(x) and x > 0 else None
        )
    
    # Generate answers
    answers = []
    total_prompts = len(df_with_prompts)
    
    for idx, row in df_with_prompts.iterrows():
        answer = answer_gen.generate_answer(row)
        answers.append(answer)
        
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
    sample_categories = ['interpretation', 'json_interpretation', 'heart_rate', 'classification']
    
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


def main(dataset_type: str = 'mimic', train_samples: int = 200000, test_samples: int = 10000):
    """Main function to process both train and test datasets
    
    Args:
        dataset_type: Type of dataset ('mimic' or 'ptbxl')
        train_samples: Number of training samples to generate
        test_samples: Number of test/validation samples to generate
    """
    
    print("GENERATING TRAIN AND TEST DATASETS")
    print("=" * 80)
    print(f"Target sizes: Train={train_samples:,}, Test={test_samples:,}")
    
    # Define paths
    test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated_with_questions.parquet'
    test_output = '/volume/ECG_tokenizer/output/mimic_test_complete_qa_dataset.parquet'
    
    train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated_with_questions.parquet'
    train_output = '/volume/ECG_tokenizer/output/mimic_train_complete_qa_dataset.parquet'
    
    try:
        # Process test dataset first (smaller)
        test_df = process_dataset(test_input, test_output, "test", sample_size=test_samples, dataset_type=dataset_type)
        
        # Process train dataset 
        train_df = process_dataset(train_input, train_output, "train", sample_size=train_samples, dataset_type=dataset_type)
        
        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Test dataset: {len(test_df)} prompts from {test_df['waveform_name'].nunique()} ECGs")
        print(f"Train dataset: {len(train_df)} prompts from {train_df['waveform_name'].nunique()} ECGs")
        
        print(f"\nFiles saved:")
        print(f"  Test: {test_output}")
        print(f"  Train: {train_output}")
        
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
    
    args = parser.parse_args()
    main(dataset_type=args.dataset, train_samples=args.train_samples, test_samples=args.test_samples)