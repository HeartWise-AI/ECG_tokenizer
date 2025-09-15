#!/usr/bin/env python3
"""
Generate train and test datasets with enhanced prompts and answers.
Drops the existing question column and generates new comprehensive Q&A pairs.
"""

import pandas as pd
import sys
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
    
    # 4. Take sample if specified
    if sample_size and len(df_merged) > sample_size:
        print(f"\n   Taking sample of {sample_size} records for testing...")
        df_sample = df_merged.head(sample_size).copy()
    else:
        print(f"\n   Processing full dataset ({len(df_merged)} records)...")
        df_sample = df_merged.copy()
    
    # 5. Generate prompts
    print(f"\n3. Generating prompts...")
    prompt_maker = ECGPromptMaker(dataset=dataset_type)
    df_with_prompts = prompt_maker.process_dataframe(df_sample, max_prompts_per_ecg=6)
    
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
    print(f"\n4. Generating answers...")
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
    print(f"\n5. Sample outputs:")
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
    print(f"\n6. Saving results...")
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


def main(dataset_type: str = 'mimic'):
    """Main function to process both train and test datasets"""
    
    print("GENERATING TRAIN AND TEST DATASETS")
    print("=" * 80)
    
    # Define paths
    test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated_with_questions.parquet'
    test_output = '/volume/ECG_tokenizer/output/mimic_test_complete_qa_dataset.parquet'
    
    train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated_with_questions.parquet'
    train_output = '/volume/ECG_tokenizer/output/mimic_train_complete_qa_dataset.parquet'
    
    try:
        # Process test dataset first (smaller)
        test_df = process_dataset(test_input, test_output, "test", sample_size=1000, dataset_type=dataset_type)  # Sample for testing
        
        # Process train dataset 
        train_df = process_dataset(train_input, train_output, "train", sample_size=5000, dataset_type=dataset_type)  # Sample for testing
        
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
    
    args = parser.parse_args()
    main(dataset_type=args.dataset)