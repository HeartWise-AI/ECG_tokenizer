#!/usr/bin/env python3
"""
Generate a Question-Answering dataset from ECG data for Hugging Face.

This script creates a QA dataset where:
- Question: "What is the diagnosis for this ECG?"
- Answer: The diagnosis from the report column
- Additional metadata includes the signal path and detected conditions
"""

import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value
from typing import Optional, Dict, List, Tuple
import argparse
from pathlib import Path
import json

# Import constants for category mappings
from constants import DEEPECG_CATEGORIES, ECG_PATTERNS_TRANSLATION


def load_parquet_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test parquet files.
    
    Args:
        train_path: Path to training parquet file
        test_path: Path to test parquet file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    return train_df, test_df


def filter_data(df: pd.DataFrame, filter_poor_quality: bool = True) -> pd.DataFrame:
    """
    Filter the dataset based on quality and other criteria.
    
    Args:
        df: Input dataframe
        filter_poor_quality: Whether to filter out poor quality ECGs
        
    Returns:
        Filtered dataframe
    """
    initial_size = len(df)
    
    if filter_poor_quality:
        # Filter out ECGs with poor quality warnings
        df = df[~df['report'].str.contains('CAUTION.*Poor ECG quality', case=False, na=False, regex=True)]
        print(f"Filtered out poor quality ECGs: {initial_size} -> {len(df)}")
    
    # Filter out records with missing essential data
    df = df.dropna(subset=['waveform_path_psa', 'report'])
    print(f"After removing missing data: {len(df)}")
    
    # Filter out records where the signal file doesn't exist
    existing_files = df['waveform_path_psa'].apply(os.path.exists)
    df = df[existing_files]
    print(f"After checking file existence: {len(df)}")
    
    return df.reset_index(drop=True)


def get_detected_conditions(row: pd.Series) -> List[str]:
    """
    Get list of detected conditions from the binary columns.
    
    Args:
        row: DataFrame row with binary condition columns
        
    Returns:
        List of detected condition names
    """
    # Get all the binary condition columns (excluding metadata columns)
    condition_columns = [col for col in row.index if col in ECG_PATTERNS_TRANSLATION.keys()]
    
    detected = []
    for col in condition_columns:
        if row[col] == 1:
            detected.append(col)
    
    return detected


def map_to_categories(detected_conditions: List[str]) -> Dict[str, List[str]]:
    """
    Map detected conditions to the 6 major categories.
    
    Args:
        detected_conditions: List of detected condition names
        
    Returns:
        Dictionary mapping category names to detected conditions in that category
    """
    category_mapping = {}
    
    for category, conditions in DEEPECG_CATEGORIES.items():
        category_conditions = [cond for cond in detected_conditions if cond in conditions]
        if category_conditions:
            category_mapping[category] = category_conditions
    
    return category_mapping


def create_qa_dataset(df: pd.DataFrame, split_name: str) -> Dataset:
    """
    Create a QA dataset from the processed dataframe.
    
    Args:
        df: Processed dataframe
        split_name: Name of the split ('train' or 'test')
        
    Returns:
        Hugging Face Dataset
    """
    print(f"Creating {split_name} QA dataset...")
    
    # Define the features for the dataset
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "signal_path": Value("string"),
        "detected_conditions": [Value("string")],
        "condition_categories": Value("string"),  # JSON string of category mapping
        "original_report": Value("string"),
    })
    
    records = []
    
    for idx, row in df.iterrows():
        try:
            # Get detected conditions
            detected_conditions = get_detected_conditions(row)
            
            # Map to categories
            category_mapping = map_to_categories(detected_conditions)
            
            # Create the QA record
            record = {
                "question": "What is the diagnosis for this ECG?",
                "answer": row['report'].strip(),
                "signal_path": row['waveform_path_psa'],
                "detected_conditions": detected_conditions,
                "condition_categories": str(category_mapping),  # Convert to string for storage
                "original_report": row['report'].strip(),
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            continue
    
    print(f"Created {len(records)} QA records for {split_name}")
    
    # Create and return the dataset
    dataset = Dataset.from_list(records, features=features)
    return dataset


def create_sample_dataset(df: pd.DataFrame, sample_size: int, split_name: str) -> Dataset:
    """
    Create a smaller sample dataset for testing.
    
    Args:
        df: Input dataframe
        sample_size: Number of samples to include
        split_name: Name of the split
        
    Returns:
        Sampled Dataset
    """
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    return create_qa_dataset(sampled_df, split_name)


def generate_qa_dataset(
    train_path: str,
    test_path: str,
    output_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    filter_poor_quality: bool = True,
    push_to_hub: bool = False,
    hub_dataset_name: Optional[str] = None,
    save_format: str = "hf"  # "hf", "json", "csv"
) -> DatasetDict:
    """
    Generate the complete QA dataset.
    
    Args:
        train_path: Path to training parquet file
        test_path: Path to test parquet file
        output_path: Path to save the dataset (optional)
        sample_size: Number of samples per split (optional, for testing)
        filter_poor_quality: Whether to filter poor quality ECGs
        push_to_hub: Whether to push dataset to Hugging Face Hub
        hub_dataset_name: Name for the dataset on Hugging Face Hub
        save_format: Format to save dataset ("hf", "json", "csv")
        
    Returns:
        DatasetDict with train and test splits
    """
    print("=== ECG QA Dataset Generation ===")
    
    # Load data
    train_df, test_df = load_parquet_data(train_path, test_path)
    
    # Filter data
    print("\nFiltering training data...")
    train_df = filter_data(train_df, filter_poor_quality)
    
    print("\nFiltering test data...")
    test_df = filter_data(test_df, filter_poor_quality)
    
    # Create datasets
    if sample_size:
        print(f"\nCreating sample datasets with {sample_size} samples each...")
        train_dataset = create_sample_dataset(train_df, sample_size, "train")
        test_dataset = create_sample_dataset(test_df, sample_size, "test")
    else:
        print("\nCreating full datasets...")
        train_dataset = create_qa_dataset(train_df, "train")
        test_dataset = create_qa_dataset(test_df, "test")
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    
    # Show sample records
    print("\n=== Sample Records ===")
    for i in range(min(3, len(dataset_dict['train']))):
        print(f"\nSample {i+1}:")
        sample = dataset_dict['train'][i]
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer'][:200]}...")  # Truncate long answers
        print(f"Detected conditions: {sample['detected_conditions'][:5]}...")  # Show first 5
        print(f"Signal path: {sample['signal_path']}")
    
    # Save if requested
    if push_to_hub and hub_dataset_name:
        print(f"\nPushing dataset to Hugging Face Hub: {hub_dataset_name} (private repository)")
        try:
            dataset_dict.push_to_hub(hub_dataset_name, private=True)
            print("Dataset successfully pushed to Hugging Face Hub as a private repository!")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("Make sure you're logged in with `huggingface-cli login`")
    
    if output_path:
        if save_format == "hf":
            print(f"\nSaving dataset to Hugging Face format: {output_path}")
            dataset_dict.save_to_disk(output_path)
            print("Dataset saved in Hugging Face format!")
        
        elif save_format == "json":
            print(f"\nSaving dataset to JSON format: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            # Save train split
            train_data = [dataset_dict['train'][i] for i in range(len(dataset_dict['train']))]
            with open(os.path.join(output_path, 'train.json'), 'w') as f:
                json.dump(train_data, f, indent=2)
            
            # Save test split
            test_data = [dataset_dict['test'][i] for i in range(len(dataset_dict['test']))]
            with open(os.path.join(output_path, 'test.json'), 'w') as f:
                json.dump(test_data, f, indent=2)
                
            print("Dataset saved in JSON format!")
        
        elif save_format == "csv":
            print(f"\nSaving dataset to CSV format: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            # Convert to pandas and save
            train_df = dataset_dict['train'].to_pandas()
            test_df = dataset_dict['test'].to_pandas()
            
            train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
            test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
            
            print("Dataset saved in CSV format!")
    
    return dataset_dict


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Generate ECG QA Dataset")
    parser.add_argument(
        "--train_path",
        type=str,
        default="/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet",
        help="Path to training parquet file"
    )
    parser.add_argument(
        "--test_path", 
        type=str,
        default="/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet",
        help="Path to test parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./ecg_qa_dataset",
        help="Output path to save the dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per split (for testing)"
    )
    parser.add_argument(
        "--no_filter_quality",
        action="store_true",
        help="Don't filter out poor quality ECGs"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_dataset_name",
        type=str,
        default=None,
        help="Name for the dataset on Hugging Face Hub (e.g., 'username/ecg-qa-dataset')"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["hf", "json", "csv"],
        default="hf",
        help="Format to save the dataset: 'hf' (Hugging Face), 'json', or 'csv'"
    )
    
    args = parser.parse_args()
    
    # Generate the dataset
    dataset = generate_qa_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        filter_poor_quality=not args.no_filter_quality,
        push_to_hub=args.push_to_hub,
        hub_dataset_name=args.hub_dataset_name,
        save_format=args.save_format
    )
    
    print("\n=== Generation Complete ===")
    return dataset


if __name__ == "__main__":
    main()
