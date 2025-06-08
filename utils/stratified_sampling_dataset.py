import pandas as pd
from data_splitter import DataSplitter

from constants import ECG_PATTERNS


if __name__ == "__main__":
        
    df = pd.read_parquet('/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet')

    columns_names = [
        'report', 
        'dataset',
        'waveform_name', 
        'waveform_path_psa', 
        'waveform_path_original', 
    ]
    feature_cols = columns_names.copy()
    columns_names.extend(ECG_PATTERNS)
    new_df = df[columns_names].copy()
    new_df[ECG_PATTERNS] = new_df[ECG_PATTERNS].astype(int)

    splitter = DataSplitter(
        df=new_df, 
        feature_col=feature_cols, 
        label_cols=ECG_PATTERNS,
        stratify_method='multilabel'
    )
    split_data = splitter.split(train_size=0.6, val_size=0.2, test_size=0.2)
    distribution_info = splitter.check_distribution(split_data)

    # Print results
    print(f"Train samples: {len(split_data['X_train'])} ({len(split_data['X_train'])/len(new_df):.2%})")
    print(f"Validation samples: {len(split_data['X_val'])} ({len(split_data['X_val'])/len(new_df):.2%})")
    print(f"Test samples: {len(split_data['X_test'])} ({len(split_data['X_test'])/len(new_df):.2%})")
    print("\nLabel distribution:")
    print(distribution_info['distributions'])
    print("\nDistribution differences:")
    print(distribution_info['differences'])
    
    # Save split DataFrames
    # split_data['train_df'].to_parquet("parquets/mhi/test_trial_v1.1_with_report_stratified_train.parquet")
    # split_data['val_df'].to_parquet("parquets/mhi/test_trial_v1.1_with_report_stratified_val.parquet")
    split_data['test_df'].to_parquet("parquets/merged_datasets/test/mimic_mhi_psa_test_updated_stratified_20%.parquet")