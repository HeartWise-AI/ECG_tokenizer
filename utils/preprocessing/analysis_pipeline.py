import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from utils.preprocessing.ecg_signal_processor import ECGSignalProcessor
from utils.constants import PTBXL_POWER_RATIO
from utils.files_handler import ECGFileHandler

class AnalysisPipeline:
    @staticmethod
    def save_and_preprocess_data(
        df: pd.DataFrame,
        output_folder: str,
        preprocessing_folder: str,
        preprocessing_n_workers: int,
        swap_leads_fn=None,
        swap_lead1=None,
        swap_lead2=None,
        path_column: str | None = None
    ) -> pd.DataFrame:
        # Initialize ECG signal processor
        ecg_signal_processor = ECGSignalProcessor()
        
        # Ensure the preprocessing folder exists
        os.makedirs(preprocessing_folder, exist_ok=True)
        
        # Resolve the ECG path column
        if path_column and path_column in df.columns:
            ecg_path_col = path_column
        else:
            path_columns = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower()]
            if not path_columns:
                raise ValueError("No column with 'path' or 'file' in its name found in the dataframe")
            # prefer standardized names if present
            for preferred in ['ecg_path', 'waveform_path_original', 'waveform_path_psa', 'ECG_path']:
                if preferred in path_columns:
                    ecg_path_col = preferred
                    break
            else:
                ecg_path_col = path_columns[0]
        print(f"Detected path column: {ecg_path_col}")
        
        # Process in batches to manage memory
        batch_size = 10000
        total_batches = (len(df) + batch_size - 1) // batch_size
        processed_df = pd.DataFrame()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
            print(f"Batch size: {end_idx - start_idx} records")
            
            try:
                batch_df = df.iloc[start_idx:end_idx].copy()
                ecgs = []
                for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Loading signals"):
                    try:
                        lead_array = ECGFileHandler.load_ecg_signal(row[ecg_path_col])
                        if np.isnan(lead_array).any():
                            continue
                        
                        file_id = os.path.basename(row[ecg_path_col]).replace(".npy", "")
                        
                        # Shape corrections
                        if lead_array.shape[-1] == 1:
                            lead_array = lead_array.squeeze(-1)
                        if lead_array.shape[0] == 12:  # transpose if needed
                            lead_array = lead_array.transpose(1, 0)
                            
                        # Handle different lengths
                        if lead_array.shape[0] != 2500:
                            if lead_array.shape[0] < 2500:
                                print(f"Warning: Skipping {file_id} - signal length {lead_array.shape[0]} < 2500")
                                continue
                            else:
                                step = lead_array.shape[0] // 2500
                                lead_array = lead_array[::step, :]
                                
                        if lead_array.shape[1] != 12:
                            print(f"Warning: Skipping {file_id} - incorrect number of leads: {lead_array.shape[1]}")
                            continue
                            
                        new_path = os.path.join(preprocessing_folder, f"{file_id}")
                        batch_df.at[index, ecg_path_col] = new_path
                        ecgs.append([new_path, lead_array])
                        
                    except Exception as e:
                        print(f"Error processing file {row[ecg_path_col]}: {str(e)}")
                        continue
                       
                # Fix: Use 'ecg_path' as column name for consistency 
                ecg_signals_df = pd.DataFrame(ecgs, columns=['ecg_path', 'ecg_signal'])
                
                if len(ecg_signals_df) == 0:
                    print(f"Warning: No valid signals in batch {batch_idx + 1}")
                    continue
                
                print("Scaling ECG signals...")
                scaled_signals_df = ecg_signal_processor.scale_ecg_signals(
                    df=ecg_signals_df, 
                    power_ratio=PTBXL_POWER_RATIO
                )

                # print("Processing ECG signals...")
                cleaned_signals_df = ecg_signal_processor.clean_and_process_ecg_leads(
                    df=scaled_signals_df,
                    max_workers=preprocessing_n_workers
                )
                
                # Save processed signals
                print("Saving processed signals...")
                for _, row in tqdm(cleaned_signals_df.iterrows(), total=len(cleaned_signals_df), desc="Saving signals"):
                    # Add .npy extension to the file path explicitly
                    save_path = f"{row['ecg_path']}.npy"
                
                    # Ensure directory exists for this file
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Apply lead swapping if function is provided
                    signal_to_save = row['ecg_signal']
                    if swap_leads_fn is not None and swap_lead1 is not None and swap_lead2 is not None:
                        try:
                            signal_to_save = swap_leads_fn(signal_to_save, swap_lead1, swap_lead2)
                            # Log every 100th swap to avoid flooding console
                            if _ % 100 == 0:
                                print(f"Swapped {swap_lead1} and {swap_lead2} leads for signal {_}")
                        except Exception as e:
                            print(f"Warning: Failed to swap leads for signal {_}: {e}")
                    
                    # Save the numpy array
                    np.save(
                        file=save_path,
                        arr=signal_to_save
                    )
                    
                    # Update the path in the dataframe to include .npy extension
                    # Find rows where the path matches (without .npy extension) and update them
                    matches = batch_df[ecg_path_col] == row['ecg_path']
                    if matches.any():
                        batch_df.loc[matches, ecg_path_col] = save_path
                
                # Append processed batch
                processed_df = pd.concat([processed_df, batch_df], ignore_index=True)
                
                # Clear memory
                del ecg_signals_df, scaled_signals_df, cleaned_signals_df, batch_df
                
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {str(e)}")
                continue
                
        if len(processed_df) == 0:
            raise ValueError("No data was successfully processed")
            
        print(f"\nCompleted processing {len(processed_df)} files")
        return processed_df
