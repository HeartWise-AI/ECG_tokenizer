import pandas as pd 
import numpy as np
from models.ECG_Gemma import ECGConfig, ECG_Gemma_config, ECG_Gemma_model, ECG_Gemma3ModelOutputWithPast, ECG_Gemma3MultiModalProjector, ECG_Gemma3ForConditionalGeneration, token_type_ids_mask_function

mimic_train = pd.read_parquet("/mnt/lost+found/parquet_achille/mimic_v4_clean_train.parquet")
mimic_train["waveform_path"] = mimic_train["waveform_path"].apply(
    lambda x: x.replace(
        "/media/data1/ravram/MIMIC-IV/1.0/files/", 
        "/mnt/lost+found/MIMIC/"
    )
)

mimic_train_copy = mimic_train.copy(deep=True) 

mimic_train_copy["ECG_signals"] = mimic_train_copy["waveform_path"].apply(np.load)

columns_to_keep = ["ECG_signals", "waveform_path", "report"]
columns = mimic_train_copy.columns.tolist() 
#print(len(columns))

columns.remove("ECG_signals")
#columns.remove("waveform_path")
columns.remove("report")

#print(columns)
#print(len(columns))

#print(mimic_train_copy["ECG_signals"][0])

mimic_train_copy = mimic_train_copy.drop(columns, axis = 1)

#print(mimic_train_copy.columns.tolist())

def normalize_zscore(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean  
    else:
        return (signal - mean) / std

mimic_train_copy["ECG_signals"] = mimic_train_copy["ECG_signals"].apply(normalize_zscore)    
#print(mimic_train_copy["ECG_signals"][0])

config = ECG_Gemma_config()
print(config)

#test_model = ECG_Gemma3ForConditionalGeneration()




