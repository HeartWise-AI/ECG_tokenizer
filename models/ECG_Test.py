import pandas as pd 
import numpy as np
from models.ECG_Gemma import ECGConfig, ECG_Gemma_config, ECG_Gemma_model, ECG_Gemma3ModelOutputWithPast, ECG_Gemma3MultiModalProjector, ECG_Gemma3ForConditionalGeneration, token_type_ids_mask_function
import torch
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.config.tokenizer_config import ECGTokenizerTrainingConfig
from utils.enums import DecoderMode, ModelName
import yaml
from transformers import AutoTokenizer
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from models.ECG_GRPO import model_4bit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformers_ecg/src')))


mimic_train = pd.read_parquet("/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet")
#path = mimic_train["waveform_path_original"][0].strip("'")
#ecg_signal = np.load(path)
#print(ecg_signal)
#print(mimic_train.columns)


mimic_train["waveform_path_original"] = mimic_train["waveform_path_original"].apply(
    lambda x: x.strip("'"))

#mimic_train_copy = mimic_train.head(100).copy(deep=True) 
mimic_train_copy = mimic_train.head(1000).copy(deep=True)

mimic_train_copy["ECG_signals"] = mimic_train_copy["waveform_path_original"].apply(np.load)
#print(mimic_train_copy["ECG_signals"][0])

columns_to_keep = ["ECG_signals", "waveform_path_original", "report"]
columns = mimic_train_copy.columns.tolist() 
#print(len(columns))

columns.remove("ECG_signals")
columns.remove("waveform_path_original")
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
#print(config)

ecg_tokenizer_wrapper = ECG_Tokenizer_Wrapper(
        decoder_name=ModelName.LINEAR_CLASSIFIER_DECODER,
		decoder_mode=DecoderMode.CLASSIFICATION)

torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])

#with open('/media/data1/models/ECG_Tokenizer/ECG_Tokenizer_Reconstruction/tfq5q94l_20250622-004552/config.yaml') as f:
#    config = yaml.safe_load(f)

#print(config)

ecg_tokenizer = ECG_Tokenizer_Wrapper(
    encoder_name="Residual_Conv_Encoder",        
    quantizer_name="ECG_Tokenizer_Quantizer",     
    decoder_name="Conv_Decoder",                  
    decoder_mode=DecoderMode.RECONSTRUCTION,      
    num_quantizers=8,                             
    codebook_size=512,                           
    num_classes=-1                                
)

checkpoint = torch.load('/media/data1/models/ECG_Tokenizer/ECG_Tokenizer_Reconstruction/tfq5q94l_20250622-004552/best_model_epoch_10.pt', map_location=torch.device('cpu'))
ecg_tokenizer_state_dict = checkpoint.get('model_state_dict', checkpoint)

#print(ecg_tokenizer_state_dict.keys())

ecg_tokenizer.load_state_dict(ecg_tokenizer_state_dict, strict=False)

ecg_model = ECG_Gemma_model(config, ecg_tokenizer)

def ecg_torch(ecg_signal):

    torch_ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)

    return torch_ecg_signal

mimic_train_copy["ECG_signals"] = mimic_train_copy["ECG_signals"].apply(ecg_torch)

sample_signal = mimic_train_copy["ECG_signals"][0]
sample_tensor = torch.tensor(sample_signal, dtype=torch.float32)
#print(sample_tensor.shape)


#features = ecg_model.ecg_tokenizer.encoder(sample_tensor)  # e.g. [T, D1, D2]
#feature_dim = features.view(features.size(0), -1).size(-1)
#print(feature_dim)

#features = ecg_model.get_ecg_features(sample_tensor)
#print(features)

#tokenized_ecg = ecg_model.tokenize_ecg(features)
#print(tokenized_ecg)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
tokenizer.add_special_tokens({"additional_special_tokens": ["<|ecg|>"]})
ecg_model.resize_token_embeddings(len(tokenizer))

ecg_model.config.ecg_token_id = tokenizer.convert_tokens_to_ids("<|ecg|>")

ecg_token_id = tokenizer.convert_tokens_to_ids("<|ecg|>")
#print("ECG token ID:", ecg_token_id)



#prompt_start = tokenizer.encode("Patient ECG reading: ", add_special_tokens=False)
#prompt_end = tokenizer.encode(" Generate a report.", add_special_tokens=False)

#input_ids_list = prompt_start + [ecg_token_id] + prompt_end
#print("Input IDs list:", input_ids_list)
#print("Decoded:", tokenizer.decode(input_ids_list))

#input_ids = torch.tensor([input_ids_list])
#attention_mask = torch.ones_like(input_ids)


def find_length(report):
    report_tokens = tokenizer(report, return_tensors="pt", add_special_tokens=True).input_ids.squeeze(0)
    length = len(report_tokens)
    return length

#test_report = mimic_train_copy["report"][0]   
#print(find_length(test_report))

mimic_train_copy["max_length"] = mimic_train_copy["report"].apply(find_length)
#print(mimic_train_copy["max_length"][0])

def full_llm_prompts(model, tokenizer, input_ids: torch.LongTensor, ecg_signals: torch.FloatTensor, attention_mask: torch.Tensor, report):
    if (input_ids is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
		 
    if ecg_signals is not None:
        ecg_features = model.get_ecg_features(ecg_signals)
        ecg_tokens = model.tokenize_ecg(ecg_features)
    else:
        raise ValueError("ecg_signals must be provided when ECG tokens are expected")

    num_special_tokens = 0
    for batch in input_ids:
        for seq in batch:
            if seq == model.config.ecg_token_id:
                num_special_tokens = num_special_tokens + 1
		
    if num_special_tokens == 0:
        raise ValueError("No special ECG tokens found in input_ids; cannot reshape ECG tokens")
		
    total_elements = ecg_tokens.numel()
    remainder = total_elements % num_special_tokens 
    missing = (-remainder) % num_special_tokens

    if missing > 0:

        extra_ecg_tokens = [model.config.ecg_token_id] * missing

        input_ids_list = []
        for batch in input_ids:
            tokens_list = batch.tolist()
            inserted = False
            for idx, tok in enumerate(tokens_list):
                if tok == model.config.ecg_token_id and not inserted:
                    tokens_list[idx:idx] = extra_ecg_tokens
                    inserted = True
            input_ids_list.append(torch.tensor(tokens_list, device=input_ids.device))

        num_special_tokens = num_special_tokens + missing

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
		
    ecg_tokens = ecg_tokens.view(num_special_tokens, -1)

    batch_size = input_ids.size(0)
    new_input_ids_list = []
    new_attention_mask_list = []

    num_special = 0
    for i in range(batch_size):
        original_tokens = input_ids[i].tolist()
        new_tokens = []
        new_attention = []

        if attention_mask is not None:
            att_mask = attention_mask[i].tolist()
        else:
            att_mask = [1] * len(original_tokens)
			
        for index, token_id in enumerate(original_tokens):
            if token_id == model.config.ecg_token_id:

                ecg_token_seq = ecg_tokens[num_special]
                new_tokens.extend(ecg_token_seq.tolist())
                new_attention.extend([1] * len(ecg_token_seq))
                num_special=num_special+1
            else:   
                new_tokens.append(token_id)
                new_attention.append(att_mask[index])
			
        new_input_ids_list.append(torch.tensor(new_tokens, device=input_ids.device))
        new_attention_mask_list.append(torch.tensor(new_attention, device=input_ids.device))
			
    padded_input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=0).to(input_ids.device)
    padded_attention_mask = pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0).to(input_ids.device)

    report_tokens = tokenizer(report, return_tensors="pt", add_special_tokens=True).input_ids.squeeze(0)
    report_attention = torch.ones_like(report_tokens)

    final_input_ids = []
    final_attention_mask = []
    labels = []
    for i in range(batch_size):
        combined_input_ids = (torch.cat([padded_input_ids[i], report_tokens.to(padded_input_ids.device)], dim=0))
        combined_attention_mask = (torch.cat([padded_attention_mask[i], report_attention.to(padded_attention_mask.device)], dim=0))

        label_ids = combined_input_ids.clone()
        label_ids[:padded_input_ids.size(1)] = -100

        final_input_ids.append(combined_input_ids.squeeze(0))
        final_attention_mask.append(combined_attention_mask.squeeze(0))
        labels.append(label_ids.squeeze(0))

    final_input_ids = pad_sequence(final_input_ids, batch_first=True, padding_value=0)
    final_attention_mask = pad_sequence(final_attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    attention_mask = final_attention_mask
    input_ids = final_input_ids

    return attention_mask, input_ids, labels

def creating_full_prompts(df, model, tokenizer):

    full_input_ids = []
    full_input_embeds = []
    full_attention_mask = []
    all_labels = []

    for index, row in df.iterrows():

        ecg_signals = row["ECG_signals"]
        report = row["report"]

        prompt_start = tokenizer.encode("Patient ECG reading: ", add_special_tokens=False)
        prompt_end = tokenizer.encode(" Generate a report.", add_special_tokens=False)

        ecg_token_id = tokenizer.convert_tokens_to_ids("<|ecg|>")
        input_ids_list = prompt_start + [ecg_token_id] + prompt_end
        input_ids = torch.tensor([input_ids_list])
        attention_mask = torch.ones_like(input_ids)


        prompt_components = full_llm_prompts(model, tokenizer, input_ids, ecg_signals, attention_mask, report)
        attention_mask, input_ids, labels = prompt_components

        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)
        labels = labels.view(-1)

        full_input_ids.append(input_ids)
        full_attention_mask.append(attention_mask)
        all_labels.append(labels)

    full_input_ids = pad_sequence(full_input_ids, batch_first=True, padding_value=0)
    full_attention_mask = pad_sequence(full_attention_mask, batch_first=True, padding_value=0)
    all_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)


    return full_input_ids, full_attention_mask, all_labels

#full_input_ids, full_attention_mask, all_labels = creating_full_prompts(mimic_train_copy, ecg_model, tokenizer)
#print(full_input_ids[0])
#print(full_attention_mask[0])
#print(all_labels[0])

#test = ecg_model.forward_ecg(full_input_ids[0], full_attention_mask[0], tokenizer)
#print(test)

def collate_fn(batch):
    input_ids = [item['input_ids'].squeeze(0) for item in batch]
    labels = [item['labels'].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'labels': labels,
        "attention_mask": attention_mask
    }

class TrainingDataset(Dataset):

    def __init__(self, full_tokenized_inputs, labels, attention_mask):
        
        #only take up to the question, not the final answer of the full text
        self.input_ids = full_tokenized_inputs

        self.attention_mask = attention_mask

        self.labels = labels.clone()

        if self.attention_mask is not None:
        # mask padding tokens, they should not be considered in loss calculation
          #padding_mask = (self.attention_mask == 0)
          #labels[padding_mask] = -100

            self.labels[self.attention_mask == 0] = -100
       
    #batch size
    def __len__(self):
        return len(self.input_ids)

   #returns item in the batch as a dictionary given it's index
    def __getitem__(self, index):
    
        item = {
        "input_ids": self.input_ids[index],
        "labels": self.labels[index]
    }
        
        if self.attention_mask is not None:
        
            item["attention_mask"] = self.attention_mask[index]
        
        return item


def rl_llm_prompts(model, tokenizer, input_ids: torch.LongTensor, ecg_signals: torch.FloatTensor, attention_mask: torch.Tensor):
    if (input_ids is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
		 
    if ecg_signals is not None:
        ecg_features = model.get_ecg_features(ecg_signals)
        ecg_tokens = model.tokenize_ecg(ecg_features)
    else:
        raise ValueError("ecg_signals must be provided when ECG tokens are expected")

    num_special_tokens = 0
    for batch in input_ids:
        for seq in batch:
            if seq == model.config.ecg_token_id:
                num_special_tokens = num_special_tokens + 1
		
    if num_special_tokens == 0:
        raise ValueError("No special ECG tokens found in input_ids; cannot reshape ECG tokens")
		
    total_elements = ecg_tokens.numel()
    remainder = total_elements % num_special_tokens 
    missing = (-remainder) % num_special_tokens

    if missing > 0:

        extra_ecg_tokens = [model.config.ecg_token_id] * missing

        input_ids_list = []
        for batch in input_ids:
            tokens_list = batch.tolist()
            inserted = False
            for idx, tok in enumerate(tokens_list):
                if tok == model.config.ecg_token_id and not inserted:
                    tokens_list[idx:idx] = extra_ecg_tokens
                    inserted = True
            input_ids_list.append(torch.tensor(tokens_list, device=input_ids.device))

        num_special_tokens = num_special_tokens + missing

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
		
    ecg_tokens = ecg_tokens.view(num_special_tokens, -1)

    batch_size = input_ids.size(0)
    new_input_ids_list = []
    new_attention_mask_list = []

    num_special = 0
    for i in range(batch_size):
        original_tokens = input_ids[i].tolist()
        new_tokens = []
        new_attention = []

        if attention_mask is not None:
            att_mask = attention_mask[i].tolist()
        else:
            att_mask = [1] * len(original_tokens)
			
        for index, token_id in enumerate(original_tokens):
            if token_id == model.config.ecg_token_id:

                ecg_token_seq = ecg_tokens[num_special]
                new_tokens.extend(ecg_token_seq.tolist())
                new_attention.extend([1] * len(ecg_token_seq))
                num_special=num_special+1
            else:   
                new_tokens.append(token_id)
                new_attention.append(att_mask[index])
			
        new_input_ids_list.append(torch.tensor(new_tokens, device=input_ids.device))
        new_attention_mask_list.append(torch.tensor(new_attention, device=input_ids.device))
			
    padded_input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=0).to(input_ids.device)
    padded_attention_mask = pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0).to(input_ids.device)

    attention_mask = padded_attention_mask
    input_ids = padded_input_ids

    return attention_mask, input_ids

def rl_prompts(df, model, tokenizer):

    full_input_ids = []
    full_input_embeds = []
    full_attention_mask = []

    for index, row in df.iterrows():

        ecg_signals = row["ECG_signals"]
        report = row["report"]

        prompt_start = tokenizer.encode("Patient ECG reading: ", add_special_tokens=False)
        prompt_end = tokenizer.encode(" Generate a report.", add_special_tokens=False)

        ecg_token_id = tokenizer.convert_tokens_to_ids("<|ecg|>")
        input_ids_list = prompt_start + [ecg_token_id] + prompt_end
        input_ids = torch.tensor([input_ids_list])
        attention_mask = torch.ones_like(input_ids)


        prompt_components = full_llm_prompts(model, tokenizer, input_ids, ecg_signals, attention_mask, report)
        attention_mask, input_ids, labels = prompt_components

        input_ids = input_ids.view(-1)
        attention_mask = attention_mask.view(-1)

        full_input_ids.append(input_ids)
        full_attention_mask.append(attention_mask)

    full_input_ids = pad_sequence(full_input_ids, batch_first=True, padding_value=0)
    full_attention_mask = pad_sequence(full_attention_mask, batch_first=True, padding_value=0)

    return full_input_ids, full_attention_mask




class RLDataset(Dataset):

  def __init__(self, reports, input_ids, attention_mask):

    self.reports = reports
    self.input_ids = input_ids
    self.attention_mask = attention_mask

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, index):

    item = {

      "input_ids": self.input_ids[index],
      "attention_mask": self.attention_mask[index],
      "reports": self.reports[index],

    }

    return item

df_sft, df_rl = train_test_split(mimic_train_copy, test_size=0.2, random_state=42)

df_sft = df_sft.sample(frac=0.1, random_state=42).reset_index(drop=True)

df_rl = df_rl.sample(frac=0.1, random_state=42).reset_index(drop=True)

full_input_ids, full_attention_mask, all_labels = creating_full_prompts(df_sft, ecg_model, tokenizer)
#print(full_input_ids[0])

#turn the tokenized inputs into batches
dataset_sft = TrainingDataset(full_input_ids, all_labels, full_attention_mask)

batch_size = 8
#make into batches
dataloader_sft = DataLoader(dataset_sft, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#print(len(dataloader_sft))

first_batch = next(iter(dataloader_sft))
input_ids = first_batch["input_ids"]
attention_mask = first_batch["attention_mask"]

output = ecg_model.forward_ecg(input_ids, attention_mask, tokenizer)
print(output)




