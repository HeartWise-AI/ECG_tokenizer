#!/usr/bin/env python3
"""
Test script to load MedGemma checkpoint and generate text for a specific ECG with a question.
"""

import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from transformers import AutoTokenizer

def test_generation():
    # Configuration
    # Try different checkpoints
    checkpoint_path = "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/cvesmlv2_20250929-113248/best_model.pt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/cvesmlv2_20250929-113248/checkpoint_epoch_1.pt"
    ecg_file = "48203929.npy"  # From validation set
    question = "Are there signs of ischemia or infarction?"
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', checkpoint.get('training_config', {}))
    
    # Handle config object vs dict
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    else:
        config_dict = config
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer_name = config_dict.get('tokenizer_name', 'google/medgemma-4b-it')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize model
    print("\nInitializing model...")
    model = ECG_Tokenizer_Wrapper(
        decoder_mode='llm',
        decoder_name='MedGemma_Decoder',
        huggingface_model_name='google/medgemma-4b-it',
        llm_input_embedding_size=2560,
        bridge_name='Llama32_ECGCodeBridge',
        num_visual_tokens=128,
        tokenizer=tokenizer
    )
    
    # Load model weights
    print("\nLoading model weights...")
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    # Handle DDP wrapped models
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    
    # Load ECG data
    print(f"\nLoading ECG file: {ecg_file}")
    
    # Load the parquet file to find the ECG path
    validation_df = pd.read_parquet('/volume/ECG_tokenizer/output/combined_test_qa_m5k_h5k.parquet')
    
    # Find row with matching waveform file
    matching_rows = validation_df[validation_df['waveform_path_psa'].str.contains(ecg_file.replace('.npy', ''))]
    
    if len(matching_rows) == 0:
        print(f"Could not find {ecg_file} in validation dataset")
        return
    
    # Get the actual path
    ecg_path = matching_rows.iloc[0]['waveform_path_psa']
    print(f"Loading ECG from: {ecg_path}")
    
    # Load and preprocess ECG
    try:
        ecg_data = np.load(ecg_path)
        # Convert to tensor and ensure correct shape [1, 12, 2500]
        if ecg_data.shape[0] == 2500 and ecg_data.shape[1] == 12:
            ecg_data = ecg_data.T  # Transpose to [12, 2500]
        ecg_tensor = torch.from_numpy(ecg_data).float().unsqueeze(0).to(device)
        print(f"ECG shape: {ecg_tensor.shape}")
    except Exception as e:
        print(f"Error loading ECG file: {e}")
        return
    
    # Prepare the question prompt
    print(f"\nQuestion: {question}")
    
    # Format prompt for instruct mode
    if config_dict.get('instruct_mode', True):
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    else:
        prompt = question
    
    # Tokenize prompt
    prompt_encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors='pt'
    )
    
    prompt_input_ids = prompt_encoding['input_ids'].to(device)
    prompt_attention_mask = prompt_encoding['attention_mask'].to(device)
    
    print(f"Prompt tokens: {prompt_input_ids.shape}")
    
    # Generate response
    print("\n" + "="*50)
    print("Generating response...")
    print("="*50)
    
    with torch.no_grad():
        # Generate with the model
        generated_ids = model.generate_report_with_question(
            x=ecg_tensor,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_token_length=config_dict.get('max_token_length', 256)
        )
        
        # Decode the generated text
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Try to extract just the assistant's response
        if "<|assistant|>" in generated_text:
            generated_text = generated_text.split("<|assistant|>")[-1].strip()
        
    print(f"\nGenerated Response:")
    print("-" * 40)
    print(generated_text if generated_text else "[EMPTY GENERATION]")
    print("-" * 40)
    
    # Check for debug messages from ECGCodeBridge
    if not generated_text:
        print("\n⚠️ Empty generation detected!")
        print("Check console output above for ECGCodeBridge debug messages.")
        print("The first 3 forward passes should show detailed debug info.")
    
    # Expected answer from the dataset (for reference)
    expected = "Yes - Previous inferior wall MI; Q waves in inferior - II, III, aVF (possible old infarct); Possible old infarction"
    print(f"\nExpected Answer (for reference):")
    print("-" * 40)
    print(expected)
    print("-" * 40)

if __name__ == "__main__":
    test_generation()
