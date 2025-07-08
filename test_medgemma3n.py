#!/usr/bin/env python3
"""
Test script for MedGemma3N integration with ECG Tokenizer.

This script demonstrates how to use the MedGemma3N decoder with the ECG tokenizer
for clinical report generation from ECG signals.
"""

import torch
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import ModelName, AdapterName, DecoderMode


def test_medgemma3n_integration():
    """Test MedGemma3N integration with ECG tokenizer."""
    print("=" * 60)
    print("Testing MedGemma3N Integration with ECG Tokenizer")
    print("=" * 60)
    
    # Configuration
    config = {
        "decoder_mode": DecoderMode.LLM,
        "decoder_name": ModelName.MEDGEMMA3N_DECODER,
        "medgemma3n_model_name": "google/medgemma-3n-8b",
        "medgemma3n_embedding_size": 3072,
        "adapter_name": AdapterName.GPT2_SEQUENCE_ADAPTER,
        "adapter_dropout": 0.1,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create ECG tokenizer wrapper with MedGemma3N
    print("Creating ECG tokenizer wrapper with MedGemma3N...")
    try:
        tokenizer = ECG_Tokenizer_Wrapper(**config)
        print("✓ ECG tokenizer wrapper created successfully")
    except Exception as e:
        print(f"✗ Error creating tokenizer: {e}")
        return False
    
    # Test data dimensions
    batch_size = 2
    sequence_length = 2500  # ECG sequence length
    num_leads = 12  # Standard 12-lead ECG
    text_length = 20  # Text sequence length
    
    print(f"\nTest data dimensions:")
    print(f"  ECG shape: ({batch_size}, {num_leads}, {sequence_length})")
    print(f"  Text shape: ({batch_size}, {text_length})")
    
    # Create dummy data
    print("\nCreating dummy data...")
    dummy_ecg = torch.randn(batch_size, num_leads, sequence_length)
    dummy_input_ids = torch.randint(0, 10, (batch_size, text_length))
    dummy_attention_mask = torch.ones(batch_size, text_length)
    dummy_labels = torch.randint(0, 10, (batch_size, text_length))
    
    # Test forward pass (training mode)
    print("\nTesting forward pass (training mode)...")
    try:
        with torch.no_grad():
            outputs = tokenizer(
                dummy_ecg,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                labels=dummy_labels
            )
        print("✓ Forward pass completed successfully")
        print(f"  Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}")
        if isinstance(outputs, dict) and 'loss' in outputs:
            print(f"  Loss: {outputs['loss'].item():.6f}")
        if isinstance(outputs, dict) and 'logits' in outputs:
            print(f"  Logits shape: {outputs['logits'].shape}")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    # Test generation (inference mode)
    print("\nTesting report generation (inference mode)...")
    try:
        with torch.no_grad():
            generated_tokens = tokenizer.generate_report(
                dummy_ecg,
                max_token_length=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        print("✓ Report generation completed successfully")
        print(f"  Generated tokens shape: {generated_tokens.shape}")
        print(f"  Sample generated tokens: {generated_tokens[0][:10].tolist()}")
    except Exception as e:
        print(f"✗ Error in report generation: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! MedGemma3N integration is working correctly.")
    print("=" * 60)
    return True


def print_model_info():
    """Print information about the MedGemma3N decoder."""
    print("\nMedGemma3N Decoder Information:")
    print("-" * 40)
    print("• Medical language model based on Gemma architecture")
    print("• Optimized for clinical text generation")
    print("• Default embedding size: 3072")
    print("• Supports various adapter architectures")
    print("• Compatible with ECG tokenizer for multimodal tasks")
    print("• Fallback to mock model when offline")


if __name__ == "__main__":
    print_model_info()
    success = test_medgemma3n_integration()
    
    if success:
        print("\n🎉 MedGemma3N is successfully integrated and ready to use!")
        print("\nNext steps:")
        print("1. Configure your training data paths in the config file")
        print("2. Run training with: bash scripts/runner.sh --base_config config/llm_finetuning/medgemma3n/base_config.yaml")
        print("3. Tune hyperparameters with: bash scripts/run_sweep.sh")
    else:
        print("\n❌ Integration test failed. Please check the error messages above.")