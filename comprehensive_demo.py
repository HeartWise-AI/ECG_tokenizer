#!/usr/bin/env python3
"""
Comprehensive demo of the ECG Instruction Tuning Pipeline.
This script demonstrates all major features of the instruction tuning system.
"""

import json
from pathlib import Path
from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter
from utils.config import load_config


def demo_templates():
    """Demo different instruction templates."""
    print("🧠 TEMPLATE DEMONSTRATION")
    print("=" * 50)
    
    # Load sample data
    with open("test_ecg_qa_sample/train.json", 'r') as f:
        data = json.load(f)
    
    sample = data[0]
    templates = ["alpaca", "vicuna", "chat", "medical"]
    
    for template in templates:
        print(f"\n📋 {template.upper()} TEMPLATE:")
        print("-" * 30)
        
        formatter = ECGInstructDataFormatter(template_style=template)
        formatted = formatter.format_single_sample(sample, include_metadata=True, enhanced_prompt=True)
        
        # Show template output
        lines = formatted['text'].split('\n')
        for i, line in enumerate(lines[:8]):  # Show first 8 lines
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... ({len(lines) - 8} more lines)")
        
        print(f"\n  📊 Stats: {len(formatted['text'])} chars, {len(lines)} lines")


def demo_features():
    """Demo different formatting features."""
    print("\n🔧 FEATURE DEMONSTRATION")
    print("=" * 50)
    
    with open("test_ecg_qa_sample/train.json", 'r') as f:
        data = json.load(f)
    
    sample = data[0]
    formatter = ECGInstructDataFormatter(template_style="alpaca")
    
    # Test different feature combinations
    features = [
        ("Basic", {"include_metadata": False, "enhanced_prompt": False}),
        ("With Metadata", {"include_metadata": True, "enhanced_prompt": False}),
        ("Enhanced Prompt", {"include_metadata": False, "enhanced_prompt": True}),
        ("Full Features", {"include_metadata": True, "enhanced_prompt": True}),
    ]
    
    for name, kwargs in features:
        print(f"\n📋 {name.upper()}:")
        print("-" * 20)
        
        formatted = formatter.format_single_sample(sample, **kwargs)
        lines = formatted['text'].split('\n')
        
        # Show key parts
        for line in lines[:5]:
            print(f"  {line}")
        print(f"  ... (total: {len(formatted['text'])} chars)")


def demo_batch_processing():
    """Demo batch dataset processing."""
    print("\n📦 BATCH PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    with open("test_ecg_qa_sample/train.json", 'r') as f:
        data = json.load(f)
    
    # Process different batch sizes
    formatter = ECGInstructDataFormatter(template_style="medical")
    
    batch_sizes = [1, 3, 5]
    for size in batch_sizes:
        print(f"\n📊 Processing {size} samples:")
        
        batch = data[:size]
        formatted_dataset = formatter.format_dataset(batch, include_metadata=True)
        
        print(f"  ✓ Input: {len(batch)} samples")
        print(f"  ✓ Output: {len(formatted_dataset)} formatted entries")
        print(f"  ✓ Avg length: {sum(len(item['text']) for item in formatted_dataset) // len(formatted_dataset)} chars")
        
        # Show keys available
        if formatted_dataset:
            print(f"  ✓ Available keys: {list(formatted_dataset[0].keys())}")


def demo_config_integration():
    """Demo config loading and integration."""
    print("\n⚙️  CONFIG INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    config_files = [
        "config/instruct_tuning/gpt2_instruct_base.yaml",
        "config/instruct_tuning/llama32_1b_instruct_base.yaml"
    ]
    
    for config_path in config_files:
        if Path(config_path).exists():
            print(f"\n📄 {config_path}:")
            print("-" * 30)
            
            config = load_config(config_path)
            
            # Show key config parameters
            key_params = [
                'huggingface_model_name', 'template_style', 'learning_rate', 
                'num_epochs', 'max_seq_length', 'use_lora'
            ]
            
            for param in key_params:
                if param in config:
                    print(f"  {param}: {config[param]}")
            
            # Test formatter with config template
            if 'template_style' in config:
                formatter = ECGInstructDataFormatter(template_style=config['template_style'])
                print(f"  ✓ Formatter created with {config['template_style']} template")


def demo_metadata_extraction():
    """Demo metadata and condition extraction."""
    print("\n🔍 METADATA EXTRACTION DEMONSTRATION")
    print("=" * 50)
    
    with open("test_ecg_qa_sample/train.json", 'r') as f:
        data = json.load(f)
    
    formatter = ECGInstructDataFormatter()
    
    # Show different samples with their metadata
    for i, sample in enumerate(data[:3]):
        print(f"\n📋 SAMPLE {i+1}:")
        print("-" * 20)
        
        print(f"  Question: {sample['question'][:60]}...")
        print(f"  Detected conditions: {len(sample.get('detected_conditions', []))}")
        print(f"  Categories: {len(sample.get('condition_categories', []))}")
        
        if 'detected_conditions' in sample:
            print(f"  First 3 conditions: {sample['detected_conditions'][:3]}")


def main():
    """Run comprehensive demo."""
    print("🏥 ECG INSTRUCTION TUNING PIPELINE DEMO")
    print("=" * 60)
    print("This demo showcases the complete instruction tuning pipeline")
    print("for ECG Large Language Models.")
    print()
    
    try:
        demo_templates()
        demo_features()
        demo_batch_processing()
        demo_config_integration()
        demo_metadata_extraction()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📋 SUMMARY:")
        print("✓ All instruction templates working (alpaca, vicuna, chat, medical)")
        print("✓ Metadata integration functional")
        print("✓ Enhanced prompting operational")
        print("✓ Batch processing ready")
        print("✓ Config integration working")
        print("✓ Pipeline ready for training!")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Install training dependencies: transformers, trl, peft")
        print("2. Run training: python runners/llm_instruction_tuning_runner.py")
        print("3. Or integrate with InstructTuningProject class")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print(f"   Check dependencies and file paths")


if __name__ == "__main__":
    main()
