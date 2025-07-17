#!/usr/bin/env python3

import sys
import json
from pathlib import Path

print("Simple Instruction Tuning Test")
print("=" * 40)

# Test 1: Import formatter
print("1. Testing imports...")
try:
    from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter
    print("   ✓ ECGInstructDataFormatter imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load sample data
print("2. Testing data loading...")
try:
    sample_file = Path("test_ecg_qa_sample/train.json")
    with open(sample_file, 'r') as f:
        data = json.load(f)
    print(f"   ✓ Loaded {len(data)} samples")
    print(f"   ✓ Sample keys: {list(data[0].keys())}")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")
    sys.exit(1)

# Test 3: Initialize formatter
print("3. Testing formatter initialization...")
try:
    formatter = ECGInstructDataFormatter(template_style="alpaca")
    print("   ✓ Formatter initialized with alpaca template")
except Exception as e:
    print(f"   ✗ Formatter initialization failed: {e}")
    sys.exit(1)

# Test 4: Format single entry
print("4. Testing single entry formatting...")
try:
    single_entry = data[0]
    formatted = formatter.format_single_sample(single_entry)
    print("   ✓ Single entry formatted successfully")
    print(f"   ✓ Output length: {len(formatted['text'])} characters")
    print(f"   ✓ First 100 chars: {formatted['text'][:100]}...")
except Exception as e:
    print(f"   ✗ Single entry formatting failed: {e}")
    sys.exit(1)

# Test 5: Format small dataset
print("5. Testing dataset formatting...")
try:
    small_dataset = data[:2]  # Just 2 entries
    formatted_dataset = formatter.format_dataset(small_dataset)
    print(f"   ✓ Dataset formatted successfully")
    print(f"   ✓ Output entries: {len(formatted_dataset)}")
    print(f"   ✓ Keys in formatted entry: {list(formatted_dataset[0].keys())}")
except Exception as e:
    print(f"   ✗ Dataset formatting failed: {e}")
    print(f"   Error details: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 40)
print("ALL TESTS PASSED! ✓")
print("Instruction tuning pipeline is working correctly.")
