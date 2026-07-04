#!/bin/bash
# Schema conversion script for ECG QA dataset

set -e

cd /volume/ECG_tokenizer

echo "=== Converting Train (weighted) ==="
python dataset_generation/convert_ecg_qa_schema.py \
    --input /volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.parquet \
    --output /volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted_schema_v1.parquet \
    --dataset combined

echo ""
echo "=== Converting Test ==="
python dataset_generation/convert_ecg_qa_schema.py \
    --input /volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet \
    --output /volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k_schema_v1.parquet \
    --dataset combined

echo ""
echo "=== Done ==="
