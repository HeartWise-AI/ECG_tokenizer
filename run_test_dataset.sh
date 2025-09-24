#!/bin/bash
# Generate 10k test dataset (5k MIMIC + 5k MHI)
python dataset_generation/generate_train_test_datasets.py \
  --dataset combined \
  --train_samples 0 \
  --mimic_test_samples 5000 \
  --mhi_test_samples 5000 \
  --max_prompts_per_ecg 10 \
  --max_normal_percentage 0.05
