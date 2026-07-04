#!/usr/bin/env bash
# Convenience wrapper to generate the full combined (MIMIC + MHI) QA datasets.
# Adjust the *_PROMPTS environment variables if you need different prompt totals.

set -euo pipefail

PROMPT_WORKERS=${PROMPT_WORKERS:-16}
ANSWER_WORKERS=${ANSWER_WORKERS:-16}
MAX_PROMPTS=${MAX_PROMPTS:-5}
MAX_NORMAL=${MAX_NORMAL:-0.05}

# Default prompt counts assume roughly 1M ECGs per source × up to 5 prompts/ECG
# => 10M prompts for training and 50k for validation.
MIMIC_TRAIN_PROMPTS=${MIMIC_TRAIN_PROMPTS:-5000000}
MHI_TRAIN_PROMPTS=${MHI_TRAIN_PROMPTS:-5000000}
MIMIC_TEST_PROMPTS=${MIMIC_TEST_PROMPTS:-25000}
MHI_TEST_PROMPTS=${MHI_TEST_PROMPTS:-25000}

TOTAL_TRAIN=$((MIMIC_TRAIN_PROMPTS + MHI_TRAIN_PROMPTS))
TOTAL_TEST=$((MIMIC_TEST_PROMPTS + MHI_TEST_PROMPTS))

echo "Generating combined dataset with:"
echo "  Train prompts: ${TOTAL_TRAIN} (MIMIC=${MIMIC_TRAIN_PROMPTS}, MHI=${MHI_TRAIN_PROMPTS})"
echo "  Test prompts : ${TOTAL_TEST} (MIMIC=${MIMIC_TEST_PROMPTS}, MHI=${MHI_TEST_PROMPTS})"
echo "  Max prompts/ECG=${MAX_PROMPTS}, Max normal fraction=${MAX_NORMAL}"
echo "  Workers      : prompt=${PROMPT_WORKERS}, answer=${ANSWER_WORKERS}"
echo

python dataset_generation/generate_train_test_datasets.py \
  --dataset combined \
  --train_samples "${TOTAL_TRAIN}" \
  --test_samples "${TOTAL_TEST}" \
  --mimic_train_samples "${MIMIC_TRAIN_PROMPTS}" \
  --mhi_train_samples "${MHI_TRAIN_PROMPTS}" \
  --mimic_test_samples "${MIMIC_TEST_PROMPTS}" \
  --mhi_test_samples "${MHI_TEST_PROMPTS}" \
  --max_prompts_per_ecg "${MAX_PROMPTS}" \
  --max_normal_percentage "${MAX_NORMAL}" \
  --prompt_workers "${PROMPT_WORKERS}" \
  --answer_workers "${ANSWER_WORKERS}" \
  "$@"
