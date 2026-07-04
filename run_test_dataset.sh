#!/usr/bin/env bash
# Lightweight wrapper to generate small combined (MIMIC+MHI) QA datasets.
# Defaults: 25k train + 25k test prompts, split evenly across MIMIC/MHI.
#
# Override with environment variables as needed, e.g.:
#   TEST_PROMPTS=10000 TRAIN_PROMPTS=0 ./run_test_dataset.sh
#   MIMIC_TEST_PROMPTS=15000 MHI_TEST_PROMPTS=10000 ./run_test_dataset.sh
#   MAX_PROMPTS=5 PROMPT_WORKERS=8 ANSWER_WORKERS=8 ./run_test_dataset.sh --drop_common_rhythms

set -euo pipefail

# Tuning knobs (env-overridable)
PROMPT_WORKERS=${PROMPT_WORKERS:-8}
ANSWER_WORKERS=${ANSWER_WORKERS:-8}
MAX_PROMPTS=${MAX_PROMPTS:-10}
MAX_NORMAL=${MAX_NORMAL:-0.05}

# Totals (env-overridable)
TEST_PROMPTS=${TEST_PROMPTS:-25000}
TRAIN_PROMPTS=${TRAIN_PROMPTS:-25000}

# If per-dataset splits are not provided, split totals evenly
if [[ -z "${MIMIC_TEST_PROMPTS:-}" || -z "${MHI_TEST_PROMPTS:-}" ]]; then
  HALF_TEST=$(( TEST_PROMPTS / 2 ))
  MIMIC_TEST_PROMPTS=${MIMIC_TEST_PROMPTS:-$HALF_TEST}
  MHI_TEST_PROMPTS=${MHI_TEST_PROMPTS:-$(( TEST_PROMPTS - HALF_TEST ))}
fi

if [[ -z "${MIMIC_TRAIN_PROMPTS:-}" || -z "${MHI_TRAIN_PROMPTS:-}" ]]; then
  HALF_TRAIN=$(( TRAIN_PROMPTS / 2 ))
  MIMIC_TRAIN_PROMPTS=${MIMIC_TRAIN_PROMPTS:-$HALF_TRAIN}
  MHI_TRAIN_PROMPTS=${MHI_TRAIN_PROMPTS:-$(( TRAIN_PROMPTS - HALF_TRAIN ))}
fi

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
