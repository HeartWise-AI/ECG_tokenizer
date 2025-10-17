#!/bin/bash

# Test script to verify NCCL timeout fixes
echo "==================================================="
echo "Testing NCCL Timeout Fixes for MedGemma Training"
echo "==================================================="
echo ""
echo "Changes implemented:"
echo "1. Proper gradient accumulation (16 steps)"
echo "2. Reduced batch size (8 instead of 16)"  
echo "3. Increased NCCL timeout (30 minutes)"
echo "4. Removed excessive sync barriers"
echo "5. NCCL async error handling enabled"
echo ""
echo "Effective batch size: 8 * 16 = 128"
echo ""
echo "Starting training with single GPU first..."
echo "==================================================="

# Run with single GPU first to ensure basic functionality
bash scripts/runner.sh \
    --base_config config/llm_finetuning/medgemma/medgemma_4b.yaml \
    --selected_gpus 0 \
    --use_wandb false \
    --run_mode train

echo ""
echo "If single GPU works, try multi-GPU with:"
echo "bash scripts/runner.sh --base_config config/llm_finetuning/medgemma/medgemma_4b.yaml --selected_gpus 0,1 --use_wandb false --run_mode train"