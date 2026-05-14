#!/bin/bash
# OpenRLHF GRPO with LLM-judge reward, 2 GPUs hybrid engine.
#
# STATUS: WIP. The reward function (services/openrlhf_judge_reward.py) and
# data builder (scripts/build_openrlhf_dataset.py) are done and unit-tested.
# What's still required before this script can run:
#
#   1. ECGCausalLMWrapper (models/openrlhf_ecg_wrapper.py) — wraps
#      ECG_Tokenizer_Wrapper as AutoModelForImageTextToText so OpenRLHF + vLLM
#      can load it. Needs:
#         - `forward(input_ids, attention_mask, **mm_inputs)` accepting ECG
#           soft tokens via mm_inputs["ecg_soft_tokens"]
#         - Standard HF generation_config / save_pretrained
#         - Register a vision_config stub so VLM auto-detection fires
#
#   2. Custom SingleTurnAgentExecutor (services/openrlhf_agent.py) — encodes
#      ECG signals to soft tokens BEFORE handing the batch to vLLM. Pulls
#      signal_path from each row, loads the .npy, runs encoder+quantizer+bridge,
#      passes the (B, T_soft, D_llm) tensor through to vLLM as multimodal input.
#
#   3. Expose `encode_to_soft_tokens(signal)` on ECG_Tokenizer_Wrapper.
#
# Reference: examples/scripts/train_vlm_math_hybrid_engine.sh (Qwen3.5-VL) shows
# the multimodal pattern we'd be mirroring.

set -e
set -x

WORKDIR=/volume/ECG_tokenizer
DATASET=$WORKDIR/data/openrlhf_train_v1.jsonl
CHECKPOINT=$WORKDIR/checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt
SAVE_DIR=$WORKDIR/checkpoints/openrlhf_grpo_v1
REWARD_FUNC=$WORKDIR/services/openrlhf_judge_reward.py

mkdir -p $SAVE_DIR
cd /volume/OpenRLHF

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$WORKDIR \
python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 --ref.num_gpus_per_node 2 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 2 \
   --vllm.num_engines 2 --vllm.tensor_parallel_size 1 \
   --vllm.gpu_memory_utilization 0.55 \
   --vllm.enable_sleep --ds.enable_sleep \
   --vllm.sync_backend nccl --vllm.enforce_eager \
   --train.colocate_all \
   --algo.kl.init_coef 1e-2 --algo.kl.estimator k3 --algo.kl.use_loss \
   --algo.advantage.estimator group_norm \
   --algo.dynamic_filtering_enable \
   --algo.dynamic_filtering_range 0.05 0.95 \
   --actor.eps_clip_low_high 0.2 0.27 \
   --actor.adam.lr 5e-7 \
   --actor.gradient_checkpointing_enable \
   --actor.model_name_or_path $CHECKPOINT \
   --reward.remote_url $REWARD_FUNC \
   --train.micro_batch_size 2 --train.batch_size 16 \
   --rollout.micro_batch_size 4 --rollout.batch_size 16 \
   --rollout.n_samples_per_prompt 5 \
   --train.max_epochs 1 \
   --data.max_len 1024 --data.max_samples 5000 \
   --data.prompt_dataset $DATASET \
   --data.input_key prompt --data.label_key label \
   --ds.zero_stage 3 --ds.param_dtype bf16 \
   --ckpt.output_dir $SAVE_DIR --ckpt.save_steps 25 --ckpt.save_hf

# Expected runtime: ~3-4 hours for 1 epoch over 5000 prompts with N=5.
# Eval: after each save_steps boundary, optionally run:
#   CUDA_VISIBLE_DEVICES=0 python scripts/rlvr_eval_subset.py \
#     --checkpoint $SAVE_DIR/global_step_25/best_model.pt \
#     --subset_parquet analysis/rlvr_eval/eval_subset_10per_cat.parquet \
#     --output_dir analysis/rlvr_eval/openrlhf_v1/step_25 --run_judge
