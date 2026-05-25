#!/bin/bash
# OpenRLHF GRPO with LLM-judge reward, 2 GPUs split engine.
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
# GPU split:
#   - `--train.colocate_actor_ref` keeps actor+reference on one training GPU.
#   - `--vllm.num_engines 1 --vllm.tensor_parallel_size 1` reserves the other
#     GPU for rollout generation.
#   - Do not use `--train.colocate_all` for this mode; OpenRLHF's colocated
#     hybrid engine shares all GPUs between vLLM and DeepSpeed instead.
#
# Reference: examples/scripts/train_vlm_math_hybrid_engine.sh (Qwen3.5-VL) shows
# the multimodal pattern we'd be mirroring; this script uses split placement
# because ECG rollout/eval should not compete with training memory.

set -e
set -x

WORKDIR=${WORKDIR:-/volume/ECG_tokenizer}
DATASET=${DATASET:-$WORKDIR/data/openrlhf_train_v1.jsonl}
CHECKPOINT=${CHECKPOINT:-/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt}
SAVE_DIR=${SAVE_DIR:-$WORKDIR/checkpoints/openrlhf_grpo_v1}
REWARD_FUNC=${REWARD_FUNC:-$WORKDIR/services/openrlhf_judge_reward.py}
# Use two visible GPUs. With GPU1 currently busy on this box, default to 0,2.
OPENRLHF_GPUS=${OPENRLHF_GPUS:-0,2}

mkdir -p $SAVE_DIR
cd /volume/OpenRLHF

CUDA_VISIBLE_DEVICES=$OPENRLHF_GPUS PYTHONPATH=$WORKDIR \
python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
   --train.colocate_actor_ref \
   --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
   --vllm.gpu_memory_utilization 0.80 \
   --vllm.sync_backend nccl --vllm.enforce_eager \
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

# Expected runtime: likely slower than colocated 2-engine mode because this uses
# one vLLM engine, but it isolates rollout memory from actor/ref training.
# Eval: after each save_steps boundary, optionally run:
#   CUDA_VISIBLE_DEVICES=2 python scripts/rlvr_eval_subset.py \
#     --checkpoint $SAVE_DIR/global_step_25/best_model.pt \
#     --subset_parquet analysis/rlvr_eval/eval_subset_10per_cat.parquet \
#     --output_dir analysis/rlvr_eval/openrlhf_v1/step_25 \
#     --device cuda:0 --run_judge
