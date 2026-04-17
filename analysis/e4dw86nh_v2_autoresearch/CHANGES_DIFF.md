# Config Diff: e4dw86nh → e4dw86nh_v2

## LoRA (8x more trainable parameters)

```diff
- lora_r: 32
+ lora_r: 96

- lora_alpha: 64
+ lora_alpha: 192

- lora_dropout: 0.05
+ lora_dropout: 0.0

- lora_top_k_layers: 12
+ lora_top_k_layers: 24

  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
+   - gate_proj
+   - up_proj
+   - down_proj
```

**Effect**: 23.8M → 196.7M LoRA params (4.38% of LLM vs 0.53%)

## Learning Rates (much more aggressive)

```diff
- llm_lr: 5.0e-5
+ llm_lr: 4.0e-4        # 8x higher

- adapter_lr: 5.0e-4
+ adapter_lr: 1.5e-3    # 3x higher

- num_warmup_percent: 0.256
+ num_warmup_percent: 0.01  # 25.6% → 1%

- adapter_weight_decay: 1.0e-5
+ adapter_weight_decay: 5.0e-5
```

## Bridge Regularization (swapped dropout locations)

```diff
- bridge_dropout: 0.1
+ bridge_dropout: 0.0

- bridge_codebook_dropout: 0.0
+ bridge_codebook_dropout: 0.05

- instruction_dropout: 0.2
+ instruction_dropout: 0.05
```

## Training Dynamics

```diff
- gradient_accumulation_steps: 16
+ gradient_accumulation_steps: 8     # 2x more frequent updates

  # Effective batch: 512 → 256
  # Optimizer steps: 28,415 → 28,415 (same with 2 GPUs)
```

## Training Phases

```diff
  training_phases:
    phase1_alignment:
-     epochs: 1
+     epochs: 0    # Skip alignment, rely on Stage-1 pretrained bridge
```

## Generation

```diff
- max_new_tokens: 96
+ max_new_tokens: 48

- repetition_penalty: 1.10
+ repetition_penalty: 1.15
```

## Unchanged (critical architecture params)

- bridge_qformer_layers: 10
- bridge_num_heads: 12
- codebook_offset: -1
- num_codebooks_kept: 8
- stage1_checkpoint_path: j4bb0w33
- huggingface_model_name: google/medgemma-4b-it
- Training data: combined_train_qa_m5000k_h5000k_weighted.parquet (7.27M rows)
- Validation data: combined_test_qa_m25k_h25k.parquet
