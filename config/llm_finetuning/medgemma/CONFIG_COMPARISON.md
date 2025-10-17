# MedGemma Configuration Comparison

Quick reference for comparing original vs optimized configurations.

## Three Available Configurations

| Config File | Description | Use Case |
|-------------|-------------|----------|
| `medgemma_4b.yaml` | **Original** (preserved) | Baseline reference |
| `medgemma_4b_optimized.yaml` | **Optimized ProjectionBridge** | Recommended first test |
| `medgemma_4b_optimized_codebridge.yaml` | **Optimized CodeBridge** | Test after ProjectionBridge |

## Complete Parameter Comparison

| Parameter | Original | Optimized | Optimized CodeBridge | Notes |
|-----------|----------|-----------|---------------------|-------|
| **num_epochs** | 10 | **15** | **15** | +50% training time |
| **llm_lr** | 5e-5 | 5e-5 | 5e-5 | Unchanged |
| **adapter_lr** | 5e-4 | **2e-4** | **2e-4** | ⭐ -60% (SIGLIP match) |
| **llm_weight_decay** | 1e-5 | **1e-3** | **1e-3** | ⭐ +100x |
| **adapter_weight_decay** | 1e-5 | **1e-2** | **1e-2** | ⭐⭐⭐ +1000x (CRITICAL) |
| **batch_size** | 14 | 14 | 14 | Unchanged |
| **gradient_accumulation_steps** | 8 | **16** | **16** | ⭐ +100% |
| **effective_batch_size** | 112 | **224** | **224** | 2x increase |
| **bridge_name** | ECGProjectionBridge | ECGProjectionBridge | **ECGCodeBridge** | Architecture change |
| **pretrained_tokenizer_path** | dss6mj14 epoch_9 | dss6mj14 epoch_9 | **tywat0ui epoch_10** | Best SIGLIP checkpoint |
| **codebook_offset** | -1 | -1 | -1 | Last 8 codebooks |
| **num_codebooks_kept** | 8 | 8 | 8 | Unchanged |
| **training_phases.adapter_lr** | 5e-4 | **2e-4** | **2e-4** | Match main adapter_lr |
| **training_phases.cross_attention_lr** | 5e-4 | **2e-4** | **2e-4** | Match adapter_lr |

⭐⭐⭐ = Critical change
⭐ = Important change

## Expected Performance Comparison

| Metric | Original (Epoch 4) | Optimized Target | CodeBridge Target |
|--------|-------------------|------------------|-------------------|
| **Overall** | 36.4% | 48-55% | 50-60% |
| **Classification** | 12.7% ❌ | 25-35% ✓ | 27-37% ✓ |
| **Interpretation** | 22.2% ⚠️ | 32-40% ✓ | 34-42% ✓ |
| **ECG Interval** | 2.9% ❌ | 8-15% ⚠️ | 10-17% ⚠️ |
| **Urgency Assessment** | 3.1% ❌ | 10-20% ⚠️ | 12-22% ⚠️ |
| **Category Rhythm** | 51.8% ✓ | 55-62% ✓ | 57-64% ✓ |
| **Category Chamber** | 74.2% ✓ | 76-82% ✓ | 78-84% ✓ |

❌ = Critical failure (< 10%)
⚠️ = Needs improvement (10-25%)
✓ = Acceptable (> 25%)

## SIGLIP Checkpoint Comparison

| Checkpoint | Architecture | Recall@5 | Used In Config |
|------------|--------------|----------|----------------|
| dss6mj14_epoch_9 | ECGProjectionBridge | 34.16% | Original, Optimized |
| tywat0ui_epoch_10 | ECGCodeBridge | 34.59% | Optimized CodeBridge |

**Performance Delta:** ECGCodeBridge +0.43pp (31 label wins, 0 losses)

## Key Insights

### Why adapter_weight_decay is Critical
- **Original:** 1e-5 allows adapter to memorize training data → poor generalization
- **Optimized:** 1e-2 forces adapter to learn general patterns → better test performance
- **Impact:** Addresses root cause of 12.7% classification and 22.2% interpretation scores

### Why adapter_lr Matters
- **High LR (5e-4) + Low WD (1e-5):** Fast learning, aggressive updates, overfitting
- **Low LR (2e-4) + High WD (1e-2):** Stable learning, smooth updates, generalization
- **SIGLIP Success:** Same combination (2e-4 + 1e-2)

### Why Batch Size Helps
- **Original:** Effective batch 112 → noisy gradients
- **Optimized:** Effective batch 224 → stable gradients
- **Target:** SIGLIP uses 512, but 224 is feasible and still helpful

### Why ECGCodeBridge Might Win
- **Discrete Codebook Embeddings:** Better capture ECG patterns as discrete tokens
- **Text Alignment:** Discrete tokens align better with discrete text embeddings
- **SIGLIP Evidence:** Consistent +0.43pp improvement across categories

## Training Command Examples

### Test Optimized (Recommended First)
```bash
python main.py --config config/llm_finetuning/medgemma/medgemma_4b_optimized.yaml
```

### Test CodeBridge (After Optimized)
```bash
python main.py --config config/llm_finetuning/medgemma/medgemma_4b_optimized_codebridge.yaml
```

### Compare Against Original (Baseline)
```bash
python main.py --config config/llm_finetuning/medgemma/medgemma_4b.yaml
```

## Success Criteria

### By Epoch 10, Expect to See:
1. Overall score > 45% (vs current 36.4%)
2. Classification score > 25% (vs current 12.7%)
3. Interpretation score > 30% (vs current 22.2%)
4. Reasonable train/val gap (no severe overfitting)

### If Successful, Next Steps:
1. Run with ECGCodeBridge for potential +2-5pp boost
2. Add data upsampling for rare categories (ECG interval, urgency)
3. Consider extending to 20 epochs
4. Explore larger bridge architectures

---

**Last Updated:** 2025-10-14

