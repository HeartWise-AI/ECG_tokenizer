# Batch Inference - Quick Start Guide

**Status:** ✅ WORKING (December 2, 2025)

---

## What Was Fixed

Batch inference now produces **identical outputs** to single-sample inference. The fix involved implementing micro-batch generation to bypass MedGemma's buggy batched `generate()` method.

---

## Quick Test

Verify the fix is working:

```bash
# Run the consistency test
python test_batch_random_position.py
```

Expected output:
```
✅ ALL TESTS PASSED - Batch inference is consistent!
Batch size  5: 3/3 matches (100%)
Batch size 20: 3/3 matches (100%)
```

---

## Usage

### Run Batch Inference

```bash
# Process entire test set with batching
python scripts/run_full_inference.py \
    --checkpoint checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/d8389lsr_20251129-073725/checkpoint_step_5500.pt \
    --parquet output/combined_test_qa_m25k_h25k.parquet \
    --output results/inference.json \
    --batch_size 20
```

### Generate All QA Pairs

```bash
# Process all QA pairs (processes one at a time, but ECG encoding is still batched)
python scripts/generate_all_qa_pairs.py \
    --checkpoint checkpoints/.../best_model.pt \
    --output_dir checkpoints/.../ \
    --max_samples 10000
```

### Single Sample Test

```bash
# Test individual ECG
python inference/generate_ecg_answer.py \
    --checkpoint checkpoints/.../checkpoint_step_5500.pt \
    --waveform /media/data1/datasets/MHI/adjusted_signals/test/0011420_04-09-2017_15-36-00.npy \
    --question "Is there anything wrong with this ECG?" \
    --device cuda
```

---

## Documentation

### For Users
- **This file** - Quick start guide
- [BATCH_INFERENCE_SOLUTION.md](BATCH_INFERENCE_SOLUTION.md) - Complete solution documentation with test results

### For Developers
- [BATCH_INFERENCE_INVESTIGATION.md](BATCH_INFERENCE_INVESTIGATION.md) - Investigation process and failed attempts
- [BATCH_INFERENCE_BUG_FIX.md](BATCH_INFERENCE_BUG_FIX.md) - Earlier fix attempt (partially successful)

---

## What Changed

### Code Changes

**File:** `/volume/ECG_tokenizer/models/decoder/medgemma_decoder.py`

- Added `_generate_with_microbatch()` method (lines 2127-2179)
- Updated generation paths to use micro-batch approach
- Default micro-batch size: 1 (processes one sample at a time during text generation)

### Performance Impact

| Operation | Before Fix | After Fix |
|-----------|------------|-----------|
| Correctness | ❌ Wrong outputs | ✅ Correct outputs |
| ECG Encoding | Batched (fast) | Batched (fast) |
| Text Generation | Batched (broken) | Micro-batched (working) |
| Overall Speed | ~2.5s/20 samples | ~5.0s/20 samples |

**Result:** Still ~6x faster than pure single-sample processing (which takes ~30s for 20 samples)

---

## Verification

### Expected Behavior

For the test ECG `0011420_04-09-2017_15-36-00.npy` with question "Is there anything wrong with this ECG?":

**Single-sample output:**
```
Yes - Abnormal ECG; Pathological findings: ST depression (lateral leads - I, avL, V5-V6); T wave inversion (anterior - V3-V4)
```

**Batch output (any position in batch):**
```
Yes - Abnormal ECG; Pathological findings: ST depression (lateral leads - I, avL, V5-V6); T wave inversion (anterior - V3-V4)
```

✅ **Outputs match exactly**

### Old (Broken) Behavior

**Single-sample:** Correct answer (as above)

**Batch:** Wrong answer like:
```
to be determined; there are findings of sinus rhythm
```

❌ **Outputs don't match**

---

## Technical Summary

### The Problem

MedGemma's `generate()` method has a bug when processing batched `inputs_embeds`:
- Works correctly when `batch_size=1`
- Produces wrong outputs when `batch_size>1`
- Root cause: Incorrect position tracking / cache position handling

### The Solution

Implemented micro-batch generation:
1. ECG encoding: ✅ Batched (fast, no issues)
2. Q-Former bridge: ✅ Batched (fast, no issues)
3. Text generation: Micro-batched with size=1 (correct, still reasonably fast)

This bypasses MedGemma's buggy batched path while maintaining most of the performance benefits.

### Why This Works

- Micro-batch size=1 → Only calls `generate()` with single samples
- MedGemma handles single-sample correctly → No position tracking bugs
- ECG encoding stays batched → Major speedup preserved

---

## Need Help?

### Common Issues

**Q: Batch inference still produces wrong outputs**

A: Check that you're using the updated code with `_generate_with_microbatch()`. Run `python test_batch_random_position.py` to verify.

**Q: Inference is slow**

A: The micro-batch approach is ~2x slower than the broken batched approach, but still ~6x faster than pure single-sample. Text generation is fast compared to ECG encoding.

**Q: Can I increase micro-batch size for speed?**

A: You can try setting `generation_microbatch_size > 1` in the decoder, but this may reintroduce the position tracking bug. Test thoroughly if you do this.

### Contact

For questions or issues:
1. Check the detailed documentation in [BATCH_INFERENCE_SOLUTION.md](BATCH_INFERENCE_SOLUTION.md)
2. Review the investigation process in [BATCH_INFERENCE_INVESTIGATION.md](BATCH_INFERENCE_INVESTIGATION.md)
3. Run the test script to verify your setup: `python test_batch_random_position.py`

---

**Last Updated:** December 2, 2025
**Status:** ✅ PRODUCTION READY
