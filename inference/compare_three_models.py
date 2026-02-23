#!/usr/bin/env python3
"""
Compare inference outputs across three MedGemma checkpoints.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference/compare_three_models.py --num_samples 20
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse functions from generate_all_qa_pairs.py
from inference.generate_all_qa_pairs import load_model, generate_answer, load_ecg_waveform

CHECKPOINTS = {
    "MedGemma_1.0_8CB": {
        "path": "checkpoints/BEST_LLM/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt",
        "description": "MedGemma 1.0 (google/medgemma-4b-it), 10 Q-Former layers, 12 heads",
    },
    "MedGemma_1.5_6L": {
        "path": "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma_1.5/2s3ll02e_20260127-003636/best_model.pt",
        "description": "MedGemma 1.5 (google/medgemma-1.5-4b-it), 6 Q-Former layers, 8 heads",
    },
    "MedGemma_1.5_10L": {
        "path": "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/a08ok2lr_20260114-213119/best_model.pt",
        "description": "MedGemma 1.5 model + 1.0 tokenizer, 10 Q-Former layers, 12 heads",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Compare three MedGemma checkpoints")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--validation_parquet", type=str,
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet")
    parser.add_argument("--output_dir", type=str, default="output/model_comparison")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load validation data and select fixed subset
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    print(f"Parquet: {args.validation_parquet}")

    val_df = pd.read_parquet(args.validation_parquet)
    print(f"Total QA pairs: {len(val_df)}")
    print(f"Unique waveforms: {val_df['waveform_name'].nunique()}")

    # Select diverse samples (different categories)
    np.random.seed(args.seed)

    # Try to get samples from different categories
    categories = val_df['prompt_category'].unique()
    samples_per_cat = max(1, args.num_samples // len(categories))
    selected_indices = []

    for cat in categories:
        cat_df = val_df[val_df['prompt_category'] == cat]
        n = min(samples_per_cat, len(cat_df))
        selected_indices.extend(cat_df.sample(n=n, random_state=args.seed).index.tolist())

    # If we need more, sample randomly
    if len(selected_indices) < args.num_samples:
        remaining = args.num_samples - len(selected_indices)
        available = val_df.index.difference(selected_indices)
        selected_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False).tolist())

    # Trim to exact number
    selected_indices = selected_indices[:args.num_samples]
    subset_df = val_df.loc[selected_indices].reset_index(drop=True)

    print(f"\nSelected {len(subset_df)} samples:")
    print(f"Categories: {subset_df['prompt_category'].value_counts().to_dict()}")

    # Save selected samples info
    sample_info = subset_df[['waveform_name', 'prompt', 'prompt_category', 'generated_answer']].copy()
    sample_info.to_csv(os.path.join(args.output_dir, "selected_samples.csv"), index=False)

    # Run inference with each checkpoint
    all_results = {}

    for model_name, ckpt_info in CHECKPOINTS.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        print(f"Checkpoint: {ckpt_info['path']}")
        print(f"Description: {ckpt_info['description']}")

        # Load model
        model, tokenizer, config = load_model(ckpt_info['path'], device)

        results = []
        waveform_cache = {}

        for idx, row in subset_df.iterrows():
            waveform_path = row['waveform_path_psa']
            question = row['prompt']
            ground_truth = row['generated_answer']

            try:
                # Load waveform (with caching)
                if waveform_path not in waveform_cache:
                    waveform = load_ecg_waveform(waveform_path)
                    waveform_cache[waveform_path] = waveform
                else:
                    waveform = waveform_cache[waveform_path]

                if waveform is None:
                    print(f"  [SKIP] Could not load waveform: {waveform_path}")
                    continue

                # Generate answer
                ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)
                generation = generate_answer(model, tokenizer, ecg_tensor, question, device, config)

                results.append({
                    'idx': idx,
                    'waveform_name': row['waveform_name'],
                    'question': question,
                    'generation': generation,
                    'ground_truth': ground_truth,
                    'prompt_category': row['prompt_category'],
                })

                print(f"  [{idx+1}/{len(subset_df)}] {row['waveform_name'][:20]}... | {row['prompt_category']}")

            except Exception as e:
                print(f"  [ERROR] {row['waveform_name']}: {e}")
                continue

        all_results[model_name] = results

        # Clear model from GPU
        del model
        torch.cuda.empty_cache()

    # Create comparison dataframe
    print(f"\n{'='*80}")
    print("CREATING COMPARISON")
    print(f"{'='*80}")

    # Build comparison rows
    comparison_rows = []
    for idx in range(len(subset_df)):
        row = {
            'idx': idx,
            'waveform_name': subset_df.iloc[idx]['waveform_name'],
            'question': subset_df.iloc[idx]['prompt'],
            'prompt_category': subset_df.iloc[idx]['prompt_category'],
            'ground_truth': subset_df.iloc[idx]['generated_answer'],
        }

        for model_name in CHECKPOINTS.keys():
            model_results = all_results.get(model_name, [])
            matching = [r for r in model_results if r['idx'] == idx]
            if matching:
                row[f'{model_name}_output'] = matching[0]['generation']
            else:
                row[f'{model_name}_output'] = '[ERROR]'

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"comparison_{timestamp}.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save detailed JSON
    json_output = {
        'metadata': {
            'timestamp': timestamp,
            'num_samples': args.num_samples,
            'parquet': args.validation_parquet,
            'seed': args.seed,
            'checkpoints': CHECKPOINTS,
        },
        'results': comparison_rows,
    }
    json_path = os.path.join(args.output_dir, f"comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Print sample outputs
    print(f"\n{'='*80}")
    print("SAMPLE OUTPUTS (first 5)")
    print(f"{'='*80}")

    for idx, row in comparison_df.head(5).iterrows():
        print(f"\n--- Sample {idx+1}: {row['waveform_name']} ({row['prompt_category']}) ---")
        print(f"Q: {row['question'][:80]}...")
        print(f"Ground Truth: {row['ground_truth'][:100]}...")
        for model_name in CHECKPOINTS.keys():
            output = row[f'{model_name}_output']
            print(f"{model_name}: {output[:100]}...")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - comparison_{timestamp}.csv")
    print(f"  - comparison_{timestamp}.json")
    print(f"  - selected_samples.csv")

    return comparison_df


if __name__ == "__main__":
    main()
