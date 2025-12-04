#!/usr/bin/env python3
"""
Generate answers for ALL question-waveform pairs in the validation dataset.

This script:
1. Loads the validation parquet with all QA pairs
2. Generates model predictions for each (waveform, question) pair
3. Saves results as JSON and CSV with proper metrics

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_all_qa_pairs.py \
        --checkpoint checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/d8389lsr_20251129-073725/best_model.pt \
        --output_dir checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/d8389lsr_20251129-073725/ \
        --max_samples 10000
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from typing import Optional

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))
    
    model = ECG_Tokenizer_Wrapper(
        encoder_name=config.encoder_name,
        quantizer_name=config.quantizer_name,
        decoder_name=config.decoder_name,
        num_quantizers=int(getattr(config, "num_quantizers", 8)),
        codebook_size=int(getattr(config, "codebook_size", 512)),
        decoder_mode=decoder_mode,
        huggingface_model_name=config.huggingface_model_name,
        llm_input_embedding_size=int(config.llm_input_embedding_size),
        bridge_name=config.bridge_name,
        num_visual_tokens=num_visual_tokens,
        bridge_mid_dim=int(getattr(config, "bridge_mid_dim", 512)),
        bridge_num_heads=int(getattr(config, "bridge_num_heads", 8)),
        bridge_dropout=float(getattr(config, "bridge_dropout", 0.1)),
        bridge_num_special_tokens=int(getattr(config, "bridge_num_special_tokens", 4)),
        bridge_qformer_layers=getattr(config, "bridge_qformer_layers", None),
        bridge_text_hidden_size=getattr(config, "bridge_text_hidden_size", None),
        stage1_checkpoint_path=getattr(config, "stage1_checkpoint_path", None),
        use_lora=bool(getattr(config, "use_lora", False)),
        lora_config=getattr(config, "lora_config", None),
        tokenizer=tokenizer,
        ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
    )
    
    state_dict = checkpoint_data["model_state_dict"]
    
    def _is_lora_key(k):
        return "lora_A" in k or "lora_B" in k or "lora_embedding" in k or k.endswith("lora_scaling")
    
    base_sd = {k: v for k, v in state_dict.items() if not _is_lora_key(k)}
    lora_sd = {k: v for k, v in state_dict.items() if _is_lora_key(k)}
    model.load_state_dict(base_sd, strict=False)
    
    if lora_sd:
        from peft import get_peft_model, set_peft_model_state_dict, LoraConfig
        
        lora_r = 32
        for k, v in lora_sd.items():
            if 'lora_A' in k and v.ndim == 2:
                lora_r = v.shape[0]
                break
        
        target_modules = set()
        for k in lora_sd.keys():
            if 'lora_A' in k or 'lora_B' in k:
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p in ('lora_A', 'lora_B') and i > 0:
                        target_modules.add(parts[i-1])
                        break
        target_modules = list(target_modules) if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        peft_llm = get_peft_model(model.decoder.llm_model, lora_config)
        
        def _clean_lora_key(k):
            if "base_model." in k:
                return k[k.index("base_model."):]
            for pref in ("decoder.", "llm_model.", "model.", "language_model.", "transformer."):
                if k.startswith(pref):
                    k = k[len(pref):]
            return k
        
        adapter_sd = {_clean_lora_key(k): v for k, v in lora_sd.items()}
        set_peft_model_state_dict(peft_llm, adapter_sd, adapter_name="default")
        model.decoder.llm_model = peft_llm.merge_and_unload()
        print("   LoRA weights merged")
    
    model.eval()
    model.to(device)
    
    return model, tokenizer, config


def generate_answer(
    model,
    tokenizer,
    ecg_tensor: torch.Tensor,
    question: str,
    device: torch.device,
    config,
) -> str:
    """Generate answer for a single (ECG, question) pair."""
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    user_content = f"<start_of_image>\n\nQuestion: {question}\n\nRespond concisely with the key finding or answer."
    prompt_text = (
        f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    encoding = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
    prompt_ids = encoding.input_ids.to(device)
    prompt_mask = encoding.attention_mask.to(device)
    
    eos_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
        eos_ids.append(end_of_turn_id)
    
    generation_kwargs = dict(getattr(config, "default_generation_kwargs", {}) or {})
    generation_kwargs.setdefault("max_new_tokens", 96)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("temperature", 0.0)
    generation_kwargs.setdefault("no_repeat_ngram_size", 5)
    generation_kwargs.setdefault("repetition_penalty", 1.1)
    generation_kwargs["eos_token_id"] = eos_ids
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    with torch.no_grad():
        generated_ids = model.generate_report(
            x=ecg_tensor,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            max_token_length=int(getattr(config, "max_token_length", 640)),
            **generation_kwargs,
        )
    
    generation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    for pattern in [r'<end_of_turn>', r'<start_of_turn>', r'<\|eot_id\|>', r'model\s*$', r'^model\s*']:
        generation = re.sub(pattern, '', generation)
    
    return generation.strip()


def load_ecg_waveform(waveform_path: str, target_length: int = 2500) -> Optional[np.ndarray]:
    """Load and preprocess ECG waveform."""
    try:
        waveform = np.load(waveform_path)
        
        if waveform.ndim == 3:
            waveform = waveform.squeeze(-1)
        
        if waveform.shape[-1] == 12:
            pass  # Already correct
        elif waveform.shape[0] == 12:
            waveform = waveform.T
        
        current_length = waveform.shape[0]
        if current_length >= target_length:
            start = (current_length - target_length) // 2
            waveform = waveform[start:start + target_length, :]
        else:
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")
        
        return waveform
    except Exception as e:
        print(f"Error loading {waveform_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate answers for all QA pairs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--validation_parquet", type=str, 
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet",
                        help="Path to validation parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--waveform_column", type=str, default="waveform_path_psa", help="Column with waveform paths")
    parser.add_argument("--question_column", type=str, default="prompt", help="Column with questions")
    parser.add_argument("--answer_column", type=str, default="generated_answer", help="Column with ground truth")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output_prefix", type=str, default="all_qa_generations", 
                        help="Prefix for output files (default: all_qa_generations)")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N samples (default: 1000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint if available")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer, config = load_model(args.checkpoint, device)
    
    print(f"\nLoading validation data from {args.validation_parquet}...")
    val_df = pd.read_parquet(args.validation_parquet)
    print(f"Total QA pairs: {len(val_df)}")
    print(f"Unique waveforms: {val_df['waveform_name'].nunique()}")
    
    if args.max_samples:
        val_df = val_df.head(args.max_samples)
        print(f"Limited to {len(val_df)} samples")
    
    checkpoint_csv = os.path.join(args.output_dir, f"{args.output_prefix}_checkpoint.csv")
    start_idx = 0
    results = []
    
    if args.resume and os.path.exists(checkpoint_csv):
        print(f"\nResuming from checkpoint: {checkpoint_csv}")
        existing_df = pd.read_csv(checkpoint_csv)
        results = existing_df.to_dict('records')
        start_idx = len(results)
        print(f"Loaded {start_idx} existing results, continuing from sample {start_idx}")
    
    print(f"\nGenerating answers (saving every {args.save_interval} samples)...")
    errors = 0
    waveform_cache = {}
    
    def save_checkpoint(results_list, is_final=False):
        if not results_list:
            return
        temp_df = pd.DataFrame(results_list)
        if is_final:
            csv_path = os.path.join(args.output_dir, f"{args.output_prefix}.csv")
        else:
            csv_path = checkpoint_csv
        temp_df.to_csv(csv_path, index=False)
        print(f"\n{'Final' if is_final else 'Checkpoint'} saved: {csv_path} ({len(results_list)} samples)")
    
    for idx, row in tqdm(val_df.iloc[start_idx:].iterrows(), total=len(val_df)-start_idx, desc="Generating", initial=start_idx):
        waveform_name = row['waveform_name']
        waveform_path = row[args.waveform_column]
        question = row[args.question_column]
        ground_truth = row[args.answer_column]
        prompt_category = row.get('prompt_category', 'unknown')
        
        try:
            if waveform_path not in waveform_cache:
                waveform = load_ecg_waveform(waveform_path)
                if waveform is None:
                    errors += 1
                    continue
                waveform_cache[waveform_path] = waveform
                if len(waveform_cache) > 1000:
                    keys_to_remove = list(waveform_cache.keys())[:500]
                    for k in keys_to_remove:
                        del waveform_cache[k]
            else:
                waveform = waveform_cache[waveform_path]
            
            ecg_tensor = torch.from_numpy(waveform.astype(np.float32)).T.unsqueeze(0).to(device)
            generation = generate_answer(model, tokenizer, ecg_tensor, question, device, config)
            
            results.append({
                'waveform_name': waveform_name,
                'waveform_path': waveform_path,
                'question': question,
                'generation': generation,
                'ground_truth': ground_truth,
                'prompt_category': prompt_category,
            })
            
            if len(results) % args.save_interval == 0:
                save_checkpoint(results)
            
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\nError on {waveform_name}: {e}")
            continue
    
    print(f"\nGenerated {len(results)} answers, {errors} errors")
    
    save_checkpoint(results, is_final=True)
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
        print(f"Removed checkpoint file: {checkpoint_csv}")
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}.csv")
    print(f"Saved CSV to {csv_path}")
    
    json_data = {}
    for waveform_name, group in results_df.groupby('waveform_name'):
        json_data[waveform_name] = []
        for _, row in group.iterrows():
            json_data[waveform_name].append({
                'Question': row['question'],
                'Generation': row['generation'],
                'Ground truth': row['ground_truth'],
                'Category': row['prompt_category'],
            })
    
    json_path = os.path.join(args.output_dir, f"{args.output_prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON to {json_path}")
    
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from nltk import word_tokenize
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoother = SmoothingFunction()
        
        rouge1, rouge2, rougeL = [], [], []
        bleu1, bleu4 = [], []
        meteor_scores = []
        
        for _, row in results_df.iterrows():
            gen = row['generation']
            ref = row['ground_truth']
            
            if gen and ref:
                scores = scorer.score(ref, gen)
                rouge1.append(scores['rouge1'].fmeasure)
                rouge2.append(scores['rouge2'].fmeasure)
                rougeL.append(scores['rougeL'].fmeasure)
                
                try:
                    ref_tokens = word_tokenize(ref.lower())
                    gen_tokens = word_tokenize(gen.lower())
                    bleu1.append(sentence_bleu([ref_tokens], gen_tokens, weights=(1,0,0,0), smoothing_function=smoother.method1))
                    bleu4.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother.method1))
                    meteor_scores.append(meteor_score([ref_tokens], gen_tokens))
                except:
                    pass
        
        print(f"\nOverall Metrics ({len(rouge1)} samples):")
        print(f"  ROUGE-1: {np.mean(rouge1):.4f}")
        print(f"  ROUGE-2: {np.mean(rouge2):.4f}")
        print(f"  ROUGE-L: {np.mean(rougeL):.4f}")
        print(f"  BLEU-1:  {np.mean(bleu1):.4f}")
        print(f"  BLEU-4:  {np.mean(bleu4):.4f}")
        print(f"  METEOR:  {np.mean(meteor_scores):.4f}")
        
        print("\nPer-Category Metrics:")
        for category in results_df['prompt_category'].unique():
            cat_df = results_df[results_df['prompt_category'] == category]
            cat_rouge = []
            for _, row in cat_df.iterrows():
                if row['generation'] and row['ground_truth']:
                    scores = scorer.score(row['ground_truth'], row['generation'])
                    cat_rouge.append(scores['rougeL'].fmeasure)
            if cat_rouge:
                print(f"  {category}: ROUGE-L={np.mean(cat_rouge):.4f} ({len(cat_rouge)} samples)")
        
    except ImportError as e:
        print(f"Could not compute metrics: {e}")
    
    print("\nDone!")
    return results_df


if __name__ == "__main__":
    main()


