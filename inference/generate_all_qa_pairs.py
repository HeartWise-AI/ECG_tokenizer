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
from typing import Optional, Any

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shim: transformers ≥5.x merged tokenization_gemma_fast into tokenization_gemma
# and renamed GemmaTokenizerFast → GemmaTokenizer.  Checkpoints saved with older
# versions pickle the old module path + class name.
import types as _types, importlib as _importlib
_gemma_tok = _importlib.import_module("transformers.models.gemma.tokenization_gemma")
_shim = _types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
_shim.GemmaTokenizerFast = _gemma_tok.GemmaTokenizer  # alias old → new
sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = _shim

from transformers import AutoTokenizer

from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    """Get config value from dict or object."""
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def _coerce_config(config_obj: Any) -> Any:
    """Convert dict config to SimpleNamespace for attribute access."""
    from types import SimpleNamespace
    if isinstance(config_obj, dict):
        return SimpleNamespace(**{k: _coerce_config(v) for k, v in config_obj.items()})
    if isinstance(config_obj, list):
        return [_coerce_config(item) for item in config_obj]
    return config_obj


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.

    Loads both the checkpoint weights and the config.yaml from the checkpoint folder
    to properly configure the bridge (e.g., num_codebooks_kept for 1CB models).
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _coerce_config(checkpoint_data["config"])
    
    # Load config.yaml from checkpoint folder for bridge configuration
    # The config.yaml may have settings not stored in the checkpoint config object
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = None
    if os.path.exists(config_yaml_path):
        print(f"Loading config.yaml from {config_yaml_path}...")
        yaml_config = load_yaml(config_yaml_path)
    
    # Tokenizer loading (offline-friendly, but backward compatible)
    base_tokenizer_dir = os.getenv("BASE_TOKENIZER_DIR", "/app/checkpoints/google-medgemma-4b-it")
    use_local = os.path.isdir(base_tokenizer_dir)
    tokenizer_source = base_tokenizer_dir if use_local else config.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=use_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))
    
    # Get num_codebooks_kept - MUST match checkpoint architecture to load weights correctly
    # Infer from checkpoint tensor shapes since config may not match actual trained architecture
    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", None)
    if num_codebooks_kept is None:
        num_codebooks_kept = _cfg_get(config, "num_codebooks_kept", None)

    # CRITICAL: Check actual checkpoint tensor shapes - mix_gate.0.weight is [hidden, hidden*num_codebooks]
    checkpoint_state_dict = checkpoint_data["model_state_dict"]
    mix_gate_key = "decoder.bridge.mix_gate.0.weight"
    if mix_gate_key in checkpoint_state_dict:
        mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
        hidden_dim = mix_gate_shape[0]
        inferred_codebooks = mix_gate_shape[1] // hidden_dim
        if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
            print(f"   OVERRIDE: Config num_codebooks_kept={num_codebooks_kept} but checkpoint has {inferred_codebooks}")
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))
    
    # Log the bridge configuration being used
    num_quantizers = int(getattr(config, "num_quantizers", 8))
    if num_codebooks_kept is not None:
        print(f"   Bridge config: num_codebooks_kept={num_codebooks_kept}, codebook_offset={codebook_offset}, num_quantizers={num_quantizers}")
    
    # Detect if checkpoint has LoRA weights even if config says use_lora=False
    # This can happen with DPO checkpoints that were saved with incomplete config
    has_lora_weights = any('lora_A' in k or 'lora_B' in k for k in checkpoint_state_dict.keys())
    use_lora = bool(getattr(config, "use_lora", False)) or has_lora_weights

    if has_lora_weights and not getattr(config, "use_lora", False):
        print(f"   OVERRIDE: Config use_lora=False but checkpoint has LoRA weights, enabling LoRA")

    # Reconstruct LoRA config if training stored scalar fields (common) instead of a full lora_config dict.
    # Use sensible defaults matching the standard training config when checkpoint has LoRA but no config.
    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and use_lora:
        # Default target modules for MedGemma/Gemma models
        default_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        lora_config = {
            "r": int(getattr(config, "lora_r", 32)),
            "lora_alpha": int(getattr(config, "lora_alpha", 64)),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.05)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or default_target_modules),
            "bias": str(getattr(config, "lora_bias", "none")),
        }
        print(f"   Using LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, targets={lora_config['target_modules']}")
        # Persist so downstream code paths can reuse it.
        try:
            setattr(config, "lora_config", lora_config)
            setattr(config, "use_lora", True)
        except Exception:
            pass

    model = ECG_Tokenizer_Wrapper(
        encoder_name=config.encoder_name,
        quantizer_name=config.quantizer_name,
        decoder_name=config.decoder_name,
        num_quantizers=num_quantizers,
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
        bridge_bias_last_codebook=_cfg_get(yaml_config, "bridge_bias_last_codebook", _cfg_get(config, "bridge_bias_last_codebook", None)),
        bridge_codebook_dropout=_cfg_get(yaml_config, "bridge_codebook_dropout", _cfg_get(config, "bridge_codebook_dropout", None)),
        bridge_cross_every=_cfg_get(yaml_config, "bridge_cross_every", _cfg_get(config, "bridge_cross_every", None)),
        instruction_dropout=_cfg_get(yaml_config, "instruction_dropout", _cfg_get(config, "instruction_dropout", 0.0)),
        stage1_checkpoint_path=None,  # Don't load stage1 - all weights are in the finetuned checkpoint
        use_lora=use_lora,
        lora_config=lora_config,
        tokenizer=tokenizer,
        ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
        num_codebooks_kept=num_codebooks_kept,
        codebook_offset=codebook_offset,
    )
    
    # Load *full* checkpoint weights using the wrapper's loader.
    # This is important for LoRA checkpoints because the correct LoRA scaling (alpha/r),
    # target modules, and naming conventions must match what was used during training.
    state_dict = checkpoint_data["model_state_dict"]
    model._load_state_dict(state_dict, strict=False)

    # Put LoRA into inference mode if present (keeps adapters attached; generation works either way).
    try:
        if use_lora:
            model.set_lora_inference_mode(True)
    except Exception:
        pass
    
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
    # Use stochastic sampling (matching validation behavior) - no do_sample/temperature override
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
    parser.add_argument("--device", type=int, default=2, help="GPU device ID")
    parser.add_argument("--output_prefix", type=str, default="all_qa_generations", 
                        help="Prefix for output files (default: all_qa_generations)")
    parser.add_argument("--save_interval", type=int, default=10,
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

    waveform_column = args.waveform_column
    if waveform_column not in val_df.columns and 'ecg_path' in val_df.columns:
        print(f"Warning: Column '{waveform_column}' not found; falling back to 'ecg_path'.")
        waveform_column = 'ecg_path'
    if waveform_column not in val_df.columns:
        raise KeyError(
            f"Column '{waveform_column}' not found in validation data. "
            f"Available columns: {list(val_df.columns)}"
        )

    if 'waveform_name' not in val_df.columns:
        val_df['waveform_name'] = val_df[waveform_column].astype(str).apply(lambda p: os.path.basename(p))
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
        waveform_path = row[waveform_column]
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
                except Exception:
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
