#!/usr/bin/env python3
"""Run inference on CoT SFT checkpoint (m6htj38p) on validation set."""
import torch, sys, numpy as np, pandas as pd, json, os, time
sys.path.insert(0, "/volume/ECG_tokenizer")

import transformers

# Trie compatibility
import transformers.tokenization_utils as _tu
if not hasattr(_tu, "Trie"):
    class _Trie:
        def __init__(self): self.data = {}
        def add(self, word):
            ref = self.data
            for ch in word:
                ref = ref.setdefault(ch, {})
            ref[""] = 1
    _tu.Trie = _Trie

try:
    from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
    _orig = GemmaTokenizer.__setstate__
    def _cs(self, d):
        if "sp_model_proto" not in d:
            self.__dict__.update(d)
            return
        _orig(self, d)
    GemmaTokenizer.__setstate__ = _cs
except (ImportError, AttributeError):
    pass

from transformers import AutoTokenizer
from utils.enums import DecoderMode
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.rewards import compute_rewards, format_reward, diagnosis_accuracy_reward, key_evidence_reward

CHECKPOINT = "checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/m6htj38p_20260223-231759/best_model.pt"
VAL_PATH = "/volume/ECG_tokenizer/output/cot_val.parquet"
NUM_SAMPLES = 20
MAX_NEW_TOKENS = 1024
OUTPUT_PATH = "/volume/ECG_tokenizer/output/cot_val_inference_m6htj38p.json"


def main(
    checkpoint=CHECKPOINT,
    val_path=VAL_PATH,
    num_samples=NUM_SAMPLES,
    max_new_tokens=MAX_NEW_TOKENS,
    output_path=OUTPUT_PATH,
):
    print(f"transformers: {transformers.__version__}")
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = cfg.decoder_mode if isinstance(cfg.decoder_mode, DecoderMode) else DecoderMode(cfg.decoder_mode)

    # Build lora_config
    lora_config = getattr(cfg, "lora_config", None)
    if lora_config is None and bool(getattr(cfg, "use_lora", False)):
        lora_config = {
            "r": int(getattr(cfg, "lora_r", 16)),
            "lora_alpha": int(getattr(cfg, "lora_alpha", getattr(cfg, "lora_r", 16))),
            "lora_dropout": float(getattr(cfg, "lora_dropout", 0.0)),
            "target_modules": list(getattr(cfg, "lora_target_modules", None) or []),
            "bias": str(getattr(cfg, "lora_bias", "none")),
            "top_k_layers": int(getattr(cfg, "lora_top_k_layers", 0)) if getattr(cfg, "lora_top_k_layers", None) else None,
        }

    model = ECG_Tokenizer_Wrapper(
        encoder_name=cfg.encoder_name, quantizer_name=cfg.quantizer_name,
        decoder_name=cfg.decoder_name, num_quantizers=int(cfg.num_quantizers),
        codebook_size=int(cfg.codebook_size), decoder_mode=decoder_mode,
        huggingface_model_name=cfg.huggingface_model_name,
        llm_input_embedding_size=int(cfg.llm_input_embedding_size),
        bridge_name=cfg.bridge_name,
        num_visual_tokens=getattr(cfg, "num_query_tokens", None),
        bridge_mid_dim=int(getattr(cfg, "bridge_mid_dim", 512)),
        bridge_num_heads=int(getattr(cfg, "bridge_num_heads", 8)),
        bridge_dropout=float(getattr(cfg, "bridge_dropout", 0.1)),
        bridge_num_special_tokens=int(getattr(cfg, "bridge_num_special_tokens", 4)),
        bridge_qformer_layers=getattr(cfg, "bridge_qformer_layers", None),
        bridge_text_hidden_size=getattr(cfg, "bridge_text_hidden_size", None),
        bridge_bias_last_codebook=getattr(cfg, "bridge_bias_last_codebook", None),
        bridge_codebook_dropout=getattr(cfg, "bridge_codebook_dropout", None),
        bridge_mix_strategy=getattr(cfg, "bridge_mix_strategy", None),
        bridge_token_axis=getattr(cfg, "bridge_token_axis", None),
        bridge_cross_every=getattr(cfg, "bridge_cross_every", None),
        instruction_dropout=0.0, stage1_checkpoint_path=None,
        use_lora=bool(getattr(cfg, "use_lora", False)),
        lora_config=lora_config,
        tokenizer=tokenizer, ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(cfg, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(cfg, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(cfg, "default_generation_kwargs", None),
        num_codebooks_kept=getattr(cfg, "num_codebooks_kept", None),
        codebook_offset=getattr(cfg, "codebook_offset", 0),
    )
    model._load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    model = model.cuda()

    # Patch SDPA
    for name, mod in model.named_modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            if mod.config._attn_implementation == "flash_attention_2":
                mod.config._attn_implementation = "sdpa"

    # Set LoRA to inference mode
    try:
        model.set_lora_inference_mode(True)
    except Exception:
        pass

    print(f"\nLoading validation data: {val_path}")
    df = pd.read_parquet(val_path)
    # Sample randomly
    df_sample = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)

    results = []
    total_rewards = {"format": 0, "diagnosis": 0, "evidence": 0, "combined": 0}

    for idx in range(len(df_sample)):
        row = df_sample.iloc[idx]
        t0 = time.time()

        # Load signal
        signal = np.load(row["waveform_path_psa"]).astype(np.float32)
        if signal.shape[0] != 12:
            signal = signal.T
        signal_tensor = torch.tensor(signal).unsqueeze(0).cuda()

        # Parse messages for prompt and ground truth
        msgs = json.loads(row["messages"])
        system_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        user_msg = user_msg.replace("<image>", "<start_of_image>")
        if "<start_of_image>" not in user_msg:
            user_msg = "<start_of_image>\n\n" + user_msg
        gt_text = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        prompt = (
            f"<start_of_turn>system\n{system_msg}<end_of_turn>\n"
            f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = enc["input_ids"].cuda()
        prompt_mask = enc["attention_mask"].cuda()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features = model.encoder(signal_tensor)
            quantized, indices, _ = model.quantizer(features)
            codes = model._extract_primary_codes(
                indices, model.decoder.num_codebooks_kept, model.decoder.codebook_offset
            )
            gen_ids = model.decoder.generate_report_with_question(
                quantized_features=quantized, quantized_codes=codes,
                prompt_input_ids=prompt_ids, prompt_attention_mask=prompt_mask,
                max_token_length=max_new_tokens, do_sample=False,
            )

        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        elapsed = time.time() - t0

        # Compute rewards
        r_fmt = format_reward(gen_text, gt_text)
        r_diag = diagnosis_accuracy_reward(gen_text, gt_text)
        r_evid = key_evidence_reward(gen_text, gt_text)
        r_all = compute_rewards(gen_text, gt_text, {"format": 0.2, "diagnosis": 0.5, "evidence": 0.3})
        r_combined = r_all["total"]

        total_rewards["format"] += r_fmt
        total_rewards["diagnosis"] += r_diag
        total_rewards["evidence"] += r_evid
        total_rewards["combined"] += r_combined

        result = {
            "idx": idx,
            "study_id": str(row.get("study_id", "")),
            "waveform": row["waveform_path_psa"],
            "generation": gen_text,
            "ground_truth": gt_text,
            "rewards": {"format": r_fmt, "diagnosis": r_diag, "evidence": r_evid, "combined": r_combined},
            "gen_tokens": len(gen_ids[0]),
            "time_s": round(elapsed, 1),
        }
        results.append(result)

        # Print progress
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}/{len(df_sample)} | {row['waveform_path_psa'].split('/')[-1]} | {elapsed:.1f}s | {len(gen_ids[0])} tokens")
        print(f"Rewards: fmt={r_fmt:.2f}, diag={r_diag:.2f}, evid={r_evid:.2f}, combined={r_combined:.3f}")
        print(f"\n--- GENERATION (first 500 chars) ---")
        print(gen_text[:500])
        print(f"\n--- GROUND TRUTH (first 500 chars) ---")
        print(gt_text[:500])

    # Summary
    n = len(df_sample)
    if n == 0:
        print(f"\nNo samples to evaluate (val_path={val_path} yielded 0 rows).")
        with open(output_path, "w") as f:
            json.dump({"summary": {k: 0.0 for k in total_rewards}, "results": []}, f, indent=2)
        print(f"Results saved to: {output_path}")
        return
    print(f"\n{'='*80}")
    print(f"SUMMARY ({n} samples)")
    print(f"{'='*80}")
    print(f"  Format reward:    {total_rewards['format']/n:.3f}")
    print(f"  Diagnosis reward: {total_rewards['diagnosis']/n:.3f}")
    print(f"  Evidence reward:  {total_rewards['evidence']/n:.3f}")
    print(f"  Combined reward:  {total_rewards['combined']/n:.3f}")

    # Save results
    with open(output_path, "w") as f:
        json.dump({"summary": {k: v/n for k, v in total_rewards.items()}, "results": results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
