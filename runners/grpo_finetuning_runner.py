import logging
import re
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from runners.rl_finetuning_base import RLFinetuningRunnerBase
from utils.enums import RunnerName
from utils.registry import RunnerRegistry
from utils.rewards import (
    BertDiagnosisReward,
    compute_rewards,
)
from utils.rewards_binary import compute_binary_rewards
from utils.rewards_labelset import LabelsetReward, compute_labelset_rewards

logger = logging.getLogger(__name__)


def _extract_answer(text: str) -> str:
    """Extract <answer>...</answer> block, or return last 200 chars."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return text[-200:].strip()


def _extract_think(text: str, max_len: int = 400) -> str:
    """Extract <think>...</think> block (the CoT reasoning), truncated."""
    m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if len(content) > max_len:
            return content[:max_len] + "..."
        return content
    return ""


@RunnerRegistry.register(RunnerName.GRPO_FINETUNING)
class GRPOFinetuningRunner(RLFinetuningRunnerBase):
    def __init__(self, config, wandb_wrapper=None, model=None, ref_model=None,
                 train_dataloader: DataLoader | None = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler=None):
        super().__init__(config, wandb_wrapper, model, ref_model,
                         train_dataloader, optimizer, scheduler)
        self._reset_metrics()

    def _reset_metrics(self):
        self._accum_loss = 0.0
        self._accum_reward = 0.0
        self._accum_reward_format = 0.0
        self._accum_reward_diagnosis = 0.0
        self._accum_reward_evidence = 0.0
        self._accum_advantage_mean = 0.0
        self._accum_kl = 0.0
        self._accum_count = 0

    def train(self):
        if self.train_dataloader is None or self.model is None:
            raise ValueError("Training requires model and train_dataloader.")

        device = torch.device(f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ref_model is not None:
            self.ref_model.to(device)

        # Ensure models use SDPA attention (flash_attention_2 may not work with
        # the installed transformers + flash-attn kernel combination).
        def _patch_attn_to_sdpa(module):
            patched = 0
            for name, mod in module.named_modules():
                cfg = getattr(mod, "config", None)
                if cfg is not None and getattr(cfg, "_attn_implementation", None) == "flash_attention_2":
                    cfg._attn_implementation = "sdpa"
                    cfg._attn_implementation_autoset = False
                    patched += 1
            return patched
        models_to_patch = [self.model] + ([self.ref_model] if self.ref_model is not None else [])
        for m in models_to_patch:
            count = _patch_attn_to_sdpa(m)
            if count > 0 and self.config.is_ref_device:
                print(f"[GRPO] Patched {count} config(s) from flash_attention_2 -> sdpa")

        precision = str(getattr(self.config, "precision", "bf16")).lower()
        use_autocast = precision in {"bf16", "fp16"}
        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        group_size = int(getattr(self.config, "group_size", 4))
        epsilon_low = float(getattr(self.config, "epsilon_low", 0.2))
        epsilon_high = float(getattr(self.config, "epsilon_high", 0.3))
        beta = float(getattr(self.config, "beta", 0.0))
        reward_weights = getattr(self.config, "reward_weights", None)
        max_new_tokens = int(getattr(self.config, "max_new_tokens", 1024))
        temperature = float(getattr(self.config, "temperature", 0.7))
        top_p = float(getattr(self.config, "top_p", 0.95))
        grad_accum = int(getattr(self.config, "grad_accum_steps", 1))
        max_grad_norm = float(getattr(self.config, "max_grad_norm", 1.0))
        save_interval = int(getattr(self.config, "save_interval", 500))
        log_interval = int(getattr(self.config, "log_interval", 10))

        tokenizer = self._get_text_tokenizer()
        if tokenizer is None:
            raise ValueError("Text tokenizer not available for GRPO training.")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Suppress repeated "Setting pad_token_id to eos_token_id" warnings
        decoder = getattr(self.model, "decoder", None)
        for m in (self.model, self.ref_model, decoder):
            llm = getattr(m, "llm_model", None) or getattr(m, "llm", None)
            if llm is not None:
                gen_cfg = getattr(llm, "generation_config", None)
                if gen_cfg is not None:
                    gen_cfg.pad_token_id = pad_token_id

        # Verifier mode: "bert" | "binary" | "labelset" | "judge"
        verifier = str(getattr(self.config, "verifier", "bert")).lower()
        bert_reward = None
        labelset_reward = None
        judge_reward = None
        if verifier == "binary":
            if self.config.is_ref_device:
                print("[GRPO] verifier=binary; using compute_binary_rewards")
        elif verifier == "labelset":
            ontology_path = getattr(self.config, "ontology_path",
                                    "/volume/LLM_JUDGE/ontology/ecg_ontology.json")
            labelset_reward = LabelsetReward(ontology_path=ontology_path)
            if self.config.is_ref_device:
                print(f"[GRPO] verifier=labelset; loaded LabelsetReward "
                      f"({len(labelset_reward.alias_to_canon)} aliases, "
                      f"{len(labelset_reward.canonicals)} canonicals)")
        elif verifier == "judge":
            from utils.rewards_judge import JudgeReward, compute_judge_rewards as _compute_judge_rewards
            judge_reward = JudgeReward()
            if self.config.is_ref_device:
                print("[GRPO] verifier=judge; loaded JudgeReward (LLM_JUDGE registry)")
        else:
            bert_reward = BertDiagnosisReward(device=str(device))
            if self.config.is_ref_device:
                print("[GRPO] Loaded BERT diagnosis classifier for reward computation")

        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.config.is_ref_device:
            print(f"[GRPO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            print(f"[GRPO] Hyperparameters: group_size={group_size}, eps_low={epsilon_low}, "
                  f"eps_high={epsilon_high}, beta={beta}, max_new_tokens={max_new_tokens}, "
                  f"temperature={temperature}, top_p={top_p}")
            print(f"[GRPO] Training: batch_size={self.config.batch_size}, grad_accum={grad_accum}, "
                  f"lr={self.config.lr}, precision={precision}")
            print(f"[GRPO] Reward weights: {reward_weights}")
            print(f"[GRPO] Dataset size: {len(self.train_dataloader)} batches")
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
                print(f"[GRPO] GPU: {torch.cuda.get_device_name(device)} ({gpu_mem:.1f} GB)")
            self._log_metrics({
                "model/trainable_params": trainable_params,
                "model/total_params": total_params,
                "model/trainable_pct": 100 * trainable_params / total_params,
            })

        total_steps = len(self.train_dataloader) * int(self.config.num_epochs)
        self.start_time = time.time()

        for epoch in range(int(self.config.num_epochs)):
            running_loss = 0.0
            epoch_start_time = time.time()
            self._reset_metrics()
            grad_norm = 0.0

            for step, batch in enumerate(self.train_dataloader):
                step_start_time = time.time()
                signal = batch["signal"].to(device)
                prompt_input_ids = batch["prompt_input_ids"].to(device)
                prompt_attention_mask = batch["prompt_attention_mask"].to(device)
                ground_truth_texts = batch["ground_truth_text"]  # list of strings
                B = signal.size(0)

                # ----------------------------------------------------------
                # 1. Generate N completions per prompt (no gradient needed)
                # ----------------------------------------------------------
                all_generated_ids = []  # list of [B, seq_len] tensors
                gen_start = time.time()
                self.model.eval()
                gen_chunk = 2  # groups per generation call
                with torch.no_grad():
                    for chunk_start in range(0, group_size, gen_chunk):
                        chunk_end = min(chunk_start + gen_chunk, group_size)
                        n_in_chunk = chunk_end - chunk_start
                        signal_rep = signal.repeat(n_in_chunk, 1, 1)
                        prompt_ids_rep = prompt_input_ids.repeat(n_in_chunk, 1)
                        prompt_mask_rep = prompt_attention_mask.repeat(n_in_chunk, 1)
                        gen_chunk_ids = self.model.generate_report_with_question(
                            x=signal_rep,
                            prompt_input_ids=prompt_ids_rep,
                            prompt_attention_mask=prompt_mask_rep,
                            max_token_length=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        for g_off in range(n_in_chunk):
                            all_generated_ids.append(gen_chunk_ids[g_off * B : (g_off + 1) * B])
                        if step == 0 and self.config.is_ref_device:
                            print(f"[GRPO DEBUG] Generation chunk {chunk_start//gen_chunk+1}: "
                                  f"B*n={B*n_in_chunk}, output shape={gen_chunk_ids.shape}")
                self.model.train()
                gen_time = time.time() - gen_start
                if step < 3 and self.config.is_ref_device:
                    print(f"[GRPO DEBUG] Step {step}: Generation took {gen_time:.1f}s "
                          f"({group_size} groups in {(group_size+gen_chunk-1)//gen_chunk} chunks)")

                # ----------------------------------------------------------
                # 2. Decode completions and compute rewards
                # ----------------------------------------------------------
                rewards = torch.zeros(B, group_size, device=device)
                completion_texts = []  # [group_size][B]
                batch_r_format = 0.0
                batch_r_diagnosis = 0.0
                batch_r_evidence = 0.0
                reward_components = []  # [group_size][B] -> (fmt, diag, evid)

                for g in range(group_size):
                    gen_ids = all_generated_ids[g]
                    texts_g = []
                    components_g = []
                    for b in range(B):
                        text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
                        texts_g.append(text)
                        if verifier == "binary":
                            r = compute_binary_rewards(text, ground_truth_texts[b], reward_weights)
                        elif verifier == "labelset":
                            r = compute_labelset_rewards(text, ground_truth_texts[b],
                                                          reward_weights, labelset_reward=labelset_reward)
                        elif verifier == "judge":
                            # Requires dataset to also pass prompt_category in batch["prompt_category"]
                            from utils.rewards_judge import compute_judge_rewards
                            cat = batch.get("prompt_category", ["classification"] * B)[b] if "prompt_category" in batch else "classification"
                            r = compute_judge_rewards(text, ground_truth_texts[b], cat,
                                                       reward_weights, judge_reward=judge_reward)
                        else:
                            r = compute_rewards(text, ground_truth_texts[b], reward_weights, bert_reward=bert_reward)
                        rewards[b, g] = r["total"]
                        batch_r_format += r["format"]
                        batch_r_diagnosis += r["diagnosis"]
                        batch_r_evidence += r["evidence"]
                        components_g.append((r["format"], r["diagnosis"], r["evidence"]))
                    completion_texts.append(texts_g)
                    reward_components.append(components_g)

                total_completions = B * group_size
                batch_r_format /= total_completions
                batch_r_diagnosis /= total_completions
                batch_r_evidence /= total_completions

                # Debug: print sample generation and reward details for first few steps
                if step < 3 and self.config.is_ref_device:
                    print(f"[GRPO DEBUG] Step {step}: Rewards matrix [B={B}, G={group_size}]:")
                    for b in range(min(B, 2)):
                        print(f"  Sample {b}: rewards={[f'{rewards[b,g].item():.3f}' for g in range(group_size)]}")
                    print(f"  Avg rewards: format={batch_r_format:.3f}, "
                          f"diagnosis={batch_r_diagnosis:.3f}, evidence={batch_r_evidence:.3f}")
                    snippet = completion_texts[0][0][:300]
                    print(f"  Sample completion (group 0, sample 0): {snippet}...")
                    gt_snippet = ground_truth_texts[0][:200]
                    print(f"  Ground truth (sample 0): {gt_snippet}...")

                # ----------------------------------------------------------
                # 3. Group-normalize advantages
                # ----------------------------------------------------------
                group_mean = rewards.mean(dim=1, keepdim=True)
                group_std = rewards.std(dim=1, keepdim=True)
                advantages = (rewards - group_mean) / (group_std + 1e-8)

                if step < 3 and self.config.is_ref_device:
                    print(f"[GRPO DEBUG] Step {step}: Advantages mean={advantages.mean().item():.4f}, "
                          f"std={advantages.std().item():.4f}, "
                          f"reward_mean={rewards.mean().item():.4f}, reward_std={rewards.std().item():.4f}")

                # ----------------------------------------------------------
                # 4. Build full sequences and compute old log-probs
                # ----------------------------------------------------------
                prompt_len = prompt_input_ids.size(1)
                all_full_ids = []
                all_full_mask = []
                all_full_labels = []
                for g in range(group_size):
                    gen_ids = all_generated_ids[g]
                    full_ids = torch.cat([prompt_input_ids, gen_ids], dim=1)
                    full_mask = (full_ids != pad_token_id).long()
                    full_labels = full_ids.clone()
                    full_labels[:, :prompt_len] = -100
                    full_labels = full_labels.masked_fill(full_ids == pad_token_id, -100)
                    all_full_ids.append(full_ids)
                    all_full_mask.append(full_mask)
                    all_full_labels.append(full_labels)

                if step == 0 and self.config.is_ref_device:
                    num_labeled = (all_full_labels[0] != -100).sum(dim=1)
                    print(f"[GRPO DEBUG] full_ids shape={all_full_ids[0].shape}, "
                          f"prompt_len={prompt_len}, gen_len={all_generated_ids[0].shape[1]}, "
                          f"labeled_tokens_per_sample={num_labeled.tolist()}")

                # Cache ECG embeddings for policy model (same signal across all groups)
                # Switch to eval() during forward passes so dropout is disabled —
                # without this, old_log_probs and policy_log_probs falsely diverge
                # at step 0 (same weights, same inputs, different dropout masks)
                # → fake per-token ratio variance → loss-clip safeguards fire.
                old_token_logps = []
                logp_start = time.time()
                self.model.eval()
                with torch.no_grad():
                    policy_ecg_emb = self._encode_ecg(
                        self.model, signal, prompt_input_ids, prompt_attention_mask
                    )
                    for g in range(group_size):
                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                            logits, labels_exp = self._forward_logits_and_labels(
                                self.model, signal,
                                all_full_ids[g], all_full_mask[g], all_full_labels[g],
                                prompt_input_ids, prompt_attention_mask,
                                ecg_embeddings=policy_ecg_emb,
                            )
                            tlp, tmask = self._token_logp(logits, labels_exp)
                            old_token_logps.append((tlp, tmask))
                # Stay in eval() for the policy forward below (need gradient but
                # not stochastic dropout). Backward & optimizer step still update
                # weights — eval() only disables dropout / batchnorm-stat updates.
                if step < 3 and self.config.is_ref_device:
                    logp_time = time.time() - logp_start
                    seq_logps = [tlp.sum(dim=-1) for tlp, _ in old_token_logps]
                    print(f"[GRPO DEBUG] Step {step}: Old log-probs computed in {logp_time:.1f}s, "
                          f"seq_values={[[f'{seq_logps[g][b].item():.1f}' for g in range(group_size)] for b in range(min(B,2))]}")

                # ----------------------------------------------------------
                # 5. Policy gradient with PER-TOKEN clipped objective
                # ----------------------------------------------------------
                batch_loss = torch.tensor(0.0, device=device)
                batch_kl = 0.0
                use_ref = beta > 0

                # Cache ref model ECG embeddings once (if KL penalty active)
                ref_ecg_emb = None
                if use_ref:
                    with torch.no_grad():
                        ref_ecg_emb = self._encode_ecg(
                            self.ref_model, signal, prompt_input_ids, prompt_attention_mask
                        )

                # Recompute policy ECG embeddings WITH gradient for backward
                policy_ecg_emb_grad = self._encode_ecg(
                    self.model, signal, prompt_input_ids, prompt_attention_mask
                )

                for g in range(group_size):
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        # Policy forward (with gradient, using cached ECG embeddings)
                        logits_policy, labels_policy = self._forward_logits_and_labels(
                            self.model, signal,
                            all_full_ids[g], all_full_mask[g], all_full_labels[g],
                            prompt_input_ids, prompt_attention_mask,
                            ecg_embeddings=policy_ecg_emb_grad,
                        )
                        token_logp_policy, token_mask = self._token_logp(logits_policy, labels_policy)

                        old_tlp, _ = old_token_logps[g]
                        min_len = min(token_logp_policy.size(1), old_tlp.size(1))
                        token_logp_pol = token_logp_policy[:, :min_len]
                        old_tlp_g = old_tlp[:, :min_len].detach()
                        tmask_g = token_mask[:, :min_len]

                        # Per-token ratio and clipping
                        token_ratio = torch.exp(token_logp_pol - old_tlp_g)
                        clipped_token_ratio = torch.clamp(token_ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

                        adv_g = advantages[:, g].detach().unsqueeze(1)  # [B, 1]
                        surr1 = token_ratio * adv_g
                        surr2 = clipped_token_ratio * adv_g
                        token_surrogate = torch.min(surr1, surr2)

                        n_valid = tmask_g.sum(dim=1).clamp_min(1)
                        per_seq_loss = -(token_surrogate * tmask_g).sum(dim=1) / n_valid
                        policy_loss = per_seq_loss.mean()

                        # Optional per-token KL penalty
                        kl = torch.tensor(0.0, device=device)
                        if use_ref:
                            with torch.no_grad():
                                logits_ref, labels_ref = self._forward_logits_and_labels(
                                    self.ref_model, signal,
                                    all_full_ids[g], all_full_mask[g], all_full_labels[g],
                                    prompt_input_ids, prompt_attention_mask,
                                    ecg_embeddings=ref_ecg_emb,
                                )
                                token_logp_ref, _ = self._token_logp(logits_ref, labels_ref)
                            ref_tlp = token_logp_ref[:, :min_len].detach()
                            token_kl = (token_logp_pol - ref_tlp) * tmask_g
                            kl = token_kl.sum(dim=1).div(n_valid).mean()
                            policy_loss = policy_loss + beta * kl

                        batch_loss = batch_loss + policy_loss
                        batch_kl += float(kl.detach())

                # Average over group
                loss = batch_loss / group_size

                if step < 3 and self.config.is_ref_device:
                    print(f"[GRPO DEBUG] Step {step}: loss={loss.item():.4f}, "
                          f"kl_avg={batch_kl/group_size:.4f}")
                    if torch.cuda.is_available():
                        alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                        reserved_gb = torch.cuda.memory_reserved(device) / 1e9
                        print(f"[GRPO DEBUG] GPU memory: allocated={alloc_gb:.2f}GB, "
                              f"reserved={reserved_gb:.2f}GB")

                # Loss clipping: skip backward if loss is extreme
                loss_val = float(loss.detach())
                max_loss_threshold = 10.0
                if loss_val > max_loss_threshold or not torch.isfinite(loss):
                    if self.config.is_ref_device:
                        print(f"[GRPO WARNING] Step {step}: loss={loss_val:.4f} exceeds threshold "
                              f"{max_loss_threshold}, skipping backward pass")
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss_scaled = loss / max(1, grad_accum)
                    loss_scaled.backward()

                # Accumulate metrics
                with torch.no_grad():
                    self._accum_loss += loss_val * B
                    self._accum_reward += float(rewards.mean().detach()) * B
                    self._accum_reward_format += batch_r_format * B
                    self._accum_reward_diagnosis += batch_r_diagnosis * B
                    self._accum_reward_evidence += batch_r_evidence * B
                    self._accum_advantage_mean += float(advantages.mean().detach()) * B
                    self._accum_kl += (batch_kl / group_size) * B
                    self._accum_count += B

                # Re-enable train mode for subsequent generation (where eval is
                # toggled internally) — we exited the forward block in eval mode.
                self.model.train()

                # Gradient step
                if (step + 1) % grad_accum == 0:
                    grad_norm_threshold = 100.0
                    if max_grad_norm > 0:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm))
                    else:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")))

                    if grad_norm > grad_norm_threshold:
                        if self.config.is_ref_device:
                            print(f"[GRPO WARNING] Step {step}: grad_norm={grad_norm:.2f} exceeds "
                                  f"threshold {grad_norm_threshold}, skipping optimizer step")
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler is not None:
                            self.scheduler.step()
                    self.global_step += 1

                running_loss += loss.item()

                if step < 3 and self.config.is_ref_device:
                    total_step_time = time.time() - step_start_time
                    print(f"[GRPO DEBUG] Step {step}: Total step time={total_step_time:.1f}s")

                # Save checkpoint
                if self.global_step > 0 and self.global_step % save_interval == 0:
                    ckpt_path = f"{self.config.output_dir}/grpo_step_{self.global_step}.pt"
                    self._save_checkpoint(
                        model=self.model, optimizer=self.optimizer, epoch=epoch,
                        loss=running_loss / max(1, step + 1), checkpoint_path=ckpt_path,
                        step=self.global_step,
                    )

                # Log metrics
                if step % log_interval == 0 and self.config.is_ref_device and self._accum_count > 0:
                    step_time = time.time() - step_start_time
                    elapsed = time.time() - self.start_time
                    samples_per_sec = self._accum_count / max(0.001, step_time * log_interval)
                    progress = (epoch * len(self.train_dataloader) + step + 1) / total_steps
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    avg_loss = self._accum_loss / self._accum_count
                    avg_reward = self._accum_reward / self._accum_count
                    avg_r_fmt = self._accum_reward_format / self._accum_count
                    avg_r_diag = self._accum_reward_diagnosis / self._accum_count
                    avg_r_evid = self._accum_reward_evidence / self._accum_count
                    avg_kl = self._accum_kl / self._accum_count

                    metrics = {
                        "train/loss": avg_loss,
                        "train/reward_mean": avg_reward,
                        "train/reward_format": avg_r_fmt,
                        "train/reward_diagnosis": avg_r_diag,
                        "train/reward_evidence": avg_r_evid,
                        "train/advantage_mean": self._accum_advantage_mean / self._accum_count,
                        "train/kl": avg_kl,
                        "optim/lr": current_lr,
                        "optim/grad_norm": grad_norm,
                        "progress/global_step": float(self.global_step),
                        "progress/epoch": float(epoch),
                        "progress/epoch_step": float(step),
                        "progress/pct_complete": progress * 100,
                        "progress/samples_per_sec": samples_per_sec,
                        "progress/elapsed_sec": elapsed,
                    }
                    self._log_metrics(metrics)

                    print(
                        f"[GRPO] epoch {epoch} step {step}/{len(self.train_dataloader)} | "
                        f"loss={avg_loss:.4f} | "
                        f"reward={avg_reward:.3f} (fmt={avg_r_fmt:.2f} diag={avg_r_diag:.2f} evid={avg_r_evid:.2f}) | "
                        f"kl={avg_kl:.4f} | "
                        f"lr={current_lr:.2e} | grad_norm={grad_norm:.2f}"
                    )

                    # Log sample generations to wandb as a table
                    try:
                        import wandb
                        if wandb.run is not None and len(completion_texts) > 0:
                            rows = []
                            for b_idx in range(min(B, 2)):
                                gt_answer = _extract_answer(ground_truth_texts[b_idx])
                                gt_cot_snip = _extract_think(ground_truth_texts[b_idx], max_len=500)
                                for g_idx in range(min(group_size, 2)):
                                    gen_text = completion_texts[g_idx][b_idx]
                                    gen_answer = _extract_answer(gen_text)
                                    gen_cot_snip = _extract_think(gen_text, max_len=500)
                                    r_val = float(rewards[b_idx, g_idx])
                                    rc = reward_components[g_idx][b_idx]
                                    rows.append([
                                        step, b_idx, g_idx,
                                        gen_cot_snip[:500], gen_answer[:300],
                                        gt_cot_snip[:500], gt_answer[:300],
                                        f"{r_val:.3f}", f"{rc[0]:.1f}", f"{rc[1]:.2f}", f"{rc[2]:.2f}",
                                    ])
                            table = wandb.Table(
                                columns=["step", "sample", "group", "gen_cot", "gen_answer",
                                         "gt_cot", "gt_answer", "reward", "fmt", "diag", "evid"],
                                data=rows,
                            )
                            wandb.log({"generations": table})
                    except Exception:
                        pass

                    if len(completion_texts) > 0:
                        gen_sample = completion_texts[0][0]
                        gt_sample = ground_truth_texts[0]
                        gen_ans = _extract_answer(gen_sample)
                        gt_ans = _extract_answer(gt_sample)
                        gen_cot = _extract_think(gen_sample, max_len=300)
                        gt_cot = _extract_think(gt_sample, max_len=300)
                        print(f"  [Gen CoT]     {gen_cot[:200]}")
                        print(f"  [Gen answer]  {gen_ans[:200]}")
                        print(f"  [GT  CoT]     {gt_cot[:200]}")
                        print(f"  [GT  answer]  {gt_ans[:200]}")

                    self._reset_metrics()

            # End of epoch
            avg_loss = running_loss / max(1, len(self.train_dataloader))
            epoch_time = time.time() - epoch_start_time

            if self.config.is_ref_device:
                print(f"[GRPO] epoch {epoch} completed | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")
                self._log_metrics({
                    "epoch/loss": avg_loss,
                    "epoch/time_sec": epoch_time,
                    "epoch/num": float(epoch),
                })

            ckpt_path = f"{self.config.output_dir}/grpo_epoch_{epoch}.pt"
            self._save_checkpoint(
                model=self.model, optimizer=self.optimizer, epoch=epoch,
                loss=avg_loss, checkpoint_path=ckpt_path, step=self.global_step,
            )

        # Final
        if self.config.is_ref_device:
            total_time = time.time() - self.start_time
            print(f"[GRPO] Training complete | total_time={total_time:.1f}s | final_step={self.global_step}")
            self._log_metrics({
                "final/total_time_sec": total_time,
                "final/total_steps": float(self.global_step),
            })

    def inference(self):
        raise NotImplementedError("GRPO inference not implemented.")
