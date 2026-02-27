import logging
import re
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from runners.base_runner import BaseRunner
from utils.enums import RunMode, RunnerName
from utils.registry import RunnerRegistry
from utils.rewards import (
    BertDiagnosisReward,
    compute_rewards,
    diagnosis_accuracy_reward,
    format_reward,
    key_evidence_reward,
)

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
class GRPOFinetuningRunner(BaseRunner):
    def __init__(
        self,
        config,
        wandb_wrapper=None,
        model=None,
        ref_model=None,
        train_dataloader: DataLoader | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        super().__init__(config, wandb_wrapper)
        self.model = model
        self.ref_model = ref_model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0
        self.start_time = None
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

    def _token_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute per-token log-probabilities and mask.

        Returns:
            token_logp: [B, seq_len-1] per-token log-probs (0 where masked)
            mask: [B, seq_len-1] boolean mask for valid (non -100) tokens
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        shift_labels = shift_labels.clamp_min(0)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask
        return token_logp, mask

    def _sequence_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sequence log-probabilities (sum of token log-probs)."""
        token_logp, mask = self._token_logp(logits, labels)
        return token_logp.sum(dim=-1)

    def _expand_labels_with_ecg(
        self,
        decoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        ecg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        merged = decoder._inject_ecg_after_image_token(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ecg_embeddings=ecg_embeddings,
            embed_layer=decoder.llm_model.get_input_embeddings(),
        )
        if merged is not None:
            _, _, _, labels_out = merged[:4]
            if labels_out is None:
                raise ValueError("Failed to expand labels with ECG injection.")
            return labels_out

        prefix_len = int(ecg_embeddings.size(1))
        if prefix_len <= 0:
            return labels
        ignore_pad = torch.full(
            (labels.size(0), prefix_len),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([ignore_pad, labels], dim=1)

    def _forward_logits_and_labels(
        self,
        model,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
    ) -> tuple:
        ecg_signal = ecg_signal.to(dtype=torch.float32)
        features = model.encoder(ecg_signal)
        quantized, indices, _ = model.quantizer(features)
        quantized_codes = model._extract_primary_codes(
            indices, model.num_codebooks_kept, model.codebook_offset
        )

        ecg_embeddings, _ = model.decoder._compute_ecg_embeddings(
            quantized,
            quantized_codes,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )

        outputs = model.decoder(
            quantized_features=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            quantized_codes=None,
            ecg_embeddings=ecg_embeddings,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        logits = outputs["logits"]
        labels_expanded = self._expand_labels_with_ecg(
            decoder=model.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ecg_embeddings=ecg_embeddings,
        )
        return logits, labels_expanded

    def _build_completion_labels(
        self,
        completion_ids: torch.Tensor,
        prompt_len: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Build labels for a completion: mask prompt tokens with -100, keep completion tokens."""
        labels = completion_ids.clone()
        labels[:, :prompt_len] = -100
        labels = labels.masked_fill(completion_ids == pad_token_id, -100)
        return labels

    def _get_text_tokenizer(self):
        if self.model is None:
            return None
        if hasattr(self.model, "_get_text_tokenizer"):
            try:
                return self.model._get_text_tokenizer()
            except Exception:
                pass
        decoder = getattr(self.model, "decoder", None)
        return getattr(decoder, "tokenizer", None)

    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn: Callable,
    ) -> dict[str, float]:
        raise NotImplementedError

    def train(self):
        if self.train_dataloader is None or self.model is None:
            raise ValueError("Training requires model and train_dataloader.")

        device = torch.device(f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ref_model is not None:
            self.ref_model.to(device)

        # Ensure models use SDPA attention (flash_attention_2 may not work with
        # the installed transformers + flash-attn kernel combination).
        # We need to patch the config on ALL sub-models that have one.
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

        # Initialize BERT diagnosis reward (tiny 0.1B model, stays on same GPU)
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
                # Batch generation in chunks of 2 groups to avoid OOM
                # (B*G=8 with full KV cache can exceed GPU memory)
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
                # Track per-component rewards for logging
                batch_r_format = 0.0
                batch_r_diagnosis = 0.0
                batch_r_evidence = 0.0

                for g in range(group_size):
                    gen_ids = all_generated_ids[g]
                    texts_g = []
                    for b in range(B):
                        text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
                        texts_g.append(text)
                        r = compute_rewards(text, ground_truth_texts[b], reward_weights, bert_reward=bert_reward)
                        rewards[b, g] = r
                        batch_r_format += format_reward(text, ground_truth_texts[b])
                        batch_r_diagnosis += bert_reward(text, ground_truth_texts[b])
                        batch_r_evidence += key_evidence_reward(text, ground_truth_texts[b])
                    completion_texts.append(texts_g)

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
                    # Print a snippet of the first completion for the first sample
                    snippet = completion_texts[0][0][:300]
                    print(f"  Sample completion (group 0, sample 0): {snippet}...")
                    # Print ground truth snippet
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
                # 4. Build full sequences (prompt + generation) and compute
                #    old log-probs (from generation step, no gradient)
                # ----------------------------------------------------------
                # generate_report_with_question returns ONLY new tokens,
                # so we need to concatenate prompt_input_ids + generated_ids
                # to form the full sequence for the forward pass.
                prompt_len = prompt_input_ids.size(1)

                # Pre-build full sequences for each group member
                all_full_ids = []   # list of [B, prompt_len + gen_len] tensors
                all_full_mask = []
                all_full_labels = []
                for g in range(group_size):
                    gen_ids = all_generated_ids[g]  # [B, gen_len]
                    # Concatenate prompt + generated tokens
                    full_ids = torch.cat([prompt_input_ids, gen_ids], dim=1)
                    full_mask = (full_ids != pad_token_id).long()
                    # Labels: mask prompt tokens with -100, keep only generated tokens
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

                # Compute old per-token log-probs (from generation step, no gradient)
                old_token_logps = []  # list of (token_logp, mask) per group
                logp_start = time.time()
                with torch.no_grad():
                    for g in range(group_size):
                        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                            logits, labels_exp = self._forward_logits_and_labels(
                                self.model,
                                signal,
                                all_full_ids[g],
                                all_full_mask[g],
                                all_full_labels[g],
                                prompt_input_ids,
                                prompt_attention_mask,
                            )
                            tlp, tmask = self._token_logp(logits, labels_exp)
                            old_token_logps.append((tlp, tmask))
                if step < 3 and self.config.is_ref_device:
                    logp_time = time.time() - logp_start
                    seq_logps = [tlp.sum(dim=-1) for tlp, _ in old_token_logps]
                    print(f"[GRPO DEBUG] Step {step}: Old log-probs computed in {logp_time:.1f}s, "
                          f"seq_values={[[f'{seq_logps[g][b].item():.1f}' for g in range(group_size)] for b in range(min(B,2))]}")

                # ----------------------------------------------------------
                # 5. Policy gradient with PER-TOKEN clipped objective
                #    (DeepSeek-R1 GRPO formula: per-token ratios, per-token clipping,
                #     sequence-level advantage, average over valid tokens)
                # ----------------------------------------------------------
                batch_loss = torch.tensor(0.0, device=device)
                batch_kl = 0.0
                use_ref = beta > 0  # skip ref model entirely when KL penalty is off

                for g in range(group_size):
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        # Policy forward (with gradient)
                        logits_policy, labels_policy = self._forward_logits_and_labels(
                            self.model,
                            signal,
                            all_full_ids[g],
                            all_full_mask[g],
                            all_full_labels[g],
                            prompt_input_ids,
                            prompt_attention_mask,
                        )
                        token_logp_policy, token_mask = self._token_logp(logits_policy, labels_policy)

                        # Get old per-token log-probs for this group
                        old_tlp, _ = old_token_logps[g]
                        # Align shapes (old may differ slightly due to label expansion)
                        min_len = min(token_logp_policy.size(1), old_tlp.size(1))
                        token_logp_pol = token_logp_policy[:, :min_len]
                        old_tlp_g = old_tlp[:, :min_len].detach()
                        tmask_g = token_mask[:, :min_len]

                        # Per-token ratio and clipping
                        token_ratio = torch.exp(token_logp_pol - old_tlp_g)
                        clipped_token_ratio = torch.clamp(token_ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

                        # Sequence-level advantage broadcast to tokens
                        adv_g = advantages[:, g].detach().unsqueeze(1)  # [B, 1]

                        surr1 = token_ratio * adv_g
                        surr2 = clipped_token_ratio * adv_g
                        token_surrogate = torch.min(surr1, surr2)

                        # Average over valid tokens per sequence, then mean over batch
                        # (1/|o_i| normalization from GRPO formula)
                        n_valid = tmask_g.sum(dim=1).clamp_min(1)  # [B]
                        per_seq_loss = -(token_surrogate * tmask_g).sum(dim=1) / n_valid
                        policy_loss = per_seq_loss.mean()

                        # Optional per-token KL penalty
                        kl = torch.tensor(0.0, device=device)
                        if use_ref:
                            with torch.no_grad():
                                logits_ref, labels_ref = self._forward_logits_and_labels(
                                    self.ref_model,
                                    signal,
                                    all_full_ids[g],
                                    all_full_mask[g],
                                    all_full_labels[g],
                                    prompt_input_ids,
                                    prompt_attention_mask,
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

                # Loss clipping: skip backward if loss is extreme (prevents divergence)
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

                # Gradient step
                if (step + 1) % grad_accum == 0:
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                    # Skip optimizer step if grad norm is extreme (spike protection)
                    grad_norm_threshold = 100.0
                    if grad_norm > grad_norm_threshold:
                        if self.config.is_ref_device:
                            print(f"[GRPO WARNING] Step {step}: grad_norm={grad_norm:.2f} exceeds "
                                  f"threshold {grad_norm_threshold}, skipping optimizer step")
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler is not None:
                            self.scheduler.step()
                    self.global_step += 1

                running_loss += float(loss.detach().cpu())

                if step < 3 and self.config.is_ref_device:
                    total_step_time = time.time() - step_start_time
                    print(f"[GRPO DEBUG] Step {step}: Total step time={total_step_time:.1f}s")

                # Save checkpoint
                if self.global_step > 0 and self.global_step % save_interval == 0:
                    ckpt_path = f"{self.config.output_dir}/grpo_step_{self.global_step}.pt"
                    self._save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=running_loss / max(1, step + 1),
                        checkpoint_path=ckpt_path,
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

                    # Console output with reward breakdown
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
                            for b_idx in range(min(B, 2)):  # log up to 2 samples
                                gt_answer = _extract_answer(ground_truth_texts[b_idx])
                                gt_cot_snip = _extract_think(ground_truth_texts[b_idx], max_len=500)
                                for g_idx in range(min(group_size, 2)):  # 2 completions per sample
                                    gen_text = completion_texts[g_idx][b_idx]
                                    gen_answer = _extract_answer(gen_text)
                                    gen_cot_snip = _extract_think(gen_text, max_len=500)
                                    r_val = float(rewards[b_idx, g_idx])
                                    rows.append([
                                        step, b_idx, g_idx,
                                        gen_cot_snip[:500],
                                        gen_answer[:300],
                                        gt_cot_snip[:500],
                                        gt_answer[:300],
                                        f"{r_val:.3f}",
                                        f"{float(format_reward(gen_text, ground_truth_texts[b_idx])):.1f}",
                                        f"{float(bert_reward(gen_text, ground_truth_texts[b_idx])):.2f}",
                                        f"{float(key_evidence_reward(gen_text, ground_truth_texts[b_idx])):.2f}",
                                    ])
                            table = wandb.Table(
                                columns=["step", "sample", "group", "gen_cot", "gen_answer",
                                         "gt_cot", "gt_answer", "reward", "fmt", "diag", "evid"],
                                data=rows,
                            )
                            wandb.log({"generations": table})
                    except Exception:
                        pass

                    # Also print a sample generation + ground truth to console
                    if len(completion_texts) > 0:
                        gen_sample = completion_texts[0][0]  # group 0, sample 0
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
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=avg_loss,
                checkpoint_path=ckpt_path,
                step=self.global_step,
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
