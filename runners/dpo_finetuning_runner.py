import math
import random
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from runners.base_runner import BaseRunner
from utils.enums import RunMode, RunnerName
from utils.registry import RunnerRegistry
from data.dpo_pair_dataset import DPOPairDataset, dpo_collate_fn


@RunnerRegistry.register(RunnerName.DPO_FINETUNING)
class DPOFinetuningRunner(BaseRunner):
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
        self.eval_dataloader: DataLoader | None = None

        # Accumulators for logging
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset metric accumulators for a new logging interval."""
        self._accum_loss = 0.0
        self._accum_dpo_loss = 0.0
        self._accum_sft_loss = 0.0
        self._accum_chosen_logp = 0.0
        self._accum_rejected_logp = 0.0
        self._accum_chosen_logp_ref = 0.0
        self._accum_rejected_logp_ref = 0.0
        self._accum_reward_margin = 0.0
        self._accum_accuracy = 0.0
        self._accum_count = 0

    def _sequence_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        shift_labels = shift_labels.clamp_min(0)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask
        return token_logp.sum(dim=-1)

    def _sequence_nll(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        shift_labels = shift_labels.clamp_min(0)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask
        token_counts = mask.sum(dim=-1).clamp_min(1)
        nll = -token_logp.sum(dim=-1) / token_counts
        return nll.mean()

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
            _, _, _, labels_out = merged
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_signal = ecg_signal.to(dtype=torch.float32)
        features = model.encoder(ecg_signal)
        quantized, indices, _ = model.quantizer(features)
        quantized_codes = model._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)

        if not hasattr(model.decoder, "_compute_ecg_embeddings"):
            raise ValueError("Decoder does not support ECG embeddings for DPO training.")

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

    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn: Callable,
    ) -> dict[str, float]:
        raise NotImplementedError

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

    def _build_eval_dataloader(self) -> Optional[DataLoader]:
        eval_path = getattr(self.config, "eval_pairs_path", None)
        if not eval_path:
            return None

        tokenizer = self._get_text_tokenizer()
        if tokenizer is None:
            raise ValueError("Tokenizer not available for DPO preference evaluation.")

        max_length = int(getattr(self.config, "max_token_length", None) or 640)
        dataset = DPOPairDataset(
            path=eval_path,
            tokenizer=tokenizer,
            config=self.config,
            max_length=max_length,
            waveform_key=self.config.waveform_key,
            prompt_key=self.config.prompt_key,
            chosen_key=self.config.chosen_key,
            rejected_key=self.config.rejected_key,
            weight_key=self.config.weight_key,
        )

        subset_size = int(getattr(self.config, "eval_subset_size", 0) or 0)
        if 0 < subset_size < len(dataset):
            seed = int(getattr(self.config, "eval_seed", 42))
            rng = random.Random(seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            dataset = Subset(dataset, indices[:subset_size])

        batch_size = int(getattr(self.config, "eval_batch_size", None) or self.config.batch_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=dpo_collate_fn,
        )

    def _run_pref_eval(
        self,
        device: torch.device,
        use_autocast: bool,
        autocast_dtype: torch.dtype,
    ) -> Optional[Dict[str, float]]:
        if self.eval_dataloader is None or self.model is None:
            return None

        self.model.eval()
        correct = 0.0
        total = 0.0
        margin_sum = 0.0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                signal = batch["signal"].to(device)
                prompt_input_ids = batch["prompt_input_ids"].to(device)
                prompt_attention_mask = batch["prompt_attention_mask"].to(device)

                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    logits_c, labels_c = self._forward_logits_and_labels(
                        self.model,
                        signal,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logits_r, labels_r = self._forward_logits_and_labels(
                        self.model,
                        signal,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logp_c = self._sequence_logp(logits_c, labels_c)
                    logp_r = self._sequence_logp(logits_r, labels_r)

                diff = logp_c - logp_r
                correct += float((diff > 0).float().sum().item())
                margin_sum += float(diff.sum().item())
                total += float(diff.numel())

        self.model.train()
        if total <= 0:
            return None

        return {
            "eval/pref_accuracy": correct / total,
            "eval/pref_margin": margin_sum / total,
            "eval/pairs": total,
        }

    def train(self):
        if self.train_dataloader is None or self.model is None or self.ref_model is None:
            raise ValueError("Training requires model, ref_model, and train_dataloader.")

        device = torch.device(f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.ref_model.to(device)

        precision = str(getattr(self.config, "precision", "bf16")).lower()
        use_autocast = precision in {"bf16", "fp16"}
        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        beta = float(getattr(self.config, "beta", 0.1))
        grad_accum = int(getattr(self.config, "grad_accum_steps", 1))
        max_grad_norm = float(getattr(self.config, "max_grad_norm", 1.0))
        save_interval = int(getattr(self.config, "save_interval", 500))
        log_interval = int(getattr(self.config, "log_interval", 10))
        eval_interval = int(getattr(self.config, "eval_interval_steps", 0) or 0)
        sft_weight = float(getattr(self.config, "sft_weight", 0.0))

        self.model.train()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.config.is_ref_device:
            print(f"[DPO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            self._log_metrics({
                "model/trainable_params": trainable_params,
                "model/total_params": total_params,
                "model/trainable_pct": 100 * trainable_params / total_params,
            })

        total_steps = len(self.train_dataloader) * int(self.config.num_epochs)
        self.start_time = time.time()

        if eval_interval > 0 and self.config.is_ref_device:
            self.eval_dataloader = self._build_eval_dataloader()
            metrics = self._run_pref_eval(device, use_autocast, autocast_dtype)
            if metrics:
                self._log_metrics(metrics)
                print(
                    f"[DPO] eval step {self.global_step} | "
                    f"pref_acc={metrics['eval/pref_accuracy']:.3f} | "
                    f"pref_margin={metrics['eval/pref_margin']:.3f}"
                )

        for epoch in range(int(self.config.num_epochs)):
            running_loss = 0.0
            epoch_start_time = time.time()
            self._reset_metrics()
            grad_norm = 0.0  # Initialize for logging before first gradient step

            for step, batch in enumerate(self.train_dataloader):
                step_start_time = time.time()
                signal = batch["signal"].to(device)
                prompt_input_ids = batch["prompt_input_ids"].to(device)
                prompt_attention_mask = batch["prompt_attention_mask"].to(device)
                weight = batch["weight"].to(device)

                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    logits_c, labels_c = self._forward_logits_and_labels(
                        self.model,
                        signal,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logits_r, labels_r = self._forward_logits_and_labels(
                        self.model,
                        signal,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                        prompt_input_ids,
                        prompt_attention_mask,
                    )
                    logp_c = self._sequence_logp(logits_c, labels_c)
                    logp_r = self._sequence_logp(logits_r, labels_r)

                    with torch.no_grad():
                        logits_c_ref, labels_c_ref = self._forward_logits_and_labels(
                            self.ref_model,
                            signal,
                            chosen_input_ids,
                            chosen_attention_mask,
                            chosen_labels,
                            prompt_input_ids,
                            prompt_attention_mask,
                        )
                        logits_r_ref, labels_r_ref = self._forward_logits_and_labels(
                            self.ref_model,
                            signal,
                            rejected_input_ids,
                            rejected_attention_mask,
                            rejected_labels,
                            prompt_input_ids,
                            prompt_attention_mask,
                        )
                        logp_c_ref = self._sequence_logp(logits_c_ref, labels_c_ref)
                        logp_r_ref = self._sequence_logp(logits_r_ref, labels_r_ref)

                    pi_logratio = logp_c - logp_r
                    ref_logratio = logp_c_ref - logp_r_ref
                    logits_diff = beta * (pi_logratio - ref_logratio)
                    dpo_loss = -torch.nn.functional.logsigmoid(logits_diff)
                    dpo_loss = (dpo_loss * weight).mean()
                    if sft_weight > 0:
                        sft_loss = self._sequence_nll(logits_c, labels_c)
                        loss = dpo_loss + sft_weight * sft_loss
                    else:
                        sft_loss = None
                        loss = dpo_loss

                    # Compute DPO-specific metrics for logging
                    with torch.no_grad():
                        reward_chosen = beta * (logp_c - logp_c_ref)
                        reward_rejected = beta * (logp_r - logp_r_ref)
                        reward_margin = (reward_chosen - reward_rejected).mean()
                        accuracy = (logits_diff > 0).float().mean()

                        # Accumulate metrics
                        batch_size = signal.size(0)
                        self._accum_loss += float(loss.detach()) * batch_size
                        self._accum_dpo_loss += float(dpo_loss.detach()) * batch_size
                        if sft_loss is not None:
                            self._accum_sft_loss += float(sft_loss.detach()) * batch_size
                        self._accum_chosen_logp += float(logp_c.mean().detach()) * batch_size
                        self._accum_rejected_logp += float(logp_r.mean().detach()) * batch_size
                        self._accum_chosen_logp_ref += float(logp_c_ref.mean().detach()) * batch_size
                        self._accum_rejected_logp_ref += float(logp_r_ref.mean().detach()) * batch_size
                        self._accum_reward_margin += float(reward_margin.detach()) * batch_size
                        self._accum_accuracy += float(accuracy.detach()) * batch_size
                        self._accum_count += batch_size

                loss_scaled = loss / max(1, grad_accum)
                loss_scaled.backward()

                if (step + 1) % grad_accum == 0:
                    # Compute gradient norm before clipping
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    if eval_interval > 0 and self.config.is_ref_device and self.global_step % eval_interval == 0:
                        metrics = self._run_pref_eval(device, use_autocast, autocast_dtype)
                        if metrics:
                            self._log_metrics(metrics)
                            print(
                                f"[DPO] eval step {self.global_step} | "
                                f"pref_acc={metrics['eval/pref_accuracy']:.3f} | "
                                f"pref_margin={metrics['eval/pref_margin']:.3f}"
                            )

                running_loss += float(loss.detach().cpu())

                # Save checkpoint at intervals
                if self.global_step > 0 and self.global_step % save_interval == 0:
                    ckpt_path = f"{self.config.output_dir}/dpo_step_{self.global_step}.pt"
                    self._save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=running_loss / max(1, step + 1),
                        checkpoint_path=ckpt_path,
                        step=self.global_step,
                    )

                # Log metrics at intervals
                if step % log_interval == 0 and self.config.is_ref_device and self._accum_count > 0:
                    step_time = time.time() - step_start_time
                    elapsed = time.time() - self.start_time
                    samples_per_sec = self._accum_count / max(0.001, step_time * log_interval)
                    progress = (epoch * len(self.train_dataloader) + step + 1) / total_steps

                    # Get current learning rate
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    metrics = {
                        # Training metrics
                        "train/loss": self._accum_loss / self._accum_count,
                        "train/dpo_loss": self._accum_dpo_loss / self._accum_count,
                        "train/chosen_logp": self._accum_chosen_logp / self._accum_count,
                        "train/rejected_logp": self._accum_rejected_logp / self._accum_count,
                        "train/chosen_logp_ref": self._accum_chosen_logp_ref / self._accum_count,
                        "train/rejected_logp_ref": self._accum_rejected_logp_ref / self._accum_count,
                        "train/reward_margin": self._accum_reward_margin / self._accum_count,
                        "train/accuracy": self._accum_accuracy / self._accum_count,
                        "train/logp_gap": (self._accum_chosen_logp - self._accum_rejected_logp) / self._accum_count,
                        # Optimization metrics
                        "optim/lr": current_lr,
                        "optim/grad_norm": grad_norm,
                        # Progress metrics
                        "progress/global_step": float(self.global_step),
                        "progress/epoch": float(epoch),
                        "progress/epoch_step": float(step),
                        "progress/pct_complete": progress * 100,
                        "progress/samples_per_sec": samples_per_sec,
                        "progress/elapsed_sec": elapsed,
                    }
                    if sft_weight > 0 and self._accum_sft_loss > 0:
                        metrics["train/sft_loss"] = self._accum_sft_loss / self._accum_count
                    self._log_metrics(metrics)

                    # Print progress
                    sft_str = ""
                    if sft_weight > 0 and self._accum_sft_loss > 0:
                        sft_str = f" | sft={self._accum_sft_loss / self._accum_count:.4f}"
                    print(
                        f"[DPO] epoch {epoch} step {step}/{len(self.train_dataloader)} | "
                        f"loss={self._accum_loss / self._accum_count:.4f} | "
                        f"dpo={self._accum_dpo_loss / self._accum_count:.4f}"
                        f"{sft_str} | "
                        f"acc={self._accum_accuracy / self._accum_count:.3f} | "
                        f"margin={self._accum_reward_margin / self._accum_count:.3f} | "
                        f"lr={current_lr:.2e} | "
                        f"{samples_per_sec:.1f} samples/sec"
                    )

                    # Reset accumulators
                    self._reset_metrics()

            # End of epoch logging
            avg_loss = running_loss / max(1, len(self.train_dataloader))
            epoch_time = time.time() - epoch_start_time

            if self.config.is_ref_device:
                print(f"[DPO] epoch {epoch} completed | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")
                self._log_metrics({
                    "epoch/loss": avg_loss,
                    "epoch/time_sec": epoch_time,
                    "epoch/num": float(epoch),
                })

            # Save epoch checkpoint
            ckpt_path = f"{self.config.output_dir}/dpo_epoch_{epoch}.pt"
            self._save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=avg_loss,
                checkpoint_path=ckpt_path,
                step=self.global_step,
            )

        # Final logging
        if self.config.is_ref_device:
            total_time = time.time() - self.start_time
            print(f"[DPO] Training complete | total_time={total_time:.1f}s | final_step={self.global_step}")
            self._log_metrics({
                "final/total_time_sec": total_time,
                "final/total_steps": float(self.global_step),
            })

    def inference(self):
        raise NotImplementedError("DPO inference not implemented.")
