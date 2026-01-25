import os
import wandb
import heapq
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Any, Callable
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LRScheduler

from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.wandb_wrapper import WandbWrapper
from utils.config import ECGTokenizerTrainingConfig
from utils.metrics.ecg_metrics import compute_metrics
from utils.schedulers import scheduler_is_per_iteration
from utils.enums import RunMode, DecoderMode, RunnerName
from utils.constants import DEEPECG_CATEGORIES, ECG_PATTERNS
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from models.local_residual_vq import ResidualVQ
from runners.base_runner import BaseRunner


@RunnerRegistry.register(RunnerName.ECG_TOKENIZER_TRAINING)
@RunnerRegistry.register(RunnerName.ECG_TOKENIZER_LINEAR_PROBING)
class ECGTokenizerRunner(BaseRunner):
    def __init__(
        self, 
        ecg_tokenizer: ECG_Tokenizer_Wrapper, 
        config: ECGTokenizerTrainingConfig, 
        scaler: GradScaler | None = None,
        scheduler: LRScheduler | None = None,
        optimizer: Optimizer | None = None,
        wandb_wrapper: WandbWrapper | None = None,
        train_dataloader: DataLoader | None = None, 
        validation_dataloader: DataLoader | None = None,
        embedding_extraction_dataloader: DataLoader | None = None,
    ):
        """
        Initialize the TokenizerRunner.
        
        Args:
            tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer.
            config (TokenizerRunnerConfig): Configuration parameters.
            dataloader: A DataLoader yielding batches of data (expected to have a "text" field).
            wandb_wrapper (WandbWrapper, optional): Wandb logging wrapper.
        """
        self.ecg_tokenizer: ECG_Tokenizer_Wrapper = ecg_tokenizer
        self.config: ECGTokenizerTrainingConfig = config
        self.train_dataloader: DataLoader | None = train_dataloader
        self.validation_dataloader: DataLoader | None = validation_dataloader
        self.wandb_wrapper: WandbWrapper | None = wandb_wrapper
        self.optimizer: Optimizer | None = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.scaler: GradScaler | None = scaler
        self.scheduler_per_iteration: bool = scheduler_is_per_iteration(self.config)
        self.criterion: nn.Module = nn.BCEWithLogitsLoss()
        self.decoder_mode: DecoderMode = getattr(self.config, 'decoder_mode', DecoderMode.RECONSTRUCTION)
        self.embedding_extraction_dataloader: DataLoader | None = embedding_extraction_dataloader
        
    def execute(
        self, 
        mode: RunMode
    ):
        """
        Execute the runner in the specified mode.
        
        Args:
            mode: The execution mode (TRAIN, INFERENCE, VALIDATE)
        """
        super().execute(mode)
        
    def train(self):
        """
        In "TRAIN" mode, iterate over the dataloader, tokenize texts, compute the average token length,
        log metrics, and save checkpoints (i.e., the tokenizer saved via save_pretrained).
        """
        best_rec_loss: float = float('inf')
        for epoch in range(1, self.config.num_epochs + 1):
            # Sync processes before starting the epoch.
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Run training epoch
            epoch_metrics: dict[str, float] = self._run_epoch(
                mode=RunMode.TRAIN, 
                epoch=epoch
            )
            
            # Log metrics via wandb if available and on the reference device.
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(
                    epoch_metrics
                )
            
            # Step the scheduler if it should be updated per-epoch
            if self.scheduler and (not self.scheduler_per_iteration):
                self.scheduler.step()
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(
                mode=RunMode.VALIDATE,
                epoch=epoch
            )
            
            # Log metrics via wandb if available and on the reference device.
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(
                    epoch_metrics
                )
            
            # Update best metric and save the tokenizer checkpoint if improved.
            if self.config.is_ref_device:
                loss_key: str = f"{RunMode.VALIDATE}/cls_loss" if self.decoder_mode == DecoderMode.CLASSIFICATION else f"{RunMode.VALIDATE}/rec_loss"
                if epoch_metrics[loss_key] < best_rec_loss:
                    best_rec_loss = epoch_metrics[loss_key]
                    self._save_model(
                        epoch=epoch,
                        loss=best_rec_loss,
                        checkpoint_path= os.path.join(
                            self.config.output_dir, 
                            f'best_model_epoch_{epoch}.pt'
                        )
                    )
            
                # Save regular tokenizer checkpoint for the current epoch.
                self._save_model(
                    epoch=epoch,
                    loss=epoch_metrics[loss_key],
                    checkpoint_path=os.path.join(
                        self.config.output_dir, 
                        f'checkpoint_epoch_{epoch}.pt'
                    )
                )
            
            # Sync after the epoch.
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )

    def _run_epoch(
        self, 
        mode: RunMode, 
        epoch: int
    ) -> dict[str, float]:
        """
        Run a single epoch over the dataloader and compute average token length.
        
        Args:
            mode (str): Execution mode (e.g., "TRAIN").
            epoch (int): Current epoch number.
            
        Returns:
            dict[str, float]: Dictionary containing metrics (e.g., average token length).
        """
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE]
        
        # Set the model to training or evaluation mode
        self.ecg_tokenizer.train(mode == RunMode.TRAIN)
        
        # Verify frozen components are handled correctly
        if self.config.is_ref_device:
            print("\n" + "="*50)
            print("VERIFYING FROZEN COMPONENT STATUS:")
            print("="*50)
            self.ecg_tokenizer.module._check_frozen_components_status()
            print("="*50 + "\n")
        
        if self.train_dataloader is None or self.validation_dataloader is None:
            raise ValueError("Train or validation dataloader is not set")
        
        # Get the dataloader and step function
        dataloader: DataLoader | None= self.train_dataloader if mode == RunMode.TRAIN else self.validation_dataloader
        step_fn: Callable = self._train_step if mode == RunMode.TRAIN else self._val_step
                
        data_iter = tqdm(
            dataloader, 
            desc=f"[GPU {self.config.device}]: {mode} epoch {epoch}/{self.config.num_epochs}",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Determine if we're in classification mode using DecoderMode enum       
        if self.decoder_mode == DecoderMode.CLASSIFICATION:
            epoch_losses = {
                "cls_loss": torch.zeros(1, device=self.config.device),
                "cmt_loss": torch.zeros(1, device=self.config.device),
                "active_codes": torch.zeros(1, device=self.config.device),
                "combined_loss": torch.zeros(1, device=self.config.device)
            }
            all_predictions = []
            all_labels = []
        else:
            epoch_losses = {
                "rec_loss": torch.zeros(1, device=self.config.device),
                "cmt_loss": torch.zeros(1, device=self.config.device),
                "active_codes": torch.zeros(1, device=self.config.device),
                "combined_loss": torch.zeros(1, device=self.config.device)
            }
        
        k_batches: int = 1
        max_signals_plot: int = 1
        # For worst indices (max heap - keeping highest losses)
        worst_heap: list[tuple[float, torch.Tensor, torch.Tensor]] = []  # Will store (loss, input_signal, reconstructed_signal) tuples
        # For best indices (min heap - keeping lowest losses)
        best_heap: list[tuple[float, torch.Tensor, torch.Tensor]] = []   # Will store (loss, input_signal, reconstructed_signal) tuples

        lr_metrics: dict[str, float] = {}

        for batch_idx, batch in enumerate(data_iter, 1):
            signals: torch.Tensor = batch["signal"].float().to(self.config.device)
            labels: torch.Tensor | None = batch["labels"].float().to(self.config.device) if "labels" in batch else None
            outputs: dict[str, torch.Tensor] = step_fn(signals=signals, labels=labels)

            for key, value in outputs.items():
                if key.startswith("lr_"):
                    lr_metrics[key] = lr_metrics.get(key, 0) + float(value)
            
            if self.decoder_mode == DecoderMode.CLASSIFICATION:
                # Classification mode
                epoch_losses["cls_loss"] += outputs["cls_loss"].item()
                epoch_losses["cmt_loss"] += outputs["cmt_loss"].item()
                epoch_losses["active_codes"] += outputs["indices"].unique().numel() / self.config.codebook_size * 100
                epoch_losses["combined_loss"] += outputs["combined_loss"].item()

                # Collect predictions and labels for metrics calculation
                all_predictions.append(outputs["predictions"].detach().cpu())
                all_labels.append(labels.detach().cpu()) if labels is not None else None
                
                # Update progress bar with current mean metrics
                data_iter.set_postfix({
                    "mean_cls_loss": f"{(epoch_losses['cls_loss'] / batch_idx).item():.4f}",
                    "mean_cmt_loss": f"{(epoch_losses['cmt_loss'] / batch_idx).item():.4f}",
                    "mean_active_codes": f"{(epoch_losses['active_codes'] / batch_idx).item():.2f}%",
                    "mean_combined_loss": f"{(epoch_losses['combined_loss'] / batch_idx).item():.4f}"
                })
            else:
                # Reconstruction mode
                # Store a copy of the signals to avoid reference issues
                rec_loss = outputs["rec_loss"].item()
                input_signal = signals.detach().cpu()
                reconstructed_signal = outputs["reconstruction"].detach().cpu()
                
                ## For worst losses (keep k highest losses)
                if len(worst_heap) < k_batches:
                    heapq.heappush(worst_heap, (rec_loss, input_signal, reconstructed_signal))
                elif rec_loss > worst_heap[0][0]:  # If current loss is worse than smallest in heap
                    heapq.heapreplace(worst_heap, (rec_loss, input_signal, reconstructed_signal))
                    
                # For best losses (keep k lowest losses)
                if len(best_heap) < k_batches:
                    heapq.heappush(best_heap, (rec_loss, input_signal, reconstructed_signal))
                elif rec_loss < best_heap[0][0]:  # If current loss is better than largest in heap
                    heapq.heapreplace(best_heap, (rec_loss, input_signal, reconstructed_signal))
    
                # Accumulate metrics on GPU
                epoch_losses["rec_loss"] += outputs["rec_loss"]
                epoch_losses["cmt_loss"] += outputs["cmt_loss"]
                epoch_losses["active_codes"] += outputs["indices"].unique().numel() / self.config.codebook_size * 100
                epoch_losses["combined_loss"] += outputs["combined_loss"]
                
                # Update progress bar with current mean metrics
                data_iter.set_postfix({
                    "mean_rec_loss": f"{(epoch_losses['rec_loss'] / batch_idx).item():.4f}",
                    "mean_cmt_loss": f"{(epoch_losses['cmt_loss'] / batch_idx).item():.4f}",
                    "mean_active_codes": f"{(epoch_losses['active_codes'] / batch_idx).item():.2f}%",
                    "mean_combined_loss": f"{(epoch_losses['combined_loss'] / batch_idx).item():.4f}"
                })
            
            # Sync across processes.
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )

        # Gather and normalize metrics once at the end of epoch
        gathered_metrics = {}
        for k, v in epoch_losses.items():
            # Normalize the accumulated losses
            gathered_metrics[f"{mode}/{k}"] = DistributedUtils.gather_loss(
                [v.item()], 
                self.config.device
            ) / len(dataloader)
        
        # Add averaged learning rate metrics to gathered_metrics gathered across GPUs
        for key, value in lr_metrics.items():
            gathered_metrics[key] = DistributedUtils.gather_loss(
                [value], 
                self.config.device
            ) / len(dataloader)
        
        # For classification mode, calculate additional metrics
        if self.decoder_mode == DecoderMode.CLASSIFICATION and all_predictions and all_labels:
            all_preds = torch.cat(all_predictions, dim=0)
            all_lbls = torch.cat(all_labels, dim=0)
            
            # Convert tensors to numpy arrays and then to pandas DataFrames
            pred_df = pd.DataFrame(
                all_preds.float().cpu().numpy(),
                columns=ECG_PATTERNS
            )
            labels_df = pd.DataFrame(
                all_lbls.float().cpu().numpy(),
                columns=ECG_PATTERNS
            )
            
            # Compute metrics
            metrics_dict = compute_metrics(labels_df, pred_df)
            
            # Add metrics to gathered_metrics
            for category in DEEPECG_CATEGORIES:
                for metric_name in metrics_dict[category]:
                    gathered_metrics[f"{mode}/{category}_{metric_name}"] = metrics_dict[category][metric_name]
        
        # Plot and log best/worst reconstructions if wandb is initialized and we're on reference device
        # and we're in reconstruction mode
        if self.decoder_mode == DecoderMode.RECONSTRUCTION and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            # Get the worst and best cases (no need for sorting since we're only using one example)
            worst_case = max(worst_heap, key=lambda x: x[0])  # Get entry with highest loss
            best_case = min(best_heap, key=lambda x: x[0])   # Get entry with lowest loss
            
            # Create figures for best and worst cases
            fig, axs = plt.subplots(self.config.num_leads, 2, figsize=(20, 30))
            
            # Get the signal pair and loss
            loss, input_sig, recon_sig = worst_case

            # Convert to numpy and ensure float32, take first signal and first lead
            for lead in range(self.config.num_leads):
                orig_sig = input_sig[0, lead].float().numpy()
                recon_sig_lead = recon_sig[0, lead].float().numpy()
                axs[lead, 0].plot(orig_sig, 'b-', label='Original', alpha=0.7)
                axs[lead, 0].set_title(f'Worst Case - Original Signal (Lead {lead})')
                axs[lead, 0].legend()
                axs[lead, 0].grid(True)
                
                axs[lead, 1].plot(recon_sig_lead, 'r-', label='Reconstruction', alpha=0.7)
                axs[lead, 1].set_title(f'Worst Case - Reconstruction (Lead {lead}, Loss: {loss:.4f})')
                axs[lead, 1].legend()
                axs[lead, 1].grid(True)
                
                plt.tight_layout()
            
            # Log to wandb
            self.wandb_wrapper.log({
                f"{mode}/worst_reconstructions": wandb.Image(fig),
                f"{mode}/worst_loss": loss
            })
            
            plt.close(fig)
            
            # Get the signal pair and loss
            loss, input_sig, recon_sig = best_case

            # Create figures for best and worst cases
            fig, axs = plt.subplots(self.config.num_leads, 2, figsize=(20, 30))
            
            # Convert to numpy and ensure float32, take first signal and first lead
            for lead in range(self.config.num_leads):
                orig_sig = input_sig[0, lead].float().numpy()
                recon_sig_lead = recon_sig[0, lead].float().numpy()
                axs[lead, 0].plot(orig_sig, 'b-', label='Original', alpha=0.7)
                axs[lead, 0].set_title(f'Best Case - Original Signal (Lead {lead})')
                axs[lead, 0].legend()
                axs[lead, 0].grid(True)
                
                axs[lead, 1].plot(recon_sig_lead, 'r-', label='Reconstruction', alpha=0.7)
                axs[lead, 1].set_title(f'Best Case - Reconstruction (Lead {lead}, Loss: {loss:.4f})')
                axs[lead, 1].legend()
                axs[lead, 1].grid(True)
                
                plt.tight_layout()
            
            # Log to wandb
            self.wandb_wrapper.log({
                f"{mode}/best_reconstructions": wandb.Image(fig),
                f"{mode}/best_loss": loss
            })
            
            plt.close(fig)
            
        # Return the epoch metrics
        return gathered_metrics
    
    def _train_step(
        self, 
        signals: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        
        assert self.optimizer is not None
        assert self.scaler is not None
        
        self.optimizer.zero_grad()
        
        alpha: float = 1.0
        with autocast('cuda', dtype=torch.bfloat16):
            # Get the output, indices, and commit loss
            out, indices, cmt_loss, _ = self.ecg_tokenizer(signals)
            # Check the mode using DecoderMode enum
            decoder_mode = getattr(self.config, 'decoder_mode', DecoderMode.RECONSTRUCTION)
            decoder_mode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)
            
            # Get learning rate metrics
            lr_metrics = {}
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"lr_{pg['name']}"] = pg["lr"]
                else:
                    # Fallback for any unnamed groups
                    lr_metrics[f"lr_group_{id(pg) % 1000}"] = pg["lr"]
            
            if decoder_mode == DecoderMode.CLASSIFICATION and labels is not None:
                # Classification mode                
                cls_loss = self.criterion(out, labels)
                combined_loss = cls_loss + alpha * cmt_loss.mean()
                
                # Apply sigmoid to get probabilities for metrics calculation
                pred_probs = torch.sigmoid(out)
                
                # Backward pass with gradient scaling
                self.scaler.scale(combined_loss).backward()
                
                # Unscale gradients and apply gradient clipping
                self.scaler.unscale_(self.optimizer)
                
                # Sync gradients across processes before optimizer step
                DistributedUtils.sync_process_group(
                    world_size=self.config.world_size,
                    device_ids=self.config.device
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                                
                # Step the scheduler if it should be updated per-iteration
                if self.scheduler and self.scheduler_per_iteration:
                    self.scheduler.step()
                    
                return {
                    "cls_loss": cls_loss,
                    "cmt_loss": cmt_loss.mean(),
                    "combined_loss": combined_loss,
                    "indices": indices,
                    "predictions": pred_probs,
                    **lr_metrics
                }
            else:
                # Reconstruction mode
                rec_loss: torch.Tensor = (out - signals).abs().mean()
                combined_loss: torch.Tensor = rec_loss + alpha * cmt_loss.mean()
                
                # Backward pass with gradient scaling
                if self.scaler is not None:
                    self.scaler.scale(combined_loss).backward()
                
                # Unscale gradients and apply gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Sync gradients across processes before optimizer step
                DistributedUtils.sync_process_group(
                    world_size=self.config.world_size,
                    device_ids=self.config.device
                )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                                
                # Step the scheduler if it should be updated per-iteration
                if self.scheduler and self.scheduler_per_iteration:
                    self.scheduler.step()
                    
                return {
                    "rec_loss": rec_loss,
                    "cmt_loss": cmt_loss.mean(),
                    "combined_loss": combined_loss,
                    "indices": indices,
                    "reconstruction": out,
                    **lr_metrics
                }

    @torch.no_grad()
    def _val_step(
        self, 
        signals: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        alpha: float = 1.0
        
        lr_metrics = {}
        for pg in self.optimizer.param_groups if self.optimizer else []:
            if "name" in pg:
                lr_metrics[f"lr_{pg['name']}"] = pg["lr"]
            else:
                # Fallback for any unnamed groups
                lr_metrics[f"lr_group_{id(pg) % 1000}"] = pg["lr"]
        
        with autocast('cuda', dtype=torch.bfloat16):
            out, indices, cmt_loss, _ = self.ecg_tokenizer(signals)
            
            # Check the mode using DecoderMode enum
            decoder_mode = getattr(self.config, 'decoder_mode', DecoderMode.RECONSTRUCTION)
            decoder_mode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)          
            
            if decoder_mode == DecoderMode.CLASSIFICATION:
                # Classification mode
                cls_loss = self.criterion(out, labels)
                combined_loss = cls_loss + alpha * cmt_loss.mean()
                pred_probs = torch.sigmoid(out)
                return {
                    "cls_loss": cls_loss,
                    "cmt_loss": cmt_loss.mean(),
                    "combined_loss": combined_loss,
                    "indices": indices,
                    "predictions": pred_probs,
                }
            else:
                # Reconstruction mode
                rec_loss: torch.Tensor = (out - signals).abs().mean()
                combined_loss: torch.Tensor = rec_loss + alpha * cmt_loss.mean()

                return {
                    "rec_loss": rec_loss,
                    "cmt_loss": cmt_loss.mean(),
                    "combined_loss": combined_loss,
                    "indices": indices,
                    "reconstruction": out,
                    **lr_metrics
                }

    def extract_embeddings(self):
        """
        Extract embeddings from the model.
        """
        save_dir: str = os.path.join(self.config.output_dir, "embeddings")
        os.makedirs(save_dir, exist_ok=True)
        
        for batch in tqdm(self.embedding_extraction_dataloader):
            signals = batch['signal'].float().to(self.config.device)
            waveform_path = batch['waveform_path']
            residual_vq_layer = None
    
            with torch.no_grad():
                out, indices, cmt_loss = self.ecg_tokenizer(signals)
                for i, layer in enumerate(self.ecg_tokenizer.module.layers):
                    if isinstance(layer, ResidualVQ):
                        residual_vq_layer = layer
                        break

            if residual_vq_layer is None:
                raise RuntimeError("No ResidualVQ layer found in the model")
                
            batch_embeddings = residual_vq_layer.get_codes_from_indices(indices)
            
            for idx in range(len(waveform_path)):
                single_embedding = batch_embeddings[:, idx:idx+1, :, :]
                single_embedding = single_embedding.squeeze(1)
                embedding_np = single_embedding.detach().cpu().numpy()
                
                original_filename = os.path.basename(waveform_path[idx])
                filename_without_ext = os.path.splitext(original_filename)[0]
                save_path = os.path.join(save_dir, f"{filename_without_ext}_embedding.npy")
                np.save(save_path, embedding_np)

    def inference(self):
        """
        Run inference by reusing the validation path.
        """
        if self.validation_dataloader is None:
            raise ValueError("Validation dataloader is not set for inference.")

        # Ensure model is in eval mode
        self.ecg_tokenizer.eval()

        # Reuse validation epoch logic (epoch index 0 for logging)
        metrics = self._run_epoch(
            mode=RunMode.VALIDATE,
            epoch=0
        )

        # Log metrics on ref device
        if self.wandb_wrapper and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            self.wandb_wrapper.log({f"Inference/{k}": v for k, v in metrics.items()})

        return metrics

    def validate(self):
        """
        Validation is not implemented for the TokenizerRunner.
        """
        raise NotImplementedError("Validate not implemented for TokenizerRunner")

    def _save_model(
        self,
        epoch: int,
        loss: float,
        checkpoint_path: str
    ):
        """
        Save the tokenizer configuration to the designated output directory.
        
        Args:
            epoch (int): The current epoch number.
            best (bool): Whether this checkpoint is the best so far.
        """
        save_dir = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # Prepare checkpoint - get the underlying model's state dict for DDP models
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': self.ecg_tokenizer.module.state_dict() if hasattr(self.ecg_tokenizer, 'module') else self.ecg_tokenizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'config': self.config
        }
                
        # Save regular model for current epoch
        torch.save(checkpoint, checkpoint_path)

        # Delete the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint_path: str = checkpoint_path.replace(f'epoch_{epoch}', f'epoch_{epoch - 1}')
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"Deleted old checkpoint: {prev_checkpoint_path}")
        
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
            # Get current learning rates
            lr_metrics = {}
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"checkpoint/lr_{pg['name']}"] = pg["lr"]
                else:
                    # Fallback for any unnamed groups
                    lr_metrics[f"checkpoint/lr_group_{id(pg) % 1000}"] = pg["lr"]
            
            self.wandb_wrapper.log({
                "checkpoint/epoch": epoch,
                "checkpoint/loss": loss,
                **lr_metrics
            })
