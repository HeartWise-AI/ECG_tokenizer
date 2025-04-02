import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.enums import RunMode
from utils.config import ECGTokenizerTrainingConfig
from utils.wandb_wrapper import WandbWrapper
from models.tokenizer import ECG_Tokenizer_Wrapper

@RunnerRegistry.register("ECG_Tokenizer_Training")
class ECGTokenizerRunner:
    def __init__(
        self, 
        ecg_tokenizer: ECG_Tokenizer_Wrapper, 
        config: ECGTokenizerTrainingConfig, 
        train_dataloader: DataLoader, 
        validation_dataloader: DataLoader, 
        wandb_wrapper: WandbWrapper | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
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
        self.train_dataloader: DataLoader = train_dataloader
        self.validation_dataloader: DataLoader = validation_dataloader
        self.wandb_wrapper: WandbWrapper | None = wandb_wrapper
        self.optimizer: optim.Optimizer | None = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.scaler: GradScaler | None = scaler

    def execute(self, mode: RunMode):
        """
        Execute the pipeline in the desired mode.
        
        Modes:
            RunMode.TRAIN: Run the tokenization training (i.e., processing and logging metrics).
            RunMode.INFERENCE: Tokenize the input texts and output the results.
            RunMode.VALIDATE: Not implemented.
        """
        if mode == RunMode.TRAIN:
            self.train()
        elif mode == RunMode.INFERENCE:
            self.inference()
        elif mode == RunMode.VALIDATE:
            self.validate()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def train(self):
        """
        In "TRAIN" mode, iterate over the dataloader, tokenize texts, compute the average token length,
        log metrics, and save checkpoints (i.e., the tokenizer saved via save_pretrained).
        """
        best_avg_token_length: float = float('-inf')
        for epoch in range(1, self.config.num_epochs + 1):
            # Sync processes before starting the epoch.
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(mode=RunMode.TRAIN, epoch=epoch)
            
            # Log metrics via wandb if available and on the reference device.
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(epoch_metrics)
            
            # # Update best metric and save the tokenizer checkpoint if improved.
            # if self.config.is_ref_device and epoch_metrics.get("TRAIN/avg_token_length", 0) > best_avg_token_length:
            #     best_avg_token_length = epoch_metrics["TRAIN/avg_token_length"]
            #     self._save_tokenizer(epoch=epoch, best=True)
            
            # # Save regular tokenizer checkpoint for the current epoch.
            # self._save_tokenizer(epoch=epoch, best=False)
            
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
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.validation_dataloader
        step_fn: callable = self._train_step if mode == RunMode.TRAIN else self._val_step
                
        data_iter = tqdm(
            dataloader, 
            desc=f"{mode} epoch {epoch}/{self.config.num_epochs}",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        epoch_metrics: dict[str, float] = {}
        
        for batch_idx, batch in enumerate(data_iter):
            signals: torch.Tensor = batch["signal"].float().to(self.config.device)
            
            # Run the step function
            outputs: dict[str, torch.Tensor] = step_fn(
                signals=signals
            )
            
            # Accumulate metrics
            epoch_metrics["rec_loss"] = epoch_metrics.get("rec_loss", 0.0) + outputs["rec_loss"].item()
            epoch_metrics["cmt_loss"] = epoch_metrics.get("cmt_loss", 0.0) + outputs["cmt_loss"].mean().item()
            epoch_metrics["active_codes"] = epoch_metrics.get("active_codes", 0.0) + outputs["indices"].unique().numel() / self.config.codebook_size * 100
            
            # Calculate running means
            current_mean_rec = epoch_metrics["rec_loss"] / (batch_idx + 1)
            current_mean_cmt = epoch_metrics["cmt_loss"] / (batch_idx + 1)
            current_mean_active = epoch_metrics["active_codes"] / (batch_idx + 1)
            
            data_iter.set_postfix({
                "rec_loss": f"{outputs['rec_loss'].item():.4f}",
                "mean_rec": f"{current_mean_rec:.4f}",
                "cmt_loss": f"{outputs['cmt_loss'].mean().item():.4f}",
                "mean_cmt": f"{current_mean_cmt:.4f}",
                "active_codes": f"{outputs['indices'].unique().numel() / self.config.codebook_size * 100:.2f}",
                "mean_active": f"{current_mean_active:.2f}"
            })
            
            # Sync across processes.
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                        
        # Normalize the epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        
        # Return the epoch metrics
        return epoch_metrics
    
    def _train_step(
        self, 
        signals: torch.Tensor
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        
        alpha: float = 1.0
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16
        ):
            out, indices, cmt_loss = self.ecg_tokenizer(signals)
            rec_loss: torch.Tensor = (out - signals).abs().mean()
            combined_loss: torch.Tensor = rec_loss + alpha * cmt_loss.mean()
            
        # Backward pass with gradient scaling
        self.scaler.scale(combined_loss).backward()
        
        # Unscale gradients and apply gradient clipping
        self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.ecg_tokenizer.parameters(), max_norm=1.0)
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "rec_loss": rec_loss,
            "cmt_loss": cmt_loss.mean(),
            "indices": indices
        }

    def inference(self):
        """
        Inference is not implemented for the ECGTokenizerRunner.
        """
        raise NotImplementedError("Inference not implemented for ECGTokenizerRunner")

    def validate(self):
        """
        Validation is not implemented for the TokenizerRunner.
        """
        raise NotImplementedError("Validate not implemented for TokenizerRunner")

    def _save_tokenizer(self, epoch: int, best: bool = False):
        """
        Save the tokenizer configuration to the designated output directory.
        
        Args:
            epoch (int): The current epoch number.
            best (bool): Whether this checkpoint is the best so far.
        """
        save_dir = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"tokenizer_epoch_{epoch}"
        if best:
            save_name += "_best"
        save_path = os.path.join(save_dir, save_name)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved tokenizer to {save_path}")
        
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
            self.wandb_wrapper.log({
                "checkpoint/epoch": epoch,
                "checkpoint/tokenizer_best": best
            })
