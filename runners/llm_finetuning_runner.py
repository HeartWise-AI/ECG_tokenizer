import os
import torch
import pandas as pd
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from transformers import GPT2Tokenizer

from utils.enums import RunMode, RunnerName
from utils.ddp import DistributedUtils
from utils.registry import (
    RunnerRegistry,
    MetricRegistry
)
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from utils.schedulers import scheduler_is_per_iteration
from utils.metrics.llm_metrics import (
    RougeMetric,
    BleuMetric,
    MeteorMetric,
    update_best_metric,
    update_worst_metric,
    update_random_batch_metric
)
from runners.base_runner import BaseRunner
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper

import random
from tqdm import tqdm
from typing import (
    Any, 
    Union, 
    Callable
)


@RunnerRegistry.register(RunnerName.LLM_FINETUNING)
class LLMFinetuningRunner(BaseRunner):
    def __init__(
        self, 
        model: ECG_Tokenizer_Wrapper,
        config: LLMFinetuningConfig, 
        validation_dataloader: DataLoader,
        wandb_wrapper: WandbWrapper | None = None,
        train_dataloader: DataLoader | None = None,
        optimizer: AdamW | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
    ):
        self.model: ECG_Tokenizer_Wrapper = model
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper | None = wandb_wrapper
        self.train_dataloader: DataLoader | None = train_dataloader
        self.validation_dataloader: DataLoader | None = validation_dataloader
        self.optimizer: AdamW | None = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.scaler: GradScaler | None = scaler
        self.scheduler_per_iteration: bool = scheduler_is_per_iteration(self.config)
        
    def execute(
        self, 
        mode: RunMode
    ):
        super().execute(mode)
        
    def train(self):
        if self.optimizer is None:
            raise ValueError("Optimizer cannot be None")
        # Note: Scaler is not required for bfloat16 training
        
        best_val_loss: float = float('inf')
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Sync before starting each epoch
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(
                RunMode.TRAIN,
                epoch
            )
                        
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
                RunMode.VALIDATE,
                epoch
            )
            
            # Save best model (only on reference device)
            if self.config.is_ref_device:
                if epoch_metrics[f'{RunMode.VALIDATE}/loss'] < best_val_loss:
                    best_val_loss = epoch_metrics[f'{RunMode.VALIDATE}/loss']
                    self._save_model(
                        epoch=epoch,
                        loss=epoch_metrics[f'{RunMode.VALIDATE}/loss'],
                        is_best=True
                    )
                
                # Also save regular checkpoint
                self._save_model(
                    epoch=epoch,
                    loss=epoch_metrics[f'{RunMode.VALIDATE}/loss'],
                    is_best=False
                )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Add learning rate metrics from training epoch metrics
                lr_metrics = {}
                for key, value in epoch_metrics.items():
                    if "lr_" in key:
                        lr_metrics[key] = value
                
                self.wandb_wrapper.log({
                    **epoch_metrics,
                    **lr_metrics,
                    f"{RunMode.VALIDATE}/best_loss": best_val_loss
                })
                
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int
    )->dict[str, float]:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE]
        
        # Set the model to training or evaluation mode
        self.model.train(mode == RunMode.TRAIN)
        
        if self.train_dataloader is None or self.validation_dataloader is None:
            raise ValueError("Train or validation dataloader is not set")
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.validation_dataloader
        step_fn: Callable | None = self._train_step if mode == RunMode.TRAIN else self._val_step
        
        # Create a progress bar for the epoch
        data_iter: tqdm = tqdm(
            dataloader, 
            desc=f"{mode} epoch {epoch}/{self.config.num_epochs}",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        # Initialize the total loss
        total_loss: float = 0.0
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Iterate over the dataloader
        epoch_metrics: dict[str, float] = {}
        
        if mode == RunMode.VALIDATE:
            worst_batch_metrics, best_batch_metrics, random_batch_metrics, random_batch_idx = self._init_validation_metrics(dataloader)
        
        for batch_idx, batch in enumerate(data_iter):            
            # Preprocess the batch
            ecg_signal: torch.Tensor = batch['signal'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = input_ids.clone()
            
            # Run the step function
            outputs: dict[str, torch.Tensor] | torch.Tensor = step_fn(
                ecg_signal=ecg_signal, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # initialize metrics
            metrics: dict[str, float] = {}
            metrics['loss'] = outputs['loss'].item()
            
            # Extract learning rate metrics
            for key, value in outputs.items():
                if key.startswith('lr_'):
                    metrics[key] = float(value) if isinstance(value, torch.Tensor) else value
            
            # Compute rouge score, bleu score, and meteor score
            if mode == RunMode.VALIDATE:
                # Metrics Rouge, Bleu, and Meteor are computed on the reference device but aggregated across all GPUs later
                metrics.update(
                    # TODO: best_batch_metrics, worst_batch_metrics and random_batch_metrics are computed on the reference device
                    # TODO: we need to gather them across all GPUs
                    self._compute_metrics( # this function returns mean metrics for the current batch
                        outputs,
                        labels,
                        dataloader,
                        best_batch_metrics, # parsed and updated by reference object - not returned
                        worst_batch_metrics, # parsed and updated by reference object - not returned
                        random_batch_metrics, # parsed and updated by reference object - not returned
                        random_batch=random_batch_idx == batch_idx
                    )
                )                  
            
            # Gather and average loss across all GPUs
            gathered_metrics: dict[str, float] = {}
            for k in metrics:
                gathered_metrics[f"{mode}/{k}"] = DistributedUtils.gather_loss(
                    [metrics[k]], 
                    self.config.device
                )
                            
            # Update the epoch metrics
            for k, v in gathered_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
            
            # Update the total loss with gathered loss
            total_loss += gathered_metrics[f'{mode}/loss']
            mean_loss: float = total_loss / (batch_idx + 1)
            
            # Log the loss to wandb
            if mode == RunMode.TRAIN:
                if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    log_dict = {
                        f"{mode}/loss": gathered_metrics[f'{mode}/loss'],  # Log the gathered loss for current batch
                        f"{mode}/mean_loss": mean_loss,  # Log the running mean loss
                    }
                    
                    # Add learning rate metrics to log_dict
                    for key, value in gathered_metrics.items():
                        if f"{mode}/lr_" in key:
                            log_dict[key] = value
                            
                    self.wandb_wrapper.log(log_dict)
            
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Update progress bar with gathered losses
            data_iter.set_postfix({
                f"{mode}/loss": f'{gathered_metrics[f"{mode}/loss"]:.4f}',
                f"{mode}/mean_loss": f'{mean_loss:.4f}'
            })
        
        # === New Block: Log best and worst metrics as HTML to wandb ===
        if mode == RunMode.VALIDATE and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            # helper function to create an HTML table given the metrics dictionary
            def create_html_table(metrics_dict, table_title):
                html = f"<h3>{table_title}</h3>"
                html += "<table border='1' cellspacing='0' cellpadding='5'>"
                html += "<tr><th>Metric</th><th>Score</th><th>Prediction(s)</th><th>Reference(s)</th></tr>"
                for metric, records in metrics_dict.items():
                    if records:
                        # pick the first record (since K=1)
                        record = records[0]
                        # Assuming record is a dict with keys: 'score', 'predictions', and 'references'
                        score = record['score']
                        # if they are lists, join them with a line break
                        predictions = "<br>".join([f"{i + 1}: {p}" for i, p in enumerate(record['predictions'])])
                        references = "<br>".join([f"{i + 1}: {r}" for i, r in enumerate(record['references'])])
                    else:
                        score, predictions, references = "", "", ""
                    html += f"<tr><td>{metric}</td><td>{score}</td><td>{predictions}</td><td>{references}</td></tr>"
                html += "</table>"
                return html

            import wandb
            best_html = create_html_table(best_batch_metrics, "Best Metrics")
            worst_html = create_html_table(worst_batch_metrics, "Worst Metrics")
            random_html = create_html_table(random_batch_metrics, "Random Metrics")
            self.wandb_wrapper.log({
                "val/best_metrics_html": wandb.Html(best_html),
                "val/worst_metrics_html": wandb.Html(worst_html),
                "val/random_metrics_html": wandb.Html(random_html)
            })
        # === End new block ===
                
        # Normalize the epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        
        # Return the epoch metrics
        return epoch_metrics

    def _train_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Clear gradients
        assert self.optimizer is not None
        
        self.optimizer.zero_grad()
        
        # Forward pass with autocast for mixed precision using bfloat16
        with autocast('cuda', dtype=torch.bfloat16):
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss: torch.Tensor = outputs['loss']

        # Backward pass - bfloat16 doesn't need gradient scaling
        loss.backward()
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Step optimizer directly without gradient scaling for bfloat16
        self.optimizer.step()
                
        # Get learning rate metrics
        lr_metrics = {}
        for pg in self.optimizer.param_groups if self.optimizer else []:
            if "name" in pg:
                lr_metrics[f"lr_{pg['name']}"] = pg["lr"]
        
        # Step the scheduler if it should be updated per-iteration
        if self.scheduler and self.scheduler_per_iteration:
            self.scheduler.step()
        
        return {
            "loss": loss,
            **lr_metrics
        }

    def _val_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss: torch.Tensor = outputs['loss']
            
            # Time the generate_report function
            if hasattr(self.model, 'module'):
                generated_ids: torch.Tensor = self.model.module.generate_report(
                    ecg_signal, 
                    max_token_length=self.config.max_token_length
                )
            else:
                generated_ids: torch.Tensor = self.model.generate_report(
                    ecg_signal, 
                    max_token_length=self.config.max_token_length
                )

            # Get learning rate metrics
            lr_metrics = {}
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"lr_{pg['name']}"] = pg["lr"]

            return {
                "loss": loss,
                "generated_ids": generated_ids,
                **lr_metrics
            }

    def _inference_step(
        self,
        ecg_signal: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.model, 'module'):
                generated_ids: torch.Tensor = self.model.module.generate_report(
                    ecg_signal=ecg_signal, 
                    max_token_length=self.config.max_token_length
                )
            else:
                generated_ids: torch.Tensor = self.model.generate_report(
                    ecg_signal=ecg_signal, 
                    max_token_length=self.config.max_token_length
                )   
        return generated_ids     

    def inference(self):
        if self.validation_dataloader is None:
            raise ValueError("Validation dataloader is not set")
            
        self.model.eval()
        
        predicted_reports: list[str] = []
        reference_reports: list[str] = []
        waveform_names: list[str] = []
        tokenizer: GPT2Tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore
        
        for batch in tqdm(self.validation_dataloader, desc="Inference", total=len(self.validation_dataloader), disable=not self.config.is_ref_device):
            ecg_signal: torch.Tensor = batch['signal'].to(self.config.device)
            labels: torch.Tensor = batch['input_ids'].to(self.config.device)
            batch_waveform_names: list[str] = batch['waveform_name']
            generated_ids: torch.Tensor = self._inference_step(
                ecg_signal=ecg_signal,
            )
                        
            for gen, lab, filename in zip(generated_ids, labels, batch_waveform_names):
                # Decode both predictions and references as strings.
                decoded_prediction = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
                decoded_reference  = tokenizer.decode(lab.tolist(), skip_special_tokens=True)
                predicted_reports.append(decoded_prediction)
                reference_reports.append(decoded_reference)
                waveform_names.append(filename)
                
        # --- Distributed Gathering using DistributedUtils ---
        results = {
            "waveform_names": waveform_names,
            "predicted_reports": predicted_reports,
            "reference_reports": reference_reports,
        }
        
        # Get the world size from DistributedUtils (using the config's world size)
        gathered_results = [None for _ in range(self.config.world_size)]
        DistributedUtils.all_gather_object(gathered_results, results)
        
        # Only the reference device (as determined by self.config.is_ref_device) consolidates and writes the output CSV.
        if self.config.is_ref_device:
            combined_waveform_names = []
            combined_predicted_reports = []
            combined_reference_reports = []
            for res in gathered_results:
                combined_waveform_names.extend(res["waveform_names"] if res is not None else [])
                combined_predicted_reports.extend(res["predicted_reports"] if res is not None else [])
                combined_reference_reports.extend(res["reference_reports"] if res is not None else [])
                    
            df = pd.DataFrame({
                'waveform_name': combined_waveform_names,
                'predicted_report': combined_predicted_reports,
                'reference_report': combined_reference_reports
            })
            os.makedirs(self.config.output_dir, exist_ok=True)
            csv_path = os.path.join(self.config.output_dir, 'inference.csv')
            df.to_csv(csv_path, index=False)

    def validate(self):
        raise NotImplementedError("Validate not implemented")

    def _save_model(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint and optionally mark as best model."""
        save_dir: str = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint - get the underlying model's state dict for DDP models
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint for current epoch
        checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Delete the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch - 1}.pt')
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"Deleted old checkpoint: {prev_checkpoint_path}")
        
        # If this is the best model, save it separately
        if is_best:
            best_model_path: str = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
            # Get current learning rates
            lr_metrics = {}
            if self.optimizer is not None:
                for pg in self.optimizer.param_groups:
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

    def _compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        dataloader: DataLoader,
        best_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        worst_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        random_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        random_batch: bool = False,
    ) -> dict[str, float]:
        """Compute metrics for validation and update best/worst batch metrics."""
        computed_metrics: dict[str, float] = {}
        for metric in self.config.metrics:
            registered_metrics: Union[
                RougeMetric, 
                BleuMetric, 
                MeteorMetric
            ] = MetricRegistry.get(metric)
            LLM_metrics: dict[str, Union[float, list[str]]] = registered_metrics.compute_score(
                outputs['generated_ids'],
                labels,
                dataloader.dataset.tokenizer  # type: ignore
            )
            for metric_name, metric_value in LLM_metrics.items():
                if metric_name not in ('predictions', 'references'):
                    # Update the best metric
                    update_best_metric(
                        metric_name=metric_name,
                        llm_metrics=LLM_metrics,
                        best_metrics=best_batch_metrics,
                        K=1  # TODO: k > 1 implemented but haven't been tested
                    )
                    # Update the worst metric
                    update_worst_metric(
                        metric_name=metric_name,
                        llm_metrics=LLM_metrics,
                        worst_metrics=worst_batch_metrics,
                        K=1  # TODO: k > 1 implemented but haven't been tested with k > 1
                    )
                    
                    if random_batch:
                        # Update the random batch metric
                        update_random_batch_metric(
                            metric_name=metric_name,
                            llm_metrics=LLM_metrics,
                            random_metrics=random_batch_metrics,
                            k=1  # TODO: k > 1 implemented but haven't been tested with k > 1
                        )
                    
                    metric_value = float(metric_value) if isinstance(metric_value, torch.Tensor) else metric_value
                    if isinstance(metric_value, float):
                        computed_metrics[metric_name] = float(metric_value)
                    
        return computed_metrics
    
    def _init_validation_metrics(
            self,
            dataloader: DataLoader
        ) -> tuple[
            dict[str, list[dict[str, Union[float, list[str]]]]],
            dict[str, list[dict[str, Union[float, list[str]]]]],
            dict[str, list[dict[str, Union[float, list[str]]]]],
            int
        ]:
            # Initialize dictionaries for best, worst, and random batch metrics.
            worst_batch_metrics = {}
            best_batch_metrics = {}
            random_batch_metrics = {}
            # Select a random batch index
            random_batch_idx = random.randint(0, len(dataloader) - 1)
            return worst_batch_metrics, best_batch_metrics, random_batch_metrics, random_batch_idx    