
import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from models.gpt2_with_embeddings import GPT2WithEmbedding
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper

from utils.registry import RunnerRegistry


@RunnerRegistry.register("LLM_finetuning_runner")
class LLMFinetuningRunner:
    def __init__(
        self, 
        config: LLMFinetuningConfig, 
        wandb_wrapper: WandbWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: AdamW,
        scheduler: LRScheduler,
        scaler: GradScaler,
        model: GPT2WithEmbedding,
        loss_fn: torch.nn.Module
    ):
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader = val_dataloader
        self.optimizer: AdamW = optimizer
        self.scheduler: LRScheduler = scheduler
        self.scaler: GradScaler = scaler
        self.model: GPT2WithEmbedding = model
        self.loss_fn: torch.nn.Module = loss_fn

    def train(self):
        print("Training...")

    def validate(self):
        raise NotImplementedError("Validation not implemented")

    def save_model(self):
        raise NotImplementedError("Saving model not implemented")