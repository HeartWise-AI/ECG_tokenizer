import torch
import torch.distributed as dist
import torch.multiprocessing as MP
from torch.utils.data import DataLoader, Sampler
import torch.utils.data.distributed as DS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group
)
from typing import Any, List, Callable, Optional, Sequence
import numpy as np


class WeightedDistributedSampler(Sampler[int]):
    """
    Distributed sampler with per-sample weights for minority class upsampling.

    This sampler partitions data across GPUs like DistributedSampler, but applies
    weighted sampling within each partition to upsample minority classes.

    Args:
        dataset: Dataset to sample from
        weights: Per-sample weights (higher = more likely to be sampled)
        num_replicas: Number of distributed processes
        rank: Current process rank
        replacement: Whether to sample with replacement (required for weighted sampling)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset,
        weights: Sequence[float],
        num_replicas: int = 1,
        rank: int = 0,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        # Store weights as numpy array for efficient sampling
        self.weights = np.array(weights, dtype=np.float64)
        if len(self.weights) != len(dataset):
            raise ValueError(
                f"Length of weights ({len(self.weights)}) must match dataset length ({len(dataset)})"
            )

        # Calculate number of samples per replica
        self.num_samples = int(np.ceil(len(dataset) / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        # Set random seed based on epoch for reproducibility
        rng = np.random.default_rng(self.seed + self.epoch)

        # Normalize weights to probabilities
        probs = self.weights / self.weights.sum()

        # Sample with replacement based on weights
        indices = rng.choice(
            len(self.dataset),
            size=self.total_size,
            replace=self.replacement,
            p=probs,
        )

        # Get this rank's subset
        indices = indices[self.rank::self.num_replicas]

        # Shuffle within the rank's samples for additional randomization
        rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling across epochs."""
        self.epoch = epoch


class DistributedUtils:
    """Utility class for Distributed Data Parallel (DDP) operations"""
    
    MP = MP
    DS = DS
    DDP = DDP
    dist = dist
    
    @staticmethod
    def ddp_setup(
        gpu_id: int, 
        world_size: int
    ):
        """
        Setup DistributedDataParallel with explicit device ID
        
        Args:
            gpu_id: The GPU ID for this process
            world_size: The total number of GPUs
        """
        # Set the device
        torch.cuda.set_device(gpu_id)
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=gpu_id,
            world_size=world_size
        )
    
    @staticmethod
    def ddp_cleanup():
        """
        Cleanup the DistributedDataParallel.
        """
        if dist.is_available() and dist.is_initialized():
            destroy_process_group()

    @staticmethod
    def sync_process_group(
        world_size: int, 
        device_ids: int
    ):
        """
        Synchronize the process group across all devices.
        """
        if world_size > 1:
            dist.barrier(device_ids=[device_ids])
    
    @staticmethod
    def gather_loss(
        loss: list, 
        device: int
    ) -> float:
        """
        Gather the loss from all devices and return the mean loss.
        """
        loss_tensor = torch.tensor(loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.mean().item() / dist.get_world_size()

    @staticmethod
    def all_gather_object(
        gather_list: List[Any], 
        obj: Any
    ):
        """
        Gathers arbitrary picklable objects from all processes and stores them in gather_list.

        :param gather_list: Pre-allocated list with length equal to the world size.
        :param obj: The picklable object to be gathered from each process.
        """
        if len(gather_list) > 1:
            dist.all_gather_object(gather_list, obj)
        else:
            # If distributed is not initialized, store the object in the first index.
            gather_list[0] = obj
    
    @staticmethod
    def get_distributed_dataloader(
        dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        collate_fn: Callable | None = None,
        sample_weights: Optional[Sequence[float]] = None,
        weighted_sampling_seed: int = 42,
    ) -> DataLoader:
        """
        Create a distributed DataLoader with optional weighted sampling.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            num_replicas: Number of distributed processes (GPUs)
            rank: Current process rank
            shuffle: Whether to shuffle data (ignored if sample_weights provided)
            collate_fn: Optional custom collate function
            sample_weights: Optional per-sample weights for weighted sampling.
                When provided, uses WeightedDistributedSampler instead of
                DistributedSampler for minority class upsampling.
            weighted_sampling_seed: Random seed for weighted sampling reproducibility

        Returns:
            DataLoader configured for distributed training
        """
        if sample_weights is not None:
            # Use weighted sampling for minority class upsampling
            sampler = WeightedDistributedSampler(
                dataset,
                weights=sample_weights,
                num_replicas=num_replicas,
                rank=rank,
                replacement=True,
                seed=weighted_sampling_seed,
            )
        else:
            # Use standard distributed sampler
            sampler = DS.DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            collate_fn=collate_fn
        )
