import os
import torch
import torch.distributed as dist
import torch.multiprocessing as MP
from torch.utils.data import DataLoader
import torch.utils.data.distributed as DS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group, 
    destroy_process_group
)
from typing import Any, List


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
        torch.distributed.init_process_group(
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
            torch.distributed.barrier(device_ids=[device_ids])
    
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
        collate_fn: callable = None
    ) -> DataLoader:
        sampler: DS.DistributedSampler = DS.DistributedSampler(
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