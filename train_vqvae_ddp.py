import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from data.dataset import ECGDataset
from models.models import (
    VQVAE, 
    SimpleVQAutoEncoder, 
    ResVQAutoEncoder
)
import os
import tqdm
import wandb
import yaml
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Author: Rohan Banerjee

Relevant issues from lucid-rains repos: #28, #44, #102
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    checkpoint_dir: str = 'checkpoints/'
) -> None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path: str = os.path.join(checkpoint_dir, f'vqvae_iteration_{iteration}.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_dir: str = 'checkpoints/',
    checkpoint_path: str | None = None
) -> tuple[int, nn.Module, optim.Optimizer]:
    if checkpoint_path:
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint: torch.Tensor = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # Dynamically adjust keys if necessary
        state_dict: torch.Tensor = checkpoint['model_state_dict']
        model_state_dict: torch.Tensor = model.state_dict()
        new_state_dict: dict = {}
        
        # for key in state_dict:
        #     if key.startswith("module.") and not any(k.startswith("module.") for k in model_state_dict.keys()):
        #         new_key = key[len("module."):]
        #     elif not key.startswith("module.") and any(k.startswith("module.") for k in model_state_dict.keys()):
        #         new_key = "module." + key
        #     else:
        #         new_key = key
        #     new_state_dict[new_key] = state_dict[key]

        # Load the state dict with strict=False
        
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
    
        optimizer_state_dict: dict = checkpoint['optimizer_state_dict']
        optimizer.param_groups.clear()
        param_list: list[torch.Tensor] = list(model.parameters()) 
        for group in optimizer_state_dict['param_groups']:
            valid_params: list[torch.Tensor] = [param_list[idx] for idx in group['params'] if isinstance(idx, int) and idx < len(param_list)]
            if valid_params:
                group['params'] = valid_params
                optimizer.add_param_group(group)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        optimizer.load_state_dict(optimizer_state_dict)

        print(f"Checkpoint {checkpoint_path} loaded successfully.")

    return checkpoint['iteration'], model, optimizer

def evaluate(
    model: nn.Module, 
    data_loader: DataLoader, 
    rank: int
) -> float:
    logging.debug("Starting evaluation...")
    model.eval()
    total_loss: float = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            logging.debug(f"Evaluating batch {i+1}/{len(data_loader)}")
            signals: torch.Tensor = batch['signal'].float().to(rank)
            with autocast():
                out, _, _ = model(signals)
                rec_loss: torch.Tensor = (out - signals).abs().mean()
            total_loss += rec_loss.item()
    model.train()
    logging.debug("Evaluation completed.")
    return total_loss / len(data_loader)

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_codes: int,
    checkpoint_dir: str,
    num_epochs: int = 10,
    alpha: float = 1.0,
    start_epoch: int = 0,
    rank: int = 0
) -> None:
    model.train()
    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)  # Important for proper shuffling
        
        # Track epoch metrics
        epoch_rec_loss = 0.0
        epoch_cmt_loss = 0.0
        epoch_active_codes = 0.0
        n_batches = len(train_loader)
        
        # Only show progress bar on rank 0
        if rank == 0:
            progress_bar = tqdm.tqdm(train_loader, total=len(train_loader), 
                                   desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(train_loader):
            signals: torch.Tensor = batch["signal"].float().to(rank)
            optimizer.zero_grad()
            out, indices, cmt_loss = model(signals)
            rec_loss: torch.Tensor = (out - signals).abs().mean()
            combined_loss: torch.Tensor = rec_loss + alpha * cmt_loss.mean()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Add barrier for gradient synchronization
            if dist.get_world_size() > 1:
                dist.barrier()

            # Accumulate metrics
            epoch_rec_loss += rec_loss.item()
            epoch_cmt_loss += cmt_loss.mean().item()
            epoch_active_codes += indices.unique().numel() / num_codes * 100

            # Calculate running means
            current_mean_rec = epoch_rec_loss / (batch_idx + 1)
            current_mean_cmt = epoch_cmt_loss / (batch_idx + 1)
            current_mean_active = epoch_active_codes / (batch_idx + 1)

            if rank == 0:
                progress_bar.set_postfix({
                    "rec_loss": f"{rec_loss.item():.4f}",
                    "mean_rec": f"{current_mean_rec:.4f}",
                    "cmt_loss": f"{cmt_loss.mean().item():.4f}",
                    "mean_cmt": f"{current_mean_cmt:.4f}",
                    "active": f"{indices.unique().numel() / num_codes * 100:.2f}%",
                    "mean_active": f"{current_mean_active:.2f}%"
                })
                progress_bar.update(1)

        torch.cuda.empty_cache()
        if rank == 0:
            progress_bar.close()

        # Calculate epoch averages
        epoch_rec_loss /= n_batches
        epoch_cmt_loss /= n_batches
        epoch_active_codes /= n_batches

        # Synchronize before logging
        if dist.get_world_size() > 1:
            dist.barrier()
        
        # Log epoch summary only on rank 0
        if rank == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            logging.info(f"Average Reconstruction Loss: {epoch_rec_loss:.4f}")
            logging.info(f"Average Commitment Loss: {epoch_cmt_loss:.4f}")
            logging.info(f"Average Active Codes (%): {epoch_active_codes:.2f}")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "epoch_avg_rec_loss": epoch_rec_loss,
                "epoch_avg_cmt_loss": epoch_cmt_loss,
                "epoch_avg_active_codes": epoch_active_codes
            })

        # Synchronize before evaluation
        if dist.get_world_size() > 1:
            dist.barrier()

        # Only save checkpoints and evaluate on rank 0
        if rank == 0:
            try:
                test_loss = evaluate(model, test_loader, rank)
                wandb.log({"test_loss": test_loss, "epoch": epoch + 1})
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
                model.train()
            except Exception as e:
                raise Exception(f"An error occurred during evaluation: {e}")
        
        # Synchronize after evaluation before next epoch
        if dist.get_world_size() > 1:
            dist.barrier()

    print("Training complete!")

def setup(rank, world_size):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        raise Exception(f"Failed to initialize process group: {e}")

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, config):
    try:
        setup(rank, world_size)
        
        # Initialize wandb only on rank 0
        if rank == 0:
            wandb.init(
                project="ECG_tokenizer",
                entity="jacques-delfrate",
                config=config
            )
        
        # Create datasets
        dataset_mimic_train = ECGDataset(parquet_file=config["train_parquet_file"])
        dataset_mimic_test = ECGDataset(parquet_file=config["test_parquet_file"])
        
        # Create model and move it to GPU with id rank
        model = ResVQAutoEncoder(
            timesteps=dataset_mimic_train.waveform_length,
            codebook_size=config["num_codes"],
            implicit_neural_codebook=True
        ).to(rank)
        
        model = DDP(model, device_ids=[rank])
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        
        # Create sampler for DDP
        train_sampler = DistributedSampler(dataset_mimic_train)
        test_sampler = DistributedSampler(dataset_mimic_test, shuffle=False)
        
        # Update DataLoader to use sampler
        train_loader = DataLoader(
            dataset_mimic_train,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=4,  # Reduce num_workers per process
            pin_memory=True
        )
        
        test_loader = DataLoader(
            dataset_mimic_test,
            batch_size=config["batch_size"],
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        train(
            model=model, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            optimizer=optimizer, 
            num_codes=config["num_codes"], 
            checkpoint_dir=config["checkpoint_dir"], 
            num_epochs=config["num_epochs"], 
            alpha=config["alpha"], 
            start_epoch=config["start_epoch"],
            rank=rank
        )
    except Exception as e:
        logging.error(f"Error in worker {rank}: {e}")
        raise e
    finally:
        cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--base_config', type=str, help='Path to base_config file', required=True)
    args = parser.parse_args()

    with open(args.base_config, 'r') as file:
        config = yaml.safe_load(file)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}")
        return
        
    world_size = 2  # Number of GPUs you want to use
    
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
