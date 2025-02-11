import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist
import argparse

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    data_loader: DataLoader
) -> float:
    logging.debug("Starting evaluation...")
    model.eval()
    total_loss: float = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            logging.debug(f"Evaluating batch {i+1}/{len(data_loader)}")
            signals: torch.Tensor = batch['signal'].float().to(device)
            with autocast():
                out, indices, cmt_loss = model(signals)
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
    start_epoch: int = 0
) -> None:
    model.train()
    for epoch in range(start_epoch, num_epochs):
        # Track epoch metrics
        epoch_rec_loss = 0.0
        epoch_cmt_loss = 0.0
        epoch_active_codes = 0.0
        n_batches = len(train_loader)
        
        progress_bar = tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(train_loader):
            signals: torch.Tensor = batch["signal"].float().to(device)
            optimizer.zero_grad()
            out, indices, cmt_loss = model(signals)
            rec_loss: torch.Tensor = (out - signals).abs().mean()
            combined_loss: torch.Tensor = rec_loss + alpha * cmt_loss.mean()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            epoch_rec_loss += rec_loss.item()
            epoch_cmt_loss += cmt_loss.mean().item()
            epoch_active_codes += indices.unique().numel() / num_codes * 100

            # Calculate running means
            current_mean_rec = epoch_rec_loss / (batch_idx + 1)
            current_mean_cmt = epoch_cmt_loss / (batch_idx + 1)
            current_mean_active = epoch_active_codes / (batch_idx + 1)

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
        progress_bar.close()

        # Log epoch summary
        epoch_rec_loss /= n_batches
        epoch_cmt_loss /= n_batches
        epoch_active_codes /= n_batches
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        logging.info(f"Average Reconstruction Loss: {epoch_rec_loss:.4f}")
        logging.info(f"Average Commitment Loss: {epoch_cmt_loss:.4f}")
        logging.info(f"Average Active Codes (%): {epoch_active_codes:.2f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_rec_loss": epoch_rec_loss,
            "epoch_avg_cmt_loss": epoch_cmt_loss,
            "epoch_avg_active_codes": epoch_active_codes
        })

        if (epoch + 1) % 1 == 0:
            try:
                test_loss = evaluate(model, test_loader)
                wandb.log({"test_loss": test_loss, "epoch": epoch + 1})
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
                model.train()  # Ensure model is back in training mode
            except Exception as e:
                raise Exception(f"An error occurred during evaluation: {e}")

    print("Training complete!")

def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train VQVAE model.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--base_config', type=str, help='Path to base_config file', required=True)
    args: argparse.Namespace = parser.parse_args()

    with open(args.base_config, 'r') as file:
        config = yaml.safe_load(file)

    wandb.init(
        project="ECG_tokenizer", 
        entity="jacques-delfrate", 
        name=config["experiment_name"], 
        config=config
    )

    dataset_mimic_train: ECGDataset = ECGDataset(parquet_file='/media/data1/datasets/DeepECG/SSL_pretraining/split/MIMIC/mimic_v4_clean_train.parquet')
    # dataset_mhi_train: ECGDataset = ECGDataset(parquet_file="/media/data1/muse_ge/train_trial_v1.1.parquet", test_size=0, split='train')
    train_loader: DataLoader = DataLoader(
        dataset_mimic_train, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=16
    )

    # combined_train_dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset([dataset_mimic_train, dataset_mhi_train])
    # combined_train_loader: DataLoader = DataLoader(combined_train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=16)

    dataset_mimic_test: ECGDataset = ECGDataset(parquet_file='/media/data1/datasets/DeepECG/SSL_pretraining/split/MIMIC/mimic_v4_clean_test.parquet')
    test_loader: DataLoader = DataLoader(
        dataset_mimic_test, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=16
    )

    lr: float = float(config["learning_rate"])
    num_epochs: int = config["num_epochs"]
    num_codes: int = config["num_codes"]
    seed: int = config["seed"]
    checkpoint_dir: str = f"checkpoints/{config["experiment_name"]}"
    torch.random.manual_seed(seed)
    model: ResVQAutoEncoder = ResVQAutoEncoder(
        timesteps=dataset_mimic_train.waveform_length,
        codebook_size=num_codes,
        implicit_neural_codebook=True
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using GPUs 2 and 3")
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        model = nn.DataParallel(model)

    opt: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # start_iteration, model, optimizer = load_checkpoint(model, opt, checkpoint_dir='checkpoints/', checkpoint_path=args.checkpoint_path)
    
    train(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        optimizer=opt, 
        num_codes=num_codes, 
        checkpoint_dir=checkpoint_dir, 
        num_epochs=num_epochs, 
        start_epoch=0
    )
    
if __name__ == '__main__':
    main()
