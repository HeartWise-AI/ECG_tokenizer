import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
import torch.distributed as dist

from data.dataset import ECGDataset
from models.vqvae import VQVAE, SimpleVQAutoEncoder, ResVQAutoEncoder
import os
from tqdm.auto import trange
import wandb
import yaml
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Author: Rohan Banerjee

Relevant issues from lucid-rains repos: #28, #44, #102
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(model, optimizer, iteration, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'vqvae_iteration_{iteration}.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return 0  

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch

def evaluate(model, data_loader):
    logging.debug("Starting evaluation...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            logging.debug(f"Evaluating batch {i+1}/{len(data_loader)}")
            signals = batch['signal'].float().to(device)
            signals = signals.permute(0, 2, 1)
            with autocast():
                out, indices, cmt_loss = model(signals)
                rec_loss = (out - signals).abs().mean()
            total_loss += rec_loss.item()
    model.train()
    logging.debug("Evaluation completed.")
    return total_loss / len(data_loader)

def train(model, train_loader, test_loader, optimizer, num_codes, checkpoint_dir, train_iterations=1000, alpha=1):
    
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            signals = batch['signal'].float().to(device)
            signals = signals.permute(0, 2, 1)
            yield signals

    for _ in (pbar := trange(train_iterations)):
        optimizer.zero_grad()
        x = next(iterate_dataset(train_loader))
        # import pdb; pdb.set_trace()
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss.mean()).backward()

        optimizer.step()
        torch.cuda.empty_cache()

        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.mean().item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

        wandb.log({
            "rec_loss": rec_loss.item(),
            "cmt_loss": cmt_loss.mean().item(),
            "active_percentage": indices.unique().numel() / num_codes * 100
        })
        
        try:
            if (_ + 1) % 100 == 0:
                test_loss = evaluate(model, test_loader)
                wandb.log({"test_loss": test_loss})
                save_checkpoint(model, optimizer, _ + 1, checkpoint_dir)
                model.train()
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            model.train()
            return

def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    wandb.init(project="ECG_tokenizer", entity="rohanbanerjee", name=config["training"]["experiment_name"])

    csv_file = config["dataset"]["csv_file"]
    dataset_mimic_train = ECGDataset(csv_file=csv_file, split='train')
    train_loader = DataLoader(dataset_mimic_train, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=16)

    dataset_mimic_test = ECGDataset(csv_file=csv_file, split='test')
    test_loader = DataLoader(dataset_mimic_test, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=16)

    lr = float(config["training"]["learning_rate"])
    train_iter = config["training"]["train_iterations"]
    num_codes = config["training"]["num_codes"]
    seed = config["training"]["seed"]
    checkpoint_dir = f"/mnt/rbanerjee/checkpoints/{config['training']['experiment_name']}"
    torch.random.manual_seed(seed)
    model = ResVQAutoEncoder(
        timesteps=dataset_mimic_train.waveform_length,
        codebook_size=num_codes,
        implicit_neural_codebook=True
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Move the model to GPU
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, train_loader, test_loader, train_iterations=train_iter, optimizer=opt, num_codes=num_codes, checkpoint_dir=checkpoint_dir)
    
if __name__ == '__main__':
    main()
