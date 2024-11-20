import torch
import matplotlib.pyplot as plt
from models.vqvae import SimpleVQAutoEncoder
from data.dataset import ECGDataset
from torch.utils.data import DataLoader
import random
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

seed = config["training"]["seed"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_file = config["dataset"]["csv_file"]
dataset_mimic_test = ECGDataset(csv_file=csv_file, split='test')
test_loader = DataLoader(dataset_mimic_test, batch_size=512, shuffle=False, num_workers=4)
num_codes = config["training"]["num_codes"]

model = SimpleVQAutoEncoder(
    timesteps=dataset_mimic_test.waveform_length,
    codebook_size=num_codes,
    rotation_trick=False 
).to(device)
checkpoint = torch.load(config["evaluation"]["model_path"], weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def load_random_test_signal(test_loader, device='cuda'):

    dataset = test_loader.dataset
    
    # import pdb; pdb.set_trace()
    dataset_size = len(dataset)
    random_index = random.randint(0, dataset_size - 1)
    sample = dataset[10]['signal']
    sample = torch.tensor(sample, dtype=torch.float32).to(device)
    sample = sample.permute(1, 0).unsqueeze(0)
    
    return sample

input_signal = load_random_test_signal(test_loader)

with torch.no_grad():
    reconstructed_signal, _, _ = model(input_signal)

    input_signal_np = input_signal.squeeze().cpu().numpy()
    reconstructed_signal_np = reconstructed_signal.squeeze().cpu().numpy()

    num_leads = input_signal_np.shape[0]
    fig, axes = plt.subplots(num_leads, 2, figsize=(15, 3 * num_leads))

    for lead in range(num_leads):

        axes[lead, 0].plot(input_signal_np[lead])
        axes[lead, 0].set_title(f'Input Signal - Lead {lead + 1}')
        

        axes[lead, 1].plot(reconstructed_signal_np[lead])
        axes[lead, 1].set_title(f'Reconstructed Signal - Lead {lead + 1}')

    plt.tight_layout()
    plt.savefig('/mnt/rbanerjee/figures/input_reconstructed_signals_2048.png')