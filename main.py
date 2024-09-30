from torch.utils.data import DataLoader
from data.dataset import ECGDataset

# Define paths
train_parquet = '/volume/mhi_dataset/train_trial_v1.1.parquet'
validation_parquet = '/volume/mhi_dataset/val_trial_v1.1.parquet'
test_parquet = '/volume/mhi_dataset/test_trial_v1.1.parquet'
npy_root_dir = '/volume/mhi_dataset/'

# Create dataset objects
train_dataset = ECGDataset(parquet_file=train_parquet, root_dir=npy_root_dir)
validation_dataset = ECGDataset(parquet_file=validation_parquet, root_dir=npy_root_dir)
test_dataset = ECGDataset(parquet_file=test_parquet, root_dir=npy_root_dir)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Example usage
for batch in train_loader:
    ecg_signals = batch['ecg_signal']  # Tensor of ECG signals

print(ecg_signals)
