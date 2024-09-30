from torch.utils.data import DataLoader
from data.dataset import ECGDataset

# Define paths
train_parquet = '/volume/mhi_dataset/train_trial_v1.1.parquet'
validation_parquet = '/volume/mhi_dataset/val_trial_v1.1.parquet'
test_parquet = '/volume/mhi_dataset/test_trial_v1.1.parquet'
npy_root_dir = '/volume/mhi_dataset/'

csv_file = '/media/data1/ravram/MIMIC-IV/mimic_index.corrected.csv'


# train_dataset = ECGDataset(parquet_file=train_parquet, root_dir=npy_root_dir)
# validation_dataset = ECGDataset(parquet_file=validation_parquet, root_dir=npy_root_dir)
# test_dataset = ECGDataset(parquet_file=test_parquet, root_dir=npy_root_dir)

train_dataset_mimic = ECGDataset(csv_file=csv_file, split='train')
test_dataset_mimic = ECGDataset(csv_file=csv_file, split='test')


train_loader_mimic = DataLoader(train_dataset_mimic, batch_size=32, shuffle=True, num_workers=4)
test_loader_mimic = DataLoader(test_dataset_mimic, batch_size=32, shuffle=False, num_workers=4)

try:
    for i, batch in enumerate(train_loader_mimic):
        signals = batch['signal']
        
        print(f"Batch {i + 1} loaded successfully.")
        print(f"Signals shape: {signals.shape}")
        
        break
except Exception as e:
    print(f"Error loading batch: {e}")