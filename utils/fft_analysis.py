import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import h5py
import glob
from tqdm import tqdm

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_ecg_data_from_parquet(parquet_path, max_samples=20000):
    """Load ECG data from parquet file, limiting to max_samples."""
    print(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    direct_waveform_cols = [col for col in df.columns if col == 'waveform' or col == 'filtered_waveform']
    
    if direct_waveform_cols:
        waveform_col = direct_waveform_cols[0]
        print(f"Using direct waveform column: {waveform_col}")
        df = df.head(max_samples)
        
        waveforms = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing waveforms"):
            try:
                waveform = row[waveform_col]
                if not isinstance(waveform, np.ndarray):
                    waveform = np.array(waveform)
                waveforms.append(waveform)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    else:
        path_cols = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower()]
        
        if not path_cols:
            raise ValueError(f"No waveform or path column found in {parquet_path}")
        
        path_col = path_cols[0]
        print(f"Using path column: {path_col}")
        df = df.head(max_samples)

        waveforms = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading waveform files"):
            try:
                path = row[path_col]
                waveform = load_waveform_from_path(path)
                if waveform is not None:
                    waveforms.append(waveform)
            except Exception as e:
                print(f"Error loading file {path}: {e}")
                continue
    
    return np.array(waveforms) if waveforms else np.array([])

def load_waveform_from_path(path):
    """Load waveform data from a file path."""
    if not os.path.exists(path):
        basename = os.path.basename(path)
        search_path = f"/**/{basename}"
        matching_files = glob.glob(search_path, recursive=True)
        
        if not matching_files:
            print(f"File not found: {path}")
            return None
        
        path = matching_files[0]
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        return np.load(path)
    elif ext == '.h5' or ext == '.hdf5':
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                return np.array(f[key])
    elif ext == '.csv':
        return np.loadtxt(path, delimiter=',')
    else:
        print(f"Unsupported file format: {ext}")
        return None

def compute_magnitude_spectrum(signal, fs=250):
    """Compute the magnitude spectrum of a signal."""
    if signal.ndim > 1:
        signal = signal[0] if signal.shape[0] <= 12 else signal[:, 0]
    
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/fs)
    magnitude_spectrum = np.abs(fft_result)
    return fft_freq, magnitude_spectrum

def plot_mean_spectrum(signals, fs=250, color='blue', label='Mean Spectrum'):
    """Plot the mean magnitude spectrum of multiple signals."""
    if len(signals) == 0:
        print(f"No signals to process for {label}")
        return None, None
    
    all_magnitude_spectra = []
    for signal in tqdm(signals, desc=f"Computing FFT for {label}"):
        try:
            fft_freq, magnitude_spectrum = compute_magnitude_spectrum(signal, fs)
            all_magnitude_spectra.append(magnitude_spectrum)
        except Exception as e:
            print(f"Error computing FFT: {e}")
            continue
    
    if not all_magnitude_spectra:
        print(f"No valid FFTs computed for {label}")
        return None, None
    
    mean_magnitude_spectrum = np.mean(all_magnitude_spectra, axis=0)
    plt.plot(fft_freq[:len(fft_freq)//2], np.log(mean_magnitude_spectrum[:len(fft_freq)//2] + 1e-10), 
             color=color, label=label)
    return fft_freq, mean_magnitude_spectrum

def main():
    config_path = "/volume/ECG_tokenizer/config/vqvae_training/base_config.yaml"
    config = load_config(config_path)
    mimic_path = config["train_parquet_MIMIC_file"]
    mhi_path = config["train_parquet_MHI_file"]
    code15_path = config["code_15_dataset_path"]
    data_plotted = False
    plt.figure(figsize=(12, 8))

    try:
        print("\n=== Processing MIMIC dataset ===")
        mimic_data = load_ecg_data_from_parquet(mimic_path, max_samples=20000)
        if len(mimic_data) > 0:
            plot_mean_spectrum(mimic_data, color='blue', label='MIMIC')
            data_plotted = True
    except Exception as e:
        print(f"Error processing MIMIC dataset: {e}")
    
    try:
        print("\n=== Processing MHI dataset ===")
        mhi_data = load_ecg_data_from_parquet(mhi_path, max_samples=20000)
        if len(mhi_data) > 0:
            plot_mean_spectrum(mhi_data, color='red', label='MHI')
            data_plotted = True
    except Exception as e:
        print(f"Error processing MHI dataset: {e}")

    try:
        code15_data = load_ecg_data_from_parquet(code15_path, max_samples=20000)
        if len(code15_data) > 0:
            plot_mean_spectrum(code15_data, color='green', label='Code_15')
            data_plotted = True
    except Exception as e:
        print(f"Error processing Code_15 dataset: {e}")
    
    if data_plotted:
        plt.title('Mean Magnitude Spectrum of ECG Datasets')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Log Magnitude')
        plt.grid(True)
        plt.legend()
        output_dir = "/volume/ECG_tokenizer/outputs"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/ecg_fft_comparison.png")
        print(f"\nPlot saved to {output_dir}/ecg_fft_comparison.png")
    else:
        print("\nNo data was successfully plotted.")

if __name__ == "__main__":
    main()
