#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_signal_processor import ECGSignalProcessor

# sample usage: python plot_dataset_fft.py --parquet /media/data1/datasets/DeepECG/SSL_pretraining/split/code15/code_15_dataset.parquet /media/data1/datasets/Harvard-Emory-ECG/preprocessed/harvard_emory_ecg_data_merged_2025_10_11_012710_n_10608417.parquet --waveform-column waveform_path --waveform-column npy_path --label code_15 --label harvard-emory --lead mean --scale log --output /volume/ECG_tokenizer/outputs/ecg_fft_comparison_MHI_scaled_with_harvard.png

def load_signals(parquet_path: str,
                 waveform_column: str,
                 max_samples: int | None,
                 lead: str,
                 target_length: int | None) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    if waveform_column not in df.columns:
        raise KeyError(f"Column '{waveform_column}' not found; available columns: {list(df.columns)}")
    if max_samples is not None:
        df = df.sample(n=max_samples, random_state=42)

    signals: list[np.ndarray] = []
    for waveform_path in df[waveform_column].dropna():
        waveform_path = Path(waveform_path)
        if not waveform_path.exists():
            print(f"Skipping missing file: {waveform_path}")
            continue

        arr = np.load(waveform_path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim != 2:
            raise ValueError(f"Expected (length, leads); found shape {arr.shape}")

        if lead == "mean":
            signal_1d = arr.mean(axis=1)
        else:
            lead_idx = int(lead)
            if lead_idx >= arr.shape[1]:
                raise ValueError(f"Lead index {lead_idx} out of range for {arr.shape[1]} leads")
            signal_1d = arr[:, lead_idx]

        if target_length and signal_1d.shape[0] != target_length:
            if signal_1d.shape[0] < target_length:
                continue  # too short, skip
            step = max(1, signal_1d.shape[0] // target_length)
            signal_1d = signal_1d[::step][:target_length]

        if np.isnan(signal_1d).any():
            continue
        signals.append(signal_1d.astype(np.float32))

    if not signals:
        raise RuntimeError("No valid waveforms loaded; check paths, NaNs, target length, or lead choice.")
    return np.stack(signals)


def main():
    parser = argparse.ArgumentParser(description="Plot mean ECG power spectrum using ECGSignalProcessor.")
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="One or more parquet files with waveform paths.",
    )
    parser.add_argument(
        "--waveform-column",
        action="append",
        help="Waveform column name; repeat to supply one per parquet. Defaults to 'waveform_path'.",
    )
    parser.add_argument("--output", default="outputs/ecg_fft_mean.png", help="Where to save the plot.")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate in Hz.")
    parser.add_argument("--lead", default="0", help="Lead index to analyze or 'mean' to average leads.")
    parser.add_argument("--target-length", type=int, default=2500, help="Downsample/trim length (set None to skip).")
    parser.add_argument("--max-samples", type=int, default=20000, help="Limit number of waveforms to load.")
    parser.add_argument(
        "--label",
        action="append",
        help="Legend label; provide once per parquet (optional). Defaults to parquet stem.",
    )
    parser.add_argument("--scale", choices=["linear", "db", "log"], default="db", help="Y-axis scale.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    labels = args.label or []
    waveform_columns = args.waveform_column or []
    processor = ECGSignalProcessor(fs=args.fs)

    plt.figure(figsize=(12, 6))
    ylabel = "Magnitude"

    for idx, parquet_path in enumerate(args.parquet):
        if waveform_columns:
            column_name = waveform_columns[idx] if idx < len(waveform_columns) else waveform_columns[-1]
        else:
            column_name = "waveform_path"

        signals = load_signals(
            parquet_path=parquet_path,
            waveform_column=column_name,
            max_samples=args.max_samples,
            lead=args.lead,
            target_length=args.target_length,
        )

        freqs, mean_mag = processor.plot_mean_spectrum(signals)
        pos_mask = freqs >= 0
        x = freqs[pos_mask]

        if args.scale == "db":
            y = 20 * np.log10(np.maximum(mean_mag[pos_mask], 1e-12))
            ylabel = "Magnitude (dB)"
        elif args.scale == "log":
            y = np.log(mean_mag[pos_mask])
            ylabel = "Log Magnitude"
        else:
            y = mean_mag[pos_mask]
            ylabel = "Magnitude"

        label = labels[idx] if idx < len(labels) else Path(parquet_path).stem
        plt.plot(x, y, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title("Mean magnitude spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.output)
    supported_exts = {".png", ".pdf", ".ps", ".eps", ".svg", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    if out_path.suffix.lower() not in supported_exts:
        print(f"Output path '{out_path}' lacks a supported image extension; defaulting to PNG.")
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
