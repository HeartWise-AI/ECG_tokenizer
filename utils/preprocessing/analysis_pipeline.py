import hashlib
import os
import re
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import PTBXL_POWER_RATIO
from utils.files_handler import ECGFileHandler
from utils.preprocessing.ecg_signal_processor import ECGSignalProcessor


class AnalysisPipeline:
    TARGET_LENGTH = 2500
    TARGET_LEADS = 12

    SwapLeadsFn = Callable[[np.ndarray, int, int], np.ndarray]

    @staticmethod
    def _resolve_path_column(df: pd.DataFrame, path_column: str | None) -> str:
        if path_column and path_column in df.columns:
            return path_column

        path_columns = [col for col in df.columns if "path" in col.lower() or "file" in col.lower()]
        if not path_columns:
            raise ValueError("No column with 'path' or 'file' in its name found in the dataframe")

        for preferred in ("ecg_path", "filepath", "waveform_path_original", "waveform_path_psa", "xml_path", "ECG_path"):
            if preferred in path_columns:
                return preferred
        return path_columns[0]

    @staticmethod
    def _sanitize_stem(stem: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        return cleaned or "ecg"

    @classmethod
    def _build_output_base_path(cls, preprocessing_folder: str, row_pos: int, source_path: str) -> str:
        stem = cls._sanitize_stem(os.path.splitext(os.path.basename(source_path))[0])
        src_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
        file_id = f"{row_pos:09d}_{stem}_{src_hash}"
        return os.path.join(preprocessing_folder, file_id)

    @staticmethod
    def _resample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
        current_length = signal.shape[0]
        if current_length == target_length:
            return signal.astype(np.float32, copy=False)
        if current_length < 2:
            raise ValueError(f"Signal length must be >= 2 for interpolation; got {current_length}")

        old_x = np.linspace(0.0, 1.0, num=current_length, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
        out = np.empty((target_length, signal.shape[1]), dtype=np.float32)
        for lead_idx in range(signal.shape[1]):
            out[:, lead_idx] = np.interp(new_x, old_x, signal[:, lead_idx]).astype(np.float32, copy=False)
        return out

    @classmethod
    def _canonicalize_signal(cls, signal: np.ndarray) -> np.ndarray:
        # MHI format can appear as (N, 12, 1)
        if signal.ndim == 3 and signal.shape[-1] == 1:
            signal = signal.squeeze(-1)

        if signal.ndim != 2:
            raise ValueError(f"Expected signal with 2 dimensions, got shape {signal.shape}")

        # Some loaders return shape (12, N). We treat (12, 12) as already
        # canonical because it is ambiguous and outside expected ECG lengths.
        if signal.shape[0] == cls.TARGET_LEADS and signal.shape[1] != cls.TARGET_LEADS:
            signal = signal.transpose(1, 0)

        if signal.shape[1] != cls.TARGET_LEADS:
            raise ValueError(f"Expected {cls.TARGET_LEADS} leads, got shape {signal.shape}")

        if signal.shape[0] != cls.TARGET_LENGTH:
            signal = cls._resample_signal(signal, target_length=cls.TARGET_LENGTH)

        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or Inf values")

        return signal.astype(np.float32, copy=False)

    @classmethod
    def _to_psa_like_signal(
        cls,
        ecg_signal_processor: ECGSignalProcessor,
        signal: np.ndarray,
    ) -> np.ndarray:
        # Per-sample spectral power normalization: match lead-0 average spectral
        # power to the PTB-XL reference.
        scaled = ecg_signal_processor.normalize_signal_spectral_power(
            signal.astype(np.float32, copy=False),
            target_power=PTBXL_POWER_RATIO,
            reference_lead=0,
        )

        return scaled

    @classmethod
    def save_and_preprocess_data(
        cls,
        df: pd.DataFrame,
        output_folder: str,
        preprocessing_folder: str,
        preprocessing_n_workers: int,
        swap_leads_fn: SwapLeadsFn | None = None,
        swap_lead1: int | None = None,
        swap_lead2: int | None = None,
        path_column: str | None = None,
    ) -> pd.DataFrame:
        del output_folder  # Kept for backward compatibility with callers.
        del preprocessing_n_workers  # Current deterministic path is intentionally single-threaded.

        ecg_signal_processor = ECGSignalProcessor()
        os.makedirs(preprocessing_folder, exist_ok=True)

        ecg_path_col = cls._resolve_path_column(df=df, path_column=path_column)
        print(f"Detected path column: {ecg_path_col}")
        print(
            f"Using deterministic preprocessing: per-sample spectral normalization "
            f"(target_power={PTBXL_POWER_RATIO})"
        )

        processed_rows: list[pd.Series] = []
        skipped = 0

        for row_pos in tqdm(range(len(df)), total=len(df), desc="Preprocessing signals"):
            row = df.iloc[row_pos]
            source_path_raw = row.get(ecg_path_col)
            if pd.isna(source_path_raw):
                skipped += 1
                print(f"Warning: Skipping row {row_pos} - missing path in '{ecg_path_col}'")
                continue

            source_path = str(source_path_raw)
            try:
                raw_signal = ECGFileHandler.load_ecg_signal_raw(source_path)
                canonical_signal = cls._canonicalize_signal(raw_signal)
                processed_signal = cls._to_psa_like_signal(
                    ecg_signal_processor=ecg_signal_processor,
                    signal=canonical_signal,
                )

                if swap_leads_fn is not None and swap_lead1 is not None and swap_lead2 is not None:
                    processed_signal = swap_leads_fn(processed_signal, swap_lead1, swap_lead2)

                output_base_path = cls._build_output_base_path(
                    preprocessing_folder=preprocessing_folder,
                    row_pos=row_pos,
                    source_path=source_path,
                )
                save_path = f"{output_base_path}.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(file=save_path, arr=processed_signal.astype(np.float32, copy=False))

                row_out = row.copy()
                row_out[ecg_path_col] = save_path
                processed_rows.append(row_out)

            except Exception as exc:
                skipped += 1
                print(f"Error processing row {row_pos} ({source_path}): {exc}")
                continue

        if not processed_rows:
            raise ValueError("No data was successfully processed")

        processed_df = pd.DataFrame(processed_rows).reset_index(drop=True)
        print(f"\nCompleted processing {len(processed_df)} files (skipped: {skipped})")
        return processed_df
