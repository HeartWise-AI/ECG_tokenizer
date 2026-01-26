# ECG Tokenizer Docker Pipeline

## Inputs
- Parquet/CSV with columns: `ecg_path` (path to .npy or WFDB .hea) and `reports`.

## Running steps
Use `docker/run_pipeline.bash` with `--step`:

1) Preprocess only  
```bash
source docker/run_pipeline.bash --step preprocess --input_file /path/to/data.csv
```

2) BERT only (on the parquet from step 1)  
```bash
source docker/run_pipeline.bash --step bert --input_file /volume/ECG_tokenizer/outputs/<preprocessed_parquet>.parquet
```

3) Full (preprocess + BERT)  
```bash
source docker/run_pipeline.bash --step all --input_file /path/to/data.csv
```

4) EfficientNet (external)  
```bash
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode inference
```

## Outputs
- Preprocessed parquet: `<output_dir>/<timestamp>[_<dataset>]_preprocessed_data.parquet`
- Cleaned signals: `<preprocessing_folder>/<timestamp>[_<dataset>]_preprocessing/*.npy`
- BERT probabilities CSV: `<preprocessed_parquet>.bert_probabilities.csv` (same base name)

## Volume mounts (Docker)
| Mount | Purpose |
|-------|---------|
| `/app/inputs` | Input CSV/Parquet |
| `/app/outputs` | Parquet outputs (preprocessed + BERT) |
| `/app/preprocessing` | Saved `.npy` signals |
| `/app/ecg_signals` | Raw ECGs (read-only) |
| `/app/checkpoints` | BERT/tokenizer checkpoints (read-only) |
| `/app/config` | Config files (read-only) |

## Checkpoints needed
- BERT classifier: `/app/checkpoints/mimic_mhi_bert`
- Tokenizer: `/app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt`
(EfficientNet checkpoint is only needed when you call `scripts/runner.sh`.)

## Notes
- BERT is mandatory for generating labels; `--step bert` or `--step all` runs it.
