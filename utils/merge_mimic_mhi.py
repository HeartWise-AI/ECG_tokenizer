
import pandas as pd

mimic_df = pd.read_parquet("parquets/mimic/mimic_v4_clean_test.parquet")
mhi_df = pd.read_parquet("parquets/mhi/test_trial_v1.1_with_report.parquet")

print(mimic_df.head())
print(mhi_df.head())


mimic_df = mimic_df[["waveform_path", "report"]]
mhi_df = mhi_df[["waveform_path", "report"]]

concat_df = pd.concat([mimic_df, mhi_df])

print(concat_df.head())

concat_df.to_parquet("parquets/mimic_mhi/mimic_mhi_test_subset.parquet")
