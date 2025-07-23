import pandas as pd 

mimic_train = pd.read_parquet("/media/data1/datasets/DeepECG/SSL_pretraining/split/MIMIC/mimic_v4_clean_train.parquet")

print(mimic_train.head())

