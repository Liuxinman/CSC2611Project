import pandas as pd
from tqdm import tqdm
from retrieve_dataset import get_month_days

dataset_dir = ...
year = ...
month = ...

for m in tqdm(month):
    df = pd.read_csv(f"{dataset_dir}/{year}/{m}/1.csv", lineterminator="\n")
    for d in tqdm(range(2, get_month_days(m) + 1)):
        df_new = pd.read_csv(f"{dataset_dir}/{year}/{m}/{d}.csv", lineterminator="\n")
        df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(f"{dataset_dir}/{year}_merged/{m}_merged.csv", index=False)
