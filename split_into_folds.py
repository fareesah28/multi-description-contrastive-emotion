import pandas as pd
from sklearn.model_selection import KFold
import os

df = pd.read_csv("/mnt/scratch/users/adhg808/MAFW/csv/data_human.csv")  
kf = KFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs("/mnt/scratch/users/adhg808/MAFW/csv/folds", exist_ok=True)

for i, (train_idx, val_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_df.to_csv(f"/mnt/scratch/users/adhg808/MAFW/csv/folds/fold_{i}_train.csv", index=False)
    val_df.to_csv(f"/mnt/scratch/users/adhg808/MAFW/csv/folds/fold_{i}_val.csv", index=False)

    print(f" Saved Fold {i} (Train: {len(train_df)} | Val: {len(val_df)})")

