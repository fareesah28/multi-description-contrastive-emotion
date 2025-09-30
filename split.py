import pandas as pd
from sklearn.model_selection import train_test_split

# path to original full training CSV
csv_path = "/mnt/scratch/users/adhg808/MAFW/Labels/data_human_train.csv"

# load original data
df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=None  
)

# save splits
train_df.to_csv("/mnt/scratch/users/adhg808/MAFW/Labels/data_human_train.csv", index=False)
val_df.to_csv("/mnt/scratch/users/adhg808/MAFW/Labels/data_human_val.csv", index=False)

print(f" Split complete: {len(train_df)} train rows, {len(val_df)} val rows.")

