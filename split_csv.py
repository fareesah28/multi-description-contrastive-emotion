import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import get_dataloaders

# split csv
df = pd.read_csv("/mnt/scratch/users/adhg808/MAFW/Labels/data_human.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train_path = "/mnt/scratch/users/adhg808/MAFW/Labels/data_human_train.csv"
test_path = "/mnt/scratch/users/adhg808/MAFW/Labels/data_human_test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f" Saved {len(train_df)} train and {len(test_df)} test rows.")

# debug test of dataloader (optional)
train_loader, test_loader = get_dataloaders(
    video_dir="/mnt/scratch/users/adhg808/MAFW/data/clips",
    train_csv=train_path,
    test_csv=test_path,
    batch_size=4,
    debug=True
)

batch = next(iter(train_loader))
print(batch["video"].shape)       # [4, 8, 3, 224, 224]
print(batch["captions"].shape)    # [4, 5, max_len]

