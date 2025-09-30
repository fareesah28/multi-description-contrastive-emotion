import os, json, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import get_dataloaders
from models.video_text_encoder_v2 import VideoTextEncoderV2

VIDEO_DIR = "/mnt/scratch/users/adhg808/MAFW/data/clips"
CSV_DIR   = "/mnt/scratch/users/adhg808/MAFW/csv/folds"
FOLD = 0
CKPT = f"checkpoints/all_unfrozen_fold{FOLD}_pool-mean_token_T1_lt.pt"
OUT  = f"embeddings/all_unfrozen_fold{FOLD}_T1_lt_val.npz"

os.makedirs("embeddings", exist_ok=True)

# dataloader (val only)
train_csv = f"{CSV_DIR}/fold_{FOLD}_train.csv"
val_csv   = f"{CSV_DIR}/fold_{FOLD}_val.csv"

_, val_loader = get_dataloaders(
    video_dir=VIDEO_DIR,
    train_csv=train_csv,
    val_csv=val_csv,
    batch_size=8,
    debug=False
)


# model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoTextEncoderV2(
    caption_pool="mean_token",
    temporal_layers=1,
    tune_policy="all_unfrozen"
).to(device).eval()
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state)

V_list, C_list, VID_list = [], [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Export val embeddings"):
        videos = batch["video"].to(device)          # [B, T, C, H, W]
        caps   = batch["captions"].to(device)       # [B, 5, L] token IDs
        v, c = model(videos, caps)                  # v: [B, D], c: [B, 5, D]
        v = F.normalize(v, dim=-1)
        c = F.normalize(c, dim=-1)                  # keep per-caption, don't mean here

        V_list.append(v.detach().cpu().numpy())
        C_list.append(c.detach().cpu().numpy())     # [B, 5, D]
        # try to record some identifier if present; else use running index
        if "video_id" in batch:
            VID_list.extend([str(x) for x in batch["video_id"]])
        elif "path" in batch:
            VID_list.extend([str(x) for x in batch["path"]])
        else:
            # fallback: sequential IDs
            start = len(VID_list)
            VID_list.extend([f"vid_{start+i}" for i in range(v.size(0))])

V = np.concatenate(V_list, axis=0)                 # [N, D]
C = np.concatenate(C_list, axis=0).reshape(-1, V.shape[1])  # [N*5, D]
VID = np.array(VID_list)                           # [N]

np.savez_compressed(OUT, V=V, C=C, VID=VID)
print("saved:", OUT, "| V:", V.shape, "C:", C.shape, "VID:", VID.shape)

