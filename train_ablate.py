import os, random, math, json, argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import wandb

from dataloader import get_dataloaders
from models.video_text_encoder_v2 import VideoTextEncoderV2
from losses import multi_positive_contrastive_loss

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", required=True)
parser.add_argument("--val_csv", required=True)
parser.add_argument("--video_dir", required=True)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--caption_pool", choices=["mean_token","eot_token"], default="mean_token")
parser.add_argument("--temporal_layers", type=int, default=1)
parser.add_argument(
    "--tune_policy",
    choices=["light","strict","vision_unfrozen","text_unfrozen","all_unfrozen"],
    default="light"
)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--logit_temp", action="store_true",
                    help="Use learnable temperature (logit_scale).")
args = parser.parse_args()

# config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, epochs, seed, patience = 8, 30, 42, 3
base_lr = 5e-5          
clip_lr = 5e-6          

torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# wandb
os.environ.setdefault("WANDB_DIR", "/mnt/scratch/users/adhg808/wandb_logs")
suffix = "_lt" if args.logit_temp else ""
run_name = f"{args.tune_policy}_fold{args.fold}_pool-{args.caption_pool}_T{args.temporal_layers}{suffix}"
wandb.init(
    project="mafw-contrastive",
    name=run_name,
    config=dict(
        lr_base=base_lr, lr_clip=clip_lr, batch_size=batch_size, epochs=epochs, seed=seed,
        fold=args.fold, caption_pool=args.caption_pool,
        temporal_layers=args.temporal_layers, tune_policy=args.tune_policy,
        logit_temp=args.logit_temp
    )
)

# data
train_loader, val_loader = get_dataloaders(
    video_dir=args.video_dir, train_csv=args.train_csv, val_csv=args.val_csv,
    batch_size=batch_size, debug=args.debug
)

# model
model = VideoTextEncoderV2(
    caption_pool=args.caption_pool,
    temporal_layers=args.temporal_layers,
    tune_policy=args.tune_policy
).to(device)
print(f" Training Fold {args.fold} | pool={args.caption_pool} | T={args.temporal_layers} | policy={args.tune_policy} | lt={args.logit_temp}")

# log param counts (optional)
def count_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
wandb.log({
    "params_total": count_params(model, False),
    "params_trainable": count_params(model, True),
})

# optimiser with param groups (LR split for pretrained CLIP)
pretrained_params, new_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # CLIP towers (pretrained)
    if n.startswith(("image_encoder", "clip_model", "ln_final", "text_projection")):
        pretrained_params.append(p)
    else:
        new_params.append(p)

optimizer = optim.AdamW(
    [
        {"params": new_params, "lr": base_lr},
        {"params": pretrained_params, "lr": clip_lr},
    ],
    weight_decay=0.01,
)
wandb.log({
    "num_params_pretrained": sum(p.numel() for p in pretrained_params),
    "num_params_new": sum(p.numel() for p in new_params),
})

# training/validation
best_recall1, no_improve_epochs, best_model = 0.0, 0, None
for epoch in range(epochs):
    model.train(); total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        videos = batch["video"].to(device)
        captions = batch["captions"]

        optimizer.zero_grad(set_to_none=True)
        v_emb, c_emb = model(videos, captions)

        temp_param = getattr(model, "logit_scale", None) if args.logit_temp else None
        loss, logits, targets, probs = multi_positive_contrastive_loss(
            v_emb, c_emb, temperature=temp_param, return_debug=True
        )
        loss.backward()
        optimizer.step()

        # clamp logit_scale like CLIP to keep exp(logit_scale) <= 100
        if args.logit_temp and hasattr(model, "logit_scale"):
            with torch.no_grad():
                model.logit_scale.data.clamp_(max=math.log(100.0))

        total_loss += loss.item()

        cos_sims = F.cosine_similarity(v_emb.unsqueeze(1), c_emb, dim=-1)
        wandb.log({
            "train_loss": loss.item(),
            "cosine_sim_mean": cos_sims.mean().item(),
            "cosine_sim_std": cos_sims.std().item()
        })

    avg_loss = total_loss / max(1, len(train_loader))
    wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})
    print(f" Epoch {epoch+1} complete | Avg Train Loss: {avg_loss:.4f}")

    # validation (mean over the 5 captions) 
    model.eval()
    recall1_total = recall5_total = recall10_total = 0
    cos_sims_total = 0.0; n_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🔎 Validating (batch mean-caption)"):
            videos = batch["video"].to(device)
            captions = batch["captions"]
            B = videos.size(0)

            v_emb, c_emb = model(videos, captions)
            cap_mean = c_emb.mean(dim=1)
            v = F.normalize(v_emb, dim=-1); c = F.normalize(cap_mean, dim=-1)

            sims = v @ c.T                              
            ranks = sims.argsort(dim=1, descending=True)
            correct = torch.arange(B, device=device)

            recall1_total  += (ranks[:, 0] == correct).sum().item()
            recall5_total  += (correct[:, None] == ranks[:, :5]).any(dim=1).sum().item()
            recall10_total += (correct[:, None] == ranks[:, :10]).any(dim=1).sum().item()

            cos_sims_total += F.cosine_similarity(v, c, dim=-1).sum().item()
            n_samples += B

    val_recall1  = recall1_total  / n_samples
    val_recall5  = recall5_total  / n_samples
    val_recall10 = recall10_total / n_samples
    val_cos_sim  = cos_sims_total / n_samples

    wandb.log({
        "val_recall@1": val_recall1,
        "val_recall@5": val_recall5,
        "val_recall@10": val_recall10,
        "val_cosine_sim": val_cos_sim
    })
    print(f"Val — R@1 {val_recall1:.4f} | R@5 {val_recall5:.4f} | R@10 {val_recall10:.4f} | Cos {val_cos_sim:.4f}")

    # checkpoint (best by R@1) 
    ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"{args.tune_policy}_fold{args.fold}_pool-{args.caption_pool}_T{args.temporal_layers}{suffix}.pt"
    if val_recall1 > best_recall1:
        best_recall1 = val_recall1; no_improve_epochs = 0
        best_model = deepcopy(model.state_dict())
        torch.save(best_model, os.path.join(ckpt_dir, ckpt_name))
        print(f"New best saved: {ckpt_name}")
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s)")
    if no_improve_epochs >= patience:
        print("Early stopping"); break

print("Training complete")

# save metrics
os.makedirs("metrics", exist_ok=True)
metrics_path = f"metrics/{args.tune_policy}_fold_{args.fold}_pool-{args.caption_pool}_T{args.temporal_layers}{suffix}.json"
with open(metrics_path, "w") as f:
    json.dump({
        "fold": args.fold,
        "caption_pool": args.caption_pool,
        "temporal_layers": args.temporal_layers,
        "tune_policy": args.tune_policy,
        "logit_temp": args.logit_temp,
        "val_recall@1": val_recall1,
        "val_recall@5": val_recall5,
        "val_recall@10": val_recall10,
        "val_cosine_sim": val_cos_sim
    }, f, indent=2)
print(f"Saved metrics → {metrics_path}")
