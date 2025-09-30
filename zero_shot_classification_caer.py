import os
import json
import glob
import argparse
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import open_clip
from models.video_text_encoder_v2 import VideoTextEncoderV2  

# descriptions 
DEFAULT_DESCRIPTIONS = {
    "Anger": [
        "a person with a jaw clenched and furrowed brows",
        "expression showing frustration and hostility",
        "nostrils slightly flared and mouth tight",
        "face showing heat and agitation",
        "face showing visible tension and irritation",
    ],
    "Disgust": [
        "upper lip raised and nose wrinkled",
        "a recoiling expression as if something smells bad",
        "eyes slightly squinting with a look of aversion",
        "expression showing distaste or repulsion",
        "eyes squinting and mouth pulled back as if repelled",
    ],
    "Fear": [
        "eyes widened and brows raised with tension",
        "mouth slightly open as if gasping",
        "a startled look with a rigid face",
        "face showing shock and anticipation of threat",
        "expression suggesting apprehension and unease",
    ],
    "Happy": [
        "a genuine smile with raised cheeks",
        "eyes crinkled with delight",
        "mouth corners lifted with a light expression",
        "face showing warmth and joy",
        "a pleasant, cheerful look",
    ],
    "Neutral": [
        "a calm, relaxed face with no pronounced movement",
        "eyes and mouth at rest with minimal tension",
        "expressionless, composed appearance",
        "brows steady and mouth closed gently",
        "a baseline, unexpressive look",
    ],
    "Sad": [
        "downturned mouth and drooping eyes",
        "a heavy, fatigued gaze with slack features",
        "chin slightly lowered and lips pressed lightly",
        "eyes looking downward with a subdued face",
        "expression conveying loss or dejection",
    ],
    "Surprise": [
        "eyes open wide with raised eyebrows",
        "mouth dropped open with a quick inhale",
        "a sudden, alert look",
        "mouth open as if startled",
        "face showing shock or amazement",
    ],
}

# CLIP normalisation stats (match training transforms)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def extract_video_frames(video_path: str, num_frames: int = 8, image_size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = max(n_frames, 1)
    idxs = np.linspace(0, n_frames - 1, num_frames).astype(int).tolist()

    frames = []
    i = 0
    next_idx = idxs[0] if idxs else 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i == next_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            idxs.pop(0)
            if not idxs:
                break
            next_idx = idxs[0]
        i += 1
    cap.release()

    if not frames:
        return torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)

    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = x.unsqueeze(0)  # [1,T,3,H,W]
    x = (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)
    return x.squeeze(0)


def compute_class_prototypes(model: VideoTextEncoderV2, tokenizer, device, classes):
    model.eval()
    protos = {}
    with torch.no_grad():
        for cls in classes:
            descs = DEFAULT_DESCRIPTIONS[cls]
            tokens = tokenizer(descs).to(device)  # [K, L]
            text_embs = model.clip_model.encode_text(tokens)  # [K, D]
            text_embs = F.normalize(text_embs, dim=-1)
            proto = text_embs.mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=-1)
            protos[cls] = proto
    return protos


def encode_video(model: VideoTextEncoderV2, frames: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = frames.unsqueeze(0).to(device)
        v = model.encode_video(x)
        v = F.normalize(v, dim=-1)
    return v


def plot_cm_heatmap(matrix: np.ndarray, classes: list, out_path: str, title: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate(model, tokenizer, video_root, out_dir, device, num_frames=8,
             max_videos_per_class: Optional[int] = None,
             subset_per_class: Optional[int] = None,
             subset_frac: Optional[float] = None,
             seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(DEFAULT_DESCRIPTIONS.keys())
    dirs_present = [d for d in classes if os.path.isdir(os.path.join(video_root, d))]
    if not dirs_present:
        raise ValueError(f"No expected class folders found in: {video_root}\nExpected: {classes}")
    classes = dirs_present

    # prototypes
    protos = compute_class_prototypes(model, tokenizer, device, classes)

    rng = np.random.RandomState(seed)
    y_true, y_pred, paths, sims = [], [], [], []

    # loop through each class dir; balanced sampling options + optional hard cap
    exts = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    for cls in classes:
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(video_root, cls, e)))
        files = sorted(files)

        # balanced subsampling:
        if subset_per_class is not None and subset_per_class > 0 and len(files) > subset_per_class:
            idx = rng.choice(len(files), size=subset_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        elif subset_frac is not None and 0 < subset_frac < 1:
            m = max(1, int(round(len(files) * subset_frac)))
            if len(files) > m:
                idx = rng.choice(len(files), size=m, replace=False)
                files = [files[i] for i in sorted(idx)]

        # legacy hard cap (after subsampling)
        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]

        for vp in tqdm(files, desc=f"Processing {cls}"):
            frames = extract_video_frames(vp, num_frames=num_frames, image_size=224)
            if frames.numel() == 0:
                continue
            v = encode_video(model, frames, device)  # [1, D]

            # stack prototypes in class order
            P = torch.cat([protos[c] for c in classes], dim=0).to(v.device)  # [C, D]
            s = (v @ P.T).squeeze(0)  # [C], cosine via dot (already normalised)
            top_idx = int(torch.argmax(s).item())
            pred = classes[top_idx]

            y_true.append(cls)
            y_pred.append(pred)
            paths.append(vp)
            sims.append(float(s[top_idx].item()))

    # save per-sample predictions
    pred_df = pd.DataFrame({
        "video_path": paths,
        "true_label": y_true,
        "pred_label": y_pred,
        "top1_sim": sims
    })
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # metrics 
    report = classification_report(
        y_true, y_pred,
        labels=classes,
        target_names=classes,
        digits=3,
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    rep_df = pd.DataFrame(report).transpose()
    rep_df.to_csv(os.path.join(out_dir, "classification_report.csv"))

    # confusion matrices
    cm_counts = confusion_matrix(y_true, y_pred, labels=classes)
    cm_counts_df = pd.DataFrame(cm_counts, index=classes, columns=classes)
    cm_counts_df.to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"))

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    cm_percents = (cm_counts / row_sums) * 100.0
    cm_percents_df = pd.DataFrame(np.round(cm_percents, 2), index=classes, columns=classes)
    cm_percents_df.to_csv(os.path.join(out_dir, "confusion_matrix_percents.csv"))

    # plots
    plot_cm_heatmap(cm_counts, classes, os.path.join(out_dir, "confusion_matrix_counts.png"),
                    title="Confusion Matrix (Counts)")
    plot_cm_heatmap(cm_percents, classes, os.path.join(out_dir, "confusion_matrix_percents.png"),
                    title="Confusion Matrix (Row %)")

    # console summary
    acc = report.get("accuracy", float("nan"))
    macro_f1 = report.get("macro avg", {}).get("f1-score", float("nan"))
    micro_f1 = report.get("weighted avg", {}).get("f1-score", float("nan"))
    print(f"\nZero-shot results on {video_root}")
    print(f"Accuracy              : {acc:.3f}")
    print(f"Macro F1              : {macro_f1:.3f}")
    print(f"Weighted (micro) F1   : {micro_f1:.3f}")
    print(f"Saved → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", type=str, default="data/CAER/test",
                    help="Root folder with 7 subdirs (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise).")
    ap.add_argument("--out_dir", type=str, default="caer_zero_shot_outputs")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to your best checkpoint (.pt) from training.")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_videos", type=int, default=-1,
                    help="Legacy: hard cap per class (<=0 means no cap).")
    ap.add_argument("--subset_per_class", type=int, default=None,
                    help="Random cap per class (balanced). If set, sample up to N videos/class with --seed.")
    ap.add_argument("--subset_frac", type=float, default=None,
                    help="Random fraction per class (0<frac<=1). Ignored if --subset_per_class is set.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic sampling.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoTextEncoderV2(
        clip_model_name="ViT-B-32",
        pretrained="openai",
        temporal_layers=1,
        tune_policy="all_unfrozen",
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    max_per_class = None if (args.max_videos is None or args.max_videos <= 0) else args.max_videos

    evaluate(
        model=model,
        tokenizer=tokenizer,
        video_root=args.video_root,
        out_dir=args.out_dir,
        device=device,
        num_frames=args.num_frames,
        max_videos_per_class=max_per_class,
        subset_per_class=args.subset_per_class,
        subset_frac=args.subset_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
