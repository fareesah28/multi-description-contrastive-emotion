import os
import json
import argparse
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import open_clip

from models.video_text_encoder_v2 import VideoTextEncoderV2

CLASSES = [
    "Anger", "Disgust", "Fear", "Happiness", "Neutral",
    "Sadness", "Surprise", "Contempt", "Anxiety", "Helplessness", "Disappointment"
]

DEFAULT_DESCRIPTIONS = {
    "Anger": [
        "jaw clenched with brows drawn together",
        "eyes narrowed and lips pressed tight",
        "nostrils slightly flared and a tense expression",
        "a hard stare with visible facial tension",
        "mouth tight and face showing irritation",
    ],
    "Disgust": [
        "upper lip raised and nose wrinkled",
        "a recoiling face as if smelling something foul",
        "eyes squinting with a look of aversion",
        "mouth pulled back in a slight sneer",
        "features showing revulsion and withdrawal",
    ],
    "Fear": [
        "eyes widened and brows lifted with tension",
        "lips parted as if gasping in alarm",
        "a rigid face with a startled look",
        "chin slightly lowered with a frozen stare",
        "features conveying apprehension and unease",
    ],
    "Happiness": [
        "a genuine smile with cheeks raised",
        "eyes crinkling at the corners in a warm look",
        "mouth corners lifted with a relaxed expression",
        "a bright, pleasant face showing positive affect",
        "features signaling joy and comfort",
    ],
    "Neutral": [
        "a calm, relaxed face with minimal movement",
        "eyes and mouth at rest without tension",
        "an even, composed expression",
        "brows steady and lips gently closed",
        "a baseline, unexpressive look",
    ],
    "Sadness": [
        "mouth corners downturned with a heavy gaze",
        "eyes looking downward with slack features",
        "chin slightly lowered and lips pressed lightly",
        "a subdued face reflecting loss",
        "features conveying dejection and low energy",
    ],
    "Surprise": [
        "eyes open wide with arched brows",
        "mouth dropped open in a quick intake of breath",
        "a sudden, alert expression",
        "forehead lifted with widened gaze",
        "features showing shock or amazement",
    ],
    "Contempt": [
        "one corner of the mouth raised in a slight smirk",
        "a sideways glance with a dismissive expression",
        "upper lip curled subtly with disdain",
        "asymmetric mouth pull suggesting scorn",
        "features signaling superiority and derision",
    ],
    "Anxiety": [
        "brows knit with restless tension in the eyes",
        "tight lips with a worried, scanning gaze",
        "subtle jaw tension and fidgety expression",
        "uneasy look with fleeting, uncertain focus",
        "features conveying persistent nervousness",
    ],
    "Helplessness": [
        "slumped features with a resigned gaze",
        "eyes unfocused and mouth slightly open",
        "a weary look suggesting loss of control",
        "subtle head tilt with defeated expression",
        "features reflecting powerlessness and fatigue",
    ],
    "Disappointment": [
        "brows lowered with a faint exhale",
        "lips pressed thin in a let-down expression",
        "eyes dipping slightly with a discouraged look",
        "a restrained sigh with downcast gaze",
        "features conveying unmet expectations",
    ],
}

# CLIP normalisation to match training
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def normalize_id(x: str, lower: bool = True, strip_ext: bool = False) -> str:
    s = str(x)
    if lower:
        s = s.lower()
    if strip_ext and "." in s:
        s = s[: s.rfind(".")]
    return s


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

    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]
    x = x.unsqueeze(0)  # [1,T,3,H,W]
    x = (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)
    return x.squeeze(0)  # [T,3,H,W]


def compute_class_prototypes(model: VideoTextEncoderV2, tokenizer, device, classes: list):
    model.eval()
    protos = {}
    with torch.no_grad():
        for cls in classes:
            descs = DEFAULT_DESCRIPTIONS.get(cls, None)
            if not descs:
                raise ValueError(f"No descriptions provided for class '{cls}'.")
            tokens = tokenizer(descs).to(device)  # [K, L]
            text_embs = model.clip_model.encode_text(tokens)  # [K, D]
            text_embs = F.normalize(text_embs, dim=-1)
            proto = text_embs.mean(dim=0, keepdim=True)  # [1, D]
            proto = F.normalize(proto, dim=-1)
            protos[cls] = proto
    return protos


def encode_video(model: VideoTextEncoderV2, frames: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = frames.unsqueeze(0).to(device)  # [1,T,3,224,224]
        v = model.encode_video(x)  # [1, D]
        v = F.normalize(v, dim=-1)
    return v


def plot_cm_heatmap(matrix: np.ndarray, classes: list, out_path: str, title: str):
    plt.figure(figsize=(max(8, 0.7 * len(classes)), max(7, 0.7 * len(classes))))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def load_mafw_split(
        clips_dir: str,
        labels_csv: str,
        id_col: str = "clip",
        label_col: str = "emotion_class",
        strip_ext: bool = False,
        max_videos_per_class: Optional[int] = None,
        subset_per_class: Optional[int] = None,
        subset_frac: Optional[float] = None,
        seed: int = 42,
):

    df = pd.read_csv(labels_csv) if labels_csv.lower().endswith(".csv") else pd.read_excel(labels_csv)
    df[id_col] = df[id_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    df = df[df[label_col].isin(CLASSES)].copy()

    vids_by_class = defaultdict(list)
    for _, row in df.iterrows():
        label = str(row[label_col])
        vid_id = row[id_col]

        # candidate paths
        candidates = [
            os.path.join(clips_dir, vid_id),
            os.path.join(clips_dir, normalize_id(vid_id, lower=True, strip_ext=False)),
        ]
        if strip_ext:
            stem = normalize_id(vid_id, lower=True, strip_ext=True)
            for ext in (".mp4", ".avi", ".mkv", ".mov"):
                candidates.append(os.path.join(clips_dir, stem + ext))

        found = None
        for c in candidates:
            if os.path.isfile(c):
                found = c
                break
        if found:
            vids_by_class[label].append(found)

    # optional balanced subsampling per class (deterministic)
    rng = np.random.RandomState(seed)
    for k in list(vids_by_class.keys()):
        files = vids_by_class[k]

        if subset_per_class is not None and subset_per_class > 0:
            if len(files) > subset_per_class:
                idx = rng.choice(len(files), size=subset_per_class, replace=False)
                files = [files[i] for i in sorted(idx)]
        elif subset_frac is not None and 0 < subset_frac < 1:
            m = max(1, int(round(len(files) * subset_frac)))
            if len(files) > m:
                idx = rng.choice(len(files), size=m, replace=False)
                files = [files[i] for i in sorted(idx)]

        # legacy hard cap (applied after sampling)
        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]

        vids_by_class[k] = files

    # restrict classes to those with files found, preserving CLASSES order
    classes_present = [c for c in CLASSES if len(vids_by_class.get(c, [])) > 0]
    if not classes_present:
        raise ValueError("No videos found after matching CSV to files. Check --clips_dir / --labels_csv / columns.")
    return classes_present, vids_by_class


def run_inference(
        model,
        tokenizer,
        classes,
        vids_by_class,
        out_dir,
        device,
        num_frames: int,
        protos: dict,
):
    y_true, y_pred, paths, sims = [], [], [], []

    for cls in classes:
        files = vids_by_class[cls]
        for vp in tqdm(files, desc=f"Processing {cls}"):
            frames = extract_video_frames(vp, num_frames=num_frames, image_size=224)
            if frames.numel() == 0:
                continue
            v = encode_video(model, frames, device)  # [1, D]

            P = torch.cat([protos[c] for c in classes], dim=0).to(v.device)  # [C, D]
            s = (v @ P.T).squeeze(0)  # [C]
            top_idx = int(torch.argmax(s).item())
            pred = classes[top_idx]

            y_true.append(cls)
            y_pred.append(pred)
            paths.append(vp)
            sims.append(float(s[top_idx].item()))

    # predictions
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
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))

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
    print(f"\n Zero-shot results on MAFW")
    print(f"Accuracy              : {acc:.3f}")
    print(f"Macro F1              : {macro_f1:.3f}")
    print(f"Weighted (micro) F1   : {micro_f1:.3f}")
    print(f"Saved → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", type=str, default="/mnt/scratch/users/adhg808/MAFW/data/clips",
                    help="Flat directory containing MAFW video files.")
    ap.add_argument("--labels_csv", type=str,
                    default="/mnt/scratch/users/adhg808/MAFW/Labels/single-set-with-classes.csv",
                    help="CSV with columns: 'clip' (filename), 'emotion_class' (class string).")
    ap.add_argument("--id_col", type=str, default="clip")
    ap.add_argument("--label_col", type=str, default="emotion_class")
    ap.add_argument("--out_dir", type=str, default="mafw_zero_shot")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to your best checkpoint (.pt).")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--strip_ext", action="store_true",
                    help="If CSV 'clip' entries are stems, try appending common extensions.")
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

    # model
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

    max_per_class = None if args.max_videos is None or args.max_videos <= 0 else args.max_videos

    # build split from csv
    classes, vids_by_class = load_mafw_split(
        clips_dir=args.clips_dir,
        labels_csv=args.labels_csv,
        id_col=args.id_col,
        label_col=args.label_col,
        strip_ext=args.strip_ext,
        max_videos_per_class=max_per_class,
        subset_per_class=args.subset_per_class,
        subset_frac=args.subset_frac,
        seed=args.seed,
    )

    # compute text prototypes for classes present
    protos = compute_class_prototypes(model, tokenizer, device, classes)

    # run and save metrics
    os.makedirs(args.out_dir, exist_ok=True)
    run_inference(
        model=model,
        tokenizer=tokenizer,
        classes=classes,
        vids_by_class=vids_by_class,
        out_dir=args.out_dir,
        device=device,
        num_frames=args.num_frames,
        protos=protos,
    )


if __name__ == "__main__":
    main()
