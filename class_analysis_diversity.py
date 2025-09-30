import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_embeddings(npz_path):
    Z = np.load(npz_path, allow_pickle=True)
    V = Z["V"]                     # [N, D] videos
    C = Z["C"]                     # [N*5, D] captions flattened
    VID = Z["VID"].astype(str)    # [N]
    N, D = V.shape
    C = C.reshape(N, -1, D)       # [N, 5, D]
    # L2-normalise
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = C / (np.linalg.norm(C, axis=2, keepdims=True) + 1e-9)
    return V, C, VID

def normalize_id(x, lower=True, strip_ext=False):
    s = str(x)
    if lower: s = s.lower()
    if strip_ext and "." in s:
        s = s[:s.rfind(".")]
    return s

def map_labels(VID, label_path, id_col, emo_col, lower=True, strip_ext=False):
    df = pd.read_csv(label_path) if label_path.lower().endswith(".csv") else pd.read_excel(label_path)
    df[id_col] = df[id_col].astype(str)
    id_to_label = {}
    for _, row in df.iterrows():
        k = normalize_id(row[id_col], lower=lower, strip_ext=strip_ext)
        id_to_label[k] = str(row[emo_col])
    labels = []
    for vid in VID:
        k = normalize_id(vid, lower=lower, strip_ext=strip_ext)
        labels.append(id_to_label.get(k, None))
    labels = np.array(labels, dtype=object)
    mask = labels != None
    return labels, mask

def caption_diversity(capt_emb):  # capt_emb: [5, D] normalised
    m = capt_emb.shape[0]
    if m < 2:
        return np.nan
    S = capt_emb @ capt_emb.T
    iu = np.triu_indices(m, k=1)
    dist = 1.0 - S[iu]  # cosine distance
    return float(np.mean(dist))

def compute_class_centroids(V, y):
    cents = {}
    for cls in sorted(set(y)):
        idx = np.where(y == cls)[0]
        if len(idx) == 0: 
            continue
        c = V[idx].mean(axis=0)
        cents[cls] = c / (np.linalg.norm(c) + 1e-9)
    return cents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--emo_col", required=True)
    ap.add_argument("--strip_ext", action="store_true")
    ap.add_argument("--out_dir", default="diversity_outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    V, C, VID = load_embeddings(args.npz)
    labels, mask = map_labels(VID, args.labels, args.id_col, args.emo_col, strip_ext=args.strip_ext)

    # filter to matched
    V, C, VID = V[mask], C[mask], VID[mask]
    y = labels[mask].astype(str)

    # class centroids from videos
    cents = compute_class_centroids(V, y)

    # per video metrics
    rows = []
    for i in range(len(V)):
        cls = y[i]
        cap_div = caption_diversity(C[i])  # mean pairwise cosine distance among 5 captions
        centroid = cents[cls]
        vid_sim = float(np.dot(V[i], centroid))  # cosine similarity to class centroid
        rows.append({
            "video_id": VID[i],
            "class": cls,
            "caption_diversity": cap_div,
            "video_to_centroid_sim": vid_sim
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_video_metrics.csv"), index=False)

    # boxplot: caption diversity by class
    plt.figure(figsize=(11,6))
    order = sorted(df["class"].unique())
    data = [df[df["class"] == cls]["caption_diversity"].values for cls in order]
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Caption diversity (mean pairwise cosine distance)")
    plt.xlabel("Emotion class")
    plt.title("Caption Diversity by Class (per video)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "box_caption_diversity_by_class.png"), dpi=220)
    plt.close()

    # boxplot: video cohesion by class
    plt.figure(figsize=(11,6))
    data2 = [df[df["class"] == cls]["video_to_centroid_sim"].values for cls in order]
    plt.boxplot(data2, labels=order, showmeans=True)
    plt.ylabel("Video → class centroid cosine similarity (higher = tighter)")
    plt.xlabel("Emotion class")
    plt.title("Video Cohesion by Class (per video)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "box_video_cohesion_by_class.png"), dpi=220)
    plt.close()

    # correlation: caption diversity vs video cohesion
    x = df["caption_diversity"].values
    yv = df["video_to_centroid_sim"].values
    # pearson r
    mx, my = np.nanmean(x), np.nanmean(yv)
    sx, sy = np.nanstd(x), np.nanstd(yv)
    if sx > 0 and sy > 0:
        r = float(np.nanmean(((x - mx) / sx) * ((yv - my) / sy)))
    else:
        r = np.nan

    plt.figure(figsize=(7,6))
    plt.scatter(x, yv)
    plt.xlabel("Caption Diversity (pairwise cosine distance)")
    plt.ylabel("Video → Class Centroid Cosine Similarity")
    plt.title(f"Per-Video: Caption Diversity vs Video Cohesion (Pearson r = {r:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_caption_diversity_vs_video_cohesion.png"), dpi=220)
    plt.close()

    # console summary
    print(f"[Done] Saved outputs in: {args.out_dir}")
    print(f"- per_video_metrics.csv  (n={len(df)})")
    print(f"- box_caption_diversity_by_class.png")
    print(f"- box_video_cohesion_by_class.png")
    print(f"- scatter_caption_diversity_vs_video_cohesion.png")
    print(f"Pearson r (caption diversity vs video cohesion): {r:.3f}")

if __name__ == "__main__":
    main()

