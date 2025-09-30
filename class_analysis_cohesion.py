import argparse
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import silhouette_samples, pairwise_distances

def load_embeddings(npz_path):
    Z = np.load(npz_path, allow_pickle=True)
    V = Z["V"]                     # [N, D]
    C = Z["C"]                     # [N*5, D] (flattened)
    VID = Z["VID"].astype(str)    # [N]
    N, D = V.shape
    C = C.reshape(N, -1, D)       # [N, 5, D]
    # normalise to unit length for cosine sims
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = C / (np.linalg.norm(C, axis=2, keepdims=True) + 1e-9)
    return V, C, VID

def normalize_id(x, lower=True, strip_ext=False):
    s = str(x)
    if lower: s = s.lower()
    if strip_ext and "." in s:
        s = s[:s.rfind(".")]
    return s

def map_labels(VID, label_csv, id_col, emo_col, lower=True, strip_ext=False):
    df = pd.read_csv(label_csv) if label_csv.lower().endswith(".csv") else pd.read_excel(label_csv)
    df[id_col] = df[id_col].astype(str)
    # normalise ids
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

def mean_pairwise_cosine_similarity(X):
    # X is [n, d], assumed L2-normalised
    if X.shape[0] < 2:
        return np.nan
    S = X @ X.T  # cosine sim matrix
    # take upper triangle without diag
    iu = np.triu_indices(S.shape[0], k=1)
    vals = S[iu]
    return float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))

def caption_diversity_per_video(Ci):
    # Ci: [m, d] m captions. return mean pairwise cosine distance (1 - sim)
    m = Ci.shape[0]
    if m < 2:
        return np.nan
    S = Ci @ Ci.T
    iu = np.triu_indices(m, k=1)
    dist = 1.0 - S[iu]
    return float(np.mean(dist)), float(np.median(dist)), float(np.std(dist))

def class_centroids(V, y):
    centroids = {}
    for cls in sorted(set(y)):
        idx = np.where(y == cls)[0]
        if len(idx) == 0: 
            continue
        centroids[cls] = np.mean(V[idx], axis=0)
        # renormalise
        centroids[cls] = centroids[cls] / (np.linalg.norm(centroids[cls]) + 1e-9)
    return centroids

def centroid_similarity_matrix(centroids):
    classes = list(centroids.keys())
    C = np.vstack([centroids[c] for c in classes])
    S = C @ C.T
    return classes, S

def main():
    ap = argparse.ArgumentParser(description="Per-class cohesion/separation analysis for video+caption embeddings.")
    ap.add_argument("--npz", required=True, help="Path to NPZ from export_embeddings.py (contains V, C, VID)")
    ap.add_argument("--labels", required=True, help="CSV/XLSX with id_col -> emo_col mapping")
    ap.add_argument("--id_col", required=True, help="Column name for video id in labels file")
    ap.add_argument("--emo_col", required=True, help="Column name for emotion label (string) in labels file")
    ap.add_argument("--strip_ext", action="store_true", help="Strip file extensions (e.g., '.mp4') when matching IDs")
    ap.add_argument("--out_dir", default="class_analysis", help="Directory to save outputs (CSVs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    V, C, VID = load_embeddings(args.npz)
    labels, mask = map_labels(VID, args.labels, args.id_col, args.emo_col, strip_ext=args.strip_ext)

    # filter to matched samples
    V = V[mask]
    C = C[mask]
    VID = VID[mask]
    y = labels[mask].astype(str)

    # per-class cohesion + silhouette
    # silhouette uses cosine distance = 1 - cosine_similarity
    if len(np.unique(y)) > 1 and len(y) >= 10:
        D = pairwise_distances(V, metric="cosine")
        sil_all = silhouette_samples(D, y, metric="precomputed")
    else:
        sil_all = np.full(len(y), np.nan)

    per_class_rows = []
    for cls in sorted(set(y)):
        idx = np.where(y == cls)[0]
        n = len(idx)
        intra_mean, intra_median, intra_std = mean_pairwise_cosine_similarity(V[idx])
        # captions: per-video diversity, then aggregate
        cap_div_means, cap_div_meds, cap_div_stds = [], [], []
        for i in idx:
            m, md, sd = caption_diversity_per_video(C[i])
            cap_div_means.append(m)
            cap_div_meds.append(md)
            cap_div_stds.append(sd)
        cap_div_mean = float(np.nanmean(cap_div_means)) if len(cap_div_means) else np.nan
        cap_div_median = float(np.nanmedian(cap_div_meds)) if len(cap_div_meds) else np.nan
        cap_div_std = float(np.nanmean(cap_div_stds)) if len(cap_div_stds) else np.nan
        sil_mean = float(np.nanmean(sil_all[idx])) if len(idx) else np.nan

        per_class_rows.append({
            "class": cls,
            "n_videos": n,
            "video_intra_sim_mean": intra_mean,
            "video_intra_sim_median": intra_median,
            "video_intra_sim_std": intra_std,
            "caption_diversity_mean": cap_div_mean,
            "caption_diversity_median": cap_div_median,
            "caption_diversity_std": cap_div_std,
            "silhouette_mean": sil_mean
        })

    df_cohesion = pd.DataFrame(per_class_rows).sort_values("class")
    df_cohesion.to_csv(os.path.join(args.out_dir, "per_class_cohesion.csv"), index=False)

    # class centroids + nearest neighbors (separation)
    cents = class_centroids(V, y)
    classes, S = centroid_similarity_matrix(cents)  # cosine sims between centroids
    df_centroid_sim = pd.DataFrame(S, index=classes, columns=classes)
    df_centroid_sim.to_csv(os.path.join(args.out_dir, "centroid_cosine_similarity.csv"))

    # nearest neighbor per class (excluding self)
    nn_rows = []
    for i, ci in enumerate(classes):
        sims = S[i].copy()
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        nn_rows.append({
            "class": ci,
            "nearest_class": classes[j],
            "centroid_cosine_sim": float(S[i, j])
        })
    pd.DataFrame(nn_rows).to_csv(os.path.join(args.out_dir, "nearest_class_by_centroid.csv"), index=False)

    # overall summary 
    overall = {
        "n_samples_used": int(len(y)),
        "n_classes": int(len(set(y))),
        "overall_silhouette_mean": float(np.nanmean(sil_all)) if len(y) else np.nan
    }
    pd.DataFrame([overall]).to_csv(os.path.join(args.out_dir, "overall_summary.csv"), index=False)

    print("[Done]")
    print(f"Saved: {os.path.join(args.out_dir, 'per_class_cohesion.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'centroid_cosine_similarity.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'nearest_class_by_centroid.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'overall_summary.csv')}")

if __name__ == "__main__":
    main()
