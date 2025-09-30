import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# optional umap
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def pca_then_embed(X, method="tsne", random_state=0, perplexity=30, n_neighbors=30, min_dist=0.1):
    d = min(50, X.shape[1])
    Xp = PCA(n_components=d, random_state=random_state).fit_transform(X)
    if method == "umap" and HAS_UMAP:
        emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit_transform(Xp)
    else:
        emb = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca",
                   random_state=random_state).fit_transform(Xp)
    return emb


def load_npz(npz_path):
    Z = np.load(npz_path, allow_pickle=True)
    V, C, VID = Z["V"], Z["C"], Z["VID"]  
    N, D = V.shape[0], V.shape[1]
    C_reshaped = C.reshape(N, -1, D)  # assume 5 captions per video
    return V, C_reshaped, VID


def plot_pair_alignment(V, C_reshaped, out_path, max_videos=200, seed=0, embed="tsne"):
    np.random.seed(seed)
    N = V.shape[0]
    K = min(max_videos, N)
    keep = np.random.choice(N, size=K, replace=False)
    Vs = V[keep]                      
    Cs = C_reshaped[keep].reshape(K * C_reshaped.shape[1], V.shape[1])  

    X = np.vstack([Vs, Cs])
    xy = pca_then_embed(X, method=embed, random_state=seed, perplexity=30)

    xy_v, xy_c = xy[:K], xy[K:]

    plt.figure(figsize=(8, 7))
    plt.scatter(xy_c[:, 0], xy_c[:, 1], s=10, alpha=0.6, label="captions", marker="^")
    plt.scatter(xy_v[:, 0], xy_v[:, 1], s=30, alpha=0.9, label="videos", marker="o")

    # draw lines from video to its captions
    m = C_reshaped.shape[1]
    for i in range(K):
        start = i * m
        end = start + m
        xs = np.vstack([np.repeat(xy_v[i][None, :], m, axis=0), xy_c[start:end]])
        plt.plot(xs[:, 0], xs[:, 1], linewidth=0.5, alpha=0.25, color="gray")

    plt.legend()
    plt.title(f"t-SNE pair alignment (K={K} videos)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("Saved:", out_path)


def compute_margin(V, C_reshaped):
    # normalise
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    Cm = C_reshaped.mean(axis=1)
    Cm = Cm / (np.linalg.norm(Cm, axis=1, keepdims=True) + 1e-9)

    S = Vn @ Cm.T                  
    pos = np.diag(S).copy()
    np.fill_diagonal(S, -1e9)
    hard_neg = S.max(axis=1)
    margin = pos - hard_neg
    return margin, Vn, Cm


def plot_margin_heatmap(V, C_reshaped, out_path, hardest=100, seed=0, embed="tsne", include_captions=False):
    margin, Vn, Cm = compute_margin(V, C_reshaped)
    N = Vn.shape[0]
    idx = np.argsort(margin)[:min(hardest, N)]  # hardest by lowest margin

    if include_captions:
        X = np.vstack([Vn[idx], Cm[idx]])  # videos + mean captions
        xy = pca_then_embed(X, method=embed, random_state=seed, perplexity=30)
        H = len(idx)
        xy_v, xy_c = xy[:H], xy[H:]
        plt.figure(figsize=(8, 7))
        sc = plt.scatter(xy_v[:, 0], xy_v[:, 1], c=margin[idx], cmap="coolwarm", s=28, marker="o", label="videos")
        plt.scatter(xy_c[:, 0], xy_c[:, 1], c="k", alpha=0.25, s=10, marker="^", label="mean captions")
        plt.colorbar(sc, label="Margin (pos − hardest neg)")
        plt.legend()
        plt.title(f"t-SNE hardest videos by margin (H={H})")
    else:
        xy = pca_then_embed(Vn[idx], method=embed, random_state=seed, perplexity=30)
        plt.figure(figsize=(8, 7))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=margin[idx], cmap="coolwarm", s=30, marker="o")
        plt.colorbar(sc, label="Margin (pos − hardest neg)")
        plt.title(f"t-SNE hardest videos by margin (H={len(idx)})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("Saved:", out_path)


def plot_emotions(V, VID, out_path, emotion_csv=None, id_col="video_id", emo_col="emotion",
                  max_videos=1000, seed=0, embed="tsne"):
    import csv
    emo_map = {}
    if emotion_csv and os.path.isfile(emotion_csv):
        with open(emotion_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if id_col not in reader.fieldnames or emo_col not in reader.fieldnames:
                print(f"[WARN] CSV missing columns: needs {id_col} and {emo_col}. Skipping emotions.")
            else:
                for row in reader:
                    emo_map[str(row[id_col])] = row[emo_col]
    else:
        print("[WARN] No emotion CSV provided/found. Skipping emotions.")
        return

    # build label array
    labels = []
    for vid in VID:
        labels.append(emo_map.get(str(vid), None))
    labels = np.array(labels)

    # filter to those that have labels
    mask = labels != np.array(None)
    V2 = V[mask]
    labels2 = labels[mask]
    N = V2.shape[0]
    if N == 0:
        print("[WARN] No matching emotion labels found. Skipping.")
        return

    # subsample for readability
    np.random.seed(seed)
    keep = np.random.choice(N, size=min(max_videos, N), replace=False)
    V3 = V2[keep]
    labels3 = labels2[keep]

    # map strings to ints for colouring 
    uniq = sorted(set(labels3.tolist()))
    label_to_int = {u: i for i, u in enumerate(uniq)}
    y = np.array([label_to_int[x] for x in labels3])

    xy = pca_then_embed(V3, method=embed, random_state=seed, perplexity=50)

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=y, cmap="tab10", s=28)
    cb = plt.colorbar(sc, ticks=range(len(uniq)))
    cb.ax.set_yticklabels(uniq)
    plt.title(f"t-SNE of videos colored by emotion (n={len(V3)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Embedding file from export (npz with V, C, VID)")
    ap.add_argument("--mode", choices=["pair", "margin", "emotion"], required=True)
    ap.add_argument("--out", default="tsne_plot.png", help="Output image path")
    ap.add_argument("--embed", choices=["tsne", "umap"], default="tsne", help="Embedding method (UMAP if installed)")
    ap.add_argument("--seed", type=int, default=0)

    # pair mode
    ap.add_argument("--max_videos", type=int, default=200, help="Pair: number of videos to plot (videos + their captions)")
    # margin mode
    ap.add_argument("--hardest", type=int, default=100, help="Margin: number of hardest videos to plot")
    ap.add_argument("--include_captions", action="store_true", help="Margin: include mean captions as gray triangles")
    # mmotion mode
    ap.add_argument("--emotion_csv", default="", help="CSV path mapping video ID to emotion label")
    ap.add_argument("--id_col", default="video_id", help="Column name in CSV for video ID")
    ap.add_argument("--emo_col", default="emotion", help="Column name in CSV for emotion label")
    ap.add_argument("--emotion_max_videos", type=int, default=1000, help="Emotion: subsample this many videos")

    args = ap.parse_args()

    V, C_reshaped, VID = load_npz(args.npz)

    if args.mode == "pair":
        plot_pair_alignment(V, C_reshaped, args.out, max_videos=args.max_videos, seed=args.seed, embed=args.embed)

    elif args.mode == "margin":
        plot_margin_heatmap(V, C_reshaped, args.out, hardest=args.hardest,
                            seed=args.seed, embed=args.embed, include_captions=args.include_captions)

    elif args.mode == "emotion":
        plot_emotions(V, VID, args.out, emotion_csv=args.emotion_csv,
                      id_col=args.id_col, emo_col=args.emo_col,
                      max_videos=args.emotion_max_videos, seed=args.seed, embed=args.embed)

    else:
        raise ValueError("Unknown mode.")


if __name__ == "__main__":
    main()
