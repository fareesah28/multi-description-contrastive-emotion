import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _annotated(ax, data, xlabels, ylabels, title, is_percent=False, cmap="YlGnBu"):
    im = ax.imshow(data, aspect='equal', cmap=cmap, vmin=0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    h, w = data.shape
    vmax = np.nanmax(data) if np.nanmax(data) > 0 else 1
    threshold = vmax / 2.0

    # annotate each cell
    for i in range(h):
        for j in range(w):
            val = data[i, j]
            txt = f"{val:.2f}" if is_percent else f"{int(round(val))}"
            color = "white" if val > threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts_csv", type=str, default="confusion_matrix_counts.csv",
                    help="CSV with raw confusion matrix counts.")
    ap.add_argument("--percents_csv", type=str, default="confusion_matrix_percents.csv",
                    help="CSV with row-normalized percentage matrix.")
    ap.add_argument("--cmap", type=str, default="YlGnBu",
                    help="Matplotlib colormap (e.g., OrRd, PuRd, viridis).")
    args = ap.parse_args()

    counts_df = pd.read_csv(args.counts_csv, index_col=0)
    classes = list(counts_df.index)
    if list(counts_df.columns) != classes:
        counts_df = counts_df.reindex(columns=classes)
    counts = counts_df.to_numpy()

    out_dir = os.path.dirname(os.path.abspath(args.counts_csv)) or "."
    counts_png = os.path.join(out_dir, "confusion_matrix_counts.png")

    fig, ax = plt.subplots(figsize=(max(9, 0.7*len(classes)), max(8, 0.7*len(classes))))
    _annotated(ax, counts, classes, classes, "Confusion Matrix (Counts)",
               is_percent=False, cmap=args.cmap)
    plt.savefig(counts_png, dpi=220); plt.close(fig)

    perc_df = pd.read_csv(args.percents_csv, index_col=0)
    if list(perc_df.columns) != classes:
        perc_df = perc_df.reindex(columns=classes).reindex(index=classes)
    perc = perc_df.to_numpy()

    perc_png = os.path.join(out_dir, "confusion_matrix_percents.png")
    fig, ax = plt.subplots(figsize=(max(9, 0.7*len(classes)), max(8, 0.7*len(classes))))
    _annotated(ax, perc, classes, classes, "Confusion Matrix (Row %)",
               is_percent=True, cmap=args.cmap)
    plt.savefig(perc_png, dpi=220); plt.close(fig)

    print(f"Saved:\n  {counts_png}\n  {perc_png}")

if __name__ == "__main__":
    main()
