import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _annotated_heatmap(ax, data, xlabels, ylabels, title, is_percent=False):
    im = ax.imshow(data, aspect='equal', cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate each cell
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            val = data[i, j]
            txt = f"{val:.2f}" if is_percent else f"{int(round(val))}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color="black")

    plt.tight_layout()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cm_csv", type=str, help="Path to confusion_matrix.csv (raw counts).")
    args = ap.parse_args()

    cm_df = pd.read_csv(args.cm_csv, index_col=0)
    classes = list(cm_df.index)
    if list(cm_df.columns) != classes:
        cm_df = cm_df.reindex(columns=classes)

    out_dir = os.path.dirname(os.path.abspath(args.cm_csv))
    counts_png = os.path.join(out_dir, "confusion_matrix_counts.png")
    perc_png   = os.path.join(out_dir, "confusion_matrix_percents.png")
    perc_csv   = os.path.join(out_dir, "confusion_matrix_percents.csv")

    # raw counts
    cm_counts = cm_df.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    _annotated_heatmap(ax, cm_counts, classes, classes, "Confusion Matrix (Counts)", is_percent=False)
    plt.savefig(counts_png, dpi=200)
    plt.close(fig)

    # row-normalised percentages
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_perc = (cm_counts / row_sums) * 100.0
    pd.DataFrame(np.round(cm_perc, 2), index=classes, columns=classes).to_csv(perc_csv)

    fig, ax = plt.subplots(figsize=(8, 7))
    _annotated_heatmap(ax, cm_perc, classes, classes, "Confusion Matrix (Row %)", is_percent=True)
    plt.savefig(perc_png, dpi=200)
    plt.close(fig)

    print(f"Saved:\n  {counts_png}\n  {perc_png}\n  {perc_csv}")

if __name__ == "__main__":
    main()
