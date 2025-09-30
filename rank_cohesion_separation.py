import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Rank emotion classes and plot bar charts.")
    ap.add_argument("--csv", required=True, help="Path to per_class_cohesion.csv")
    ap.add_argument("--out_dir", default="class_analysis_min", help="Output directory")
    ap.add_argument("--figsize", type=float, nargs=2, default=(10, 5), help="Matplotlib figsize, e.g. 10 5")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # rankings
    rank_video = (
        df[["class", "n_videos", "video_intra_sim_mean"]]
        .sort_values("video_intra_sim_mean", ascending=False)
        .reset_index(drop=True)
    )
    rank_caption = (
        df[["class", "n_videos", "caption_diversity_mean"]]
        .sort_values("caption_diversity_mean", ascending=True)
        .reset_index(drop=True)
    )
    rank_silhouette = (
        df[["class", "n_videos", "silhouette_mean"]]
        .sort_values("silhouette_mean", ascending=False)
        .reset_index(drop=True)
    )

    # combined CSV
    combined = (
        rank_video[["class", "video_intra_sim_mean"]]
        .rename(columns={"video_intra_sim_mean": "rank_metric_video"})
        .merge(
            rank_caption[["class", "caption_diversity_mean"]]
            .rename(columns={"caption_diversity_mean": "rank_metric_caption"}),
            on="class",
        )
        .merge(
            rank_silhouette[["class", "silhouette_mean"]]
            .rename(columns={"silhouette_mean": "rank_metric_silhouette"}),
            on="class",
        )
    )

    combined_path = os.path.join(args.out_dir, "rankings_combined.csv")
    combined.to_csv(combined_path, index=False)

    # plot helpers
    def plot_bar(x, y, ylabel, title, outfile, rotate=45, invert_for_caption=False):
        plt.figure(figsize=tuple(args.figsize))
        plt.bar(x, y)
        plt.ylabel(ylabel)
        plt.xlabel("Emotion class")
        plt.title(title)
        plt.xticks(rotation=rotate, ha="right")

        ax = plt.gca()
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, rotation=0)

        plt.tight_layout()
        plt.savefig(outfile, dpi=220)
        plt.close()

    # bar charts
    # video cohesion
    plot_bar(
        rank_video["class"],
        rank_video["video_intra_sim_mean"],
        ylabel="Video intra-class cosine similarity (mean)",
        title="Video Cohesion by Class (higher = tighter)",
        outfile=os.path.join(args.out_dir, "bar_video_cohesion.png"),
    )

    # caption consistency
    plot_bar(
        rank_caption["class"],
        rank_caption["caption_diversity_mean"],
        ylabel="Caption diversity (mean pairwise cosine distance)",
        title="Caption Consistency by Class (lower = more consistent)",
        outfile=os.path.join(args.out_dir, "bar_caption_consistency.png"),
    )

    # class separation
    plot_bar(
        rank_silhouette["class"],
        rank_silhouette["silhouette_mean"],
        ylabel="Silhouette score (cosine distance)",
        title="Class Separation by Silhouette (higher = better)",
        outfile=os.path.join(args.out_dir, "bar_class_separation.png"),
    )

    # console summary
    def print_rank(label, df_sub, col, ascending=False):
        print(f"\n=== {label} ===")
        for i, r in df_sub.iterrows():
            print(f"{i+1}. {r['class']}: {col}={r[col]:.6f} (n={int(r.get('n_videos', 0))})")

    print_rank("Video Cohesion (higher is better)", rank_video, "video_intra_sim_mean")
    print_rank("Caption Consistency (lower is better)", rank_caption, "caption_diversity_mean")
    print_rank("Class Separation (higher is better)", rank_silhouette, "silhouette_mean")

    print("\nSaved:")
    print(" -", combined_path)
    print(" -", os.path.join(args.out_dir, "bar_video_cohesion.png"))
    print(" -", os.path.join(args.out_dir, "bar_caption_consistency.png"))
    print(" -", os.path.join(args.out_dir, "bar_class_separation.png"))


if __name__ == "__main__":
    main()

