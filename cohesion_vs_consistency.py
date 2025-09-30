import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_class_cohesion.csv")

x = df["video_intra_sim_mean"] # video cohesion
y = df["caption_diversity_mean"] # caption diversity
c = df["silhouette_mean"] # silhouete

plt.figure(figsize=(9,7))
sc = plt.scatter(x, y, c=c, cmap="coolwarm", s=120, edgecolor="k")

# annotate each point with the class name
for i, row in df.iterrows():
    plt.text(row["video_intra_sim_mean"]+0.001,
             row["caption_diversity_mean"]+0.001,
             row["class"], fontsize=8)

plt.xlabel("Video Cohesion (intra-class similarity ↑ = tighter)")
plt.ylabel("Caption Diversity (pairwise distance ↓ = more consistent)")
plt.title("Class Cohesion vs Caption Consistency\n(Color = Separation by Silhouette Score)")

# colourbar for silhouette
cbar = plt.colorbar(sc)
cbar.set_label("Silhouette Score (↑ = better separation)")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_cohesion_vs_caption_silhouette.png", dpi=220)
plt.show()
