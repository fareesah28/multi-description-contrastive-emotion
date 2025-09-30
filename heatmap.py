import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# load centroid cosine similarity matrix
df_centroid = pd.read_csv("centroid_cosine_similarity.csv", index_col=0)

# cluster the classes 
dist = 1 - df_centroid.values  # cosine similarity → distance
link = linkage(squareform(dist, checks=False), method="average")
order = leaves_list(link)

df_ordered = df_centroid.iloc[order, order]
ordered_classes = df_ordered.index.tolist()

# heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    df_ordered,
    cmap="coolwarm",
    annot=True, fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Cosine Similarity"}
)
plt.title("Centroid Cosine Similarities (Class Separation)")
plt.tight_layout()
plt.savefig("centroid_similarity_heatmap.png", dpi=220)
plt.show()

# find strongest & weakest pairs 
pairs = []
for i in range(len(ordered_classes)):
    for j in range(i+1, len(ordered_classes)):
        sim = df_ordered.iloc[i, j]
        pairs.append((ordered_classes[i], ordered_classes[j], sim))

pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

print("Strongest Similarity Pairs")
for p in pairs[:2]:
    print(f"{p[0]} ↔ {p[1]} = {p[2]:.3f}")

print("\nWeakest Similarity Pairs")
for p in pairs[-2:]:
    print(f"{p[0]} ↔ {p[1]} = {p[2]:.3f}")
