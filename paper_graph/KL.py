import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE

# =========================
# 1. 创建输出目录
# =========================
out_dir = "paper_graph"
os.makedirs(out_dir, exist_ok=True)

# =========================
# 2. KL散度矩阵（按你论文填）
# =========================
chains = ["ETH", "BSC", "Polygon", "Fantom"]

kl_matrix = np.array([
    [0.00, 0.31, 0.89, 1.47],
    [0.31, 0.00, 0.72, 1.21],
    [0.89, 0.72, 0.00, 0.95],
    [1.47, 1.21, 0.95, 0.00]
])

# =========================
# 3. 图1：KL Heatmap（论文标准）
# =========================
plt.figure(figsize=(6,5))
sns.heatmap(
    kl_matrix,
    annot=True,
    fmt=".2f",
    cmap="Reds",
    xticklabels=chains,
    yticklabels=chains,
    cbar_kws={'label': 'KL Divergence'}
)
plt.title("Cross-chain Distribution Shift (KL Divergence)")
plt.tight_layout()

heatmap_path = os.path.join(out_dir, "kl_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"[OK] Saved: {heatmap_path}")

# =========================
# 4. 图2：KL Graph（你想要的那种图）
# =========================
G = nx.Graph()

# 加节点
for c in chains:
    G.add_node(c)

# 加边（KL值）
for i in range(len(chains)):
    for j in range(i+1, len(chains)):
        G.add_edge(chains[i], chains[j], weight=kl_matrix[i][j])

# layout（关键：越远表示KL越大）
pos = nx.spring_layout(G, weight='weight', seed=42)

plt.figure(figsize=(6,6))

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

# 节点
nx.draw_networkx_nodes(G, pos, node_size=2000)

# 边（宽度 = KL）
nx.draw_networkx_edges(
    G, pos,
    width=[w*2 for w in weights],
    alpha=0.6
)

# 标签
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 边权显示
edge_labels = {(u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Cross-chain Distribution Graph (KL Divergence)")
plt.axis('off')

graph_path = os.path.join(out_dir, "kl_graph.png")
plt.savefig(graph_path, dpi=300)
plt.close()

print(f"[OK] Saved: {graph_path}")

# =========================
# 5. 图3：t-SNE 分布图（加分项）
# =========================
# ⚠️ 这里用模拟数据（你后面可以换成真实14维特征）

np.random.seed(42)

# 模拟每条链数据（你可以替换成真实特征）
n = 200

eth = np.random.normal(loc=0.0, scale=1.0, size=(n, 14))
bsc = np.random.normal(loc=1.0, scale=1.0, size=(n, 14))
polygon = np.random.normal(loc=2.0, scale=1.2, size=(n, 14))
fantom = np.random.normal(loc=3.0, scale=1.3, size=(n, 14))

X = np.vstack([eth, bsc, polygon, fantom])
labels = (["ETH"]*n + ["BSC"]*n + ["Polygon"]*n + ["Fantom"]*n)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_emb = tsne.fit_transform(X)

plt.figure(figsize=(6,5))

colors = {
    "ETH": "red",
    "BSC": "blue",
    "Polygon": "green",
    "Fantom": "purple"
}

for chain in chains:
    idx = [i for i,l in enumerate(labels) if l==chain]
    plt.scatter(
        X_emb[idx,0],
        X_emb[idx,1],
        label=chain,
        alpha=0.6,
        s=10,
        color=colors[chain]
    )

plt.legend()
plt.title("Cross-chain Distribution Visualization (t-SNE)")
plt.tight_layout()

tsne_path = os.path.join(out_dir, "tsne_distribution.png")
plt.savefig(tsne_path, dpi=300)
plt.close()

print(f"[OK] Saved: {tsne_path}")

print("\n✅ All figures generated in folder: paper_graph/")