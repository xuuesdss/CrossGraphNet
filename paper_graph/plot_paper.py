import os
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

# =========================
# 全局风格
# =========================
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

OUT_DIR = "paper_graph"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 颜色配置
# =========================
COLORS = {
    "main": "#c44e52",         # CrossGraphNet 主色
    "main_soft": "#e9b4b6",
    "baseline_dark": "#4c72b0",
    "baseline_mid": "#8172b2",
    "baseline_soft": "#64b5cd",
    "gray1": "#2f2f2f",
    "gray2": "#666666",
    "gray3": "#aaaaaa",
    "gray4": "#dddddd",
    "eth": "#d95f5f",
    "bsc": "#4c72b0",
    "polygon": "#55a868",
    "fantom": "#8172b2",
}

CHAINS = ["ETH", "BSC", "Polygon", "Fantom"]


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


# =========================
# 图1：跨链 KL 力导向关系图
# =========================
def plot_kl_graph():
    kl_matrix = np.array([
        [0.00, 0.31, 0.89, 1.47],
        [0.31, 0.00, 0.72, 1.21],
        [0.89, 0.72, 0.00, 0.95],
        [1.47, 1.21, 0.95, 0.00]
    ])

    G = nx.Graph()
    for c in CHAINS:
        G.add_node(c)

    for i in range(len(CHAINS)):
        for j in range(i + 1, len(CHAINS)):
            G.add_edge(CHAINS[i], CHAINS[j], weight=kl_matrix[i, j])

    # 权重大表示距离大，所以这里用自定义位置而不是完全依赖spring_layout
    # 先用 spring_layout，再做轻微拉伸
    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.3, iterations=200)

    # 节点样式
    node_sizes = {
        "ETH": 2400,
        "BSC": 1800,
        "Polygon": 1600,
        "Fantom": 1200,
    }
    node_colors = {
        "ETH": COLORS["eth"],
        "BSC": COLORS["bsc"],
        "Polygon": COLORS["polygon"],
        "Fantom": COLORS["fantom"],
    }

    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    ax.set_facecolor("white")

    # 边
    edges = list(G.edges(data=True))
    weights = [e[2]["weight"] for e in edges]
    w_min, w_max = min(weights), max(weights)

    for u, v, d in edges:
        w = d["weight"]
        norm = (w - w_min) / (w_max - w_min + 1e-8)
        width = 1.5 + 5.0 * norm
        alpha = 0.25 + 0.35 * norm
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color="#c9b2b2"
        )

    # 节点发光效果
    for n in G.nodes():
        x, y = pos[n]
        plt.scatter(x, y, s=node_sizes[n] * 1.8, color=node_colors[n], alpha=0.08, zorder=2)
        plt.scatter(x, y, s=node_sizes[n] * 1.3, color=node_colors[n], alpha=0.12, zorder=3)
        plt.scatter(x, y, s=node_sizes[n], color=node_colors[n], alpha=0.95,
                    edgecolors="white", linewidths=1.5, zorder=4)

    # 标签
    for n, (x, y) in pos.items():
        plt.text(x, y, n, ha="center", va="center", fontsize=12,
                 color="white", fontweight="bold", zorder=5)

    # 边权标注
    for u, v, d in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(xm, ym, f"{d['weight']:.2f}", fontsize=9, color=COLORS["gray2"],
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75))

    plt.title("Cross-chain Distribution Graph", pad=12)
    plt.axis("off")
    savefig("fig_kl_graph.png")


# =========================
# 图2：跨链 t-SNE 云图
# 如果你后面有真实14维特征，把这里模拟数据替换掉
# =========================
def plot_chain_embedding_tsne():
    np.random.seed(42)

    n = 220
    eth = np.random.normal(loc=0.0, scale=1.00, size=(n, 14))
    bsc = np.random.normal(loc=0.8, scale=1.05, size=(n, 14))
    polygon = np.random.normal(loc=1.8, scale=1.10, size=(n, 14))
    fantom = np.random.normal(loc=2.6, scale=1.20, size=(n, 14))

    X = np.vstack([eth, bsc, polygon, fantom])
    y = (["ETH"] * n) + (["BSC"] * n) + (["Polygon"] * n) + (["Fantom"] * n)

    tsne = TSNE(n_components=2, random_state=42, perplexity=35, init="pca")
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6.5))
    ax = plt.gca()
    ax.set_facecolor("white")

    color_map = {
        "ETH": COLORS["eth"],
        "BSC": COLORS["bsc"],
        "Polygon": COLORS["polygon"],
        "Fantom": COLORS["fantom"],
    }

    for chain in CHAINS:
        idx = [i for i, c in enumerate(y) if c == chain]
        pts = emb[idx]
        # 发光层
        plt.scatter(pts[:, 0], pts[:, 1], s=40, color=color_map[chain], alpha=0.06, linewidths=0)
        # 实点层
        plt.scatter(pts[:, 0], pts[:, 1], s=14, color=color_map[chain], alpha=0.55,
                    linewidths=0, label=chain)

        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        plt.scatter(cx, cy, s=180, color=color_map[chain], edgecolors="white",
                    linewidths=1.5, zorder=4)
        plt.text(cx, cy, chain, fontsize=10, ha="center", va="center",
                 color="white", fontweight="bold", zorder=5)

    plt.title("Cross-chain Distribution Visualization (t-SNE)", pad=12)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(frameon=False, ncol=2)
    savefig("fig_tsne_distribution.png")


# =========================
# 图3：RQ1 总体方法对比 —— Cleveland Dot Plot
# =========================
def plot_rq1_method_comparison():
    methods = [
        "Devign", "ReGVD", "ESCORT", "AMEVulDetector",
        "SCVHunter", "LineVul", "ContraBERT-FT", "CrossGraphNet"
    ]

    scores = {
        "ETH":      [0.741, 0.769, 0.756, 0.762, 0.778, 0.723, 0.804, 0.921],
        "BSC":      [0.768, 0.795, 0.783, 0.791, 0.801, 0.751, 0.826, 0.938],
        "Polygon":  [0.692, 0.721, 0.708, 0.715, 0.729, 0.681, 0.751, 0.894],
        "Fantom":   [0.731, 0.758, 0.749, 0.756, 0.763, 0.712, 0.788, 0.921],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    chain_colors = [COLORS["eth"], COLORS["bsc"], COLORS["polygon"], COLORS["fantom"]]

    for ax, chain, c in zip(axes, ["ETH", "BSC", "Polygon", "Fantom"], chain_colors):
        vals = scores[chain]
        y = np.arange(len(methods))

        # 轻灰背景线
        for yi, v in zip(y, vals):
            ax.plot([0.65, v], [yi, yi], color=COLORS["gray4"], linewidth=1.2, zorder=1)

        # 先画普通方法
        for yi, m, v in zip(y, methods, vals):
            if m != "CrossGraphNet":
                ax.scatter(v, yi, s=70, color=COLORS["gray3"], edgecolors="white",
                           linewidths=1.0, zorder=3)
            else:
                ax.scatter(v, yi, s=180, color=COLORS["main"], edgecolors="white",
                           linewidths=1.4, zorder=4)
                ax.scatter(v, yi, s=340, color=COLORS["main"], alpha=0.12, zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_xlim(0.65, 0.96)
        ax.set_title(chain, color=c, pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle("RQ1: Overall Comparison Across Chains", fontsize=15, y=1.02)
    fig.supxlabel("F1 Score", y=0.03)
    savefig("fig_rq1_method_comparison.png")


# =========================
# 图4：RQ2 静态工具对比 —— 哑铃图
# =========================
def plot_rq2_static_tool_dumbbell():
    chains = ["Ethereum", "BSC", "Polygon", "Fantom"]
    slither = np.array([0.412, 0.822, 0.790, 0.844])
    mythril = np.array([0.438, 0.791, 0.762, 0.819])
    cross = np.array([0.921, 0.938, 0.894, 0.921])

    y = np.arange(len(chains))

    plt.figure(figsize=(9, 5.8))

    # Slither -> Cross
    for i in range(len(chains)):
        plt.plot([slither[i], cross[i]], [y[i] + 0.08, y[i] + 0.08],
                 color="#d8c4c4", linewidth=2.5, alpha=0.9, zorder=1)
        plt.plot([mythril[i], cross[i]], [y[i] - 0.08, y[i] - 0.08],
                 color="#d5d5e8", linewidth=2.5, alpha=0.9, zorder=1)

    plt.scatter(slither, y + 0.08, s=80, color=COLORS["baseline_dark"], label="Slither",
                edgecolors="white", linewidths=1.0, zorder=3)
    plt.scatter(mythril, y - 0.08, s=80, color=COLORS["baseline_mid"], label="Mythril",
                edgecolors="white", linewidths=1.0, zorder=3)
    plt.scatter(cross, y, s=170, color=COLORS["main"], label="CrossGraphNet",
                edgecolors="white", linewidths=1.3, zorder=4)
    plt.scatter(cross, y, s=320, color=COLORS["main"], alpha=0.10, zorder=2)

    for xv, yv in zip(cross, y):
        plt.text(xv + 0.008, yv, f"{xv:.3f}", va="center", fontsize=9, color=COLORS["gray2"])

    plt.yticks(y, chains)
    plt.xlabel("F1 Score")
    plt.title("RQ2: Comparison with Static Analysis Tools", pad=12)
    plt.xlim(0.35, 0.98)
    plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.legend(frameon=False, ncol=3, loc="lower right")
    savefig("fig_rq2_static_tool_dumbbell.png")


# =========================
# 图5：RQ3 KL vs F1 散点关系图
# =========================
def plot_rq3_kl_vs_f1():
    kl = np.array([0.31, 0.89, 1.47])
    f1_direct = np.array([0.929, 0.785, 0.731])
    f1_proto = np.array([0.941, 0.847, 0.879])
    labels = ["ETH→BSC", "ETH→Polygon", "ETH→Fantom"]
    colors = [COLORS["bsc"], COLORS["polygon"], COLORS["fantom"]]

    plt.figure(figsize=(8.2, 6))

    # 连线表示原型学习提升
    for x, y1, y2, lab, c in zip(kl, f1_direct, f1_proto, labels, colors):
        plt.plot([x, x], [y1, y2], color=c, linewidth=2.8, alpha=0.75)
        plt.scatter(x, y1, s=80, color="white", edgecolors=c, linewidths=2.0, zorder=3)
        plt.scatter(x, y2, s=130, color=c, edgecolors="white", linewidths=1.2, zorder=4)
        plt.text(x + 0.02, y2 + 0.005, lab, fontsize=9, color=COLORS["gray2"])

    # 趋势线（直接迁移）
    z = np.polyfit(kl, f1_direct, 1)
    p = np.poly1d(z)
    xx = np.linspace(0.25, 1.55, 100)
    plt.plot(xx, p(xx), linestyle="--", color=COLORS["gray2"], alpha=0.8, linewidth=1.6)

    plt.xlabel("KL Divergence")
    plt.ylabel("F1 Score")
    plt.title("RQ3: Impact of Distribution Shift on Cross-chain Generalization", pad=12)
    plt.grid(linestyle="--", alpha=0.25)

    # 手工图例
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=COLORS["gray2"], markeredgewidth=1.8, markersize=8, label='Direct Transfer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["main"],
               markeredgecolor='white', markersize=10, label='+ Prototype (10-shot)')
    ]
    plt.legend(handles=handles, frameon=False, loc="lower left")
    savefig("fig_rq3_kl_vs_f1.png")


# =========================
# 图6：RQ3 few-shot 高级曲线图
# =========================
def plot_rq3_fewshot_curve():
    shots = np.array([5, 10, 20, 50])

    f1_bsc = np.array([0.932, 0.941, 0.946, 0.948])
    f1_polygon = np.array([0.801, 0.847, 0.859, 0.862])
    f1_fantom = np.array([0.816, 0.879, 0.891, 0.894])

    no_proto = {
        "ETH→BSC": 0.929,
        "ETH→Polygon": 0.785,
        "ETH→Fantom": 0.731,
    }

    plt.figure(figsize=(8.5, 6))

    def draw_curve(x, y, color, marker, label, baseline=None):
        # 柔和发光线
        plt.plot(x, y, color=color, linewidth=8, alpha=0.08)
        plt.plot(x, y, color=color, linewidth=2.6, marker=marker, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.1, label=label)
        if baseline is not None:
            plt.axhline(y=baseline, color=color, linestyle="--", linewidth=1.3, alpha=0.5)

    draw_curve(shots, f1_bsc, COLORS["bsc"], "^", "ETH→BSC", baseline=no_proto["ETH→BSC"])
    draw_curve(shots, f1_polygon, COLORS["polygon"], "s", "ETH→Polygon", baseline=no_proto["ETH→Polygon"])
    draw_curve(shots, f1_fantom, COLORS["fantom"], "o", "ETH→Fantom", baseline=no_proto["ETH→Fantom"])

    # 标出10-shot
    plt.axvline(10, linestyle=":", color=COLORS["gray3"], linewidth=1.3)
    plt.text(10.5, 0.944, "10-shot sweet spot", fontsize=10, color=COLORS["gray2"])

    plt.xlabel("Number of Shots")
    plt.ylabel("F1 Score")
    plt.title("RQ3: Few-shot Prototype Adaptation", pad=12)
    plt.xticks(shots)
    plt.grid(linestyle="--", alpha=0.22)
    plt.legend(frameon=False, loc="lower right")
    savefig("fig_rq3_fewshot_curve.png")


# =========================
# 图7：RQ4 消融实验 —— Waterfall风格
# =========================
def plot_rq4_ablation_waterfall():
    labels = [
        "AST only",
        "CFG only",
        "DFG only",
        "Naive Fusion",
        "+ Gating",
        "+ CodeBERT-FT",
        "+ Chain-aware Agg.",
        "Full Model"
    ]
    values = [0.684, 0.712, 0.698, 0.890, 0.903, 0.914, 0.919, 0.921]

    plt.figure(figsize=(10.5, 5.8))
    x = np.arange(len(labels))

    # 颜色：前面灰，最后高亮
    bar_colors = [COLORS["gray3"]] * len(labels)
    bar_colors[3] = "#c9d7ef"
    bar_colors[4] = "#b9d9c0"
    bar_colors[5] = "#d6c7e8"
    bar_colors[6] = "#f1d8d8"
    bar_colors[7] = COLORS["main"]

    # 画柱
    bars = plt.bar(x, values, color=bar_colors, edgecolor="white", linewidth=1.0, zorder=3)

    # 发光突出完整模型
    plt.scatter(x[-1], values[-1], s=700, color=COLORS["main"], alpha=0.10, zorder=2)

    # 连接线
    plt.plot(x, values, color=COLORS["gray2"], linestyle="--", linewidth=1.4, alpha=0.75, zorder=4)

    for xi, yi in zip(x, values):
        plt.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.ylim(0.64, 0.95)
    plt.ylabel("F1 Score")
    plt.title("RQ4: Incremental Contribution of Model Components", pad=12)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.22)
    savefig("fig_rq4_ablation_waterfall.png")


# =========================
# 图8：RQ5 联邦策略 —— Pareto散点图
# x=方差（越小越好），y=平均F1（越大越好）
# =========================
def plot_rq5_federated_pareto():
    methods = ["Centralized", "FedAvg", "FedProx", "SCAFFOLD", "FedNova", "CrossGraphNet"]
    avg_f1 = np.array([0.921, 0.899, 0.903, 0.908, 0.905, 0.919])
    variance = np.array([0.031, 0.019, 0.017, 0.015, 0.016, 0.009])

    plt.figure(figsize=(8.3, 6))
    for m, x, y in zip(methods, variance, avg_f1):
        if m == "CrossGraphNet":
            plt.scatter(x, y, s=260, color=COLORS["main"], edgecolors="white",
                        linewidths=1.4, zorder=4)
            plt.scatter(x, y, s=520, color=COLORS["main"], alpha=0.10, zorder=2)
            plt.text(x + 0.0008, y + 0.0008, m, fontsize=10, color=COLORS["main"], fontweight="bold")
        elif m == "Centralized":
            plt.scatter(x, y, s=150, color=COLORS["baseline_dark"], edgecolors="white",
                        linewidths=1.2, zorder=3)
            plt.text(x + 0.0008, y + 0.0006, m, fontsize=9, color=COLORS["gray2"])
        else:
            plt.scatter(x, y, s=120, color=COLORS["gray3"], edgecolors="white",
                        linewidths=1.0, zorder=3)
            plt.text(x + 0.0008, y + 0.0006, m, fontsize=9, color=COLORS["gray2"])

    # 标注更优区域
    plt.annotate("Better",
                 xy=(0.010, 0.918), xytext=(0.020, 0.911),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["gray2"]),
                 fontsize=10, color=COLORS["gray2"])

    plt.xlabel("Variance Across Chains (lower is better)")
    plt.ylabel("Average F1 Score (higher is better)")
    plt.title("RQ5: Performance–Stability Trade-off of Federated Strategies", pad=12)
    plt.grid(linestyle="--", alpha=0.22)
    savefig("fig_rq5_federated_pareto.png")


# =========================
# 图9：RQ6 错误分析 —— 环形图
# =========================
def plot_rq6_error_donut():
    labels = ["Indirect Reentrancy\n(Proxy Pattern)", "Conditional Overflow", "Rare Patterns"]
    values = [56, 28, 16]
    colors = [COLORS["fantom"], COLORS["polygon"], COLORS["baseline_soft"]]

    plt.figure(figsize=(7, 6.4))
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
        textprops=dict(color=COLORS["gray1"], fontsize=10)
    )

    # 中心圆
    centre_circle = plt.Circle((0, 0), 0.34, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.text(0, 0.04, "RQ6", ha="center", va="center", fontsize=16,
             color=COLORS["main"], fontweight="bold")
    plt.text(0, -0.11, "Error Patterns", ha="center", va="center", fontsize=10,
             color=COLORS["gray2"])

    plt.title("RQ6: Distribution of False Negative Patterns", pad=12)
    savefig("fig_rq6_error_donut.png")


# =========================
# 主函数
# =========================
def main():
    plot_kl_graph()
    plot_chain_embedding_tsne()
    plot_rq1_method_comparison()
    plot_rq2_static_tool_dumbbell()
    plot_rq3_kl_vs_f1()
    plot_rq3_fewshot_curve()
    plot_rq4_ablation_waterfall()
    plot_rq5_federated_pareto()
    plot_rq6_error_donut()
    print("\n✅ All figures have been generated in folder: paper_graph/")


if __name__ == "__main__":
    main()