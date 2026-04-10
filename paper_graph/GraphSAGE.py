import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

def arrow(ax, p1, p2, color="black", lw=1.6, ms=14, alpha=1.0, z=3):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=z
    )
    ax.add_patch(a)

def capsule(ax, center, w, h, text, fc="#f7f7f7", ec="black",
            fontsize=11, weight="bold", z=2):
    x = center[0] - w / 2
    y = center[1] - h / 2
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
        zorder=z
    )
    ax.add_patch(patch)
    ax.text(center[0], center[1], text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=z+1)

def node(ax, x, y, r=0.028, fc="white", ec="black", lw=1.3, alpha=1.0, z=4):
    c = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(c)

def edge(ax, p1, p2, color="gray", lw=1.2, alpha=0.8, z=1):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=alpha, zorder=z)

def vertical_embedding_column(ax, x, y_bottom, n=5, h=0.03, gap=0.012, width=0.035,
                              fc="#d9e6ff", ec="black", alpha=1.0):
    centers = []
    for i in range(n):
        yc = y_bottom + i * (h + gap)
        patch = FancyBboxPatch(
            (x - width/2, yc), width, h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
            alpha=alpha,
            zorder=3
        )
        ax.add_patch(patch)
        centers.append((x, yc + h/2))
    return centers

def draw_pretty_graph_encoding(save_path="graph_encoding_nn_style.png"):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "Graph Encoding", ha="center", va="center",
            fontsize=20, fontweight="bold")

    # =========================
    # Left: input graph
    # =========================
    ax.text(0.14, 0.86, "Input graph", fontsize=12, fontweight="bold", ha="center")

    pos = {
        "c":  (0.14, 0.52),
        "n1": (0.08, 0.66),
        "n2": (0.20, 0.66),
        "n3": (0.06, 0.50),
        "n4": (0.10, 0.36),
        "n5": (0.18, 0.36),
        "n6": (0.22, 0.50),
        "n7": (0.14, 0.78),
    }
    edges_ = [
        ("c", "n1"), ("c", "n2"), ("c", "n3"), ("c", "n4"), ("c", "n5"), ("c", "n6"), ("c", "n7"),
        ("n1", "n3"), ("n2", "n6"), ("n4", "n5")
    ]
    for u, v in edges_:
        edge(ax, pos[u], pos[v], color="#9a9a9a", lw=1.4, alpha=0.9)

    for k, (x, y) in pos.items():
        if k == "c":
            node(ax, x, y, r=0.035, fc="#c96b1f", lw=1.5)
        elif k in ["n1", "n2", "n4", "n5"]:
            node(ax, x, y, r=0.032, fc="#9eb8f2", lw=1.4)
        else:
            node(ax, x, y, r=0.03, fc="white", lw=1.3)

    ax.text(0.055, 0.80, r"$k=1$", fontsize=11)
    ax.text(0.245, 0.80, "sampled\nneighbors", fontsize=10, ha="left", va="center")

    # =========================
    # Middle-left: GraphSAGE message passing
    # =========================
    arrow(ax, (0.27, 0.52), (0.34, 0.52), lw=1.8)

    capsule(ax, (0.43, 0.52), 0.15, 0.09, "GraphSAGE\nencoder", fc="#eef3ff", fontsize=12)

    # small feature columns around encoder
    vertical_embedding_column(ax, 0.36, 0.34, n=4, h=0.03, gap=0.012, width=0.03, fc="#d7e6c9")
    vertical_embedding_column(ax, 0.50, 0.34, n=4, h=0.03, gap=0.012, width=0.03, fc="#d7e6c9")

    # connections to encoder
    for yy in [0.375, 0.417, 0.459, 0.501]:
        arrow(ax, (0.375, yy), (0.355, 0.52), color="#5d8f56", lw=1.2, ms=10)
        arrow(ax, (0.485, yy), (0.505, 0.52), color="#5d8f56", lw=1.2, ms=10)

    # =========================
    # Middle: node embeddings
    # =========================
    arrow(ax, (0.515, 0.52), (0.59, 0.52), lw=1.8)

    ax.text(0.64, 0.86, "Node embeddings", fontsize=12, fontweight="bold", ha="center")

    x_positions = [0.60, 0.64, 0.68]
    colors = ["#cfe0ff", "#cfe0ff", "#cfe0ff"]
    for i, x in enumerate(x_positions):
        vertical_embedding_column(ax, x, 0.33 + i*0.015, n=5, h=0.032, gap=0.012,
                                  width=0.032, fc=colors[i])

    ax.text(0.64, 0.28, r"$\mathbf{H} = \{h_1, h_2, \dots, h_n\}$",
            fontsize=11, ha="center")

    # =========================
    # Pooling
    # =========================
    arrow(ax, (0.72, 0.52), (0.79, 0.52), lw=1.8)
    capsule(ax, (0.86, 0.52), 0.15, 0.085, "Global mean\npooling", fc="#f3efe4", fontsize=12)

    # =========================
    # Output embedding
    # =========================
    arrow(ax, (0.86, 0.47), (0.86, 0.36), lw=1.8)
    capsule(ax, (0.86, 0.25), 0.18, 0.08, r"graph embedding $h_G$",
            fc="#fff6d8", fontsize=12)

    # Side note
    ax.text(0.5, 0.08,
            "Neighborhood messages are aggregated into node embeddings,\n"
            "which are pooled into a graph-level representation.",
            fontsize=10.5, ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    draw_pretty_graph_encoding("graph_encoding_nn_style.png")