import os
import matplotlib.pyplot as plt

# =========================
# 输出目录
# =========================
out_dir = "paper_graph"
os.makedirs(out_dir, exist_ok=True)

# =========================
# Few-shot数据（按你论文填）
# =========================
shots = [5, 10, 20, 50]

# 你可以替换成真实结果
f1_fantom = [0.816, 0.879, 0.891, 0.894]
f1_polygon = [0.801, 0.847, 0.859, 0.862]
f1_bsc = [0.932, 0.941, 0.946, 0.948]

# =========================
# 画图
# =========================
plt.figure(figsize=(6,5))

plt.plot(shots, f1_fantom, marker='o', label="ETH→Fantom")
plt.plot(shots, f1_polygon, marker='s', label="ETH→Polygon")
plt.plot(shots, f1_bsc, marker='^', label="ETH→BSC")

# 标注关键点（10-shot）
plt.scatter([10], [0.879], s=80)
plt.text(10, 0.879+0.005, "10-shot", ha='center')
plt.axhline(y=0.731, linestyle='--', label="Without Prototype")
plt.xlabel("Number of Shots")
plt.ylabel("F1 Score")
plt.title("Few-shot Adaptation Performance")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

save_path = os.path.join(out_dir, "fewshot_curve.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"[OK] Saved: {save_path}")