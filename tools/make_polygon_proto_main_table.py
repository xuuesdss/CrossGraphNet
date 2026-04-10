import pandas as pd

df = pd.read_csv("results/polygon_evalplus_by_seed.csv")

# group by proto
agg = df.groupby("use_proto").agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    auc_mean=("auc", "mean"),
    auc_std=("auc", "std"),
    ap_mean=("ap", "mean"),
    ap_std=("ap", "std"),
    best_t_mean=("best_t", "mean"),
    best_t_std=("best_t", "std"),
).reset_index()

# delta (proto1 - proto0)
p0 = agg[agg["use_proto"] == 0].iloc[0]
p1 = agg[agg["use_proto"] == 1].iloc[0]

delta = {
    "Δf1": p1["f1_mean"] - p0["f1_mean"],
    "Δauc": p1["auc_mean"] - p0["auc_mean"],
    "Δap": p1["ap_mean"] - p0["ap_mean"],
    "Δbest_t_std": p1["best_t_std"] - p0["best_t_std"],
}

print("=== Polygon Proto Main Table ===")
print(agg)
print("\n=== Delta (Proto1 - Proto0) ===")
for k, v in delta.items():
    print(f"{k}: {v:+.4f}")

agg.to_csv("results/polygon_proto_main_table.csv", index=False)
pd.DataFrame([delta]).to_csv("results/polygon_proto_delta.csv", index=False)
