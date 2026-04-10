import pandas as pd
import numpy as np

INCSV = "results/polygon_evalplus_by_seed.csv"
OUT1  = "results/polygon_oracle_by_seed.csv"
OUT2  = "results/polygon_oracle_summary.csv"

df = pd.read_csv(INCSV)

# 只保留 evalplus 的 6 条（如果你 CSV 已经是干净的可删掉这行）
df = df[df["file"].str.contains("evalplus")].copy()

# 解析 seed（你现在 seed=None，因为日志里没写 seed）
# 直接从文件名里抓 seed1/seed7/seed42
df["seed"] = df["file"].str.extract(r"seed(\d+)").astype(int)

# Oracle gap
df["oracle_gap"] = df["best_f1"] - df["f1"]

# 便于读表
df["method"] = df["use_proto"].map({0: "FedProx", 1: "FedProx + Prototypes"})

df_out = df[["seed","method","use_proto","f1","auc","ap","best_f1","best_t","oracle_gap"]].sort_values(["use_proto","seed"])
print("\n=== Polygon Oracle-F1 (by seed) ===")
print(df_out)

# 汇总（mean±std over seeds）
def mean_std(x):
    return float(np.mean(x)), float(np.std(x, ddof=1))

rows = []
for u in [0, 1]:
    sub = df[df["use_proto"] == u]
    f1m, f1s = mean_std(sub["f1"])
    of1m, of1s = mean_std(sub["best_f1"])
    gapm, gaps = mean_std(sub["oracle_gap"])
    tm, ts = mean_std(sub["best_t"])
    rows.append({
        "method": "FedProx" if u == 0 else "FedProx + Prototypes",
        "use_proto": u,
        "F1@0.5_mean": f1m, "F1@0.5_std": f1s,
        "OracleF1_mean": of1m, "OracleF1_std": of1s,
        "OracleGap_mean": gapm, "OracleGap_std": gaps,
        "best_t_mean": tm, "best_t_std": ts,
    })

summary = pd.DataFrame(rows).sort_values("use_proto")
print("\n=== Summary (mean±std over seeds) ===")
print(summary)

df_out.to_csv(OUT1, index=False)
summary.to_csv(OUT2, index=False)
print(f"\nSaved:\n- {OUT1}\n- {OUT2}")
