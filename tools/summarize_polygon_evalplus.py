import glob, json
import pandas as pd
import os

LOGDIR = "logs/fl_evalplus" 

rows = []

for fp in glob.glob(os.path.join(LOGDIR, "*eth2poly*_evalplus*.jsonl")):
    with open(fp, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
        last = lines[-1]  # round=10

        # 关键：从 per_chain["Polygon"] 里取
        poly = last["per_chain"]["Polygon"]

        rows.append({
            "file": os.path.basename(fp),
            "seed": last.get("seed", None),
            "use_proto": last["use_proto"],

            # Polygon-specific metrics
            "f1": poly["f1"],
            "auc": poly["auc"],
            "ap": poly["ap"],
            "best_f1": poly["best_f1"],
            "best_t": poly["best_t"],
        })

df = pd.DataFrame(rows).sort_values(["use_proto", "seed"])

print("\n=== Polygon eval-plus (by seed) ===")
print(df)

os.makedirs("results", exist_ok=True)
df.to_csv("results/polygon_evalplus_by_seed.csv", index=False)

print("\nSaved to: results/polygon_evalplus_by_seed.csv")
