import json
import glob

base = "results/baselines/linevul"
rows = []

for path in glob.glob(f"{base}/*/metrics.json"):
    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)

    task = path.split("/")[-2]
    rows.append({
        "task": task,
        "seed": x["seed"],
        "f1": x["test"]["eval_f1"],
        "auc": x["test"]["eval_auc"],
    })

for r in sorted(rows, key=lambda z: (z["task"], z["seed"])):
    print(r)