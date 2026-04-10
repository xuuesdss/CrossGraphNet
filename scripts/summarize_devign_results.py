#汇总devign的结果
import json
import os
import glob

base = "results/baselines/devign"

rows = []

for path in glob.glob(f"{base}/*/metrics.json"):
    with open(path) as f:
        x = json.load(f)

    task = path.split("/")[-2]

    rows.append({
        "task": task,
        "seed": x["seed"],
        "f1": x["test"]["f1"],
        "auc": x["test"]["auc"]
    })

for r in sorted(rows, key=lambda x:(x["task"], x["seed"])):
    print(r)