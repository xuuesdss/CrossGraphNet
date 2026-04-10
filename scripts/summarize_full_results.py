import os
import json
import csv

ROOT = "results/crossgraphnet_full"
OUT_CSV = os.path.join(ROOT, "summary_full.csv")

rows = []

if not os.path.exists(ROOT):
    raise SystemExit(f"Not found: {ROOT}")

for name in sorted(os.listdir(ROOT)):
    run_dir = os.path.join(ROOT, name)
    if not os.path.isdir(run_dir):
        continue

    best_path = os.path.join(run_dir, "best_metrics.json")
    final_path = os.path.join(run_dir, "final_metrics.json")

    if not os.path.exists(best_path):
        continue

    with open(best_path, "r", encoding="utf-8") as f:
        best = json.load(f)

    final = {}
    if os.path.exists(final_path):
        with open(final_path, "r", encoding="utf-8") as f:
            final = json.load(f)

    row = {
        "run": name,
        "seed": best.get("seed"),
        "train_path": best.get("train_path"),
        "test_path": best.get("test_path"),
        "best_epoch": best.get("best_epoch"),
        "f1": best.get("f1"),
        "auc": best.get("auc"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "acc": best.get("acc"),
        "loss": best.get("loss"),
        "train_size_after_attach": best.get("train_size_after_attach"),
        "test_size_after_attach": best.get("test_size_after_attach"),
        "device_used": best.get("device_used"),
        "final_epoch": final.get("final_epoch"),
    }
    rows.append(row)

fieldnames = [
    "run",
    "seed",
    "train_path",
    "test_path",
    "best_epoch",
    "f1",
    "auc",
    "precision",
    "recall",
    "acc",
    "loss",
    "train_size_after_attach",
    "test_size_after_attach",
    "device_used",
    "final_epoch",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] saved: {OUT_CSV}")
for row in rows:
    print(row["run"], "F1=", row["f1"], "AUC=", row["auc"])