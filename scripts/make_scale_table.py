import pandas as pd
import re

INPUT = "results/crossgraphnet_lite_matrix/summary_lite.csv"
OUTPUT = "results/crossgraphnet_lite_matrix/scale_sensitivity_table.csv"


def parse_exp(exp_name):
    # 例子: eth_to_BSC_500_seed42
    m = re.match(r"eth_to_(.+)_(\d+)_seed(\d+)", exp_name)
    if not m:
        return None, None, None
    chain = m.group(1)
    size = int(m.group(2))
    seed = int(m.group(3))
    return chain, size, seed


def main():
    df = pd.read_csv(INPUT)

    rows = []
    for _, row in df.iterrows():
        exp = row["experiment"] if "experiment" in df.columns else row.iloc[0]
        f1 = row["f1"] if "f1" in df.columns else row["F1"]
        auc = row["auc"] if "auc" in df.columns else row["AUC"]

        chain, size, seed = parse_exp(exp)
        if chain is None:
            continue

        rows.append({
            "chain": chain,
            "size": size,
            "seed": seed,
            "f1": f1,
            "auc": auc,
        })

    out = pd.DataFrame(rows)
    summary = (
        out.groupby(["size", "chain"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        )
    )

    pivot = summary.pivot(index="size", columns="chain", values="f1_mean")
    pivot.to_csv(OUTPUT)

    print("[OK] saved:", OUTPUT)
    print("\nDetailed summary:")
    print(summary)
    print("\nPivot table (F1 mean):")
    print(pivot)


if __name__ == "__main__":
    main()