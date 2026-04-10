import json
import pandas as pd
import argparse

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize(path, exp_name):
    rows = read_jsonl(path)

    best = max(rows, key=lambda x: x["f1_mean"])

    return {
        "experiment": exp_name,
        "best_round": best["round"],
        "best_f1": best["f1_mean"],
        "best_auc": best["auc_mean"]
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--out", default="results/summary_all_runs.csv")

    args = parser.parse_args()

    rows = []

    for path, name in zip(args.inputs, args.names):
        rows.append(summarize(path, name))

    df = pd.DataFrame(rows)

    df.to_csv(args.out, index=False)

    print("\nSummary saved to:", args.out)
    print(df)


if __name__ == "__main__":
    main()