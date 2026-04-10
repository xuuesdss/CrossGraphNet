import json
import argparse
from collections import defaultdict
import pandas as pd

def summarize_dataset(file_path, chain_name):
    total = 0
    vuln = 0
    non_vuln = 0
    vuln_types = defaultdict(int)

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            total += 1

            label = data.get("label", 0)

            if label == 1:
                vuln += 1
                vtype = data.get("vuln_type", "unknown")
                vuln_types[vtype] += 1
            else:
                non_vuln += 1

    return {
        "chain": chain_name,
        "contracts": total,
        "vulnerable": vuln,
        "non_vulnerable": non_vuln,
        "vuln_types": dict(vuln_types)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth")
    parser.add_argument("--bsc")
    parser.add_argument("--polygon")
    parser.add_argument("--fantom")
    parser.add_argument("--out", default="results/dataset_stats.csv")

    args = parser.parse_args()

    rows = []

    rows.append(summarize_dataset(args.eth, "Ethereum"))
    rows.append(summarize_dataset(args.bsc, "BSC"))
    rows.append(summarize_dataset(args.polygon, "Polygon"))
    rows.append(summarize_dataset(args.fantom, "Fantom"))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\nDataset statistics saved to:", args.out)
    print(df)


if __name__ == "__main__":
    main()