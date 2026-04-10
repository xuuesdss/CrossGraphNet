##数据从500扩充到750,
'''
Dataset statistics saved to: results/dataset_stats.csv
      chain  contracts  vulnerable  non_vulnerable          vuln_types
0  Ethereum      27284       10636           16648  {'unknown': 10636}
1       BSC      12179        6974            5205   {'unknown': 6974}
2   Polygon       7048        1967            5081   {'unknown': 1967}
3    Fantom        775         366             409    {'unknown': 366}
'''


import json
import argparse
import random
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(rows, n, seed=42, label_key="label"):
    random.seed(seed)

    pos = [r for r in rows if int(r.get(label_key, 0)) == 1]
    neg = [r for r in rows if int(r.get(label_key, 0)) == 0]

    total = len(rows)
    pos_n = round(n * len(pos) / total)
    neg_n = n - pos_n

    if pos_n > len(pos) or neg_n > len(neg):
        raise ValueError(
            f"Not enough samples for stratified sampling: "
            f"need pos={pos_n}, neg={neg_n}, have pos={len(pos)}, neg={len(neg)}"
        )

    sampled = random.sample(pos, pos_n) + random.sample(neg, neg_n)
    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth", required=True)
    parser.add_argument("--bsc", required=True)
    parser.add_argument("--polygon", required=True)
    parser.add_argument("--fantom", required=True)
    parser.add_argument("--size", type=int, required=True, help="subset size per chain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data/subsets")
    args = parser.parse_args()

    chain_files = {
        "ETH": args.eth,
        "BSC": args.bsc,
        "Polygon": args.polygon,
        "Fantom": args.fantom,
    }

    for chain, file_path in chain_files.items():
        rows = load_jsonl(file_path)
        if len(rows) < args.size:
            raise ValueError(
                f"{chain} only has {len(rows)} samples, cannot sample {args.size}"
            )
        sampled = stratified_sample(rows, args.size, seed=args.seed)
        out_path = Path(args.out_dir) / f"{chain}_{args.size}_seed{args.seed}.jsonl"
        save_jsonl(sampled, out_path)
        print(f"[OK] {chain}: {len(sampled)} samples -> {out_path}")


if __name__ == "__main__":
    main()