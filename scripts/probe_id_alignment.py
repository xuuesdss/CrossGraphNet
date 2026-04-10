import json
import argparse
from pathlib import Path
from collections import Counter


def normalize(x):
    if x is None:
        return None
    return str(x).strip().replace("\\", "/")


def short(x, n=120):
    x = str(x)
    return x if len(x) <= n else x[:n] + "..."


def collect_subset_keys(path, max_show=3):
    exact = set()
    base = set()
    shown = 0

    print(f"\n[SUBSET] {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            if shown < max_show:
                print(f"\nSUBSET SAMPLE {shown}")
                print("keys:", list(obj.keys()))
                print("id:", obj.get("id"))
                print("src_path:", obj.get("src_path"))
                ast_id = obj.get("graphs", {}).get("ast", {}).get("id")
                print("graphs.ast.id:", ast_id)
                shown += 1

            for cand in [
                obj.get("id"),
                obj.get("src_path"),
                obj.get("graphs", {}).get("ast", {}).get("id"),
            ]:
                cand = normalize(cand)
                if cand:
                    exact.add(cand)
                    base.add(Path(cand).name)

    print(f"\n[SUBSET STATS] exact={len(exact)} basename={len(base)}")
    return exact, base


def probe_dfg(path, subset_exact, subset_base, max_show=10):
    print(f"\n[DFG] {path}")
    field_counter = Counter()
    match_counter = Counter()

    candidate_keys = [
        "id", "sid", "sample_id", "contract_id",
        "src_path", "path", "file", "filename",
        "contract_path", "source_path", "graph_path"
    ]

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            if i < max_show:
                print(f"\nDFG SAMPLE {i}")
                print("keys:", list(obj.keys()))

            for k in candidate_keys:
                if k in obj:
                    field_counter[k] += 1
                    v = normalize(obj.get(k))
                    if v:
                        if i < max_show:
                            print(f"  {k}: {short(v)}")

                        if v in subset_exact:
                            match_counter[f"{k}:exact"] += 1
                        if Path(v).name in subset_base:
                            match_counter[f"{k}:basename"] += 1

            meta = obj.get("meta")
            if isinstance(meta, dict):
                for k in ["src_path", "path", "file", "filename"]:
                    if k in meta:
                        field_counter[f"meta.{k}"] += 1
                        v = normalize(meta.get(k))
                        if v:
                            if i < max_show:
                                print(f"  meta.{k}: {short(v)}")
                            if v in subset_exact:
                                match_counter[f"meta.{k}:exact"] += 1
                            if Path(v).name in subset_base:
                                match_counter[f"meta.{k}:basename"] += 1

    print("\n[DFG FIELD COUNTS]")
    for k, v in field_counter.most_common():
        print(f"{k}: {v}")

    print("\n[POTENTIAL MATCH COUNTS AGAINST SUBSET]")
    if match_counter:
        for k, v in match_counter.most_common():
            print(f"{k}: {v}")
    else:
        print("No overlap found on any probed field.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--dfg_json", required=True)
    args = parser.parse_args()

    subset_exact, subset_base = collect_subset_keys(args.subset_json)
    probe_dfg(args.dfg_json, subset_exact, subset_base)


if __name__ == "__main__":
    main()