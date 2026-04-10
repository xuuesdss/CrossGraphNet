import json
import argparse
from pathlib import Path


def load_ids_from_split(split_json):
    ids = set()
    total = 0
    with open(split_json, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            sid = obj.get("id")
            if sid is not None:
                ids.add(str(sid))
    return ids, total


def normalize_id(x):
    if x is None:
        return None
    return str(x).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True, help="subset jsonl with selected ids")
    parser.add_argument("--dfg_json", required=True, help="full DFG jsonl")
    parser.add_argument("--out_json", required=True, help="filtered DFG jsonl")
    args = parser.parse_args()

    print(f"[INFO] Loading split ids from {args.split_json}")
    split_ids, split_total = load_ids_from_split(args.split_json)
    print(f"[INFO] split entries: {split_total}, unique ids: {len(split_ids)}")

    print(f"[INFO] Filtering DFG from {args.dfg_json}")
    kept = 0
    seen = 0

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.dfg_json, "r", encoding="utf-8") as fin, \
         open(args.out_json, "w", encoding="utf-8") as fout:
        for line in fin:
            seen += 1
            obj = json.loads(line)

            cand_ids = [
                obj.get("id"),
                obj.get("contract_id"),
                obj.get("sample_id"),
                obj.get("sid"),
            ]

            matched = False
            for cid in cand_ids:
                cid = normalize_id(cid)
                if cid is not None and cid in split_ids:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    matched = True
                    break

            if not matched:
                # try filename/source fallback if needed
                pass

    print(f"[RESULT] seen={seen}, kept={kept}, out={args.out_json}")


if __name__ == "__main__":
    main()