# baselines/linevul/prepare_linevul_data.py
import os
import json
import argparse
from typing import Dict, List, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                arr.append(json.loads(line))
    return arr

def load_id_file(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(x)
    return ids

def normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(raw.get("id")),
        "label": int(raw.get("label", 0)),
        "source": str(raw.get("source", raw.get("code", raw.get("contract_source", ""))))
    }

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--train_ids", type=str, required=True)
    parser.add_argument("--val_ids", type=str, required=True)
    parser.add_argument("--test_ids", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    all_samples = load_jsonl(args.input_jsonl)
    id2sample = {}
    for x in all_samples:
        sid = str(x.get("id"))
        if sid:
            id2sample[sid] = normalize_sample(x)

    split_map = {
        "train": load_id_file(args.train_ids),
        "val": load_id_file(args.val_ids),
        "test": load_id_file(args.test_ids),
    }

    for split_name, ids in split_map.items():
        out = []
        miss = 0
        for sid in ids:
            if sid in id2sample:
                out.append(id2sample[sid])
            else:
                miss += 1
        save_json(os.path.join(args.out_dir, f"{split_name}.json"), out)
        print(f"[{split_name}] kept={len(out)} missing={miss}")

if __name__ == "__main__":
    main()