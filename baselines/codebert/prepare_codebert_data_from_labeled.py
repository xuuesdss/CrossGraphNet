import os
import json
import random
import argparse
from typing import List, Dict, Any


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def read_source_code(src_path: str) -> str:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_label(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return 1 if v != 0 else 0
    if isinstance(v, float):
        return 1 if v != 0 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "vul", "vulnerable", "buggy"}:
            return 1
        if s in {"0", "false", "no", "clean", "safe", "non_vul", "non-vulnerable"}:
            return 0
    raise ValueError(f"Unsupported label value: {v}")


def convert_record(item: Dict[str, Any]) -> Dict[str, Any]:
    if "src_path" not in item:
        raise KeyError(f"Missing 'src_path' in item keys: {list(item.keys())}")
    if "label" not in item:
        raise KeyError(f"Missing 'label' in item keys: {list(item.keys())}")

    src_path = item["src_path"]
    code = read_source_code(src_path)
    label = normalize_label(item["label"])

    out = {
        "id": item.get("id", ""),
        "chain": item.get("chain", ""),
        "src_path": src_path,
        "code": code,
        "label": label,
    }

    return out


def split_train_val(data: List[Dict[str, Any]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    data = data[:]
    rng.shuffle(data)
    n = len(data)
    n_val = int(n * val_ratio)
    val = data[:n_val]
    train = data[n_val:]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_train", type=str, required=True)
    parser.add_argument("--src_test", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_train = read_jsonl(args.src_train)
    src_test = read_jsonl(args.src_test)

    train_conv = [convert_record(x) for x in src_train]
    test_conv = [convert_record(x) for x in src_test]

    train_data, val_data = split_train_val(train_conv, args.val_ratio, args.seed)

    write_jsonl(train_data, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(val_data, os.path.join(args.out_dir, "val.jsonl"))
    write_jsonl(test_conv, os.path.join(args.out_dir, "test.jsonl"))

    print(f"[OK] train={len(train_data)} val={len(val_data)} test={len(test_conv)}")
    print(f"[OK] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()