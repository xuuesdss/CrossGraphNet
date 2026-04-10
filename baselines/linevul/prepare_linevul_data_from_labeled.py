import os
import json
import random
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


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_label(row: Dict[str, Any]) -> int:
    for key in ["label", "target", "y"]:
        if key in row:
            return int(row[key])
    return 0


def read_source_from_path(src_path: str) -> str:
    if not src_path:
        return ""

    # 1) 先按原路径读取
    candidate_paths = [src_path]

    # 2) 如果是相对路径，再尝试相对项目根目录
    if not os.path.isabs(src_path):
        candidate_paths.append(os.path.join(".", src_path))
        candidate_paths.append(os.path.join(os.getcwd(), src_path))

    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(p, "r", encoding="latin-1") as f:
                        return f.read()
                except Exception:
                    pass
            except Exception:
                pass

    return ""


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    missing_source = 0
    missing_path = 0

    for row in rows:
        sid = str(row.get("id", row.get("uid", row.get("sample_id", ""))))
        if not sid:
            continue

        label = get_label(row)
        src_path = str(row.get("src_path", "")).strip()

        if not src_path:
            missing_path += 1
            continue

        source = read_source_from_path(src_path)

        if not source.strip():
            missing_source += 1
            continue

        out.append({
            "id": sid,
            "label": int(label),
            "source": source,
            "src_path": src_path
        })

    print(f"[INFO] converted={len(out)} missing_path={missing_path} missing_source={missing_source}")
    return out


def split_train_val(train_samples: List[Dict[str, Any]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(train_samples)))
    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val])

    train_out, val_out = [], []
    for i, x in enumerate(train_samples):
        if i in val_idx:
            val_out.append(x)
        else:
            train_out.append(x)

    return train_out, val_out


def label_stats(name: str, arr: List[Dict[str, Any]]):
    pos = sum(int(x["label"]) for x in arr)
    neg = len(arr) - pos
    print(f"[{name}] total={len(arr)} pos={pos} neg={neg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_labeled", type=str, required=True)
    parser.add_argument("--test_labeled", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_labeled)
    test_rows = load_jsonl(args.test_labeled)

    print(f"[INFO] raw_train_rows={len(train_rows)} raw_test_rows={len(test_rows)}")

    train_all = convert_rows(train_rows)
    test = convert_rows(test_rows)
    train, val = split_train_val(train_all, args.val_ratio, args.seed)

    label_stats("train", train)
    label_stats("val", val)
    label_stats("test", test)

    save_json(os.path.join(args.out_dir, "train.json"), train)
    save_json(os.path.join(args.out_dir, "val.json"), val)
    save_json(os.path.join(args.out_dir, "test.json"), test)

    print(f"[OK] saved to {args.out_dir}")


if __name__ == "__main__":
    main()