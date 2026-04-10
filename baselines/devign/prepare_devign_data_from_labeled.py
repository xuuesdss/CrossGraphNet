import os
import json
import random
import argparse
from typing import Dict, List, Any, Set


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_label(row: Dict[str, Any]) -> int:
    for key in ["label", "target", "y"]:
        if key in row:
            return int(row[key])
    return 0


def build_needed_labels(train_labeled_path: str, test_labeled_path: str):
    train_rows = load_jsonl(train_labeled_path)
    test_rows = load_jsonl(test_labeled_path)

    id2label: Dict[str, int] = {}
    needed_ids: Set[str] = set()

    for row in train_rows:
        sid = str(row.get("id"))
        if sid:
            id2label[sid] = get_label(row)
            needed_ids.add(sid)

    for row in test_rows:
        sid = str(row.get("id"))
        if sid:
            id2label[sid] = get_label(row)
            needed_ids.add(sid)

    print(f"[INFO] raw_train_rows={len(train_rows)} raw_test_rows={len(test_rows)}")
    print(f"[INFO] needed_ids={len(needed_ids)}")

    return train_rows, test_rows, id2label, needed_ids


def normalize_ast_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(raw.get("id"))

    raw_nodes = raw.get("ast_nodes", [])
    nodes = []
    for n in raw_nodes:
        if isinstance(n, dict):
            ntype = n.get("type", n.get("node_type", "UNK"))
        else:
            ntype = "UNK"
        nodes.append({"type": str(ntype)})

    raw_edges = raw.get("ast_edges", [])
    edges = []
    for e in raw_edges:
        if isinstance(e, dict):
            src = int(e.get("src", e.get("u", e.get("from", 0))))
            dst = int(e.get("dst", e.get("v", e.get("to", 0))))
        elif isinstance(e, list) and len(e) >= 2:
            src = int(e[0])
            dst = int(e[1])
        else:
            continue
        edges.append({"src": src, "dst": dst, "etype": "AST"})

    return {
        "id": sid,
        "nodes": nodes,
        "edges": edges,
    }


def build_graph_index_streaming(graph_paths: List[str], needed_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    id2graph: Dict[str, Dict[str, Any]] = {}
    total_seen = 0
    total_hit = 0

    for path in graph_paths:
        print(f"[SCAN GRAPH] {path}")
        for raw in iter_jsonl(path):
            total_seen += 1
            sid = str(raw.get("id"))
            if not sid or sid not in needed_ids:
                continue
            if sid in id2graph:
                continue
            id2graph[sid] = normalize_ast_graph(raw)
            total_hit += 1

        print(f"[INFO] after {path}: collected={len(id2graph)} / needed={len(needed_ids)}")

        if len(id2graph) == len(needed_ids):
            print("[INFO] all needed ids found, early stop.")
            break

    print(f"[INFO] total_seen_graph_rows={total_seen}, matched_graph_ids={total_hit}, unique_graph_ids={len(id2graph)}")
    return id2graph


def build_samples(rows: List[Dict[str, Any]], id2graph: Dict[str, Dict[str, Any]], id2label: Dict[str, int]) -> List[Dict[str, Any]]:
    out = []
    miss = 0

    for row in rows:
        sid = str(row.get("id"))
        if sid not in id2graph:
            miss += 1
            continue

        g = id2graph[sid]
        out.append({
            "id": sid,
            "label": int(id2label[sid]),
            "nodes": g["nodes"],
            "edges": g["edges"],
        })

    print(f"[INFO] build_samples: kept={len(out)} missing_graph={miss}")
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
    parser.add_argument("--graph_paths", nargs="+", required=True)
    parser.add_argument("--train_labeled", type=str, required=True)
    parser.add_argument("--test_labeled", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_rows, test_rows, id2label, needed_ids = build_needed_labels(
        args.train_labeled, args.test_labeled
    )

    id2graph = build_graph_index_streaming(args.graph_paths, needed_ids)

    train_all = build_samples(train_rows, id2graph, id2label)
    test = build_samples(test_rows, id2graph, id2label)
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