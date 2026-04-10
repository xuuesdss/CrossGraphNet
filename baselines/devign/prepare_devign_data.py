# baselines/devign/prepare_devign_data.py
import os
import json
import argparse
from typing import Dict, List, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # 情况1：标准 JSONL
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) > 1:
        try:
            return [json.loads(line) for line in lines]
        except Exception:
            pass

    # 情况2：整个文件是一个 JSON 数组
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # 如果是 {"data":[...]} 之类
            for k in ["data", "samples", "items"]:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
            return [obj]
    except Exception:
        pass

    raise ValueError(f"Unsupported file format: {path}")

def load_id_file(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(x)
    return ids

def normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    适配 AST 图格式，统一输出为：
    {
      "id": str,
      "label": int,
      "nodes": [{"type": "..."} ...],
      "edges": [{"src":0,"dst":1,"etype":"AST"} ...]
    }
    """

    sid = str(raw.get("id"))

    # 1) label：如果图文件本身没有 label，先默认取 0
    # 后面我们会再从 split / labeled 文件里补真实 label
    label = int(raw.get("label", 0))

    # 2) 节点：优先读取 ast_nodes
    raw_nodes = raw.get("ast_nodes", raw.get("nodes", []))
    nodes = []
    for n in raw_nodes:
        if isinstance(n, dict):
            ntype = n.get("type", n.get("node_type", "UNK"))
        else:
            ntype = "UNK"
        nodes.append({"type": str(ntype)})

    # 3) 边：优先读取 ast_edges
    raw_edges = raw.get("ast_edges", raw.get("edges", []))
    edges = []
    for e in raw_edges:
        if isinstance(e, dict):
            src = int(e.get("src", e.get("u", e.get("from", 0))))
            dst = int(e.get("dst", e.get("v", e.get("to", 0))))
            etype = str(e.get("etype", e.get("type", "AST")))
        elif isinstance(e, list) and len(e) >= 2:
            src = int(e[0])
            dst = int(e[1])
            etype = str(e[2]) if len(e) >= 3 else "AST"
        else:
            continue
        edges.append({"src": src, "dst": dst, "etype": etype})

    return {
        "id": sid,
        "label": label,
        "nodes": nodes,
        "edges": edges
    }

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, nargs="+", required=True, help="one or more graph jsonl paths")
    parser.add_argument("--train_ids", type=str, required=True)
    parser.add_argument("--val_ids", type=str, required=True)
    parser.add_argument("--test_ids", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    id2sample = {}
    total_raw = 0
    nonempty_graphs = 0

    for gpath in args.graphs:
        print(f"[LOAD] {gpath}")
        all_samples = load_jsonl(gpath)
        total_raw += len(all_samples)

        for x in all_samples:
            sid = str(x.get("id"))
            if not sid:
                continue
            norm = normalize_sample(x)
            if len(norm["nodes"]) > 0:
                nonempty_graphs += 1
            id2sample[sid] = norm

    print(f"[INFO] loaded raw samples={total_raw}, unique ids={len(id2sample)}, nonempty_graphs={nonempty_graphs}")

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