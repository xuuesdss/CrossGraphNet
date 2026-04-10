#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from tqdm import tqdm

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

SOLIDITY_KEYWORDS = {
    "if","else","for","while","return","emit",
    "require","revert","assert",
    "mapping","struct","event","function",
    "pragma","contract","interface","library",
    "memory","storage","calldata","public",
    "private","internal","external","view",
    "pure","payable","virtual","override",
    "returns","modifier","using","new","delete",
    # types-ish
    "uint","uint8","uint16","uint32","uint64","uint128","uint256",
    "int","int8","int16","int32","int64","int128","int256",
    "address","bool","bytes","bytes32","string",
}


def extract_vars(expr: str):
    if not expr or not isinstance(expr, str):
        return []
    toks = IDENT_RE.findall(expr)
    out = []
    for t in toks:
        if t.lower() in SOLIDITY_KEYWORDS:
            continue
        out.append(t)
    return out


def get_cfg_nodes(sample: dict):
    """
    兼容不同 schema：
    - sample["graphs"]["cfg"]["cfg_nodes"]
    - sample["graphs"]["cfg"]["nodes"]
    - sample["cfg_nodes"]
    """
    g = sample.get("graphs", {}) or {}
    cfg = g.get("cfg", {}) or {}
    nodes = cfg.get("cfg_nodes", None)
    if nodes is None:
        nodes = cfg.get("nodes", None)
    if nodes is None:
        nodes = sample.get("cfg_nodes", None)
    return nodes or []


def build_dfg_from_cfg_nodes(cfg_nodes):
    """
    输出：
      dfg_nodes: [{id: 0..N-1, type, expression, contract, function, orig_id?}]
      dfg_edges: [{src:int, dst:int, var:str}]
    注意：强制重编号，保证后续 DFGDataset 能正常 one-hot
    """
    # 1) 重编号 old_id -> new_id
    old2new = {}
    dfg_nodes = []
    for new_id, n in enumerate(cfg_nodes):
        old_id = n.get("id", new_id)
        old2new[old_id] = new_id
        dfg_nodes.append({
            "id": new_id,
            "orig_id": old_id,
            "type": n.get("type", "UNK"),
            "expression": n.get("expression", ""),
            "contract": n.get("contract", ""),
            "function": n.get("function", ""),
        })

    # 2) 变量出现序列：var -> [new_node_id...]
    var2nodes = {}
    for n in cfg_nodes:
        old_id = n.get("id", None)
        if old_id not in old2new:
            continue
        nid = old2new[old_id]
        expr = n.get("expression", "")
        for v in extract_vars(expr):
            var2nodes.setdefault(v, []).append(nid)

    # 3) 连链式边：同一变量在节点序列中的相邻出现连边
    dfg_edges = []
    for v, lst in var2nodes.items():
        # 去重保持顺序
        seen = set()
        seq = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                seq.append(x)
        for i in range(len(seq) - 1):
            dfg_edges.append({"src": seq[i], "dst": seq[i + 1], "var": v})

    return dfg_nodes, dfg_edges


def main():
    import argparse
    p = argparse.ArgumentParser("Build DFG jsonl directly from labeled jsonl (cfg_nodes).")
    p.add_argument("--in_jsonl", type=str, required=True, help="e.g. data/train/crossgraphnet_lite_labeled/BSC_500.jsonl")
    p.add_argument("--out_jsonl", type=str, required=True, help="e.g. data/graphs_dfg/BSC_500.jsonl")
    p.add_argument("--chain", type=str, default="BSC", help="write chain field")
    args = p.parse_args()

    in_p = Path(args.in_jsonl)
    out_p = Path(args.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    wrote = 0
    empty_cfg = 0

    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"DFG-from-labeled {in_p.name}"):
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            cid = obj.get("id", "")
            label = obj.get("label", None)

            cfg_nodes = get_cfg_nodes(obj)
            if not cfg_nodes:
                empty_cfg += 1
                # 也写一条空 DFG，保证样本对齐（Full 训练时至少不缺文件）
                fout.write(json.dumps({
                    "id": cid,
                    "chain": args.chain,
                    "label": label,
                    "dfg_nodes": [{"id": 0, "orig_id": 0, "type": "PAD", "expression": "", "contract": "", "function": ""}],
                    "dfg_edges": [],
                }, ensure_ascii=False) + "\n")
                wrote += 1
                continue

            dfg_nodes, dfg_edges = build_dfg_from_cfg_nodes(cfg_nodes)
            fout.write(json.dumps({
                "id": cid,
                "chain": args.chain,
                "label": label,
                "dfg_nodes": dfg_nodes,
                "dfg_edges": dfg_edges,
            }, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[OK] in={in_p} total={total} wrote={wrote} empty_cfg={empty_cfg} out={out_p}")


if __name__ == "__main__":
    main()