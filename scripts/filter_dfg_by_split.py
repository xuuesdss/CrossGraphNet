import json
from pathlib import Path

def norm_id(s: str) -> str:
    # 必须与 build_graphs_dfg.py 的 normalize_cid 保持一致（你现在输出 id 已经是 normalize 过的）
    return str(s).strip()

def load_ids(label_jsonl: Path):
    ids = set()
    with label_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id", "")
            if cid:
                ids.add(norm_id(cid))
    return ids

def filter_dfg(dfg_jsonl: Path, keep_ids: set, out_jsonl: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = 0
    with dfg_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            seen += 1
            obj = json.loads(line)
            cid = norm_id(obj.get("id", ""))
            if cid in keep_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[OK] filter done: seen={seen}, kept={kept}, out={out_jsonl}")

if __name__ == "__main__":
    # 你按实际路径改这三行即可
    LABEL_SPLIT = Path("data/train/crossgraphnet_lite_labeled/BSC_500.jsonl")
    DFG_ALL     = Path("data/graphs_dfg/BSC.jsonl")
    OUT_DFG     = Path("data/graphs_dfg/BSC_500.jsonl")

    keep = load_ids(LABEL_SPLIT)
    print(f"[INFO] split ids={len(keep)} from {LABEL_SPLIT}")
    filter_dfg(DFG_ALL, keep, OUT_DFG)