import json
import re
from pathlib import Path


HASH40_RE = re.compile(r"[a-fA-F0-9]{40}")


def safe_str(x):
    if x is None:
        return ""
    return str(x).strip()


def extract_hash40(s):
    s = safe_str(s)
    if not s:
        return None
    m = HASH40_RE.search(s)
    if m:
        return m.group(0).lower()
    return None


def basename_no_ext(s):
    s = safe_str(s)
    if not s:
        return None
    return Path(Path(s).name).stem.lower()


def basename_only(s):
    s = safe_str(s)
    if not s:
        return None
    return Path(s).name.lower()


def prefix_before_underscore(s):
    s = safe_str(s)
    if not s:
        return None
    stem = Path(Path(s).name).stem
    if "_" in stem:
        return stem.split("_", 1)[0].lower()
    return stem.lower()


def normalize_full_path(s):
    s = safe_str(s)
    if not s:
        return None
    return s.replace("\\", "/").lower()


def candidate_keys_from_value(x):
    s = safe_str(x)
    if not s:
        return []

    cands = []

    cands.append(s.lower())

    fp = normalize_full_path(s)
    if fp:
        cands.append(fp)

    bn = basename_only(s)
    if bn:
        cands.append(bn)

    stem = basename_no_ext(s)
    if stem:
        cands.append(stem)

    pref = prefix_before_underscore(s)
    if pref:
        cands.append(pref)

    h = extract_hash40(s)
    if h:
        cands.append(h)

    out = []
    seen = set()
    for z in cands:
        if z and z not in seen:
            seen.add(z)
            out.append(z)

    return out


def candidate_keys_from_obj(obj):
    vals = []
    if "id" in obj:
        vals.append(obj.get("id"))
    if "src_path" in obj:
        vals.append(obj.get("src_path"))

    out = []
    seen = set()
    for v in vals:
        for k in candidate_keys_from_value(v):
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def build_subset(main_path, dfg_full_path, out_path):
    # 1) 读取主样本子集的 key 集合
    target_keys = set()
    target_rows = 0

    with open(main_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            target_rows += 1
            for k in candidate_keys_from_obj(obj):
                target_keys.add(k)

    print("=" * 80)
    print(f"[MAIN] {main_path}")
    print(f"rows={target_rows} unique_keys={len(target_keys)}")

    # 2) 从全量 DFG 中筛出匹配子集
    kept = 0
    seen_ids = set()

    with open(dfg_full_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            keys = candidate_keys_from_obj(obj)

            matched = any(k in target_keys for k in keys)
            if not matched:
                continue

            row_id = obj.get("id", line[:80])
            if row_id in seen_ids:
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            seen_ids.add(row_id)

    print(f"[DFG ] {dfg_full_path}")
    print(f"[OUT ] {out_path}")
    print(f"kept={kept}")


def main():
    build_subset(
        main_path="data/train/crossgraphnet_lite_labeled/Polygon_500.jsonl",
        dfg_full_path="data/graphs_dfg/Polygon.jsonl",
        out_path="data/graphs_dfg/Polygon_500.jsonl",
    )

    build_subset(
        main_path="data/train/crossgraphnet_lite_labeled/Fantom_500.jsonl",
        dfg_full_path="data/graphs_dfg/Fantom.jsonl",
        out_path="data/graphs_dfg/Fantom_500.jsonl",
    )


if __name__ == "__main__":
    main()