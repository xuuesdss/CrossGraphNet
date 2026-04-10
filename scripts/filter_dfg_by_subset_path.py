import json
import argparse
from pathlib import Path


def normalize_path(p: str):
    if p is None:
        return None
    p = str(p).strip().replace("\\", "/")
    return p


def basename_only(p: str):
    if p is None:
        return None
    return Path(p).name


def load_subset_keys(subset_json):
    exact_paths = set()
    basenames = set()

    total = 0
    with open(subset_json, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)

            # 1) src_path
            src_path = normalize_path(obj.get("src_path"))
            if src_path:
                exact_paths.add(src_path)
                basenames.add(basename_only(src_path))

            # 2) graphs.ast.id
            ast_id = (
                obj.get("graphs", {})
                  .get("ast", {})
                  .get("id")
            )
            ast_id = normalize_path(ast_id)
            if ast_id:
                exact_paths.add(ast_id)
                basenames.add(basename_only(ast_id))

    return exact_paths, basenames, total


def candidate_paths_from_dfg_obj(obj):
    cands = []

    # 常见字段
    for key in [
        "src_path", "path", "file", "filename",
        "contract_path", "source_path", "graph_path",
        "id", "sid", "sample_id", "contract_id"
    ]:
        v = obj.get(key)
        if isinstance(v, str):
            cands.append(v)

    # 有些数据会嵌在 meta 里
    meta = obj.get("meta", {})
    if isinstance(meta, dict):
        for key in ["src_path", "path", "file", "filename"]:
            v = meta.get(key)
            if isinstance(v, str):
                cands.append(v)

    return [normalize_path(x) for x in cands if x is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--dfg_json", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    exact_paths, basenames, subset_total = load_subset_keys(args.subset_json)
    print(f"[INFO] subset entries={subset_total}")
    print(f"[INFO] subset exact_paths={len(exact_paths)}, basenames={len(basenames)}")

    seen = 0
    kept = 0

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.dfg_json, "r", encoding="utf-8") as fin, \
         open(args.out_json, "w", encoding="utf-8") as fout:
        for line in fin:
            seen += 1
            obj = json.loads(line)

            matched = False
            cands = candidate_paths_from_dfg_obj(obj)

            for c in cands:
                if c in exact_paths or basename_only(c) in basenames:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    matched = True
                    break

    print(f"[RESULT] seen={seen}, kept={kept}, out={args.out_json}")


if __name__ == "__main__":
    main()