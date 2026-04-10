import json
import argparse
from pathlib import Path


def norm_path(x):
    if x is None:
        return None
    return str(x).strip().replace("\\", "/")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_subset_paths(subset_json):
    src_paths = set()
    basenames = set()

    with open(subset_json, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            cands = [
                obj.get("src_path"),
                obj.get("graphs", {}).get("ast", {}).get("id"),
            ]
            for c in cands:
                c = norm_path(c)
                if c:
                    src_paths.add(c)
                    basenames.add(Path(c).name)

    return src_paths, basenames


def build_target_ids(src_paths, basenames, src2id, filename2id):
    target_ids = set()

    # 1) exact src_path -> id
    for p in src_paths:
        if p in src2id:
            target_ids.add(str(src2id[p]))

    # 2) basename -> id
    for b in basenames:
        if b in filename2id:
            target_ids.add(str(filename2id[b]))

    return target_ids


def keep_if_match(obj, target_ids):
    cand_ids = [
        obj.get("id"),
        obj.get("sid"),
        obj.get("sample_id"),
        obj.get("contract_id"),
    ]
    for cid in cand_ids:
        if cid is not None and str(cid) in target_ids:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_json", required=True)
    parser.add_argument("--dfg_json", required=True)
    parser.add_argument("--src2id_json", required=True)
    parser.add_argument("--filename2id_json", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    src_paths, basenames = load_subset_paths(args.subset_json)
    src2id = load_json(args.src2id_json)
    filename2id = load_json(args.filename2id_json)

    print(f"[INFO] subset src_paths={len(src_paths)}, basenames={len(basenames)}")
    print(f"[INFO] src2id entries={len(src2id)}, filename2id entries={len(filename2id)}")

    target_ids = build_target_ids(src_paths, basenames, src2id, filename2id)
    print(f"[INFO] mapped target_ids={len(target_ids)}")

    seen = 0
    kept = 0

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.dfg_json, "r", encoding="utf-8") as fin, \
         open(args.out_json, "w", encoding="utf-8") as fout:
        for line in fin:
            seen += 1
            obj = json.loads(line)
            if keep_if_match(obj, target_ids):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[RESULT] seen={seen}, kept={kept}, out={args.out_json}")


if __name__ == "__main__":
    main()