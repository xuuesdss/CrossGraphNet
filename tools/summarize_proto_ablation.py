import os
import re
import glob
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# ====== CONFIG ======
LOGDIR = "logs/fl"  # 改成你的真实路径
OUTDIR = "results/ablation_proto_all"

# 你现在的 tag 规范是：eth2bsc / eth2poly / eth2ftm
# 但为了防止你历史命名不一致，这里做了 alias 映射
TARGET_ALIASES = {
    "bsc": "BSC",
    "poly": "Polygon",
    "polygon": "Polygon",
    "ftm": "Fantom",
    "fantom": "Fantom",
}

# round 取哪个：正式实验是 10
FINAL_ROUND = 10

# ====== UTILS ======
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"JSONL parse error in {path}:{ln} -> {e}")
    return rows


def infer_seed_useproto_target_from_filename(path: str) -> Tuple[int, int, str]:
    """
    Parse from filename like:
      fedprox_stats_eth2bsc_n500_e1_r10_seed42_proto1.jsonl
      fedprox_stats_eth2poly_n500_e1_r10_seed7_proto0.jsonl
      fedprox_stats_eth2ftm_n500_e1_r10_seed1_proto1.jsonl

    Returns: (seed, use_proto, target_canonical)
    """
    base = os.path.basename(path).lower()

    # seed
    m = re.search(r"seed(\d+)", base)
    if not m:
        raise ValueError(f"Cannot parse seed from filename: {base}")
    seed = int(m.group(1))

    # use_proto
    if "_proto1" in base:
        use_proto = 1
    elif "_proto0" in base:
        use_proto = 0
    else:
        raise ValueError(f"Cannot parse proto flag (_proto0/_proto1) from filename: {base}")

    # target
    # support eth2bsc / eth2poly / eth2polygon / eth2ftm / eth2fantom
    m2 = re.search(r"eth2([a-z]+)", base)
    if not m2:
        raise ValueError(f"Cannot parse target from filename (expect eth2XXX): {base}")
    raw = m2.group(1)

    # strip possible suffixes like _n500...
    raw = re.split(r"[_\-]", raw)[0]
    target = TARGET_ALIASES.get(raw, raw)

    # normalize common raws
    if target not in ("BSC", "Polygon", "Fantom"):
        # 你如果未来加别的链，这里不会直接崩，而是保留原样
        target = str(target)

    return seed, use_proto, target


def pick_round(rows: List[Dict[str, Any]], final_round: int) -> Dict[str, Any]:
    """
    Robustly pick a specific round record:
    - Prefer exact round match.
    - If multiple entries for same round, take the last occurrence.
    - If no exact round exists, fall back to the max round (warn).
    """
    by_round = {}
    for r in rows:
        rr = r.get("round")
        if rr is None:
            continue
        by_round.setdefault(int(rr), []).append(r)

    if final_round in by_round:
        return by_round[final_round][-1]

    # fallback: choose max round available
    if len(by_round) == 0:
        raise ValueError("No 'round' field found in jsonl records.")
    max_r = max(by_round.keys())
    print(f"[WARN] final_round={final_round} not found; fallback to max_round={max_r}")
    return by_round[max_r][-1]


def extract_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    # tolerate keys: auc_mean vs test_auc_mean (you用的是 auc_mean)
    out = {}
    out["round"] = rec.get("round")
    out["f1_mean"] = rec.get("f1_mean")
    out["f1_std"] = rec.get("f1_std")
    out["auc_mean"] = rec.get("auc_mean")
    out["auc_std"] = rec.get("auc_std")
    out["train_loss_mean"] = rec.get("train_loss_mean")
    out["use_proto"] = rec.get("use_proto")
    out["proto_enabled_this_round"] = rec.get("proto_enabled_this_round", rec.get("proto_enabled", None))
    out["proto_lambda"] = rec.get("proto_lambda", None)
    out["proto_warmup_rounds"] = rec.get("proto_warmup_rounds", None)
    out["proto_counts"] = rec.get("proto_counts", None)
    return out


# ====== MAIN ======
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(LOGDIR, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No jsonl files found under {LOGDIR}")

    # filter only those with eth2* and proto0/proto1 and fedprox (or your algo name)
    cand = []
    for fp in files:
        b = os.path.basename(fp).lower()
        if "eth2" not in b:
            continue
        if "_proto0" not in b and "_proto1" not in b:
            continue
        # 如果你日志中还有 fedavg，也能一起汇总；这里只聚焦 FedProx 可加条件
        # if "fedprox" not in b:
        #     continue
        cand.append(fp)

    if not cand:
        raise FileNotFoundError(f"No matching eth2* proto jsonl files found under {LOGDIR}")

    # build per-file records
    curve_rows = []
    final_rows = []

    for fp in cand:
        seed, use_proto, target = infer_seed_useproto_target_from_filename(fp)
        rows = read_jsonl(fp)

        # curve rows (round-by-round)
        for rec in rows:
            if rec.get("round") is None:
                continue
            m = extract_metrics(rec)
            curve_rows.append({
                "file": os.path.basename(fp),
                "target": target,
                "seed": seed,
                "use_proto": use_proto,
                **m,
            })

        # final round (robust pick by round)
        final_rec = pick_round(rows, FINAL_ROUND)
        mfin = extract_metrics(final_rec)

        final_rows.append({
            "file": os.path.basename(fp),
            "target": target,
            "seed": seed,
            "use_proto": use_proto,
            "final_round": mfin.get("round"),
            "f1_mean_rfinal": mfin.get("f1_mean"),
            "f1_std_rfinal": mfin.get("f1_std"),
            "auc_mean_rfinal": mfin.get("auc_mean"),
            "auc_std_rfinal": mfin.get("auc_std"),
            "train_loss_rfinal": mfin.get("train_loss_mean"),
            "proto_lambda": mfin.get("proto_lambda"),
            "proto_warmup_rounds": mfin.get("proto_warmup_rounds"),
        })

    curve_df = pd.DataFrame(curve_rows)
    final_df = pd.DataFrame(final_rows)

    # sanity checks
    # each (target, seed, use_proto) should have 1 file
    dup = final_df.duplicated(subset=["target", "seed", "use_proto"], keep=False)
    if dup.any():
        print("[WARN] Duplicate runs detected for same (target, seed, use_proto).")
        print(final_df.loc[dup, ["file", "target", "seed", "use_proto", "final_round"]].sort_values(["target","seed","use_proto"]))

    # summary over seeds for each target and setting
    summary = (
        final_df
        .groupby(["target", "use_proto"], as_index=False)
        .agg(
            f1_mean=("f1_mean_rfinal", "mean"),
            f1_std=("f1_mean_rfinal", "std"),
            auc_mean=("auc_mean_rfinal", "mean"),
            auc_std=("auc_mean_rfinal", "std"),
        )
    )
    summary["setting"] = summary["use_proto"].map({0: "FedProx", 1: "FedProx + Prototypes"})

    # delta table: proto1 - proto0 per target (mean-wise)
    pivot = summary.pivot(index="target", columns="use_proto", values=["f1_mean","auc_mean","f1_std","auc_std"])
    pivot.columns = [f"{a}_proto{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    # compute deltas if both exist
    for met in ["f1_mean", "auc_mean", "f1_std", "auc_std"]:
        c0 = f"{met}_proto0"
        c1 = f"{met}_proto1"
        if c0 in pivot.columns and c1 in pivot.columns:
            pivot[f"delta_{met}"] = pivot[c1] - pivot[c0]

    # save
    os.makedirs(OUTDIR, exist_ok=True)
    curve_path = os.path.join(OUTDIR, "curve_rounds_all.csv")
    final_path = os.path.join(OUTDIR, "final_r{0}_by_seed_all.csv".format(FINAL_ROUND))
    summary_path = os.path.join(OUTDIR, "summary_over_seeds_all.csv")
    delta_path = os.path.join(OUTDIR, "delta_proto1_minus_proto0_by_target.csv")

    curve_df.sort_values(["target","use_proto","seed","round"]).to_csv(curve_path, index=False)
    final_df.sort_values(["target","seed","use_proto"]).to_csv(final_path, index=False)
    summary[["target","setting","use_proto","f1_mean","f1_std","auc_mean","auc_std"]].sort_values(["target","use_proto"]).to_csv(summary_path, index=False)
    pivot.sort_values(["target"]).to_csv(delta_path, index=False)

    print("Saved:")
    print(f" - {curve_path}")
    print(f" - {final_path}")
    print(f" - {summary_path}")
    print(f" - {delta_path}")

    summary_out = summary[["target","setting","use_proto","f1_mean","f1_std","auc_mean","auc_std"]].sort_values(["target","use_proto"])
    summary_out.to_csv(summary_path, index=False)

    print("\n=== Summary (mean±std over seeds) ===")
    print(summary_out)

    print(summary[["target","setting","use_proto","f1_mean","f1_std","auc_mean","auc_std"]].sort_values(["target","use_proto"]))


    print("\n=== Delta (proto1 - proto0) by target ===")
    show_cols = ["target"]
    for c in ["delta_f1_mean","delta_auc_mean","delta_f1_std","delta_auc_std"]:
        if c in pivot.columns:
            show_cols.append(c)
    print(pivot[show_cols].sort_values("target"))


if __name__ == "__main__":
    main()
