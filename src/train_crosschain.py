from pathlib import Path
import json
import re
import gc
import os
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

from src.data_lite import MultiGraphJsonlDataset
from src.data_dfg import DFGDataset
from src.model import CrossGraphNetLite, CrossGraphNetLiteConfig, CrossGraphNetFull

from torch_geometric.data import Batch, Data


# ==============================
# simple vocab
# ==============================
class SimpleVocab:
    def __init__(self, tokens):
        uniq = ["<PAD>", "<UNK>"]
        seen = set(uniq)

        for t in tokens:
            t = str(t)
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        self.itos = uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, token):
        return self.stoi.get(str(token), self.stoi["<UNK>"])

    def __len__(self):
        return len(self.itos)


# ==============================
# AST encoder wrapper
# ==============================
class ASTWrapper(nn.Module):
    def __init__(self, lite_model):
        super().__init__()
        self.ast_enc = lite_model.ast_enc
        self.fuse_ast_cfg = lite_model.fuse_ast_cfg

    def encode(self, ast_data):
        h_ast = self.ast_enc(
            ast_data.node_type,
            ast_data.edge_index,
            ast_data.batch,
        )
        return h_ast


# ==============================
# lightweight dataset
# ==============================
class CompactMainDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ==============================
# compact main items
# ==============================
def compact_main_items(raw_items):
    compact = []

    for obj in raw_items:
        graphs = obj.get("graphs", {}) or {}
        ast = graphs.get("ast", {}) or {}
        ast_nodes = ast.get("ast_nodes", []) or []

        ast_types = [str(n.get("type", "<UNK>")) for n in ast_nodes]

        compact.append(
            {
                "id": obj.get("id"),
                "src_path": obj.get("src_path"),
                "label": int(obj["label"]),
                "ast_types": ast_types,
            }
        )

    return compact


# ==============================
# build vocab from compact items
# ==============================
def build_ast_vocab_from_items(items):
    counter = Counter()

    for obj in items:
        for t in obj.get("ast_types", []):
            counter[str(t)] += 1

    vocab = SimpleVocab(counter.keys())
    return vocab


# ==============================
# collate
# ==============================
def collate_full(batch, ast_vocab):
    ast_graphs = []
    labels = []
    dfg_graphs = []

    for obj in batch:
        labels.append(int(obj["label"]))

        ast_types = obj.get("ast_types", []) or []
        ast_x = torch.tensor(
            [ast_vocab.encode(t) for t in ast_types],
            dtype=torch.long,
        )

        if ast_x.numel() == 0:
            ast_x = torch.zeros((1,), dtype=torch.long)

        ast_edge_index = torch.empty((2, 0), dtype=torch.long)

        ast_graphs.append(
            Data(node_type=ast_x, edge_index=ast_edge_index)
        )

        dfg_graphs.append(obj["dfg_data"])

    return (
        Batch.from_data_list(ast_graphs),
        Batch.from_data_list(dfg_graphs),
        torch.tensor(labels, dtype=torch.long),
    )


# ==============================
# evaluate
# ==============================
def evaluate(model, loader, device):
    model.eval()

    ys = []
    probs = []

    with torch.no_grad():
        for ast_b, dfg_b, y in loader:
            ast_b = ast_b.to(device)
            dfg_b = dfg_b.to(device)
            y = y.to(device)

            logits = model(ast_b, dfg_b)
            p = torch.softmax(logits, dim=-1)[:, 1]

            ys.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)

    out = {
        "f1": float(f1_score(ys, preds)),
        "auc": float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan"),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "acc": float(accuracy_score(ys, preds)),
    }
    return out


# ==============================
# key utils
# ==============================
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


def candidate_keys_from_main_item(item):
    vals = []
    if "id" in item:
        vals.append(item.get("id"))
    if "src_path" in item:
        vals.append(item.get("src_path"))

    out = []
    seen = set()
    for v in vals:
        for k in candidate_keys_from_value(v):
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def candidate_keys_from_dfg_json(obj):
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


def load_dfg_key_lists(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            rows.append(candidate_keys_from_dfg_json(obj))

    return rows


# ==============================
# attach DFG by multi keys
# ==============================
def attach_dfg_by_keys(main_items, dfg_ds, dfg_path, split_name="train"):
    dfg_key_lists = load_dfg_key_lists(dfg_path)

    if len(dfg_key_lists) != len(dfg_ds.samples):
        print(
            f"[WARN][{split_name}] key_rows={len(dfg_key_lists)} "
            f"dfg_samples={len(dfg_ds.samples)}",
            flush=True,
        )

    pair_n = min(len(dfg_key_lists), len(dfg_ds.samples))

    dfg_map = {}
    dup_cnt = 0
    empty_key_rows = 0

    for i in range(pair_n):
        keys = dfg_key_lists[i]
        if not keys:
            empty_key_rows += 1
            continue

        for k in keys:
            if k in dfg_map:
                dup_cnt += 1
            dfg_map[k] = dfg_ds.samples[i]

    kept = []
    missing = []
    matched_by = []

    for item in main_items:
        item_keys = candidate_keys_from_main_item(item)

        found = None
        found_key = None

        for k in item_keys:
            if k in dfg_map:
                found = dfg_map[k]
                found_key = k
                break

        if found is not None:
            item["dfg_data"] = found
            kept.append(item)
            matched_by.append(found_key)
        else:
            missing.append(
                {
                    "id": item.get("id"),
                    "src_path": item.get("src_path"),
                    "candidate_keys": item_keys[:8],
                }
            )

    print(
        f"[DFG][{split_name}] "
        f"main={len(main_items)} "
        f"dfg_map={len(dfg_map)} "
        f"kept={len(kept)} "
        f"missing={len(missing)} "
        f"dup={dup_cnt} "
        f"empty_key_rows={empty_key_rows}",
        flush=True,
    )

    if kept:
        print(f"[DFG][{split_name}] first_matched_keys={matched_by[:5]}", flush=True)

    if missing:
        print(f"[DFG][{split_name}] first_missing={missing[:3]}", flush=True)

    if len(kept) == 0:
        raise RuntimeError(
            f"No samples left after DFG attach ({split_name}). "
            f"Main keys and DFG keys still do not overlap."
        )

    return kept


# ==============================
# save helpers
# ==============================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ==============================
# main
# ==============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--dfg_train_path", type=str, required=True)
    parser.add_argument("--dfg_test_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="results/full_run")

    args = parser.parse_args()

    ensure_dir(args.save_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_config = vars(args).copy()
    dump_json(run_config, os.path.join(args.save_dir, "config.json"))

    # ===== load AST =====
    print("[STAGE] loading AST datasets...", flush=True)

    raw_train_ds = MultiGraphJsonlDataset(
        Path(args.train_path), limit=args.train_limit
    )
    raw_test_ds = MultiGraphJsonlDataset(
        Path(args.test_path), limit=args.test_limit
    )

    print("AST train:", len(raw_train_ds.items), flush=True)
    print("AST test :", len(raw_test_ds.items), flush=True)

    # ===== compact =====
    print("[STAGE] compacting main datasets...", flush=True)

    train_items = compact_main_items(raw_train_ds.items)
    test_items = compact_main_items(raw_test_ds.items)

    del raw_train_ds
    del raw_test_ds
    gc.collect()

    print("Compact train:", len(train_items), flush=True)
    print("Compact test :", len(test_items), flush=True)

    # ===== vocab =====
    print("[STAGE] building AST vocab...", flush=True)
    ast_vocab = build_ast_vocab_from_items(train_items)
    print("[OK] AST vocab size =", len(ast_vocab), flush=True)

    # ===== load DFG =====
    print("[STAGE] loading DFG datasets...", flush=True)

    dfg_train = DFGDataset(args.dfg_train_path)
    dfg_test = DFGDataset(args.dfg_test_path)

    print("DFG train:", len(dfg_train.samples), flush=True)
    print("DFG test :", len(dfg_test.samples), flush=True)

    # ===== attach =====
    print("[STAGE] attaching DFG by keys...", flush=True)

    train_items = attach_dfg_by_keys(
        train_items,
        dfg_train,
        args.dfg_train_path,
        "train",
    )

    test_items = attach_dfg_by_keys(
        test_items,
        dfg_test,
        args.dfg_test_path,
        "test",
    )

    print("[OK] train after attach =", len(train_items), flush=True)
    print("[OK] test  after attach =", len(test_items), flush=True)

    train_ds = CompactMainDataset(train_items)
    test_ds = CompactMainDataset(test_items)

    # ===== dataloaders =====
    print("[STAGE] building dataloaders...", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_full(b, ast_vocab),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_full(b, ast_vocab),
    )

    print("[OK] dataloaders ready", flush=True)

    # ===== model =====
    print("[STAGE] building model...", flush=True)

    lite_cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(ast_vocab),
        num_cfg_types=1,
        sem_dim=0,
    )

    lite_model = CrossGraphNetLite(lite_cfg)
    ast_model = ASTWrapper(lite_model)

    model = CrossGraphNetFull(
        ast_model=ast_model,
        dfg_in_dim=dfg_train.num_node_types,
        hidden_dim=64,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    print("[OK] model ready", flush=True)

    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for ast_b, dfg_b, y in train_loader:
            ast_b = ast_b.to(device)
            dfg_b = dfg_b.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(ast_b, dfg_b)
            loss = crit(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        test_metrics = evaluate(model, test_loader, device)
        epoch_loss = float(np.mean(losses)) if losses else float("nan")

        row = {
            "epoch": ep,
            "loss": epoch_loss,
            **test_metrics,
        }
        history.append(row)

        print(
            f"Epoch {ep:02d} | "
            f"loss={epoch_loss:.4f} | "
            f"F1={test_metrics['f1']:.4f} "
            f"AUC={test_metrics['auc']:.4f}",
            flush=True,
        )

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_epoch = ep
            best_metrics = {
                "best_epoch": ep,
                "loss": epoch_loss,
                **test_metrics,
                "seed": args.seed,
                "train_path": args.train_path,
                "test_path": args.test_path,
                "dfg_train_path": args.dfg_train_path,
                "dfg_test_path": args.dfg_test_path,
                "device_used": str(device),
                "train_size_after_attach": len(train_ds),
                "test_size_after_attach": len(test_ds),
            }

            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_full_best.pt"))
            dump_json(best_metrics, os.path.join(args.save_dir, "best_metrics.json"))

        dump_json(history, os.path.join(args.save_dir, "history.json"))

    final_metrics = {
        "final_epoch": args.epochs,
        "best_epoch": best_epoch,
        "best_f1": best_metrics["f1"] if best_metrics else None,
        "best_auc": best_metrics["auc"] if best_metrics else None,
        "seed": args.seed,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "dfg_train_path": args.dfg_train_path,
        "dfg_test_path": args.dfg_test_path,
        "device_used": str(device),
        "train_size_after_attach": len(train_ds),
        "test_size_after_attach": len(test_ds),
    }
    dump_json(final_metrics, os.path.join(args.save_dir, "final_metrics.json"))

    print("Training complete.", flush=True)
    print(f"[SAVE] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()