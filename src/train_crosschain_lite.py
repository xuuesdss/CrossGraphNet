from pathlib import Path
import json
import gc
import os
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.data_lite import MultiGraphJsonlDataset
from src.model import CrossGraphNetLite, CrossGraphNetLiteConfig

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
# lightweight dataset
# ==============================
class CompactLiteDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ==============================
# compact items
# ==============================
def compact_main_items(raw_items):
    compact = []

    for obj in raw_items:
        graphs = obj.get("graphs", {}) or {}

        ast = graphs.get("ast", {}) or {}
        cfg = graphs.get("cfg", {}) or {}

        ast_nodes = ast.get("ast_nodes", []) or []
        cfg_nodes = cfg.get("cfg_nodes", []) or []

        ast_types = [str(n.get("type", "<UNK>")) for n in ast_nodes]
        cfg_types = [str(n.get("type", "<UNK>")) for n in cfg_nodes]

        compact.append(
            {
                "id": obj.get("id"),
                "src_path": obj.get("src_path"),
                "label": int(obj["label"]),
                "ast_types": ast_types,
                "cfg_types": cfg_types,
            }
        )

    return compact


# ==============================
# build vocabs
# ==============================
def build_vocabs_from_items(items):
    ast_counter = Counter()
    cfg_counter = Counter()

    for obj in items:
        for t in obj.get("ast_types", []):
            ast_counter[str(t)] += 1
        for t in obj.get("cfg_types", []):
            cfg_counter[str(t)] += 1

    ast_vocab = SimpleVocab(ast_counter.keys())
    cfg_vocab = SimpleVocab(cfg_counter.keys())

    return ast_vocab, cfg_vocab


# ==============================
# collate lite
# ==============================
def collate_lite(batch, ast_vocab, cfg_vocab):
    ast_graphs = []
    cfg_graphs = []
    labels = []

    for obj in batch:
        labels.append(int(obj["label"]))

        ast_types = obj.get("ast_types", []) or []
        cfg_types = obj.get("cfg_types", []) or []

        ast_x = torch.tensor(
            [ast_vocab.encode(t) for t in ast_types],
            dtype=torch.long,
        )
        if ast_x.numel() == 0:
            ast_x = torch.zeros((1,), dtype=torch.long)

        cfg_x = torch.tensor(
            [cfg_vocab.encode(t) for t in cfg_types],
            dtype=torch.long,
        )
        if cfg_x.numel() == 0:
            cfg_x = torch.zeros((1,), dtype=torch.long)

        # 轻量版先不给显式边，保留空 edge_index
        ast_edge_index = torch.empty((2, 0), dtype=torch.long)
        cfg_edge_index = torch.empty((2, 0), dtype=torch.long)

        ast_graphs.append(Data(node_type=ast_x, edge_index=ast_edge_index))
        cfg_graphs.append(Data(node_type=cfg_x, edge_index=cfg_edge_index))

    return (
        Batch.from_data_list(ast_graphs),
        Batch.from_data_list(cfg_graphs),
        torch.tensor(labels, dtype=torch.long),
    )


# ==============================
# robust lite forward
# ==============================
def lite_forward(model, ast_b, cfg_b):
    """
    兼容不同的 CrossGraphNetLite.forward 签名：
    1) model(ast_batch, cfg_batch)
    2) model(ast_type, ast_edge, ast_batch, cfg_type, cfg_edge, cfg_batch)
    3) model(ast_type, ast_edge, ast_batch, cfg_type, cfg_edge, cfg_batch, sem_feat)
    """

    # 情况1：直接吃两个 Batch
    try:
        return model(ast_b, cfg_b)
    except TypeError:
        pass

    args6 = (
        ast_b.node_type,
        ast_b.edge_index,
        ast_b.batch,
        cfg_b.node_type,
        cfg_b.edge_index,
        cfg_b.batch,
    )

    # 情况2：吃 6 个张量参数
    try:
        return model(*args6)
    except TypeError as e6:
        # 情况3：还需要 sem_feat
        num_graphs = getattr(ast_b, "num_graphs", None)
        if num_graphs is None:
            if ast_b.batch.numel() == 0:
                num_graphs = 1
            else:
                num_graphs = int(ast_b.batch.max().item()) + 1

        sem_feat = torch.zeros(
            (num_graphs, 0),
            dtype=torch.float32,
            device=ast_b.node_type.device,
        )

        try:
            return model(*args6, sem_feat)
        except TypeError:
            raise e6


# ==============================
# evaluate
# ==============================
def evaluate(model, loader, device):
    model.eval()

    ys = []
    probs = []

    with torch.no_grad():
        for ast_b, cfg_b, y in loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            y = y.to(device)

            logits = lite_forward(model, ast_b, cfg_b)
            p = torch.softmax(logits, dim=-1)[:, 1]

            ys.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)

    return {
        "f1": float(f1_score(ys, preds)),
        "auc": float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan"),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "acc": float(accuracy_score(ys, preds)),
    }


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

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="results/lite_run")

    args = parser.parse_args()

    ensure_dir(args.save_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_config = vars(args).copy()
    dump_json(run_config, os.path.join(args.save_dir, "config.json"))

    print("[STAGE] loading lite datasets...", flush=True)

    raw_train_ds = MultiGraphJsonlDataset(
        Path(args.train_path), limit=args.train_limit
    )
    raw_test_ds = MultiGraphJsonlDataset(
        Path(args.test_path), limit=args.test_limit
    )

    print("RAW train:", len(raw_train_ds.items), flush=True)
    print("RAW test :", len(raw_test_ds.items), flush=True)

    print("[STAGE] compacting datasets...", flush=True)

    train_items = compact_main_items(raw_train_ds.items)
    test_items = compact_main_items(raw_test_ds.items)

    del raw_train_ds
    del raw_test_ds
    gc.collect()

    print("Compact train:", len(train_items), flush=True)
    print("Compact test :", len(test_items), flush=True)

    print("[STAGE] building vocabs...", flush=True)
    ast_vocab, cfg_vocab = build_vocabs_from_items(train_items)

    print(f"[OK] AST vocab size = {len(ast_vocab)}", flush=True)
    print(f"[OK] CFG vocab size = {len(cfg_vocab)}", flush=True)

    train_ds = CompactLiteDataset(train_items)
    test_ds = CompactLiteDataset(test_items)

    print("[STAGE] building dataloaders...", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_lite(b, ast_vocab, cfg_vocab),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_lite(b, ast_vocab, cfg_vocab),
    )

    print("[OK] dataloaders ready", flush=True)

    print("[STAGE] building lite model...", flush=True)

    cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(ast_vocab),
        num_cfg_types=len(cfg_vocab),
        sem_dim=0,
    )

    model = CrossGraphNetLite(cfg).to(device)

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

        for ast_b, cfg_b, y in train_loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = lite_forward(model, ast_b, cfg_b)
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
                "device_used": str(device),
                "train_size": len(train_ds),
                "test_size": len(test_ds),
            }

            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_lite_best.pt"))
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
        "device_used": str(device),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }
    dump_json(final_metrics, os.path.join(args.save_dir, "final_metrics.json"))

    print("Training complete.", flush=True)
    print(f"[SAVE] {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()