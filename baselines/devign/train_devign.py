# baselines/devign/train_devign.py
import os
import json
import math
import random
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vocab(train_data: List[Dict[str, Any]]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for item in train_data:
        for n in item["nodes"]:
            t = n.get("type", "UNK")
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def encode_dataset(data_list: List[Dict[str, Any]], vocab: Dict[str, int]) -> List[Data]:
    pyg_list = []
    for item in data_list:
        node_types = [vocab.get(n.get("type", "UNK"), vocab["<UNK>"]) for n in item["nodes"]]
        if len(node_types) == 0:
            node_types = [vocab["<UNK>"]]

        x = torch.tensor(node_types, dtype=torch.long).view(-1, 1)

        edges = item.get("edges", [])
        if len(edges) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            src = []
            dst = []
            num_nodes = len(node_types)
            for e in edges:
                u = int(e["src"])
                v = int(e["dst"])
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    src.append(u)
                    dst.append(v)
            if len(src) == 0:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            else:
                edge_index = torch.tensor([src, dst], dtype=torch.long)

        y = torch.tensor([int(item["label"])], dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, y=y)
        pyg_list.append(d)
    return pyg_list

class DevignLite(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(emb_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x = data.x.squeeze(-1)
        x = self.emb(x)

        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        h_mean = global_mean_pool(x, data.batch)
        h_max = global_max_pool(x, data.batch)
        h = torch.cat([h_mean, h_max], dim=-1)
        logits = self.cls(h)
        return logits

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)

        ys.extend(batch.y.cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())

    f1 = f1_score(ys, preds, zero_division=0)
    prec = precision_score(ys, preds, zero_division=0)
    rec = recall_score(ys, preds, zero_division=0)

    try:
        auc = roc_auc_score(ys, probs)
    except Exception:
        auc = float("nan")

    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc)
    }

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_num = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()

        bs = batch.y.size(0)
        total_loss += loss.item() * bs
        total_num += bs

    return total_loss / max(total_num, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    train_raw = load_json(os.path.join(args.data_dir, "train.json"))
    val_raw = load_json(os.path.join(args.data_dir, "val.json"))
    test_raw = load_json(os.path.join(args.data_dir, "test.json"))

    vocab = build_vocab(train_raw)
    with open(os.path.join(args.out_dir, "node_type_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    train_data = encode_dataset(train_raw, vocab)
    val_data = encode_dataset(val_raw, vocab)
    test_data = encode_dataset(test_raw, vocab)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DevignLite(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_epoch = -1
    wait = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early stop] epoch={epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt"), map_location=device))
    val_best = evaluate(model, val_loader, device)
    test_best = evaluate(model, test_loader, device)

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "best_epoch": best_epoch,
            "val": val_best,
            "test": test_best
        }, f, ensure_ascii=False, indent=2)

    print("[BEST]")
    print(json.dumps({
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val": val_best,
        "test": test_best
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()