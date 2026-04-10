# src/federated/adapters.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data_lite import MultiGraphJsonlDataset, build_vocabs, collate_fn
from src.model import CrossGraphNetLite, CrossGraphNetLiteConfig
from src.train_crosschain import evaluate as eval_crosschain


@dataclass
class FLContext:
    chains: List[str]
    data_root: Path
    emb_root: Path
    per_chain_n: Optional[int]
    seed: int
    train_ratio: float
    batch_size: int
    num_workers: int
    ast_vocab: Any
    cfg_vocab: Any


_CTX: Optional[FLContext] = None


def _resolve_chain_jsonl(data_root: Path, chain: str) -> Path:
    p1 = data_root / f"{chain}.jsonl"
    if p1.exists():
        return p1
    p2 = data_root / f"{chain}_500.jsonl"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find jsonl for chain={chain} under {data_root} (tried {p1.name}, {p2.name})")


def _resolve_emb_dir(emb_root: Path, chain: str) -> Path:
    p1 = emb_root / chain
    if p1.exists():
        return p1
    p2 = emb_root / f"{chain}_500"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find emb_dir for chain={chain} under {emb_root} (tried {p1.name}, {p2.name})")


def prepare_federated_context(
    chains: List[str],
    semantic_mode: str,
    data_root: str = "data/train/crossgraphnet_lite_labeled",
    emb_root: str = "data/embeddings",
    per_chain_n: Optional[int] = 500,
    seed: int = 42,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 0,
) -> None:
    global _CTX

    if semantic_mode not in ("none", "stats", "llm"):
        raise ValueError(f"semantic_mode must be none|stats|llm, got {semantic_mode}")

    data_root_p = Path(data_root)
    emb_root_p = Path(emb_root)

    merged_items = []
    for ch in chains:
        path = _resolve_chain_jsonl(data_root_p, ch)
        ds = MultiGraphJsonlDataset(path, limit=per_chain_n)
        merged_items.extend(ds.items)

    class _Tmp:
        def __init__(self, items):
            self.items = items

    ast_vocab, cfg_vocab = build_vocabs(_Tmp(merged_items))

    _CTX = FLContext(
        chains=list(chains),
        data_root=data_root_p,
        emb_root=emb_root_p,
        per_chain_n=per_chain_n,
        seed=seed,
        train_ratio=train_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        ast_vocab=ast_vocab,
        cfg_vocab=cfg_vocab,
    )


def build_model(semantic_mode: str, **kwargs) -> Any:
    if _CTX is None:
        raise RuntimeError("Call prepare_federated_context(...) before build_model().")

    if semantic_mode == "none":
        sem_dim = 0
    elif semantic_mode == "stats":
        sem_dim = 8
    elif semantic_mode == "llm":
        sem_dim = 768
    else:
        raise ValueError(f"Unknown semantic_mode={semantic_mode}")

    cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(_CTX.ast_vocab),
        num_cfg_types=len(_CTX.cfg_vocab),
        sem_dim=sem_dim,
        emb_dim=64,
        hidden_dim=64,
        num_classes=2,
        dropout=0.1,
    )
    return CrossGraphNetLite(cfg)


def build_loaders(chain: str, semantic_mode: str, **kwargs) -> Tuple[Any, Any, int]:
    if _CTX is None:
        raise RuntimeError("Call prepare_federated_context(...) before build_loaders().")

    per_chain_n = kwargs.get("per_chain_n", _CTX.per_chain_n)
    batch_size = kwargs.get("batch_size", _CTX.batch_size)
    num_workers = kwargs.get("num_workers", _CTX.num_workers)

    path = _resolve_chain_jsonl(_CTX.data_root, chain)
    full_ds = MultiGraphJsonlDataset(path, limit=per_chain_n)

    n_total = len(full_ds)
    n_train = int(n_total * _CTX.train_ratio)
    n_test = max(n_total - n_train, 1)

    g = torch.Generator().manual_seed(_CTX.seed)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)

    if semantic_mode == "none":
        sem_mode = "none"
        emb_dir = None
    elif semantic_mode == "stats":
        sem_mode = "stats"
        emb_dir = None
    elif semantic_mode == "llm":
        sem_mode = "llm"
        emb_dir = str(_resolve_emb_dir(_CTX.emb_root, chain))
    else:
        raise ValueError(f"Unknown semantic_mode={semantic_mode}")

    def _cf(b, sm=sem_mode, ed=emb_dir):
        return collate_fn(b, _CTX.ast_vocab, _CTX.cfg_vocab, sem_mode=sm, emb_dir=ed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_cf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_cf,
    )
    return train_loader, test_loader, n_train


def _forward_logits_and_feat(model: Any, ast_b, cfg_b, sem):
    """
    Compatibility layer:
    - If model returns (logits, feat): use it.
    - Else: logits is returned; use logits as feat (minimal viable for proto ablation).
    """
    out = model(
        ast_b.node_type, ast_b.edge_index, ast_b.batch,
        cfg_b.node_type, cfg_b.edge_index, cfg_b.batch,
        sem,
    )
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        logits, feat = out[0], out[1]
        return logits, feat
    logits = out
    feat = logits
    return logits, feat


@torch.no_grad()
def compute_prototypes(
    model: Any,
    loader: Any,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """
    Compute local class prototypes (mean feature) for binary classification {0,1}.
    Returns:
      protos: {0: (D,), 1: (D,)} (missing classes are omitted)
      counts: {0: n0, 1: n1}
    """
    model.eval()
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    bidx = 0
    for ast_b, cfg_b, sem, y in loader:
        ast_b = ast_b.to(device)
        cfg_b = cfg_b.to(device)
        sem = sem.to(device)
        y = y.to(device)

        _, feat = _forward_logits_and_feat(model, ast_b, cfg_b, sem)
        # feat: (B, D)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        for cls in [0, 1]:
            mask = (y == cls)
            if torch.any(mask):
                f = feat[mask].detach()
                s = torch.sum(f, dim=0)
                if cls not in sums:
                    sums[cls] = s
                    counts[cls] = int(mask.sum().item())
                else:
                    sums[cls] = sums[cls] + s
                    counts[cls] = counts[cls] + int(mask.sum().item())

        bidx += 1
        if max_batches is not None and bidx >= int(max_batches):
            break

    protos: Dict[int, torch.Tensor] = {}
    for cls, s in sums.items():
        n = max(counts.get(cls, 0), 1)
        protos[cls] = (s / float(n)).detach()
    return protos, counts


def aggregate_global_prototypes(
    local_protos_list: List[Dict[int, torch.Tensor]],
    local_counts_list: List[Dict[int, int]],
    device: str = "cuda",
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """
    Server-side: weighted average per class, weights are per-class counts.
    """
    sum_vec: Dict[int, torch.Tensor] = {}
    sum_cnt: Dict[int, int] = {}

    for lp, lc in zip(local_protos_list, local_counts_list):
        for cls in [0, 1]:
            if cls in lp and cls in lc and lc[cls] > 0:
                v = lp[cls].to(device)
                w = int(lc[cls])
                if cls not in sum_vec:
                    sum_vec[cls] = v * float(w)
                    sum_cnt[cls] = w
                else:
                    sum_vec[cls] = sum_vec[cls] + v * float(w)
                    sum_cnt[cls] = sum_cnt[cls] + w

    global_protos: Dict[int, torch.Tensor] = {}
    for cls, sv in sum_vec.items():
        n = max(sum_cnt.get(cls, 0), 1)
        global_protos[cls] = (sv / float(n)).detach()
    return global_protos, sum_cnt


def train_one_epoch(
    model: Any,
    train_loader: Any,
    optimizer: Any,
    device: str = "cuda",
    algo: str = "fedavg",
    mu: float = 0.0,
    global_params=None,
    # proto knobs
    use_proto: bool = False,
    proto_lambda: float = 0.1,
    global_protos: Optional[Dict[int, torch.Tensor]] = None,
) -> float:
    """
    FedAvg : loss = CE
    FedProx: loss = CE + (mu/2)*||theta - theta_global||^2
    Proto  : loss += proto_lambda * MSE(feat, proto_global[y])
    """
    model.train()
    crit = nn.CrossEntropyLoss()
    mse = nn.MSELoss(reduction="mean")
    losses = []

    use_prox = (algo.lower() == "fedprox") and (global_params is not None) and (mu is not None) and (mu > 0)
    use_proto_eff = bool(use_proto) and (global_protos is not None) and (0 in global_protos or 1 in global_protos)

    # move protos once
    gp0 = global_protos.get(0).to(device) if (use_proto_eff and global_protos and 0 in global_protos) else None
    gp1 = global_protos.get(1).to(device) if (use_proto_eff and global_protos and 1 in global_protos) else None

    for ast_b, cfg_b, sem, y in train_loader:
        ast_b = ast_b.to(device)
        cfg_b = cfg_b.to(device)
        sem = sem.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, feat = _forward_logits_and_feat(model, ast_b, cfg_b, sem)
        loss = crit(logits, y)

        if use_prox:
            prox = 0.0
            for p, p0 in zip(model.parameters(), global_params):
                prox = prox + torch.sum((p - p0) ** 2)
            loss = loss + 0.5 * float(mu) * prox

        if use_proto_eff:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            # only align samples whose class prototype exists
            mask = torch.zeros(y.shape[0], dtype=torch.bool, device=device)
            targets = torch.empty_like(feat)

            if gp0 is not None:
                m0 = (y == 0)
                if torch.any(m0):
                    targets[m0] = gp0.unsqueeze(0).expand(int(m0.sum().item()), -1)
                    mask |= m0

            if gp1 is not None:
                m1 = (y == 1)
                if torch.any(m1):
                    targets[m1] = gp1.unsqueeze(0).expand(int(m1.sum().item()), -1)
                    mask |= m1

            if torch.any(mask):
                l_proto = mse(feat[mask], targets[mask])
                loss = loss + float(proto_lambda) * l_proto

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


# attach prototype computation capability to train_one_epoch_fn (used in client.py)
train_one_epoch.compute_prototypes = compute_prototypes  # type: ignore[attr-defined]


#def evaluate(model: Any, test_loader: Any, device: str = "cuda") -> Dict[str, float]:
#    return eval_crosschain(model, test_loader, device)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

@torch.no_grad()
def evaluate(model: Any, test_loader: Any, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    ys, scores = [], []

    for ast_b, cfg_b, sem, y in test_loader:
        ast_b = ast_b.to(device)
        cfg_b = cfg_b.to(device)
        sem = sem.to(device)
        y = y.to(device)

        logits, _ = _forward_logits_and_feat(model, ast_b, cfg_b, sem)
        p1 = torch.softmax(logits, dim=-1)[:, 1]  # prob of class=1

        ys.append(y.detach().cpu().numpy())
        scores.append(p1.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)

    # ROC-AUC / AP require both classes present
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    else:
        auc = float("nan")
        ap = float("nan")

    # F1 at threshold=0.5 (what you've been using implicitly)
    y_pred_05 = (y_score >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred_05, zero_division=0))

    # Best F1 by threshold sweep (diagnostic, not necessarily the reported metric)
    ths = np.linspace(0.01, 0.99, 99)
    f1s = []
    for t in ths:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_t = float(ths[best_idx])

    return {
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "best_t": best_t,
    }

