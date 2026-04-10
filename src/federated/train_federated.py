# src/federated/train_federated.py
from __future__ import annotations

import os
import json
import time
import argparse
import copy
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from .fedavg import fedavg
from .client import FLClient
from . import adapters


def _as_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict) and "loss" in x:
        try:
            return float(x["loss"])
        except Exception:
            return None
    return None


def _mean_std(metrics: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    xs = [float(m[key]) for m in metrics]
    return float(np.mean(xs)), float(np.std(xs))


def make_optimizer_fn(lr: float, weight_decay: float):
    def _fn(model):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return _fn


def _safe_tensor_to_list(x: Optional[torch.Tensor]):
    if x is None:
        return None
    return x.detach().cpu().tolist()


def run_fl(
    clients: List[FLClient],
    global_model,
    rounds: int,
    local_epochs: int,
    device: str,
    out_jsonl: str,
    lr: float,
    weight_decay: float,
    algo: str = "fedavg",
    mu: float = 0.001,
    # proto knobs
    use_proto: int = 0,
    proto_lambda: float = 0.1,
    proto_warmup_rounds: int = 0,
    proto_max_batches: Optional[int] = None,  # SMOKE 可设很小
) -> Any:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    optimizer_fn = make_optimizer_fn(lr=lr, weight_decay=weight_decay)

    model = copy.deepcopy(global_model).to(device)

    # global prototypes (updated each round if enabled)
    global_protos: Optional[Dict[int, torch.Tensor]] = None
    global_proto_counts: Optional[Dict[int, int]] = None

    for r in range(1, rounds + 1):
        t0 = time.time()

        local_states = []
        local_weights = []
        local_losses = []

        # collect local protos for server aggregation
        local_protos_list = []
        local_proto_counts_list = []

        proto_enabled_this_round = (use_proto == 1) and (r > int(proto_warmup_rounds))

        for c in clients:
            sd, loss, proto_pack = c.local_train(
                global_model=model,
                train_one_epoch_fn=adapters.train_one_epoch,
                optimizer_fn=optimizer_fn,
                local_epochs=local_epochs,
                device=device,
                algo=algo,
                mu=mu,
                # proto
                use_proto=proto_enabled_this_round,
                proto_lambda=float(proto_lambda),
                global_protos=global_protos,
                proto_max_batches=proto_max_batches,
            )
            local_states.append(sd)
            local_weights.append(c.num_samples)
            local_losses.append(_as_float(loss))

            if proto_enabled_this_round and proto_pack is not None:
                lp, lc = proto_pack
                local_protos_list.append(lp)
                local_proto_counts_list.append(lc)

        # aggregation (FedAvg aggregation is still used for FedProx)
        new_sd = fedavg(local_states, local_weights)
        model.load_state_dict(new_sd, strict=True)

        # aggregate global prototypes AFTER local training
        if proto_enabled_this_round:
            global_protos, global_proto_counts = adapters.aggregate_global_prototypes(
                local_protos_list,
                local_proto_counts_list,
                device=device,
            )

        # evaluation on each client test set
        per_chain = {}
        metrics_list = []
        for c in clients:
            m = adapters.evaluate(model, c.test_loader, device=device)
            if "f1" not in m or "auc" not in m:
                raise ValueError(f"evaluate() must return keys 'f1' and 'auc'. Got keys={list(m.keys())}")
            per_chain[c.name] = m
            metrics_list.append(m)

        f1_mean, f1_std = _mean_std(metrics_list, "f1")
        auc_mean, auc_std = _mean_std(metrics_list, "auc")

        loss_vals = [x for x in local_losses if x is not None]
        train_loss_mean = float(np.mean(loss_vals)) if len(loss_vals) > 0 else None

        algo_name = "FedProx" if algo.lower() == "fedprox" else "FedAvg"

        rec = {
            "algo": algo_name,
            "round": r,
            "local_epochs": local_epochs,
            "mu": float(mu) if algo.lower() == "fedprox" else 0.0,
            "chains": [c.name for c in clients],
            "weights_num_train": {c.name: int(c.num_samples) for c in clients},
            "train_loss_mean": train_loss_mean,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "per_chain": per_chain,
            "time_sec": time.time() - t0,
            # proto log (paper-ready)
            "use_proto": int(use_proto),
            "proto_enabled_this_round": int(proto_enabled_this_round),
            "proto_lambda": float(proto_lambda),
            "proto_warmup_rounds": int(proto_warmup_rounds),
            "proto_counts": global_proto_counts,
            "proto_global_0": _safe_tensor_to_list(global_protos[0]) if global_protos and 0 in global_protos else None,
            "proto_global_1": _safe_tensor_to_list(global_protos[1]) if global_protos and 1 in global_protos else None,
        }

        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mu_str = f" mu={mu}" if algo.lower() == "fedprox" else ""
        proto_str = f" +Proto(λ={proto_lambda},warmup={proto_warmup_rounds})" if proto_enabled_this_round else ""
        print(
            f"[{algo_name}{mu_str}{proto_str}] "
            f"[R{r:03d}] F1 {f1_mean:.4f}±{f1_std:.4f} | "
            f"AUC {auc_mean:.4f}±{auc_std:.4f} | "
            f"loss {train_loss_mean} | "
            f"{rec['time_sec']:.1f}s"
        )

    return model


def main():
    p = argparse.ArgumentParser()

    # FL setup
    p.add_argument("--clients", nargs="+", default=["Ethereum", "BSC"], help="e.g., Ethereum BSC Polygon Fantom")
    p.add_argument("--semantic", default="stats", choices=["none", "stats", "llm"])
    p.add_argument("--algo", default="fedavg", choices=["fedavg", "fedprox"])
    p.add_argument("--mu", type=float, default=0.001, help="FedProx proximal strength (typical 1e-3)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--device", default="cuda")

    # Proto (ablation knobs)
    p.add_argument("--use_proto", type=int, default=0, help="0: off, 1: on")
    p.add_argument("--proto_lambda", type=float, default=0.1, help="alignment loss weight")
    p.add_argument("--proto_warmup_rounds", type=int, default=0, help="enable proto after warmup rounds")
    p.add_argument("--proto_max_batches", type=int, default=0, help="0 means no limit; SMOKE can set small like 2")

    # Data / batching
    p.add_argument("--per_chain_n", type=int, default=500, help="max samples per chain (limit)")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    # Paths
    p.add_argument("--data_root", default="data/train/crossgraphnet_lite_labeled")
    p.add_argument("--emb_root", default="data/embeddings")
    p.add_argument("--logdir", default="logs/fl")
    p.add_argument("--tag", default="exp")

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    args = p.parse_args()

    out_jsonl = os.path.join(args.logdir, f"{args.algo}_{args.semantic}_{args.tag}.jsonl")

    adapters.prepare_federated_context(
        chains=args.clients,
        semantic_mode=args.semantic,
        data_root=args.data_root,
        emb_root=args.emb_root,
        per_chain_n=args.per_chain_n,
        seed=args.seed,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    clients: List[FLClient] = []
    for chain in args.clients:
        train_loader, test_loader, n_train = adapters.build_loaders(
            chain=chain,
            semantic_mode=args.semantic,
            per_chain_n=args.per_chain_n,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        clients.append(FLClient(chain, train_loader, test_loader, n_train))

    global_model = adapters.build_model(args.semantic)

    proto_max_batches = None if int(args.proto_max_batches) <= 0 else int(args.proto_max_batches)

    run_fl(
        clients=clients,
        global_model=global_model,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        device=args.device,
        out_jsonl=out_jsonl,
        lr=args.lr,
        weight_decay=args.weight_decay,
        algo=args.algo,
        mu=args.mu,
        use_proto=int(args.use_proto),
        proto_lambda=float(args.proto_lambda),
        proto_warmup_rounds=int(args.proto_warmup_rounds),
        proto_max_batches=proto_max_batches,
    )


if __name__ == "__main__":
    main()