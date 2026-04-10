# src/federated/client.py
import copy
from typing import Optional, Dict, Tuple

import torch


class FLClient:
    def __init__(self, name, train_loader, test_loader, num_samples):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_samples = int(num_samples)

    def local_train(
        self,
        global_model,
        train_one_epoch_fn,
        optimizer_fn,
        local_epochs,
        device,
        algo: str = "fedavg",
        mu: float = 0.0,
        # proto knobs
        use_proto: bool = False,
        proto_lambda: float = 0.1,
        global_protos: Optional[Dict[int, torch.Tensor]] = None,
        proto_max_batches: Optional[int] = None,
    ):
        """
        algo: 'fedavg' or 'fedprox'
        mu: FedProx proximal strength (typical: 1e-3 ~ 1e-2)

        Prototype Learning (minimal):
        - If use_proto=True and global_protos is not None, training adds alignment loss.
        - After local training, compute local class prototypes and return to server for aggregation.
        """
        model = copy.deepcopy(global_model).to(device)
        opt = optimizer_fn(model)

        # snapshot global params for FedProx proximal term
        global_params = None
        if algo.lower() == "fedprox":
            global_params = [p.detach().clone().to(device) for p in global_model.parameters()]

        last_loss = None
        for _ in range(local_epochs):
            last_loss = train_one_epoch_fn(
                model,
                self.train_loader,
                opt,
                device=device,
                algo=algo,
                mu=mu,
                global_params=global_params,
                # proto
                use_proto=use_proto,
                proto_lambda=proto_lambda,
                global_protos=global_protos,
            )

        proto_pack = None
        if use_proto:
            # compute local prototypes for server aggregation
            lp, lc = train_one_epoch_fn.compute_prototypes(  # type: ignore[attr-defined]
                model,
                self.train_loader,
                device=device,
                max_batches=proto_max_batches,
            )
            proto_pack = (lp, lc)

        return model.state_dict(), last_loss, proto_pack
