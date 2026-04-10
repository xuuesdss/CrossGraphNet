import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class DFGDataset(Dataset):
    """
    读取 DFG jsonl：
      {
        "id": "...",
        "label": 0/1,
        "dfg_nodes": [{"id":0..N-1, "type": "...", ...}, ...],
        "dfg_edges": [{"src":0..N-1, "dst":0..N-1, "var":"x"}, ...]
      }

    输出：PyG Data(x, edge_index, y) ，其中 y 是标量 long（0/1）
    """
    def __init__(self, jsonl_path: str):
        self.samples = []
        self.node_type_map = {}
        self.num_node_types = 0
        self._build_dataset(jsonl_path)

    def _build_dataset(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        # 1) 收集 node type
        type_set = set()
        parsed = []
        for line in lines:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            nodes = obj.get("dfg_nodes", []) or []
            for node in nodes:
                type_set.add(str(node.get("type", "UNK")))
            parsed.append(obj)

        self.node_type_map = {t: i for i, t in enumerate(sorted(type_set))}
        self.num_node_types = len(self.node_type_map)

        # 2) 构造样本
        for obj in parsed:
            nodes = obj.get("dfg_nodes", []) or []
            edges = obj.get("dfg_edges", []) or []

            num_nodes = len(nodes)
            if num_nodes <= 0:
                # 至少给一个节点，避免 PyG 报错
                num_nodes = 1
                nodes = [{"id": 0, "type": "PAD"}]

            # 节点 one-hot
            x = torch.zeros((num_nodes, self.num_node_types), dtype=torch.float32)
            for node in nodes:
                idx = int(node.get("id", 0))
                if idx < 0 or idx >= num_nodes:
                    continue
                t = str(node.get("type", "UNK"))
                if t in self.node_type_map:
                    x[idx, self.node_type_map[t]] = 1.0

            # 边
            src, dst = [], []
            for e in edges:
                s = e.get("src", None)
                d = e.get("dst", None)
                if s is None or d is None:
                    continue
                s = int(s); d = int(d)
                if 0 <= s < num_nodes and 0 <= d < num_nodes:
                    src.append(s); dst.append(d)

            if len(src) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([src, dst], dtype=torch.long)

            # label
            yv = obj.get("label", None)
            if yv is None:
                # 兼容某些数据用 "y"
                yv = obj.get("y", 0)
            y = torch.tensor(int(yv), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]