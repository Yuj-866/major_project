"""models.py -- shared T-GCN + GAT core with a swappable input adapter and head.

    input adapter (in_dim -> hidden)      <- differs per dataset (17 DGraph, 11 cloud)
        -> T-GCN (GRU with graph-conv gates, run over the T time slices)
        -> GAT on the last slice
    head                                  <- differs per task
        "node":  per-node logits              (DGraph fraud vs normal)
        "graph": mean+max pool -> logits      (cloud LOW / MEDIUM / HIGH)

Only the core (T-GCN + GAT) is saved after pretraining and reused for fine-tuning.

Batch format expected by RiskModel.forward:
    {"x": [T tensors [N, in_dim]], "edge_index": [T tensors [2, E_t]],
     "edge_weight": [T tensors [E_t]] or None,
     "batch": LongTensor [N] (graph id per node; needed for task="graph"),
     "num_graphs": int}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool


class TGCNCell(nn.Module):
    """T-GCN cell: a GRU whose linear maps are replaced by graph convolutions."""

    def __init__(self, hidden):
        super().__init__()
        self.conv_gates = GCNConv(2 * hidden, 2 * hidden)   # update + reset gates
        self.conv_cand = GCNConv(2 * hidden, hidden)        # candidate state

    def forward(self, x, h, edge_index, edge_weight=None):
        ur = torch.sigmoid(self.conv_gates(torch.cat([x, h], -1), edge_index, edge_weight))
        u, r = ur.chunk(2, dim=-1)
        c = torch.tanh(self.conv_cand(torch.cat([x, r * h], -1), edge_index, edge_weight))
        return u * h + (1 - u) * c


class TGCNGATCore(nn.Module):
    def __init__(self, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.hidden = hidden
        self.cell = TGCNCell(hidden)
        self.gat = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, xs, edge_indices, edge_weights=None):
        h = xs[0].new_zeros(xs[0].size(0), self.hidden)
        for t, x in enumerate(xs):
            ew = None if edge_weights is None else edge_weights[t]
            h = self.cell(x, h, edge_indices[t], ew)
        g = F.elu(self.gat(h, edge_indices[-1]))
        return self.norm(h + self.drop(g))


class RiskModel(nn.Module):
    def __init__(self, in_dim, out_dim, task, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        assert task in ("node", "graph")
        self.task, self.hidden, self.heads = task, hidden, heads
        self.adapter = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.core = TGCNGATCore(hidden, heads, dropout)
        if task == "node":
            self.head = nn.Linear(hidden, out_dim)
        else:
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, b):
        cache, xs = {}, []
        for x in b["x"]:                       # static x is adapted once, not T times
            if id(x) not in cache:
                cache[id(x)] = self.adapter(x)
            xs.append(cache[id(x)])
        z = self.core(xs, b["edge_index"], b.get("edge_weight"))
        if self.task == "node":
            return self.head(z)
        n = b["num_graphs"]
        pooled = torch.cat([global_mean_pool(z, b["batch"], n),
                            global_max_pool(z, b["batch"], n)], dim=-1)
        return self.head(pooled)

    # ---- helpers for pretrain -> fine-tune
    def load_core(self, state_dict):
        self.core.load_state_dict(state_dict)

    def set_core_trainable(self, flag):
        for p in self.core.parameters():
            p.requires_grad = flag


def to_device(b, device):
    out = dict(b)
    cache = {}
    def mv(t):
        if id(t) not in cache:
            cache[id(t)] = t.to(device)
        return cache[id(t)]
    out["x"] = [mv(t) for t in b["x"]]
    out["edge_index"] = [t.to(device) for t in b["edge_index"]]
    if b.get("edge_weight") is not None:
        out["edge_weight"] = [t.to(device) for t in b["edge_weight"]]
    if b.get("batch") is not None:
        out["batch"] = b["batch"].to(device)
    return out