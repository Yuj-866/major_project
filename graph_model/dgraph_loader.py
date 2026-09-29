"""dgraph_loader.py -- DGraph-Fin -> small temporal subgraphs for pretraining.

DGraph-Fin is far too big to use whole (3.7M nodes), so each training sample is:
  1. a few seed users (half fraud, half normal, from the official split),
  2. their 2-hop neighbourhood (neighbours capped, so a subgraph has a few thousand nodes),
  3. split into T cumulative time slices by edge timestamp
     (slice k = all edges with timestamp <= k-th quantile), matching the cloud
     sequence length of 5. Node features are static in DGraph, only edges change.

Expects dgraphfin.npz with keys: x, y, edge_index (E,2), edge_timestamp,
train_mask / valid_mask / test_mask (node index arrays or boolean masks).
"""
import numpy as np
import torch


class DGraphData:
    def __init__(self, path):
        d = np.load(path)
        x = d["x"].astype(np.float32)
        mu, sd = x.mean(0), x.std(0)
        sd[sd < 1e-6] = 1.0
        self.x = torch.from_numpy((x - mu) / sd)          # standardised (missing = -1 kept as a value)
        self.y = d["y"].astype(np.int64)
        self.N = x.shape[0]

        e = d["edge_index"].astype(np.int64)
        if e.shape[0] == 2 and e.shape[1] != 2:
            e = e.T
        ts = d["edge_timestamp"] if "edge_timestamp" in d.files else d["edge_time"]
        ts = ts.astype(np.float64)

        # symmetric CSR built by hand (scipy would SUM duplicate timestamps)
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        tt = np.concatenate([ts, ts])
        order = np.argsort(src, kind="stable")
        self.nbr, self.ts = dst[order], tt[order]
        self.indptr = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=self.N), out=self.indptr[1:])

        self.pools = {}
        for name in ["train", "valid", "test"]:
            m = d[f"{name}_mask"]
            idx = np.flatnonzero(m) if m.dtype == bool else m.astype(np.int64)
            idx = idx[self.y[idx] <= 1]                    # only normal(0) / fraud(1) are targets
            self.pools[name] = (idx[self.y[idx] == 1], idx[self.y[idx] == 0])
        for name, (fr, no) in self.pools.items():
            print(f"[DGraph] {name}: {len(fr)} fraud, {len(no)} normal labeled nodes")

    def _expand(self, nodes, cap, rng):
        S, Nb, T = [], [], []
        for v in nodes:
            lo, hi = self.indptr[v], self.indptr[v + 1]
            deg = hi - lo
            if deg == 0:
                continue
            pos = np.arange(lo, hi) if deg <= cap else lo + rng.choice(deg, cap, replace=False)
            S.append(np.full(len(pos), v)); Nb.append(self.nbr[pos]); T.append(self.ts[pos])
        if not S:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0)
        return np.concatenate(S), np.concatenate(Nb), np.concatenate(T)

    def sample(self, split, rng, n_seeds=32, cap1=15, cap2=8, slices=5):
        fr, no = self.pools[split]
        half = n_seeds // 2
        seeds = np.concatenate([rng.choice(fr, half, replace=False),
                                rng.choice(no, n_seeds - half, replace=False)])
        s1, n1, t1 = self._expand(seeds, cap1, rng)
        s2, n2, t2 = self._expand(np.unique(n1), cap2, rng)
        src = np.concatenate([s1, s2]); dst = np.concatenate([n1, n2]); ts = np.concatenate([t1, t2])

        nodes, inv = np.unique(np.concatenate([seeds, src, dst]), return_inverse=True)
        S, E = len(seeds), len(src)
        seed_local, a, b = inv[:S], inv[S:S + E], inv[S + E:]
        keep = a != b
        a, b, ts = a[keep], b[keep], ts[keep]
        lo_, hi_ = np.minimum(a, b), np.maximum(a, b)      # undirected, deduplicated
        order = np.argsort(ts, kind="stable")
        key = lo_[order] * len(nodes) + hi_[order]
        _, first = np.unique(key, return_index=True)
        sel = order[first]
        lo_, hi_, ts = lo_[sel], hi_[sel], ts[sel]

        edge_index = []
        for q in np.linspace(1.0 / slices, 1.0, slices):
            thr = np.quantile(ts, q) if len(ts) else 0
            m = ts <= thr
            ei = np.stack([np.concatenate([lo_[m], hi_[m]]), np.concatenate([hi_[m], lo_[m]])])
            edge_index.append(torch.from_numpy(ei).long())
        return {"x": self.x[torch.from_numpy(nodes)], "edge_index": edge_index,
                "seed_local": torch.from_numpy(seed_local).long(),
                "y_seed": torch.from_numpy(self.y[seeds]).long()}


def collate(items):
    T = len(items[0]["edge_index"])
    xs, seeds, ys = [], [], []
    eis = [[] for _ in range(T)]
    off = 0
    for it in items:
        xs.append(it["x"])
        for t in range(T):
            eis[t].append(it["edge_index"][t] + off)
        seeds.append(it["seed_local"] + off)
        ys.append(it["y_seed"])
        off += it["x"].size(0)
    x = torch.cat(xs)
    batch = {"x": [x] * T, "edge_index": [torch.cat(e, 1) for e in eis],
             "edge_weight": None, "batch": None, "num_graphs": len(items)}
    return batch, torch.cat(seeds), torch.cat(ys)