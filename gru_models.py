"""PyTorch GRU regressor and the five adaptation strategies.

All strategies start from the SAME pre-trained GRU and receive the SAME adaptation
schedule from the detector (see engine.py), so differences in error come from the
adaptation method only.

  Static                 frozen baseline
  Transfer - frozen GRU  recurrent layer frozen, only the regression head is re-trained
  Full retraining        brand-new GRU trained on all labelled history so far
  Continual + replay K   fine-tune the current GRU on the new window + K replayed old rows
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class GRUConfig:
    seq_len: int = 12
    hidden: int = 48
    batch: int = 256
    base_epochs: int = 30
    base_lr: float = 3e-3
    transfer_epochs: int = 25
    transfer_lr: float = 3e-3
    replay_epochs: int = 12
    replay_lr: float = 1e-3
    replay_sizes: tuple = (100, 500)
    buffer_cap: int = 2000
    full_epochs: int = 15
    full_cap: int = 20000        # max rows used by one full retraining (uniform subsample of all history)

    @classmethod
    def fast(cls):
        """Quick smoke-test setting (a few minutes on a laptop CPU)."""
        return cls(base_epochs=8, transfer_epochs=8, replay_epochs=4, full_epochs=4, full_cap=6000)


class GRURegressor(nn.Module):
    def __init__(self, n_features, hidden=48):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit(model, data, idx, epochs, lr, batch, params=None, seed=0):
    """Mini-batch Adam on the sequences that end at rows `idx` (scaled target)."""
    idx = np.asarray(idx)
    if len(idx) == 0 or epochs <= 0:
        return
    params = list(model.parameters()) if params is None else list(params)
    opt = torch.optim.Adam(params, lr=lr)
    xs = torch.from_numpy(data.seq(idx).astype(np.float32))
    ys = torch.from_numpy(data.y[idx].astype(np.float32))
    gen = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(xs), generator=gen)
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            opt.zero_grad()
            F.mse_loss(model(xs[b]), ys[b]).backward()
            opt.step()


def train_base(data, cfg, seed=0):
    """Pre-train the GRU on the normal history only (rows before train_end)."""
    torch.manual_seed(seed)
    model = GRURegressor(len(data.features), cfg.hidden)
    fit(model, data, np.arange(data.train_end), cfg.base_epochs, cfg.base_lr, cfg.batch, seed=seed)
    return model


class Reservoir:
    """Fixed-size uniform sample of row indices seen so far."""

    def __init__(self, cap, init_idx, rng):
        self.cap, self.rng = int(cap), rng
        init_idx = np.asarray(init_idx)
        self.items = list(rng.choice(init_idx, min(self.cap, len(init_idx)), replace=False))
        self.n = len(init_idx)

    def add(self, idx):
        for i in np.asarray(idx):
            self.n += 1
            if len(self.items) < self.cap:
                self.items.append(int(i))
            else:
                j = int(self.rng.integers(0, self.n))
                if j < self.cap:
                    self.items[j] = int(i)

    def sample(self, k):
        if not self.items:
            return np.empty(0, int)
        return np.asarray(self.rng.choice(self.items, min(int(k), len(self.items)), replace=False))


class _Strategy:
    name = "base"

    def __init__(self, name, base, data, cfg, seed=0):
        self.name, self.d, self.cfg, self.seed = name, data, cfg, int(seed)
        self.model = copy.deepcopy(base)
        self.n_updates = 0

    @torch.no_grad()
    def predict(self, idx):
        self.model.eval()
        out = []
        for s in range(0, len(idx), 2048):
            x = torch.from_numpy(self.d.seq(np.asarray(idx)[s:s + 2048]).astype(np.float32))
            out.append(self.model(x).numpy())
        return np.concatenate(out) * self.d.y_sd + self.d.y_mu

    def update(self, idx):
        self.n_updates += 1
        return self._update(np.asarray(idx))

    def _update(self, idx):
        return 0


class Static(_Strategy):
    pass


class TransferFrozen(_Strategy):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for p in self.model.gru.parameters():
            p.requires_grad = False

    def _update(self, idx):
        c = self.cfg
        fit(self.model, self.d, idx, c.transfer_epochs, c.transfer_lr, c.batch,
            params=self.model.head.parameters(), seed=self.seed + self.n_updates)
        return len(idx)


class FullRetrain(_Strategy):
    def _update(self, idx):
        c = self.cfg
        hist = np.arange(0, int(idx[-1]) + 1)
        if len(hist) > c.full_cap:
            hist = np.sort(np.random.default_rng(self.seed + self.n_updates)
                           .choice(hist, c.full_cap, replace=False))
        torch.manual_seed(self.seed + 1000 + self.n_updates)
        self.model = GRURegressor(len(self.d.features), c.hidden)
        fit(self.model, self.d, hist, c.full_epochs, c.base_lr, c.batch, seed=self.seed + self.n_updates)
        return len(hist)


class Replay(_Strategy):
    def __init__(self, name, base, data, cfg, k, seed=0):
        super().__init__(name, base, data, cfg, seed)
        self.k = int(k)
        self.rng = np.random.default_rng(seed + 2 + self.k)
        self.mem = Reservoir(cfg.buffer_cap, np.arange(data.train_end), self.rng)

    def _update(self, idx):
        c = self.cfg
        old = self.mem.sample(self.k)
        train_idx = np.concatenate([idx, old]) if len(old) else idx
        fit(self.model, self.d, train_idx, c.replay_epochs, c.replay_lr, c.batch,
            seed=self.seed + self.n_updates)
        self.mem.add(idx)
        return len(train_idx)


def build_strategies(data, cfg, seed=0, base=None):
    base = base if base is not None else train_base(data, cfg, seed)
    out = [
        Static("Static (no adaptation)", base, data, cfg, seed),
        TransferFrozen("Transfer - frozen GRU", base, data, cfg, seed),
        FullRetrain("Full retraining", base, data, cfg, seed),
    ]
    out += [Replay(f"Continual + replay {k}", base, data, cfg, k, seed) for k in cfg.replay_sizes]
    return out
