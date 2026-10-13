"""PyTorch GRU regressor and the adaptation strategies.

All strategies start from the SAME pre-trained GRU and receive the SAME adaptation
schedule from the detector (see engine.py), so differences in error come from the
adaptation method only.

  Static                      frozen baseline
  Transfer                    lower GRU layer(s) frozen; the regression head and (by default) the TOP GRU
                              layer are re-trained on ALL rows of the current drift episode, and the model
                              returns to the pre-trained weights when the drift ends (see TransferFrozen)
  Full retraining             brand-new GRU trained on all labelled history so far
  Continual + replay K        fine-tune the current GRU on the new window + K replayed old rows

Improvements added on top of the original five:
  * Output clipping           every prediction is clipped to the valid target range (cfg.clip_range)
  * "... + rollback" variants when the detector says the old behaviour is back (recovery flag),
                              the model returns to the original pre-trained weights
"""
import copy
import time
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
    n_layers: int = 2                  # stacked GRU layers (2 gives transfer a generic lower layer to freeze)
    batch: int = 256
    base_epochs: int = 30
    base_lr: float = 3e-3
    transfer_epochs: int = 25
    transfer_lr: float = 3e-3
    transfer_unfreeze: str = "top"     # "head" = only the MLP head (original) | "top" = head + top GRU layer | "all"
    transfer_gru_lr_mult: float = 0.3  # unfrozen GRU layers learn at this fraction of transfer_lr (less forgetting)
    transfer_episode: bool = True      # train on every row of the current drift episode, not just the newest window
    transfer_episode_cap: int = 3000   # max rows kept for that (uniform subsample beyond it)
    transfer_replay: int = 0           # OLD-regime rows mixed in. 0 on purpose: after a relational drift they carry
                                       # labels that contradict the new relation (measured: replay 300 is ~35% worse)
    transfer_rollback: bool = True     # back to the pre-trained weights when the detector reports recovery
    replay_epochs: int = 12
    replay_lr: float = 1e-3
    replay_sizes: tuple = (100, 500)
    buffer_cap: int = 2000
    full_epochs: int = 15
    full_cap: int = 20000        # max rows used by one full retraining (uniform subsample of all history)
    clip_range: tuple = (0.0, 100.0)   # valid range of the target; set to None to disable clipping
    rollback_variants: bool = True     # add "+ rollback" copies of Transfer and Replay 500
    huber_beta: float = 0.5            # Huber loss (in target std units): robust to outliers; 0 = plain MSE
    hard_drift_mult: float = 1.5       # relational / both drift: this many times more epochs per update
    adapt_until_recovery: bool = True  # keep adapting while drift is active, not just 3 windows
    max_adapt_windows: int = 12        # safety cap for the above

    @classmethod
    def fast(cls):
        """Quick smoke-test setting (a few minutes on a laptop CPU)."""
        return cls(base_epochs=8, transfer_epochs=8, replay_epochs=4, full_epochs=4, full_cap=6000)


class GRURegressor(nn.Module):
    def __init__(self, n_features, hidden=48, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=int(n_layers), batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit(model, data, idx, epochs, lr, batch, params=None, seed=0, huber_beta=0.0):
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
            out = model(xs[b])
            loss = F.smooth_l1_loss(out, ys[b], beta=huber_beta) if huber_beta and huber_beta > 0 \
                else F.mse_loss(out, ys[b])
            loss.backward()
            opt.step()


def train_base(data, cfg, seed=0):
    """Pre-train the GRU on the normal history only (rows before train_end)."""
    torch.manual_seed(seed)
    model = GRURegressor(len(data.features), cfg.hidden, cfg.n_layers)
    fit(model, data, np.arange(data.train_end), cfg.base_epochs, cfg.base_lr, cfg.batch, seed=seed,
        huber_beta=cfg.huber_beta)
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
        self.base_state = copy.deepcopy(base.state_dict())   # original weights, kept for rollback
        self.rollback_on_recovery = False
        self.diagnosis = "none"          # set by the engine before every update
        self.n_updates = 0
        self.n_rollbacks = 0
        self.train_seconds = 0.0         # wall-clock time spent inside update()
        self.train_samples = 0           # rows seen by the optimiser, summed over updates (not times epochs)

    def _epochs(self, base_epochs):
        """More epochs when the relationship itself has changed (relational or both)."""
        hard = self.diagnosis in ("relational", "both")
        return max(1, int(round(base_epochs * (self.cfg.hard_drift_mult if hard else 1.0))))

    @torch.no_grad()
    def predict(self, idx):
        self.model.eval()
        out = []
        for s in range(0, len(idx), 2048):
            x = torch.from_numpy(self.d.seq(np.asarray(idx)[s:s + 2048]).astype(np.float32))
            out.append(self.model(x).numpy())
        p = np.concatenate(out) * self.d.y_sd + self.d.y_mu
        if self.cfg.clip_range is not None:
            p = np.clip(p, self.cfg.clip_range[0], self.cfg.clip_range[1])
        return p

    def update(self, idx):
        self.n_updates += 1
        t0 = time.perf_counter()
        used = self._update(np.asarray(idx))
        self.train_seconds += time.perf_counter() - t0
        self.train_samples += int(used)
        return used

    def _update(self, idx):
        return 0

    def stored_rows(self):
        """Raw rows this strategy must keep to do its next update (0 = nothing beyond the model)."""
        return 0

    def memory_mb(self):
        """ESTIMATED state a deployment must hold: model weights, the rollback copy, and stored rows.
        It is not process RSS and ignores transient training buffers; it is for comparing strategies."""
        n_par = sum(p.numel() for p in self.model.parameters())
        copies = 2 if self.rollback_on_recovery else 1
        row_bytes = 4 * (len(self.d.features) + 1)
        return (4 * n_par * copies + row_bytes * self.stored_rows()) / 1e6

    def rollback(self):
        """Return to the original pre-trained weights (the old behaviour is valid again)."""
        self.model.load_state_dict(self.base_state)
        self.n_rollbacks += 1


class Static(_Strategy):
    pass


class TransferFrozen(_Strategy):
    """Transfer learning from the pre-trained GRU.

    The lower GRU layers stay frozen (generic features). What is re-trained depends on cfg.transfer_unfreeze:
      "head" - only the MLP head                      (original behaviour)
      "top"  - the head and the top GRU layer          (default; the GRU layer learns at a lower rate)
      "all"  - everything (then it is just continual fine-tuning)

    Three things decide whether transfer beats Static (found with the stand-in study, surrogate_study.py):
      1. Train on the whole drift episode seen so far (cfg.transfer_episode). One 600-row window is a narrow,
         strongly autocorrelated slice of input space, so a head fitted to it extrapolates badly.
      2. Do NOT replay old-regime rows (cfg.transfer_replay = 0). After a relational drift their labels
         contradict the new relation and drag the model back towards the old one.
      3. Roll back to the pre-trained weights on recovery (the engine does this when rollback_on_recovery
         is set; cfg.transfer_rollback). Without it the model stays tuned to a regime that has ended.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        mode = self.cfg.transfer_unfreeze
        if mode not in ("head", "top", "all"):
            raise ValueError(f"transfer_unfreeze must be 'head', 'top' or 'all', got {mode!r}")
        top = f"_l{self.model.gru.num_layers - 1}"
        for name, p in self.model.gru.named_parameters():
            p.requires_grad = mode == "all" or (mode == "top" and name.endswith(top))
        self.rng = np.random.default_rng(self.seed + 3)
        self.mem = (Reservoir(self.cfg.buffer_cap, np.arange(self.d.train_end), self.rng)
                    if self.cfg.transfer_replay > 0 else None)
        self.ep_rows = np.empty(0, int)          # rows of the current drift episode

    def rollback(self):
        super().rollback()
        self.ep_rows = np.empty(0, int)          # a new episode starts from the original model and fresh data

    def _update(self, idx):
        c = self.cfg
        train_idx = np.asarray(idx)
        if c.transfer_episode:
            rows = np.concatenate([self.ep_rows, train_idx])
            if len(rows) > c.transfer_episode_cap:
                rows = np.sort(self.rng.choice(rows, c.transfer_episode_cap, replace=False))
            self.ep_rows = rows
            train_idx = rows
        if self.mem is not None:
            old = self.mem.sample(c.transfer_replay)
            if len(old):
                train_idx = np.concatenate([train_idx, old])
        groups = [{"params": list(self.model.head.parameters()), "lr": c.transfer_lr}]
        gru_params = [p for p in self.model.gru.parameters() if p.requires_grad]
        if gru_params:
            groups.append({"params": gru_params, "lr": c.transfer_lr * c.transfer_gru_lr_mult})
        fit(self.model, self.d, train_idx, self._epochs(c.transfer_epochs), c.transfer_lr, c.batch,
            params=groups, seed=self.seed + self.n_updates, huber_beta=c.huber_beta)
        if self.mem is not None:
            self.mem.add(idx)
        return len(train_idx)

    def stored_rows(self):
        return len(self.ep_rows) + (len(self.mem.items) if self.mem is not None else 0)


class FullRetrain(_Strategy):
    def _update(self, idx):
        c = self.cfg
        hist = np.arange(0, int(idx[-1]) + 1)
        self._seen = len(hist)           # full retraining must keep all labelled history
        if len(hist) > c.full_cap:
            hist = np.sort(np.random.default_rng(self.seed + self.n_updates)
                           .choice(hist, c.full_cap, replace=False))
        torch.manual_seed(self.seed + 1000 + self.n_updates)
        self.model = GRURegressor(len(self.d.features), c.hidden, c.n_layers)
        fit(self.model, self.d, hist, c.full_epochs, c.base_lr, c.batch, seed=self.seed + self.n_updates,
            huber_beta=c.huber_beta)
        return len(hist)

    def stored_rows(self):
        return self._seen if hasattr(self, "_seen") else self.d.train_end


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
        fit(self.model, self.d, train_idx, self._epochs(c.replay_epochs), c.replay_lr, c.batch,
            seed=self.seed + self.n_updates, huber_beta=c.huber_beta)
        self.mem.add(idx)
        return len(train_idx)

    def stored_rows(self):
        return len(self.mem.items)


def build_strategies(data, cfg, seed=0, base=None):
    base = base if base is not None else train_base(data, cfg, seed)
    tlabel = {"head": "frozen GRU", "top": "top layer unfrozen", "all": "fine-tune all"}[cfg.transfer_unfreeze]
    out = [
        Static("Static (no adaptation)", base, data, cfg, seed),
        TransferFrozen(f"Transfer - {tlabel}", base, data, cfg, seed),
        FullRetrain("Full retraining", base, data, cfg, seed),
    ]
    out += [Replay(f"Continual + replay {k}", base, data, cfg, k, seed) for k in cfg.replay_sizes]

    out[1].rollback_on_recovery = cfg.transfer_rollback      # the main Transfer rolls back by default
    if cfg.rollback_variants:
        k = max(cfg.replay_sizes)
        rp = Replay(f"Continual + replay {k} + rollback", base, data, cfg, k, seed)
        rp.rollback_on_recovery = True
        out.append(rp)
        if not cfg.transfer_rollback:                         # otherwise Transfer already IS the rollback version
            tr = TransferFrozen(f"Transfer - {tlabel} + rollback", base, data, cfg, seed)
            tr.rollback_on_recovery = True
            out.append(tr)

    return out
