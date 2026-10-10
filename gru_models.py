"""PyTorch GRU regressor and the adaptation strategies.

All strategies start from the SAME pre-trained GRU and receive the SAME adaptation
schedule from the detector (see engine.py), so differences in error come from the
adaptation method only.

  Static                      frozen baseline
  Transfer - frozen GRU       recurrent layer frozen, only the regression head is re-trained
  Full retraining             brand-new GRU trained on all labelled history so far
  Continual + replay K        fine-tune the current GRU on the new window + K replayed old rows

Improvements added on top of the original five:
  * Output clipping           every prediction is clipped to the valid target range (cfg.clip_range)
  * "... + rollback" variants when the detector says the old behaviour is back (recovery flag),
                              the model returns to the original pre-trained weights
  * "Blend (error-weighted)"  averages all other strategies, trusting each in proportion to how
                              well it did on recent windows
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
    clip_range: tuple = (0.0, 100.0)   # valid range of the target; set to None to disable clipping
    rollback_variants: bool = True     # add "+ rollback" copies of Transfer and Replay 500
    blend: bool = True                 # add the error-weighted blend of all other strategies
    blend_decay: float = 0.3           # memory of the blend's error average (0 = only the last window; higher = slower)
    blend_power: float = 6.0           # higher = the blend leans harder on the currently best model
    huber_beta: float = 0.5            # Huber loss (in target std units): robust to outliers; 0 = plain MSE
    hard_drift_mult: float = 1.5       # relational / both drift: this many times more epochs per update
    adapt_until_recovery: bool = True  # keep adapting while drift is active, not just 3 windows
    max_adapt_windows: int = 12        # safety cap for the above

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
    model = GRURegressor(len(data.features), cfg.hidden)
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
        return self._update(np.asarray(idx))

    def _update(self, idx):
        return 0

    def rollback(self):
        """Return to the original pre-trained weights (the old behaviour is valid again)."""
        self.model.load_state_dict(self.base_state)
        self.n_rollbacks += 1


class Static(_Strategy):
    pass


class TransferFrozen(_Strategy):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for p in self.model.gru.parameters():
            p.requires_grad = False

    def _update(self, idx):
        c = self.cfg
        fit(self.model, self.d, idx, self._epochs(c.transfer_epochs), c.transfer_lr, c.batch,
            params=self.model.head.parameters(), seed=self.seed + self.n_updates, huber_beta=c.huber_beta)
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
        fit(self.model, self.d, hist, c.full_epochs, c.base_lr, c.batch, seed=self.seed + self.n_updates,
            huber_beta=c.huber_beta)
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
        fit(self.model, self.d, train_idx, self._epochs(c.replay_epochs), c.replay_lr, c.batch,
            seed=self.seed + self.n_updates, huber_beta=c.huber_beta)
        self.mem.add(idx)
        return len(train_idx)


class Blend(_Strategy):
    """Error-weighted average of other strategies.

    After every window (labels revealed, same as the detector) each member's MAE is folded into a
    running average. The next window's prediction weights members by 1 / error^power, so the blend
    follows whichever model is currently doing best, without needing to know the drift type.
    It never trains anything itself; the members are adapted by the engine as usual.
    """

    def __init__(self, name, members, data, cfg, seed=0):
        self.name, self.d, self.cfg, self.seed = name, data, cfg, int(seed)
        self.members = list(members)
        self.rollback_on_recovery = False
        self.diagnosis = "none"
        self.n_updates = 0
        self.n_rollbacks = 0
        self.ema = None
        self.weight_history = []          # one row per window: the weights used for that window
        self._w = np.full(len(self.members), 1.0 / len(self.members))

    def weights(self):
        if self.ema is None:
            return np.full(len(self.members), 1.0 / len(self.members))
        w = np.maximum(self.ema, 1e-6) ** (-self.cfg.blend_power)
        return w / w.sum()

    def predict(self, idx):
        self._w = self.weights()
        preds = np.stack([np.asarray(m.predict(idx), float) for m in self.members])
        return self._w @ preds

    def observe(self, idx, y_true):
        """Called by the engine after each window's predictions, before any adaptation."""
        err = np.array([float(np.mean(np.abs(np.asarray(m.predict(idx), float) - y_true))) for m in self.members])
        dcy = self.cfg.blend_decay
        self.ema = err if self.ema is None else dcy * self.ema + (1 - dcy) * err
        self.weight_history.append({"window": len(self.weight_history),
                                    **{m.name: float(w) for m, w in zip(self.members, self._w)}})

    def update(self, idx):
        return 0

    def rollback(self):
        pass


def build_strategies(data, cfg, seed=0, base=None):
    base = base if base is not None else train_base(data, cfg, seed)
    out = [
        Static("Static (no adaptation)", base, data, cfg, seed),
        TransferFrozen("Transfer - frozen GRU", base, data, cfg, seed),
        FullRetrain("Full retraining", base, data, cfg, seed),
    ]
    out += [Replay(f"Continual + replay {k}", base, data, cfg, k, seed) for k in cfg.replay_sizes]

    if cfg.rollback_variants:
        tr = TransferFrozen("Transfer - frozen GRU + rollback", base, data, cfg, seed)
        tr.rollback_on_recovery = True
        k = max(cfg.replay_sizes)
        rp = Replay(f"Continual + replay {k} + rollback", base, data, cfg, k, seed)
        rp.rollback_on_recovery = True
        out += [tr, rp]

    if cfg.blend:
        out.append(Blend("Blend (error-weighted)", list(out), data, cfg, seed))
    return out
