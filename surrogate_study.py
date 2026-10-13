"""Torch-free stand-in study: tests WHY Transfer loses, using the real data, detector and streaming engine.

The model is a numpy MLP (encoder layers = stand-in for the GRU, head = the regression head) because the
target depends only on the current row, so the GRU's memory adds nothing the MLP lacks. It is NOT the GRU:
use it to judge the mechanisms (replay, unfreezing, capacity), then confirm with diagnose_transfer.py.
"""
import copy, sys, time
import numpy as np, pandas as pd
from data_generator import FEATURES, MONITORED, TARGET, TRAIN_END, WINDOW, generate
from engine import StreamData, StreamRunner
from evaluation import strategy_metrics


class Reservoir:   # same logic as gru_models.Reservoir (copied so this file needs no PyTorch)
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
        return np.asarray(self.rng.choice(self.items, min(int(k), len(self.items)), replace=False))


class MLP:
    def __init__(self, dims, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.tanh(z))
        return acts

    def predict(self, X):
        return self.forward(X)[-1][:, 0]

    def fit(self, X, y, epochs, lr, batch, layers, lr_mult, rng, beta=0.5):
        L = len(self.W)
        m = {i: [np.zeros_like(self.W[i]), np.zeros_like(self.b[i])] for i in layers}
        v = {i: [np.zeros_like(self.W[i]), np.zeros_like(self.b[i])] for i in layers}
        t = 0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(order), batch):
                bi = order[s:s + batch]
                acts = self.forward(X[bi])
                err = acts[-1][:, 0] - y[bi]
                g = np.clip(err, -beta, beta) / len(bi)          # Huber gradient
                delta = g[:, None]
                t += 1
                for i in range(L - 1, -1, -1):
                    gW, gb = acts[i].T @ delta, delta.sum(0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (1 - acts[i] ** 2)
                    if i in layers:
                        for k, (p, gr) in enumerate(((self.W, gW), (self.b, gb))):
                            m[i][k] = 0.9 * m[i][k] + 0.1 * gr
                            v[i][k] = 0.999 * v[i][k] + 0.001 * gr ** 2
                            mh, vh = m[i][k] / (1 - 0.9 ** t), v[i][k] / (1 - 0.999 ** t)
                            p[i] -= lr * lr_mult.get(i, 1.0) * mh / (np.sqrt(vh) + 1e-8)


DIMS = [8, 64, 48, 32, 1]           # layers 0-1 = encoder (GRU stand-in), 2-3 = head
ENC, HEAD, TOP = [0, 1], [2, 3], [1]


def mk_x(d, idx):
    return d.seq(np.asarray(idx))[:, -1, :]


class S:
    def __init__(self, name, base, d, seed, mode="none", replay=0, lr=3e-3, epochs=25, gru_mult=0.3,
                 rollback=False, full=False, skip_data=False, episode=False, ep_cap=3000, guard=0.0):
        self.name, self.d, self.mode, self.replay, self.full = name, d, mode, replay, full
        self.model, self.base = copy.deepcopy(base), copy.deepcopy(base)
        self.lr, self.epochs, self.mult = lr, epochs, gru_mult
        self.rng = np.random.default_rng(seed + 11)
        self.rollback_on_recovery, self.diagnosis = rollback, "none"
        self.mem = Reservoir(2000, np.arange(d.train_end), self.rng) if replay else None
        self.seed = seed
        self.skip_data, self.episode, self.ep_cap, self.ep_rows = skip_data, episode, ep_cap, []
        self.guard, self.dirty, self.n_guard = guard, False, 0

    def predict(self, idx):
        return np.clip(self.model.predict(mk_x(self.d, idx)) * self.d.y_sd + self.d.y_mu, 0, 100)

    def rollback(self):
        self.model = copy.deepcopy(self.base)
        self.ep_rows, self.dirty = [], False

    def observe(self, idx, y_true):
        """Labels of the window just predicted are known: if the ORIGINAL model now beats the adapted one,
        the old behaviour is back, so go back to it (faster than waiting for the detector's recovery flag)."""
        if not (self.guard and self.dirty):
            return
        x = mk_x(self.d, idx)
        f = lambda m: np.mean(np.abs(np.clip(m.predict(x) * self.d.y_sd + self.d.y_mu, 0, 100) - y_true))
        if f(self.model) > self.guard * f(self.base):
            self.rollback()
            self.n_guard += 1

    def update(self, idx):
        d = self.d
        if self.mode == "none":
            return 0
        ep = int(round(self.epochs * (1.5 if self.diagnosis in ("relational", "both") else 1)))
        if self.full:
            hist = np.arange(0, int(idx[-1]) + 1)
            if len(hist) > 20000:
                hist = np.sort(self.rng.choice(hist, 20000, replace=False))
            self.model = MLP(DIMS, self.rng)
            self.model.fit(mk_x(d, hist), d.y[hist], 15, 3e-3, 256, ENC + HEAD, {}, self.rng)
            return len(hist)
        if self.skip_data and self.diagnosis == "data":
            return 0          # relation unchanged: Static's mapping is still right, do not touch it
        tr = np.asarray(idx)
        if self.episode:      # train on every row of this drift episode, not just the latest window
            self.ep_rows.append(tr)
            tr = np.concatenate(self.ep_rows)
            if len(tr) > self.ep_cap:
                tr = np.sort(self.rng.choice(tr, self.ep_cap, replace=False))
        if self.mem is not None:
            old = self.mem.sample(self.replay)
            tr = np.concatenate([tr, old]) if len(old) else tr
        layers = {"head": HEAD, "top": TOP + HEAD, "all": ENC + HEAD}[self.mode]
        self.model.fit(mk_x(d, tr), d.y[tr], ep, self.lr, 256, layers, {i: self.mult for i in ENC}, self.rng)
        if self.mem is not None:
            self.mem.add(idx)
        self.dirty = True
        return len(tr)


def base_model(d, seed, dims=DIMS):
    rng = np.random.default_rng(seed)
    m = MLP(dims, rng)
    tr = np.arange(d.train_end)
    m.fit(mk_x(d, tr), d.y[tr], 30, 3e-3, 256, list(range(len(dims) - 1)), {}, rng)
    return m


def run(df, seed, variants):
    d = StreamData(df, FEATURES, MONITORED, TARGET, TRAIN_END, 12)
    base = base_model(d, seed)
    strats = [S("Static", base, d, seed)] + [S(n, base, d, seed, **kw) for n, kw in variants]
    win, pred = StreamRunner(d, strats, WINDOW).run()
    m = strategy_metrics(pred, df, TARGET)
    return m.pivot_table(index="Strategy", columns="Phase", values="MAE"), win


POLICY = [
    ("P1 top, skip data, episode buf, no old replay, rollback", dict(mode="top", skip_data=True, episode=True, rollback=True)),
    ("P2 = P1 + replay100", dict(mode="top", skip_data=True, episode=True, rollback=True, replay=100)),
    ("P3 head-only version of P1", dict(mode="head", skip_data=True, episode=True, rollback=True)),
    ("P4 all-layers version of P1", dict(mode="all", skip_data=True, episode=True, rollback=True)),
    ("P5 = P1 without episode buffer", dict(mode="top", skip_data=True, rollback=True)),
    ("P6 = P1 without skip_data", dict(mode="top", episode=True, rollback=True)),
    ("P7 = P1 without rollback", dict(mode="top", skip_data=True, episode=True)),
]
FINAL = [
    ("T-A ORIGINAL transfer (head only, window only)", dict(mode="head")),
    ("N1 top + episode buf + rollback", dict(mode="top", episode=True, rollback=True)),
    ("N2 = N1 + guard 1.10", dict(mode="top", episode=True, rollback=True, guard=1.10)),
    ("N3 = N1 + guard 1.25", dict(mode="top", episode=True, rollback=True, guard=1.25)),
    ("N4 all layers + guard 1.10", dict(mode="all", episode=True, rollback=True, guard=1.10)),
    ("N5 head only + guard 1.10", dict(mode="head", episode=True, rollback=True, guard=1.10)),
    ("Replay 500 (all layers)", dict(mode="all", replay=500, lr=1e-3, epochs=12)),
    ("Full retraining", dict(mode="all", full=True)),
]
VARIANTS = [
    ("T-A head only, no replay (ORIGINAL)", dict(mode="head")),
    ("T-B head only + replay300", dict(mode="head", replay=300)),
    ("T-C top+head, no replay", dict(mode="top")),
    ("T-D top+head + replay300", dict(mode="top", replay=300)),
    ("T-E top+head + replay300 + rollback", dict(mode="top", replay=300, rollback=True)),
    ("T-F all layers + replay300", dict(mode="all", replay=300)),
    ("Replay 500 (all layers)", dict(mode="all", replay=500, lr=1e-3, epochs=12)),
    ("Full retraining", dict(mode="all", full=True)),
]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--policy"]
    if "--final" in sys.argv:
        VARIANTS = FINAL
        args = [a for a in args if a != "--final"]
    elif "--policy" in sys.argv:
        VARIANTS = POLICY + [VARIANTS[4], VARIANTS[6]]
    seeds = [int(a) for a in args] or [0, 1, 2]
    df = generate()
    res = []
    for sd in seeds:
        t0 = time.time()
        piv, win = run(df, sd, VARIANTS)
        res.append(piv)
        print(f"seed {sd} done in {time.time()-t0:.0f}s", flush=True)
    avg = sum(res) / len(res)
    cols = [c for c in ["overall", "stable", "data_drift", "relational_drift", "both"] if c in avg.columns]
    avg = avg[cols]
    gain = (1 - avg.div(avg.loc["Static"], axis=1)) * 100
    print("\nMAE (mean over seeds)"); print(avg.round(2).to_string())
    print("\nGain vs Static, % (negative = worse than Static)"); print(gain.round(1).to_string())
    import os; os.makedirs("results", exist_ok=True); avg.to_csv("results/surrogate_mae.csv")
