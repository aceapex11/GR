"""Streaming loop: predict -> detect -> diagnose -> adapt (test-then-train).

The engine is model-agnostic.  A *strategy* is any object with

    name : str
    predict(idx: np.ndarray) -> np.ndarray   # predictions on the raw target scale
    update(idx: np.ndarray) -> int           # learn from rows `idx` (labels now revealed)

and, optionally,

    observe(idx, y_true)    # called after every window's predictions (used by the blend)
    rollback()              # called when the detector reports recovery, if
                            # strategy.rollback_on_recovery is True

so the same loop runs the GRU strategies (gru_models.py) or any stand-in.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from detectors import DriftDetector

ADAPT_WINDOWS = 3          # a new alarm triggers adaptation on this many consecutive windows


class StreamData:
    """Scaled arrays shared by the detector and all strategies (scaler fitted on normal history only)."""

    def __init__(self, df, features, monitored, target, train_end, seq_len, input_clip=5.0):
        self.df, self.features, self.monitored, self.target = df, list(features), list(monitored), target
        self.train_end, self.seq_len = int(train_end), int(seq_len)
        self.input_clip = input_clip       # model inputs are limited to +-this many std (tames sensor spikes)
        Xraw = df[self.features].to_numpy(float)
        self.yraw = df[target].to_numpy(float)
        self.scaler = StandardScaler().fit(Xraw[:train_end])
        self.X = self.scaler.transform(Xraw)
        self.y_mu = float(self.yraw[:train_end].mean())
        self.y_sd = float(self.yraw[:train_end].std()) or 1.0
        self.y = (self.yraw - self.y_mu) / self.y_sd
        self.mon_idx = [self.features.index(f) for f in self.monitored]

    def seq(self, idx):
        """(n, seq_len, n_features) windows ending at each row in idx (features only, no future rows)."""
        idx = np.asarray(idx)
        rows = np.clip(idx[:, None] + np.arange(-self.seq_len + 1, 1)[None, :], 0, None)
        out = self.X[rows]
        # only the models see the clipped version; the detector keeps reading the raw scaled sensors (self.X)
        return np.clip(out, -self.input_clip, self.input_clip) if self.input_clip else out


def regression_metrics(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    denom = float(np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float("nan") if denom == 0 else 1.0 - float(np.sum(err ** 2)) / denom}


class StreamRunner:
    def __init__(self, data, strategies, window, adapt_windows=ADAPT_WINDOWS, until_recovery=True, max_adapt=12):
        self.d, self.strategies = data, list(strategies)
        self.window, self.adapt_windows = int(window), int(adapt_windows)
        # until_recovery: keep adapting for as long as the detector says drift is active (up to max_adapt
        # windows), instead of stopping after adapt_windows. Same schedule for every strategy.
        self.until_recovery, self.max_adapt = bool(until_recovery), int(max_adapt)
        self.detector = DriftDetector(data.X[:data.train_end], data.y[:data.train_end],
                                      data.mon_idx, data.monitored, self.window)

    def run(self, progress=None):
        d, n = self.d, len(self.d.df)
        starts = list(range(d.train_end, n, self.window))
        pred = {s.name: np.full(n, np.nan) for s in self.strategies}
        rows, active, ep_age = [], 0, 0

        for k, s0 in enumerate(starts):
            e0 = min(s0 + self.window, n)
            idx = np.arange(s0, e0)

            # 1) predict first -- labels are not used yet
            row = {"window": k, "start": s0, "end": e0}
            for s in self.strategies:
                p = np.asarray(s.predict(idx), float)
                pred[s.name][idx] = p
                row[f"mae[{s.name}]"] = float(np.mean(np.abs(p - d.yraw[idx])))

            # labels for this window are now known: let strategies that track their own error look at them
            for s in self.strategies:
                if hasattr(s, "observe"):
                    s.observe(idx, d.yraw[idx])

            # 2) detect (sensors + frozen-rule residual only)
            det = self.detector.update(d.X[idx], d.y[idx])
            row.update(det)

            diagnosis = det.get("diagnosis", "none")
            for s in self.strategies:
                s.diagnosis = diagnosis     # lets a strategy adapt harder to relational / both drift

            # 2b) recovery: the original behaviour is valid again, so roll back the strategies that opted in
            rolled = bool(det.get("recovery"))
            if rolled:
                for s in self.strategies:
                    if getattr(s, "rollback_on_recovery", False):
                        s.rollback()
            row["rolled_back"] = rolled

            # 3) adapt every strategy on the same schedule
            drifting = diagnosis != "none"
            ep_age = ep_age + 1 if drifting else 0
            if det["alarm"]:
                active = self.adapt_windows
            keep_going = self.until_recovery and drifting and ep_age <= self.max_adapt
            adapt_now = active > 0 or keep_going
            row["adapted"] = adapt_now
            if adapt_now:
                for s in self.strategies:
                    if rolled and getattr(s, "rollback_on_recovery", False):
                        continue          # just restored: don't immediately adapt it away again
                    s.update(idx)
            if active > 0:
                active -= 1

            row["timestamp"] = str(d.df["timestamp"].iloc[s0]) if "timestamp" in d.df else s0
            rows.append(row)
            if progress:
                progress(k + 1, len(starts))

        pred_df = pd.DataFrame(pred, index=np.arange(n)).iloc[d.train_end:]
        return pd.DataFrame(rows), pred_df
