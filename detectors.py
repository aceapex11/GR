"""Model-independent drift detection (classic ML / statistics, no deep learning).

Why this is separate from the GRU
---------------------------------
Detection only looks at the incoming sensors and at the residual of a frozen,
simple "old rule" (poly-2 Ridge fitted on the normal history).  It never uses the
GRU's own error, so a weak or half-trained GRU cannot make the detector noisy,
and every GRU strategy gets exactly the same alarms (fair comparison).

Signals
-------
* Data drift      : per-sensor KS test on one 600-row window vs the normal history.
                    The alarm level of each sensor is calibrated from the normal
                    training period: max(KS_FLOOR, 1.2 x largest KS of a normal block).
                    PSI is a second, independent trigger.
                    Needs >= DATA_MIN_FEATURES sensors to move for DATA_CONFIRM windows.
* Relational drift: Page-Hinkley on the clipped, normalised residual of the frozen rule.
* Diagnosis       : none / data / relational / both.  One contiguous drift episode
                    produces ONE alarm; a "recovery" is flagged when diagnosis returns to none.

Ground-truth columns (regime, data_drift, concept_drift, ...) are never read here.
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

KS_FLOOR = 0.15
PSI_THRESHOLD = 0.25
PH_PARAMS = dict(min_instances=100, delta=0.5, threshold=50.0, alpha=0.9999)
DATA_MIN_FEATURES = 3
DATA_CONFIRM_WINDOWS = 2
RESID_CLIP = 4.0
RULE_BACK = 1.25          # mean residual below this => the old rule works again


class PageHinkley:
    """Page-Hinkley test for an upward change in the mean of a stream."""

    def __init__(self, min_instances=100, delta=0.05, threshold=60.0, alpha=0.9999):
        self.min_n, self.delta, self.threshold, self.alpha = min_instances, delta, threshold, alpha
        self.reset()

    def reset(self):
        self.n, self.mean, self.sum, self.stat = 0, 0.0, 0.0, 0.0

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.sum = max(0.0, self.alpha * self.sum + (x - self.mean - self.delta))
        self.stat = self.sum
        return self.n >= self.min_n and self.sum > self.threshold


def psi(ref, cur, bins=10):
    """Population Stability Index of `cur` against `ref` (bins from ref quantiles)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(q) < 2:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    p0 = np.histogram(ref, q)[0] / len(ref) + 1e-6
    p1 = np.histogram(cur, q)[0] / len(cur) + 1e-6
    return float(np.sum((p1 - p0) * np.log(p1 / p0)))


class DriftDetector:
    """Feed it one window at a time with `update(Xb, yb)` (scaled arrays)."""

    def __init__(self, X_ref, y_ref, monitored_idx, monitored_names, window):
        self.ref = X_ref
        self.mon_idx = list(monitored_idx)
        self.mon_names = list(monitored_names)

        # frozen "old rule": what the normal relationship looked like
        self.rule = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)).fit(X_ref, y_ref)
        res = y_ref - self.rule.predict(X_ref)
        self.rule_sd = float(np.sqrt(np.mean(res ** 2))) or 1.0
        self.normal_res = float(np.mean(np.abs(res) / self.rule_sd)) or 1.0

        # per-sensor KS alarm level from the normal blocks of the SAME window size
        blocks = range(0, len(X_ref) - window + 1, window)
        self.ks_thr = {}
        for name, j in zip(self.mon_names, self.mon_idx):
            worst = max((ks_2samp(X_ref[b:b + window, j], X_ref[:, j]).statistic for b in blocks),
                        default=0.0)
            self.ks_thr[name] = max(KS_FLOOR, 1.2 * worst)

        self.ph = PageHinkley(**PH_PARAMS)
        self.relational_on = False
        self.data_latched = False
        self.data_streak = 0
        self.episode_latched = False
        self.prev_diag = "none"

    def update(self, Xb, yb):
        ks = {n: float(ks_2samp(self.ref[:, j], Xb[:, j]).statistic)
              for n, j in zip(self.mon_names, self.mon_idx)}
        ps = {n: psi(self.ref[:, j], Xb[:, j]) for n, j in zip(self.mon_names, self.mon_idx)}

        # ---- data drift: several sensors must move, and stay moved ----
        moved = [n for n in self.mon_names if ks[n] > self.ks_thr[n] or ps[n] > PSI_THRESHOLD]
        self.data_streak = self.data_streak + 1 if len(moved) >= DATA_MIN_FEATURES else 0
        data_on = self.data_streak >= DATA_CONFIRM_WINDOWS
        data_alarm = data_on and not self.data_latched
        self.data_latched = data_on

        # ---- relational drift: Page-Hinkley on frozen-rule residual ----
        resid = np.minimum(np.abs(yb - self.rule.predict(Xb)) / self.rule_sd / self.normal_res,
                           RESID_CLIP)
        fired = False
        for v in resid:
            if self.ph.update(float(v)):
                fired = True
                self.ph.reset()
        rel_alarm = fired and not self.relational_on
        if fired:
            self.relational_on = True
        elif self.relational_on and resid.mean() < RULE_BACK:
            self.relational_on = False
            self.ph.reset()

        diagnosis = {(0, 0): "none", (1, 0): "data", (0, 1): "relational", (1, 1): "both"}[
            (int(data_on), int(self.relational_on))]

        # ---- one drift episode = one alarm ----
        alarm = (data_alarm or rel_alarm) and not self.episode_latched
        if alarm:
            self.episode_latched = True
        recovery = self.prev_diag != "none" and diagnosis == "none"
        if recovery:
            self.episode_latched = False
        self.prev_diag = diagnosis

        return {
            "diagnosis": diagnosis, "alarm": bool(alarm), "recovery": bool(recovery),
            "data_alarm": bool(data_alarm), "relational_alarm": bool(rel_alarm),
            "data_active": bool(data_on), "relational_active": bool(self.relational_on),
            "worst_ks": max(ks.values()), "worst_psi": max(ps.values()),
            "n_moved": len(moved), "features_moved": ",".join(moved),
            "ph_stat": float(self.ph.stat), "mean_resid": float(resid.mean()),
        }
