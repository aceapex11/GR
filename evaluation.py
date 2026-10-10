"""Offline scoring.  Ground-truth columns are used HERE ONLY, after the stream has run."""
import numpy as np
import pandas as pd

from engine import regression_metrics

MAX_DELAY_WINDOWS = 2


def _truth(df, s, e):
    return (bool(df.loc[s:e - 1, "data_drift"].astype(bool).any()),
            bool(df.loc[s:e - 1, "concept_drift"].astype(bool).any()))


def true_episodes(win, df):
    flags = [any(_truth(df, int(r.start), int(r.end))) for r in win.itertuples()]
    eps, i = [], 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        eps.append((i, j))
        i = j + 1
    return eps


def _episode_type(win, df, a, b):
    dd = cc = False
    for i in range(a, b + 1):
        d, c = _truth(df, int(win.start.iloc[i]), int(win.end.iloc[i]))
        dd |= d
        cc |= c
    return "both" if dd and cc else "data" if dd else "relational" if cc else "none"


def detector_scorecard(win, df, max_delay=MAX_DELAY_WINDOWS):
    """Episode-level precision / recall / delay + drift-type accuracy."""
    eps = true_episodes(win, df)
    alarms = [i for i, a in enumerate(win.alarm) if a]
    matched, delays, type_rows = set(), [], []
    for n, (a, b) in enumerate(eps, 1):
        cand = [x for x in alarms if x not in matched and a <= x <= b + max_delay]
        hit = min(cand) if cand else None
        if hit is not None:
            matched.add(hit)
            delays.append(max(0, hit - a))
        seen = set()
        for i in range(a, min(len(win) - 1, b + max_delay) + 1):
            if win.data_active.iloc[i]:
                seen.add("data")
            if win.relational_active.iloc[i]:
                seen.add("relational")
        detected = "both" if seen == {"data", "relational"} else (next(iter(seen)) if len(seen) == 1 else "none")
        true_t = _episode_type(win, df, a, b)
        type_rows.append({"episode": n, "true_type": true_t, "detected_type": detected,
                          "type_correct": detected == true_t, "alarm_window": hit,
                          "episode_start_window": a, "episode_end_window": b,
                          "delay_windows": None if hit is None else max(0, hit - a)})
    tp = len(matched)
    fp = len([x for x in alarms if x not in matched])
    fn = len(eps) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    score = {"Precision": prec, "Recall": rec, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
             "True drift episodes": len(eps), "Alarms raised": len(alarms),
             "Mean detection delay (windows)": float(np.mean(delays)) if delays else float("nan")}
    return score, pd.DataFrame(type_rows)


def strategy_metrics(pred_df, df, target):
    """Overall + per-regime MAE/RMSE/R2 for every strategy (stream part only)."""
    y = df[target].to_numpy(float)
    regime = df["regime"].to_numpy() if "regime" in df else np.array(["all"] * len(df))
    rows = []
    for name in pred_df.columns:
        idx = pred_df.index.to_numpy()
        phases = [("overall", np.ones(len(idx), bool))] + [
            (r, regime[idx] == r) for r in pd.unique(regime[idx])]
        for phase, m in phases:
            if m.sum() < 2:
                continue
            rows.append({"Strategy": name, "Phase": phase, "Rows": int(m.sum()),
                         **{k.upper() if k != "r2" else "R2": v for k, v in
                            regression_metrics(y[idx][m], pred_df[name].to_numpy()[m]).items()}})
    return pd.DataFrame(rows)
