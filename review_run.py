"""Multi-seed run that feeds the dashboard's "Model review" tab.

    python review_run.py --fast            # quick smoke test, 3 seeds
    python review_run.py                   # full training, 3 seeds
    python review_run.py --seeds 0 1 2 3 4

It does NOT touch the files written by experiment.py / run_experiment.py. Everything goes to
results/review/:

    metrics.csv    seed x strategy x phase    MAE / RMSE            (review_metrics.metrics_long)
    episodes.csv   seed x strategy x episode  error + recovery      (review_metrics.episode_table)
    cost.csv       seed x strategy            time / updates / samples / memory estimate
    meta.json      what was run (seeds, fast flag, window, stream length)

Seeds change the GRU initialisation, batch order and replay sampling. The synthetic stream and the
drift detector are the same for every seed, so the drift episodes are identical across seeds: seeds
measure training noise, not "different worlds". The dashboard says so.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data_generator import FEATURES, MONITORED, TARGET, TRAIN_END, WINDOW
from engine import StreamData, StreamRunner
from experiment import load_dataset
from gru_models import GRUConfig, build_strategies, train_base
from review_metrics import episode_table, metrics_long

ROOT = Path(__file__).resolve().parent


def cost_table(strategies, seed):
    by_name = {s.name: s for s in strategies}
    rows = []
    for s in strategies:
        members = getattr(s, "members", None)
        group = [by_name[m.name] for m in members] if members else [s]
        rows.append({
            "seed": seed, "Strategy": s.name,
            "updates": int(sum(g.n_updates for g in group)),
            "train_samples": int(sum(g.train_samples for g in group)),
            "train_seconds": float(sum(g.train_seconds for g in group)),
            "memory_mb": float(s.memory_mb()),
            "stored_rows": int(sum(g.stored_rows() for g in group)),
            "rollbacks": int(sum(g.n_rollbacks for g in group)),
            # a blend does no training of its own: it pays for every member it runs
            "includes_members": bool(members)})
    return pd.DataFrame(rows)


def run_seed(df, data, cfg, seed, t0):
    print(f"[seed {seed}] pre-training the GRU ({cfg.base_epochs} epochs)...")
    base = train_base(data, cfg, seed)
    strategies = build_strategies(data, cfg, seed, base=base)
    win, pred = StreamRunner(data, strategies, WINDOW, until_recovery=cfg.adapt_until_recovery,
                             max_adapt=cfg.max_adapt_windows).run(
        lambda i, n: print(f"[seed {seed}]   window {i}/{n} ({time.time() - t0:.0f}s)") if i % 20 == 0 or i == n else None)
    actual, regime = df[TARGET], df["regime"]
    return (metrics_long(pred, actual, regime, seed),
            episode_table(win, pred, actual, df["data_drift"], df["concept_drift"], seed),
            cost_table(strategies, seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "drift_regression_dataset_large.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "review"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = GRUConfig.fast() if args.fast else GRUConfig()
    df = load_dataset(args.data)
    data = StreamData(df, FEATURES, MONITORED, TARGET, TRAIN_END, cfg.seq_len)
    t0 = time.time()

    M, E, C = [], [], []
    for seed in args.seeds:
        m, e, c = run_seed(df, data, cfg, seed, t0)
        M.append(m), E.append(e), C.append(c)
        # written after every seed, so a stopped run still leaves usable (but honestly labelled) results
        pd.concat(M).to_csv(out / "metrics.csv", index=False)
        pd.concat(E).to_csv(out / "episodes.csv", index=False)
        pd.concat(C).to_csv(out / "cost.csv", index=False)
        (out / "meta.json").write_text(json.dumps(
            {"seeds": [int(s) for s in args.seeds[:len(M)]], "requested_seeds": [int(s) for s in args.seeds],
             "fast": bool(args.fast), "window": WINDOW, "train_end": TRAIN_END, "rows": len(df)}, indent=2))
    print(f"Done in {time.time() - t0:.0f}s. Wrote {out}")


if __name__ == "__main__":
    main()
