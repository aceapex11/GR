"""Why does Transfer lose to Static?  Ablation over the three suspects.

    python diagnose_transfer.py --fast      # smoke test
    python diagnose_transfer.py             # full training, seeds 0 1 2

Variants add one change at a time to the original Transfer (1-layer GRU, only the head re-trained, newest
window only, never rolled back): rollback on recovery, whole-episode training, unfreezing the GRU, a second
GRU layer, and (B, H) the old-row replay that the stand-in study found harmful.

For every variant the script trains its own base GRU, so Static is re-measured on the same base and the
gain is a like-for-like number.  Output: results/transfer_ablation.csv (mean over seeds).
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from data_generator import FEATURES, MONITORED, TARGET, TRAIN_END, WINDOW
from engine import StreamData, StreamRunner
from evaluation import strategy_metrics
from experiment import load_dataset
from gru_models import GRUConfig, Replay, Static, TransferFrozen, train_base

ROOT = Path(__file__).resolve().parent
VARIANTS = [
    ("A original: 1 layer, head only, newest window, no rollback",
     dict(n_layers=1, transfer_unfreeze="head", transfer_episode=False, transfer_replay=0, rollback=False)),
    ("B A + replay 300 old rows",
     dict(n_layers=1, transfer_unfreeze="head", transfer_episode=False, transfer_replay=300, rollback=False)),
    ("C A + rollback on recovery",
     dict(n_layers=1, transfer_unfreeze="head", transfer_episode=False, transfer_replay=0, rollback=True)),
    ("D C + whole-episode training",
     dict(n_layers=1, transfer_unfreeze="head", transfer_episode=True, transfer_replay=0, rollback=True)),
    ("E D + GRU unfrozen (1 layer)",
     dict(n_layers=1, transfer_unfreeze="top", transfer_episode=True, transfer_replay=0, rollback=True)),
    ("F 2 layers, top unfrozen + episode + rollback (new default)",
     dict(n_layers=2, transfer_unfreeze="top", transfer_episode=True, transfer_replay=0, rollback=True)),
    ("G F without rollback",
     dict(n_layers=2, transfer_unfreeze="top", transfer_episode=True, transfer_replay=0, rollback=False)),
    ("H F + replay 300 old rows",
     dict(n_layers=2, transfer_unfreeze="top", transfer_episode=True, transfer_replay=300, rollback=True)),
]


def run_variant(df, label, kw, base_cfg, seed):
    kw = dict(kw)
    rollback = kw.pop("rollback")
    cfg = replace(base_cfg, **kw)
    data = StreamData(df, FEATURES, MONITORED, TARGET, TRAIN_END, cfg.seq_len)
    base = train_base(data, cfg, seed)
    strategies = [Static("Static", base, data, cfg, seed),
                  TransferFrozen("Transfer", base, data, cfg, seed),
                  Replay("Replay 500", base, data, cfg, 500, seed)]     # reference: the strongest simple method
    strategies[1].rollback_on_recovery = rollback
    _, pred = StreamRunner(data, strategies, WINDOW, until_recovery=cfg.adapt_until_recovery,
                           max_adapt=cfg.max_adapt_windows).run()
    m = strategy_metrics(pred, df, TARGET).pivot_table(index="Strategy", columns="Phase", values="MAE")
    row = {"variant": label, "seed": seed}
    for ph in [c for c in ["overall", "stable", "data_drift", "relational_drift", "both"] if c in m.columns]:
        row[f"Static {ph}"] = m.loc["Static", ph]
        row[f"Transfer {ph}"] = m.loc["Transfer", ph]
        row[f"Replay500 {ph}"] = m.loc["Replay 500", ph]
        row[f"Transfer gain vs Static {ph} (%)"] = 100 * (1 - m.loc["Transfer", ph] / m.loc["Static", ph])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "drift_regression_dataset_large.csv"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    base_cfg = GRUConfig.fast() if args.fast else GRUConfig()
    df = load_dataset(args.data)
    rows = []
    for label, kw in VARIANTS:
        for seed in args.seeds:
            print(f"{label}  seed {seed}", flush=True)
            rows.append(run_variant(df, label, kw, base_cfg, seed))
    res = pd.DataFrame(rows)
    out = res.drop(columns="seed").groupby("variant", sort=False).mean().round(3)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "transfer_ablation.csv")
    show = [c for c in out.columns if c.startswith("Transfer gain")]
    print("\nTransfer gain vs Static (% MAE reduction; negative = worse than Static), mean over seeds")
    print(out[show].to_string())


if __name__ == "__main__":
    main()
