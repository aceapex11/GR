"""Run the whole GRU drift experiment and write results/*.csv

    python run_experiment.py            # full training
    python run_experiment.py --fast     # quick smoke test
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_generator import FEATURES, MONITORED, TARGET, TRAIN_END, WINDOW, generate
from engine import StreamData, StreamRunner
from evaluation import detector_scorecard, strategy_metrics
from gru_models import GRUConfig, build_strategies, train_base

ROOT = Path(__file__).resolve().parent


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        print(f"{path} not found - generating the 60,000-row synthetic stream (seed 7).")
        path.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "drift_regression_dataset_large.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--fast", action="store_true", help="few epochs, for a quick check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = GRUConfig.fast() if args.fast else GRUConfig()
    t0 = time.time()

    df = load_dataset(args.data)
    data = StreamData(df, FEATURES, MONITORED, TARGET, TRAIN_END, cfg.seq_len)

    print(f"Pre-training the GRU on the first {TRAIN_END:,} normal rows ({cfg.base_epochs} epochs)...")
    base = train_base(data, cfg, args.seed)
    strategies = build_strategies(data, cfg, args.seed, base=base)

    def progress(i, n):
        if i % 10 == 0 or i == n:
            print(f"  window {i}/{n}  ({time.time() - t0:.0f}s)")

    print(f"Streaming {len(df) - TRAIN_END:,} rows in windows of {WINDOW}...")
    win, pred = StreamRunner(data, strategies, WINDOW).run(progress)

    # ---------- offline scoring (the only place ground truth is read) ----------
    score, types = detector_scorecard(win, df)
    metrics = strategy_metrics(pred, df, TARGET)

    win.to_csv(out / "stream_log.csv", index=False)
    pd.concat([df[["timestamp"]].loc[pred.index], df[[TARGET]].loc[pred.index].rename(
        columns={TARGET: "actual"}), pred], axis=1).rename_axis("row").to_csv(out / "predictions.csv")
    df[["timestamp", "regime", "data_drift", "concept_drift"]].to_csv(
        out / "ground_truth_offline_only.csv", index_label="row")
    metrics.to_csv(out / "strategy_metrics.csv", index=False)
    pd.DataFrame([score]).to_csv(out / "detector_scorecard.csv", index=False)
    types.to_csv(out / "drift_type_evaluation.csv", index=False)
    win[win.alarm | win.recovery | win.adapted].to_csv(out / "adaptation_events.csv", index=False)
    (out / "config.json").write_text(json.dumps(
        {"fast": args.fast, "seed": args.seed, "window": WINDOW, "train_end": TRAIN_END,
         "gru": asdict(cfg)}, indent=2))

    print("\nDetector scorecard:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in score.items()})
    print(metrics.query("Phase == 'overall'").sort_values("MAE").to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Files written to {out}")


if __name__ == "__main__":
    main()
