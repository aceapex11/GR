"""End-to-end test of detection + streaming loop using sklearn stand-in strategies
(no PyTorch needed).  Run from the project root:  python tests/test_engine.py
The detector never touches the model, so this checks exactly the part that was failing."""
import os
import sys

import numpy as np
from sklearn.linear_model import SGDRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import FEATURES, MONITORED, TARGET, TRAIN_END, WINDOW, generate   # noqa: E402
from engine import StreamData, StreamRunner                                           # noqa: E402
from evaluation import detector_scorecard, strategy_metrics                           # noqa: E402


class StandIn:
    """Flat-feature SGD model with the same predict/update interface as the GRU strategies."""

    def __init__(self, name, data, adapt):
        self.name, self.d, self.adapt = name, data, adapt
        self.m = SGDRegressor(alpha=1e-4, learning_rate="constant", eta0=0.01, tol=None, random_state=0)
        tr = np.arange(data.train_end)
        for _ in range(30):
            self.m.partial_fit(data.X[tr], data.y[tr])

    def predict(self, idx):
        return self.m.predict(self.d.X[idx]) * self.d.y_sd + self.d.y_mu

    def update(self, idx):
        if self.adapt:
            for _ in range(5):
                self.m.partial_fit(self.d.X[idx], self.d.y[idx])
        return len(idx)


def main():
    df = generate()
    data = StreamData(df, FEATURES, MONITORED, TARGET, TRAIN_END, seq_len=12)
    runner = StreamRunner(data, [StandIn("Static", data, False), StandIn("Adaptive", data, True)], WINDOW)
    win, pred = runner.run()
    score, types = detector_scorecard(win, df)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in score.items()})
    print(types.to_string(index=False))
    print(strategy_metrics(pred, df, TARGET).query("Phase == 'overall'").to_string(index=False))

    assert score["True drift episodes"] == 6, score
    assert score["Recall"] == 1.0, f"missed a drift: {score}"
    assert score["FP"] == 0, f"false alarms: {score}"
    assert not win.loc[(win.start >= 30000) & (win.end <= 31200), "alarm"].any(), "alarm on the sensor glitch"
    assert df.index.equals(pred.index.union(df.index[:TRAIN_END]))
    print("\nOK: all 6 drift episodes detected, no false alarms, glitch ignored.")


if __name__ == "__main__":
    main()
