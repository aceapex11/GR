"""
Synthetic industrial-machine stream for drift research.

60,000 hourly rows (~6.8 years), 8 sensors, 11 labelled regimes covering every drift family:
sudden / gradual, data / relational / both, a recurring concept, plus a harmless sensor glitch
that a good detector should IGNORE.  Same column names as the original notebooks, so it is a
drop-in replacement for drift_regression_dataset.csv.

Run:  python data_generator.py   ->  data/drift_regression_dataset_large.csv
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

FEATURES = ["temperature", "vibration", "pressure", "load",
            "operating_hours", "humidity", "rpm", "oil_viscosity"]
# operating_hours is a maintenance counter (saw-tooth), so it is excluded from distribution monitoring
MONITORED = [f for f in FEATURES if f != "operating_hours"]
TARGET = "health_deterioration"
N_ROWS = 60_000
TRAIN_END = 14_400            # rows before this = normal training history
WINDOW = 600                  # the stream arrives 600 hours at a time

NOM_MEAN = dict(temperature=70, vibration=2.0, pressure=30, load=60, operating_hours=2000,
                humidity=45, rpm=1500, oil_viscosity=40)
NOM_SD = dict(temperature=6, vibration=0.5, pressure=3, load=12, operating_hours=1155,
              humidity=10, rpm=120, oil_viscosity=4)

# target = bias + linear + 3 interaction/curvature terms (+ small overheating knee)
#          [bias, temp, vib, press, load, hours, hum, rpm, oil, temp*load, vib^2, vib*press]
W_BASE = np.array([38, 7, 10, 5, 8, 7, 1, 3, -4, 3.5, 3.0, 0.0])
W_R1 = np.array([38, 7, 20, 5, -3, 7, 1, 3, -4, 3.5, 3.0, 7.0])    # vibration x2, load effect flips, new vib*press term
W_R2 = np.array([46, 2, 10, 9, 8, 14, 1, -5, -4, -3.5, 3.0, 0.0])  # wear dominated by hours, temp weak, interaction flips

# sensor shifts in units of the normal std: (mean shift, scale multiplier)
SHIFT_A = dict(temperature=(2.0, 1.0), vibration=(1.2, 1.3), rpm=(1.0, 1.0))
SHIFT_B = dict(load=(1.8, 1.0), temperature=(0.0, 1.6), humidity=(1.5, 1.0), pressure=(-1.0, 1.0))
SHIFT_C = dict(temperature=(1.5, 1.0), load=(1.2, 1.0), oil_viscosity=(-1.5, 1.2), vibration=(0.8, 1.0))

# start, end, regime, mode, data shift, relation
EVENTS = [
    (18_000, 21_000, "data_drift",       "sudden",  SHIFT_A, W_BASE),
    (25_200, 29_400, "relational_drift", "sudden",  None,    W_R1),
    (32_400, 37_200, "both",             "gradual", SHIFT_C, W_R2),   # ramps in over the first 2,400 rows
    (39_600, 42_600, "relational_drift", "sudden",  None,    W_R1),   # RECURRING concept (same as event 2)
    (45_000, 51_000, "data_drift",       "sudden",  SHIFT_B, W_BASE),
    (52_800, 60_000, "both",             "sudden",  SHIFT_A, W_R2),
]
GRADUAL_RAMP = 2_400
GLITCH = (30_600, 30_612)     # 12 rows of sensor spikes in a stable period: NOT a drift


def _ar1(n, phi, rng):
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n)) * np.sqrt(1 - phi ** 2)


def generate(n=N_ROWS, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    day = np.sin(2 * np.pi * t / 24)
    year = np.sin(2 * np.pi * t / 8760)

    a_data = np.zeros(n)                 # how far the sensor distribution has moved (0..1)
    W = np.tile(W_BASE, (n, 1))          # relationship weights, row by row
    regime = np.array(["stable"] * n, dtype=object)
    data_flag, concept_flag = np.zeros(n, int), np.zeros(n, int)
    shifts = [None] * n
    drift_point = np.full(n, np.nan)
    for s, e, name, mode, shift, rel in EVENTS:
        ramp = np.clip((t[s:e] - s) / GRADUAL_RAMP, 0, 1) if mode == "gradual" else np.ones(e - s)
        regime[s:e] = name
        drift_point[s] = 1.0
        if shift is not None:
            a_data[s:e] = ramp; data_flag[s:e] = 1
            for i in range(s, e): shifts[i] = shift
        if name in ("relational_drift", "both"):
            W[s:e] = (1 - ramp)[:, None] * W_BASE + ramp[:, None] * rel
            concept_flag[s:e] = 1

    raw = {}
    z = {  # standardised "normal" signals before any drift
        "temperature": _ar1(n, 0.92, rng) * 0.8 + 0.4 * day + 0.2 * year,
        "vibration": _ar1(n, 0.90, rng) + 0.15 * day,
        "pressure": _ar1(n, 0.93, rng) - 0.1 * year,
        "load": _ar1(n, 0.88, rng) * 0.8 + 0.5 * day,
        "humidity": _ar1(n, 0.94, rng) * 0.9 - 0.2 * year,
        "rpm": _ar1(n, 0.85, rng) * 0.9 + 0.2 * rng.normal(size=n),
        "oil_viscosity": _ar1(n, 0.95, rng) * 0.9,
    }
    # slow shared factor: hot + heavy load tends to go with high vibration (realistic correlation)
    z["vibration"] += 0.35 * (z["temperature"] + z["load"]) / 2
    for f, v in z.items():
        mu_shift, sc = np.zeros(n), np.ones(n)
        for i in np.flatnonzero(a_data > 0):
            m, s_ = shifts[i].get(f, (0.0, 1.0)); mu_shift[i] = a_data[i] * m; sc[i] = 1 + a_data[i] * (s_ - 1)
        raw[f] = NOM_MEAN[f] + NOM_SD[f] * (v * sc + mu_shift)
    raw["operating_hours"] = (t % 4000).astype(float)               # maintenance reset every 4,000 h

    # sensor glitch (spikes only in readings; the machine itself is fine)
    g0, g1 = GLITCH
    raw["temperature"][g0:g1] += 6 * NOM_SD["temperature"]
    raw["vibration"][g0:g1] += 5 * NOM_SD["vibration"]

    u = {f: (raw[f] - NOM_MEAN[f]) / NOM_SD[f] for f in FEATURES}
    u_clean = {f: u[f].copy() for f in FEATURES}                      # the machine's true state ignores the glitch
    u_clean["temperature"][g0:g1] = ((raw["temperature"][g0:g1] - 6 * NOM_SD["temperature"]) - NOM_MEAN["temperature"]) / NOM_SD["temperature"]
    u_clean["vibration"][g0:g1] = ((raw["vibration"][g0:g1] - 5 * NOM_SD["vibration"]) - NOM_MEAN["vibration"]) / NOM_SD["vibration"]
    terms = np.column_stack([np.ones(n), u_clean["temperature"], u_clean["vibration"], u_clean["pressure"], u_clean["load"],
                             u_clean["operating_hours"], u_clean["humidity"], u_clean["rpm"], u_clean["oil_viscosity"],
                             u_clean["temperature"] * u_clean["load"], u_clean["vibration"] ** 2,
                             u_clean["vibration"] * u_clean["pressure"]])
    knee = 2.5 / (1 + np.exp(-3 * (u_clean["temperature"] - 1.2)))   # overheating knee (not polynomial)
    y = (W * terms).sum(1) + knee + rng.normal(0, 2.5, n)
    y = np.clip(y, 0, 100)

    df = pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n, freq="h")})
    for f in FEATURES: df[f] = raw[f]
    df[TARGET] = y
    df["regime"], df["data_drift"], df["concept_drift"], df["true_drift_point"] = regime, data_flag, concept_flag, drift_point
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    d = generate()
    d.to_csv("data/drift_regression_dataset_large.csv", index=False)
    print(d.shape); print(d.groupby("regime", sort=False)[TARGET].agg(["size", "mean", "std"]).round(2))
