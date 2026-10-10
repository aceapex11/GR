"""GRU drift dashboard (Streamlit only - no other server needed).

    streamlit run app.py

Reads results/*.csv written by run_experiment.py.  The sidebar button runs the experiment.
Ground-truth drift labels are only used to shade the charts and to score the detector.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from detectors import DATA_MIN_FEATURES, PH_PARAMS, RULE_BACK

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REGION_COLOR = {"data_drift": "rgba(245,166,35,0.18)", "relational_drift": "rgba(80,120,255,0.16)",
                "both": "rgba(230,60,60,0.16)"}
LINE_COLORS = ["#9aa5b1", "#17a2b8", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

st.set_page_config(page_title="GRU Drift Intelligence", page_icon="📈", layout="wide")
st.title("GRU Drift Intelligence")
st.caption("Predict → Detect → Diagnose → Adapt → Evaluate  |  PyTorch GRU  |  detector: KS + PSI + Page-Hinkley")


def read(name):
    p = RESULTS / name
    if not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


with st.sidebar:
    st.header("Experiment")
    fast = st.checkbox("Fast mode (few epochs)", value=True)
    if st.button("Run GRU experiment", type="primary", use_container_width=True):
        cmd = [sys.executable, str(ROOT / "run_experiment.py")] + (["--fast"] if fast else [])
        with st.spinner("Running... (this can take several minutes)"):
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            st.success("Done")
            st.code(r.stdout[-3000:])
        else:
            st.error("Experiment failed")
            st.code((r.stderr + r.stdout)[-6000:])
        st.rerun()

log, pred, truth = read("stream_log.csv"), read("predictions.csv"), read("ground_truth_offline_only.csv")
score, metrics, types = read("detector_scorecard.csv"), read("strategy_metrics.csv"), read("drift_type_evaluation.csv")
if log is None or pred is None:
    st.info("No results yet. Click **Run GRU experiment** in the sidebar (or run `python run_experiment.py`).")
    st.stop()

log["timestamp"] = pd.to_datetime(log["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])
strategy_cols = [c for c in pred.columns if c not in ("row", "timestamp", "actual")]
alarms = log[log["alarm"]]


def drift_blocks():
    """Contiguous ground-truth drift regions as (start_ts, end_ts, regime)."""
    if truth is None:
        return []
    t = truth.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t[t["regime"].isin(REGION_COLOR)]
    if t.empty:
        return []
    grp = (t["regime"] != t["regime"].shift()) | (t["row"].diff() != 1)
    return [(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], g["regime"].iloc[0])
            for _, g in t.groupby(grp.cumsum())]


def decorate(fig, row=None, show_alarms=True):
    kw = {} if row is None else {"row": row, "col": 1}
    for a, b, regime in drift_blocks():
        fig.add_vrect(x0=a, x1=b, fillcolor=REGION_COLOR[regime], line_width=0, layer="below", **kw)
    if show_alarms:
        for ts in alarms["timestamp"]:
            fig.add_shape(type="line", x0=ts, x1=ts, y0=0, y1=1, yref="paper",
                          line=dict(color="#ff7f0e", dash="dash", width=1.2))


tab_over, tab_pred, tab_det, tab_strat, tab_ev = st.tabs(
    ["Overview", "Predictions", "Detector", "Strategies", "Events"])

# ------------------------------------------------------------------ Overview
with tab_over:
    if score is not None:
        s = score.iloc[0]
        c = st.columns(5)
        c[0].metric("Drift episodes", int(s["True drift episodes"]))
        c[1].metric("Alarms raised", int(s["Alarms raised"]))
        c[2].metric("Precision", f"{s['Precision']:.2f}")
        c[3].metric("Recall", f"{s['Recall']:.2f}")
        c[4].metric("Mean delay (windows)", "n/a" if pd.isna(s["Mean detection delay (windows)"])
                    else f"{s['Mean detection delay (windows)']:.2f}")
        st.caption("Episode-level scoring: one alarm per drift episode counts as a hit if it fires "
                   "inside the episode or ≤2 windows after it ends.")
    if metrics is not None:
        st.subheader("Overall error by strategy")
        ov = metrics[metrics["Phase"] == "overall"].sort_values("MAE")
        st.dataframe(ov, hide_index=True, use_container_width=True)
    if types is not None:
        st.subheader("Drift-type diagnosis")
        st.dataframe(types, hide_index=True, use_container_width=True)

# --------------------------------------------------------------- Predictions
with tab_pred:
    c1, c2 = st.columns([1, 2])
    res = c1.radio("Resolution", ["Daily mean", "Hourly"], horizontal=True)
    chosen = c2.multiselect("Strategies", strategy_cols, default=strategy_cols)
    d = pred.set_index("timestamp")[["actual"] + chosen]
    if res == "Daily mean":
        d = d.resample("D").mean()
    else:
        d = d.iloc[:: max(1, len(d) // 6000)]
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["actual"], name="Actual", line=dict(color="#1f4e9c", width=2))
    for i, col in enumerate(chosen):
        fig.add_scatter(x=d.index, y=d[col], name=col, line=dict(color=LINE_COLORS[(i + 1) % 6], width=1.2))
    decorate(fig)
    fig.update_layout(height=520, yaxis_title="Health deterioration", xaxis_title="Time",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shaded: ground-truth drift (amber = data, blue = relational, red = both).  "
               "Orange dashed lines: detector alarms (one per drift episode).")

# ------------------------------------------------------------------ Detector
with tab_det:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Sensors moved (KS / PSI) - alarm needs ≥%d for 2 windows" % DATA_MIN_FEATURES,
                                        "Page-Hinkley statistic (frozen-rule residual)",
                                        "Mean normalised residual of the frozen rule"))
    fig.add_scatter(x=log["timestamp"], y=log["n_moved"], mode="lines+markers", name="sensors moved", row=1, col=1)
    fig.add_hline(y=DATA_MIN_FEATURES, line_dash="dot", row=1, col=1)
    fig.add_scatter(x=log["timestamp"], y=log["ph_stat"], mode="lines", name="PH statistic", row=2, col=1)
    fig.add_hline(y=PH_PARAMS["threshold"], line_dash="dot", row=2, col=1)
    fig.add_scatter(x=log["timestamp"], y=log["mean_resid"], mode="lines", name="mean residual", row=3, col=1)
    fig.add_hline(y=RULE_BACK, line_dash="dot", row=3, col=1)
    for r in (1, 2, 3):
        decorate(fig, row=r)
    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(log[["window", "timestamp", "diagnosis", "alarm", "recovery", "n_moved",
                      "features_moved", "worst_ks", "worst_psi", "ph_stat", "mean_resid"]],
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Strategies
with tab_strat:
    if metrics is not None:
        phase = st.selectbox("Phase", metrics["Phase"].unique().tolist())
        view = metrics[metrics["Phase"] == phase].sort_values("MAE")
        st.plotly_chart(go.Figure(go.Bar(x=view["Strategy"], y=view["MAE"])).update_layout(
            title=f"MAE by strategy - {phase}", yaxis_title="MAE"), use_container_width=True)
        st.dataframe(view, hide_index=True, use_container_width=True)
        pivot = metrics.pivot(index="Strategy", columns="Phase", values="MAE").round(2)
        st.subheader("MAE by regime")
        st.dataframe(pivot, use_container_width=True)
    win_cols = [c for c in log.columns if c.startswith("mae[")]
    if win_cols:
        fig = go.Figure()
        for i, c in enumerate(win_cols):
            fig.add_scatter(x=log["timestamp"], y=log[c], name=c[4:-1], line=dict(color=LINE_COLORS[i % 6]))
        decorate(fig)
        fig.update_layout(title="Window MAE over time", height=420, yaxis_title="MAE")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------- Events
with tab_ev:
    ev = read("adaptation_events.csv")
    if ev is None:
        st.info("No alarm / adaptation events were recorded.")
    else:
        st.dataframe(ev[["window", "start", "end", "timestamp", "diagnosis", "alarm", "recovery", "adapted",
                         "features_moved", "ph_stat"]], hide_index=True, use_container_width=True)
