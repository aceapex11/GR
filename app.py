"""Drift monitor - Streamlit dashboard for the GRU drift experiment.

    streamlit run app.py

Reads results/*.csv written by experiment.py (the sidebar button runs it).
Ground-truth drift labels are used only to shade charts and to score the detector.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from detectors import DATA_MIN_FEATURES, PH_PARAMS, RULE_BACK

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# ------------------------------------------------------------------ design tokens
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD_TEXT, CRITICAL = "#006300", "#d03b3b"
BAND = "rgba(11,11,11,0.045)"
FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif'
# one fixed colour per strategy (categorical slots 1-3 and 7 of the validated palette); Static is the grey baseline
STRATEGY_COLORS = [("static", MUTED), ("transfer", "#2a78d6"), ("full", "#eb6834"),
                   ("replay 100", "#1baf7a"), ("replay 500", "#6250d6")]
REGIME_LABEL = {"stable": "Stable", "data_drift": "Data", "relational_drift": "Relational", "both": "Both"}


def color_for(name):
    low = name.lower()
    for key, col in STRATEGY_COLORS:
        if key in low:
            return col
    return INK2


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ------------------------------------------------------------------ data
def read(name):
    p = RESULTS / name
    if not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


log, pred, truth = read("stream_log.csv"), read("predictions.csv"), read("ground_truth_offline_only.csv")
has_results = log is not None and pred is not None

st.set_page_config(page_title="Drift monitor", page_icon=None, layout="wide",
                   initial_sidebar_state="collapsed" if has_results else "expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [class*="st-"], button, input, textarea { font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp { background: #fcfcfb; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: #0b0b0b; }
.app-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap;
            padding-bottom: 1.1rem; border-bottom: 1px solid #e1e0d9; }
.app-title { font-size: 1.65rem; font-weight: 600; line-height: 1.15; color: #0b0b0b; }
.app-sub { font-size: .95rem; color: #52514e; margin-top: .3rem; }
.app-meta { font-size: .85rem; color: #898781; text-align: right; }
.verdict { display: flex; align-items: center; gap: .6rem; font-size: 1.05rem; font-weight: 500; margin: 1.4rem 0 .2rem; }
.verdict svg { flex: none; }
.kpi-strip { display: flex; flex-wrap: wrap; border-top: 1px solid #e1e0d9; border-bottom: 1px solid #e1e0d9; margin: 1rem 0 1.6rem; }
.kpi { flex: 1 1 170px; padding: 14px 20px; border-left: 1px solid #e1e0d9; }
.kpi:first-child { border-left: none; padding-left: 0; }
.kpi-label { font-size: .8rem; color: #52514e; }
.kpi-value { font-size: 1.75rem; font-weight: 600; line-height: 1.25; font-variant-numeric: tabular-nums; color: #0b0b0b; }
.kpi-note { font-size: .78rem; color: #898781; }
.sec { font-size: 1.02rem; font-weight: 600; margin: 1.8rem 0 .15rem; color: #0b0b0b; }
.sec-note { font-size: .85rem; color: #52514e; margin-bottom: .6rem; max-width: 70ch; }
.takeaway { font-size: .95rem; color: #0b0b0b; margin: .25rem 0; max-width: 75ch; }
button[data-baseweb="tab"] { font-size: .92rem; font-weight: 500; padding-left: .2rem; padding-right: .2rem; margin-right: 1.4rem; }
[data-testid="stSidebar"] { border-right: 1px solid #e1e0d9; }
@media (max-width: 640px) { .kpi { flex-basis: 45%; border-left: none; padding-left: 0; } .app-meta { text-align: left; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def wide(fn, *args, **kwargs):
    """Call a Streamlit widget at full width on both old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show_fig(fig):
    wide(st.plotly_chart, fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def show_df(df, **kwargs):
    wide(st.dataframe, df, hide_index=True, **kwargs)


def section(title, note=None):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sec-note">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def kpi_strip(items):
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div>'
                    f'<div class="kpi-note">{n}</div></div>' for l, v, n in items)
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=legend,
        font=dict(family=FONT, size=12, color=INK2), margin=dict(l=56, r=24, t=40, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2))
    return fig


def ts(x):
    return pd.Timestamp(x).isoformat()


# ------------------------------------------------------------------ sidebar: run controls
cfg = {}
if (RESULTS / "config.json").is_file():
    try:
        cfg = json.loads((RESULTS / "config.json").read_text())
    except Exception:
        cfg = {}

with st.sidebar:
    st.markdown("**Run experiment**")
    st.caption("Trains the GRU, streams the data and rewrites the results folder.")
    full_training = st.checkbox("Full training (slower)", value=False)
    if wide(st.button, "Run GRU experiment", type="primary"):
        cmd = [sys.executable, str(ROOT / "experiment.py")] + (["--full"] if full_training else [])
        with st.spinner("Running. This can take several minutes."):
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            st.success("Finished.")
        else:
            st.error("The experiment failed. Details below.")
            st.code((r.stderr + r.stdout)[-6000:])
        st.rerun()

meta = ""
if cfg:
    meta = f'{"Fast" if cfg.get("fast") else "Full"} run, seed {cfg.get("seed", 0)}'
st.markdown(
    f'<div class="app-head"><div><div class="app-title">Drift monitor</div>'
    f'<div class="app-sub">GRU regression on a 60,000-row industrial sensor stream, with drift detection and model adaptation</div></div>'
    f'<div class="app-meta">{meta}</div></div>', unsafe_allow_html=True)

if not has_results:
    section("No results yet", "Open the sidebar and select <b>Run GRU experiment</b>, or run <code>python experiment.py</code> "
                              "in a terminal. The fast setting takes a few minutes.")
    st.stop()

# ------------------------------------------------------------------ load + derive
score, metrics, types = read("detector_scorecard.csv"), read("strategy_metrics.csv"), read("drift_type_evaluation.csv")
summary, recovery, events = read("performance_summary.csv"), read("episode_recovery.csv"), read("adaptation_events.csv")

log["timestamp"] = pd.to_datetime(log["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])
strategy_cols = [c for c in pred.columns if c not in ("row", "timestamp", "actual")]
alarms = log[log["alarm"]]
row_ts = None
if truth is not None:
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    row_ts = truth.set_index("row")["timestamp"]


def drift_blocks():
    """Contiguous ground-truth drift regions: (start, end, regime)."""
    if truth is None:
        return []
    t = truth[truth["regime"] != "stable"]
    if t.empty:
        return []
    new = (t["regime"] != t["regime"].shift()) | (t["row"].diff() != 1)
    return [(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], g["regime"].iloc[0]) for _, g in t.groupby(new.cumsum())]


BLOCKS = drift_blocks()


def bands(fig, row=None, labels=True):
    kw = {} if row is None else {"row": row, "col": 1}
    for a, b, regime in BLOCKS:
        fig.add_vrect(x0=ts(a), x1=ts(b), fillcolor=BAND, line_width=0, layer="below", **kw)
        if labels:
            mid = a + (b - a) / 2
            fig.add_annotation(x=ts(mid), y=1, yref="paper", yanchor="top", text=REGIME_LABEL.get(regime, regime),
                               showarrow=False, font=dict(size=10.5, color=MUTED))


def alarm_lines(fig, row=None):
    kw = {} if row is None else {"row": row, "col": 1}
    for t in alarms["timestamp"]:
        fig.add_vline(x=ts(t), line_width=1, line_color=INK, opacity=0.55, **kw)


def window_mae_cols():
    return [c for c in log.columns if c.startswith("mae[")]


# ------------------------------------------------------------------ tabs
tab_over, tab_model, tab_det, tab_pred, tab_ev = st.tabs(
    ["Overview", "Model performance", "Detector", "Predictions", "Events"])

# ================================================================== Overview
with tab_over:
    if score is not None:
        s = score.iloc[0]
        caught, total, fp = int(s["TP"]), int(s["True drift episodes"]), int(s["FP"])
        ok = caught == total and fp == 0
        icon = (f'<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="{GOOD_TEXT}" stroke-width="2.2" '
                f'stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="8.5"/><path d="M6 10.4l2.6 2.6L14 7.6"/></svg>'
                if ok else
                f'<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="{CRITICAL}" stroke-width="2.2" '
                f'stroke-linecap="round"><circle cx="10" cy="10" r="8.5"/><path d="M10 5.5v5.5M10 14.2v.1"/></svg>')
        verdict = ("Detector caught every drift episode with no false alarms." if ok else
                   f"Detector caught {caught} of {total} drift episodes and raised {fp} false alarm{'s' if fp != 1 else ''}.")
        st.markdown(f'<div class="verdict">{icon}<span>{verdict}</span></div>', unsafe_allow_html=True)
        delay = s["Mean detection delay (windows)"]
        win_hours = int(cfg.get("window", 600))
        best = summary.iloc[0] if summary is not None and len(summary) else None
        kpi_strip([
            ("Drift episodes caught", f"{caught} / {total}", "one alarm per episode"),
            ("False alarms", f"{fp}", "alarms outside any episode"),
            ("Mean detection delay", "n/a" if pd.isna(delay) else f"{delay:.1f} windows",
             "" if pd.isna(delay) else f"about {delay * win_hours:.0f} hours"),
            ("Best model, overall error", best["Strategy"] if best is not None else "n/a",
             f'MAE {best["MAE"]:.2f}' if best is not None else ""),
        ])

    section("Drift timeline", "What really drifted, when the detector reacted, and when the models were updated.")
    fig = go.Figure()
    rows_order = ["Models adapting", "Detector alarm", "True drift"]

    def segments(name, items, color, width):
        xs, ys, mids, mid_txt = [], [], [], []
        for a, b, label in items:
            xs += [ts(a), ts(b), None]
            ys += [name, name, None]
            mids.append(ts(a + (b - a) / 2))
            mid_txt.append(label)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=width, color=color), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=mids, y=[name] * len(mids), mode="markers", marker=dict(size=16, opacity=0),
                                 hovertext=mid_txt, hoverinfo="text", showlegend=False))
        return mids

    mids = segments("True drift", [(a, b, f"{REGIME_LABEL.get(r, r)} drift") for a, b, r in BLOCKS], "rgba(82,81,78,0.35)", 16)
    fig.add_trace(go.Scatter(x=mids, y=["True drift"] * len(mids), mode="text",
                             text=[REGIME_LABEL.get(r, r) for _, _, r in BLOCKS], textposition="top center",
                             textfont=dict(size=11, color=INK2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[ts(t) for t in alarms["timestamp"]], y=["Detector alarm"] * len(alarms), mode="markers",
                             marker=dict(symbol="triangle-up", size=12, color=INK, line=dict(width=2, color=SURFACE)),
                             hovertext=[f"Window {w}: {d} drift" for w, d in zip(alarms["window"], alarms["diagnosis"])],
                             hoverinfo="text", showlegend=False))
    ad = log[log["adapted"]]
    if len(ad) and row_ts is not None:
        grp = (ad["window"].diff() != 1).cumsum()
        items = []
        for _, g in ad.groupby(grp):
            a = g["timestamp"].iloc[0]
            b = row_ts.get(int(g["end"].iloc[-1]) - 1, g["timestamp"].iloc[-1])
            items.append((a, b, f"Adapting for {len(g)} windows"))
        segments("Models adapting", items, "#2a78d6", 12)
    style(fig, height=230, legend=False)
    fig.update_layout(margin=dict(l=120, r=24, t=16, b=40))
    fig.update_yaxes(categoryorder="array", categoryarray=rows_order, showgrid=False, tickfont=dict(color=INK2, size=12))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    show_fig(fig)

    if summary is not None and len(summary):
        left, right = st.columns([1.15, 1])
        with left:
            section("Overall error by model", "Mean absolute error on every streamed row. Lower is better.")
            order = summary.sort_values("MAE", ascending=False)
            fig = go.Figure(go.Bar(y=order["Strategy"], x=order["MAE"], orientation="h",
                                   marker_color=[color_for(n) for n in order["Strategy"]],
                                   text=[f"{v:.2f}" for v in order["MAE"]], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            style(fig, height=250, legend=False)
            fig.update_layout(bargap=0.55, margin=dict(l=160, r=40, t=8, b=36))
            fig.update_xaxes(showgrid=True, gridcolor=GRID, title="MAE")
            fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
            show_fig(fig)
        with right:
            section("What stands out")
            best = summary.iloc[0]
            st.markdown(f'<div class="takeaway">{best["Strategy"]} has the lowest overall error (MAE {best["MAE"]:.2f}).</div>',
                        unsafe_allow_html=True)
            if "Gain vs Static, drift (%)" in summary:
                top = summary.sort_values("MAE drift").iloc[0]
                if top["Strategy"].lower().startswith("static"):
                    msg = "No adaptation strategy beat the static model while drift was active."
                else:
                    msg = (f'While drift was active, {top["Strategy"]} cut the error by '
                           f'{top["Gain vs Static, drift (%)"]:.0f}% compared with the static model.')
                st.markdown(f'<div class="takeaway">{msg}</div>', unsafe_allow_html=True)
                worst = summary.assign(ratio=summary["MAE stable"] / summary["MAE normal"]).sort_values("ratio").iloc[-1]
                if worst["ratio"] > 1.15 and not worst["Strategy"].lower().startswith("static"):
                    st.markdown(f'<div class="takeaway">After drift ended, {worst["Strategy"]} stayed '
                                f'{worst["ratio"]:.1f} times above its pre-drift error, which suggests it forgot part of the '
                                f'original behaviour.</div>', unsafe_allow_html=True)
            st.caption("Details and per-episode results are on the Model performance tab.")

# ================================================================== Model performance
with tab_model:
    if summary is None:
        st.info("This tab needs performance_summary.csv. Run the experiment again to create it.")
    else:
        chosen = st.selectbox("Model", summary["Strategy"].tolist(), index=0)
        r0 = summary[summary["Strategy"] == chosen].iloc[0]
        gain = r0.get("Gain vs Static, drift (%)", np.nan)
        kpi_strip([
            ("MAE", f'{r0["MAE"]:.2f}', "average absolute error"),
            ("RMSE", f'{r0["RMSE"]:.2f}', "penalises large misses"),
            ("R squared", f'{r0["R2"]:.3f}', "share of variance explained"),
            ("Error during drift", f'{r0["Drift error (x normal)"]:.2f}x', "relative to pre-drift error"),
            ("Gain vs Static in drift", "n/a" if pd.isna(gain) else f"{gain:+.0f}%", "negative means worse"),
        ])

        section("Leaderboard", "Accuracy, how each model behaves around drift, and what its updates cost.")
        cols = ["Rank", "Strategy", "MAE", "RMSE", "R2", "MAE normal", "MAE stable", "MAE drift", "Gain vs Static, drift (%)",
                "Updates", "Training samples", "Training time (s)"]
        show_df(summary[[c for c in cols if c in summary]].round(2))

        left, right = st.columns(2)
        with left:
            section("Error by regime", "Mean absolute error in each period of the stream.")
            if metrics is not None:
                piv = (metrics[metrics["Phase"] != "overall"].pivot(index="Strategy", columns="Phase", values="MAE")
                       .reindex(summary["Strategy"]))
                piv = piv[[c for c in ["stable", "data_drift", "relational_drift", "both"] if c in piv.columns]]
                fig = go.Figure(go.Heatmap(
                    z=piv.values, x=[REGIME_LABEL.get(c, c) for c in piv.columns], y=piv.index, zmin=0,
                    colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#104281"]], showscale=False,
                    text=np.round(piv.values, 1), texttemplate="%{text}", xgap=2, ygap=2,
                    textfont=dict(size=12), hovertemplate="%{y}, %{x}: %{z:.2f}<extra></extra>"))
                style(fig, height=290, legend=False)
                fig.update_layout(margin=dict(l=150, r=12, t=8, b=36))
                fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK2))
                fig.update_xaxes(side="top", tickfont=dict(color=INK2))
                show_fig(fig)
        with right:
            section("Peak error per drift episode", "Highest window error reached during each episode.")
            if recovery is not None and len(recovery):
                fig = go.Figure()
                for name in summary["Strategy"]:
                    d = recovery[recovery["Strategy"] == name]
                    fig.add_trace(go.Bar(x=[f'{int(e)} {REGIME_LABEL.get(t.split("_")[0].title().lower(), t)}'
                                            if False else f"Ep {int(e)}" for e in d["episode"]],
                                         y=d["peak_mae"], name=name, marker_color=color_for(name),
                                         hovertemplate="%{x}: %{y:.2f}<extra>" + name + "</extra>"))
                style(fig, height=290)
                fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06, margin=dict(l=56, r=12, t=40, b=36))
                fig.update_yaxes(title="Peak window MAE")
                show_fig(fig)

        section("Error over time", "Window error for every model. Shaded bands are the true drift periods; vertical lines are detector alarms.")
        fig = go.Figure()
        for c in window_mae_cols():
            name = c[4:-1]
            on = name == chosen
            fig.add_trace(go.Scatter(x=log["timestamp"], y=log[c], name=name, mode="lines",
                                     line=dict(width=2.4 if on else 1.4, color=color_for(name)),
                                     opacity=1 if on else 0.65, hovertemplate="%{y:.2f}<extra>" + name + "</extra>"))
        bands(fig)
        alarm_lines(fig)
        style(fig, height=380)
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title="Window MAE", rangemode="tozero")
        show_fig(fig)

        left, right = st.columns(2)
        merged = pred[["row", "actual", chosen]].copy()
        if truth is not None:
            merged = merged.merge(truth[["row", "regime"]], on="row", how="left")
        else:
            merged["regime"] = "stable"
        merged["resid"] = merged[chosen] - merged["actual"]
        with left:
            section(f"Predicted against actual: {chosen}", "Points on the diagonal are perfect predictions.")
            sample = merged.sample(min(3500, len(merged)), random_state=0)
            calm, hot = sample[sample["regime"] == "stable"], sample[sample["regime"] != "stable"]
            lo, hi = float(merged["actual"].min()), float(merged["actual"].max())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=calm["actual"], y=calm[chosen], mode="markers", name="Stable periods",
                                     marker=dict(size=5, color=AXIS, opacity=0.6, line=dict(width=0))))
            fig.add_trace(go.Scatter(x=hot["actual"], y=hot[chosen], mode="markers", name="During drift",
                                     marker=dict(size=5, color=color_for(chosen), opacity=0.55, line=dict(width=0))))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                     line=dict(width=1, color=INK2)))
            style(fig, height=360)
            fig.update_xaxes(title="Actual", showgrid=True, gridcolor=GRID)
            fig.update_yaxes(title="Predicted")
            show_fig(fig)
        with right:
            section("Prediction error by regime", "Predicted minus actual. A box centred on zero means no bias.")
            fig = go.Figure()
            for reg in ["stable", "data_drift", "relational_drift", "both"]:
                d = merged[merged["regime"] == reg]["resid"]
                if len(d):
                    fig.add_trace(go.Box(y=d, name=REGIME_LABEL[reg], boxpoints=False, line=dict(width=1.5, color=color_for(chosen)),
                                         fillcolor=rgba(color_for(chosen).replace("#898781", "#52514e"), 0.18)))
            fig.add_hline(y=0, line_width=1, line_color=INK2)
            style(fig, height=360, legend=False)
            fig.update_yaxes(title="Error")
            fig.update_xaxes(showgrid=False)
            show_fig(fig)

        if recovery is not None and len(recovery):
            section("Recovery speed", "Windows needed, after an episode starts, for a model's error to return within 25% of its "
                                      "own pre-drift level. A dash means it did not recover while the episode lasted.")
            rec = recovery.copy()
            rec["Episode"] = rec["episode"].astype(int).astype(str) + " (" + rec["true_type"] + ")"
            tbl = rec.pivot(index="Episode", columns="Strategy", values="windows_to_recover")
            tbl = tbl[[n for n in summary["Strategy"] if n in tbl.columns]]
            show_df(tbl.fillna(-1).astype(int).astype(str).replace("-1", "-").reset_index())

# ================================================================== Detector
with tab_det:
    section("How the detector decides",
            f"Data drift needs at least {DATA_MIN_FEATURES} sensors to move for two windows in a row. Relational drift is flagged when "
            "the Page-Hinkley statistic of the frozen-rule residual crosses its limit. Thin lines mark each alarm.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Sensors that moved (KS or PSI)", "Page-Hinkley statistic",
                                        "Mean residual of the frozen rule"))
    line = dict(width=1.8, color="#2a78d6")
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["n_moved"], mode="lines", line=dict(width=1.8, color="#2a78d6", shape="hv"),
                             name="Sensors moved", hovertemplate="%{y}<extra>Sensors moved</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["ph_stat"], mode="lines", line=line, name="Page-Hinkley",
                             hovertemplate="%{y:.1f}<extra>Page-Hinkley</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["mean_resid"], mode="lines", line=line, name="Mean residual",
                             hovertemplate="%{y:.2f}<extra>Mean residual</extra>"), row=3, col=1)
    for r, (level, label) in enumerate([(DATA_MIN_FEATURES, "alarm level"), (PH_PARAMS["threshold"], "alarm level"),
                                        (RULE_BACK, "old rule works again below this")], start=1):
        fig.add_hline(y=level, line_width=1, line_color=MUTED, row=r, col=1,
                      annotation_text=label, annotation_position="top left", annotation_font=dict(size=10.5, color=MUTED))
        bands(fig, row=r, labels=False)
        alarm_lines(fig, row=r)
    style(fig, height=640, legend=False)
    fig.update_layout(margin=dict(l=56, r=24, t=44, b=40))
    for a in fig.layout.annotations:
        if a.text in ("Sensors that moved (KS or PSI)", "Page-Hinkley statistic", "Mean residual of the frozen rule"):
            a.update(x=0, xanchor="left", font=dict(size=13, color=INK))
    show_fig(fig)

    section("Window by window")
    show_df(log[["window", "timestamp", "diagnosis", "alarm", "recovery", "n_moved", "features_moved", "worst_ks",
                 "worst_psi", "ph_stat", "mean_resid"]].round(3))

# ================================================================== Predictions
with tab_pred:
    c1, c2 = st.columns([1, 3])
    res = c1.radio("Resolution", ["Daily mean", "Hourly"], horizontal=True)
    chosen_cols = c2.multiselect("Models", strategy_cols, default=strategy_cols)
    d = pred.set_index("timestamp")[["actual"] + chosen_cols]
    d = d.resample("D").mean() if res == "Daily mean" else d.iloc[:: max(1, len(d) // 6000)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["actual"], name="Actual", mode="lines", line=dict(width=2, color=INK),
                             hovertemplate="%{y:.1f}<extra>Actual</extra>"))
    for name in chosen_cols:
        fig.add_trace(go.Scatter(x=d.index, y=d[name], name=name, mode="lines", line=dict(width=1.3, color=color_for(name)),
                                 hovertemplate="%{y:.1f}<extra>" + name + "</extra>"))
    bands(fig)
    alarm_lines(fig)
    style(fig, height=520)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Health deterioration")
    show_fig(fig)
    st.caption("Shaded bands are the true drift periods (offline ground truth). Thin vertical lines are detector alarms.")

# ================================================================== Events
with tab_ev:
    if types is not None:
        section("Drift episodes and what the detector said", "Compared against the ground-truth labels, which the live detector never sees.")
        t = types.copy()
        t["type_correct"] = np.where(t["type_correct"], "Yes", "No")
        show_df(t.rename(columns={"episode": "Episode", "true_type": "True type", "detected_type": "Detected type",
                                  "type_correct": "Type matched", "alarm_window": "Alarm window",
                                  "episode_start_window": "Episode start", "episode_end_window": "Episode end",
                                  "delay_windows": "Delay (windows)"}))
    if events is not None:
        section("Alarms, recoveries and adaptation windows")
        show_df(events[["window", "start", "end", "timestamp", "diagnosis", "alarm", "recovery", "adapted", "features_moved",
                        "ph_stat"]].round(2))
