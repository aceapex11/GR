"""Drift monitor - Streamlit dashboard for the GRU drift experiment.

    streamlit run app.py

Reads results/*.csv written by run_experiment.py (the sidebar button runs it).
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
INK, INK2, MUTED = "#0f172a", "#475569", "#94a3b8"
GRID, AXIS, SURFACE = "#eef0f6", "#d5d9e4", "#ffffff"
PAGE_BG, BORDER = "#f5f6fb", "#e5e8f0"
ACCENT = "#4f46e5"
GOOD_TEXT, CRITICAL = "#047857", "#e11d48"
DRIFT = "#f59e0b"
BAND = "rgba(245,158,11,0.10)"
FONT = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
# one fixed colour per strategy; Static is the neutral grey baseline
STRATEGY_COLORS = [("blend", "#0d9488"), ("rollback", "#d946ef"),
                   ("static", "#94a3b8"), ("transfer", "#3b82f6"), ("full", "#f97316"),
                   ("replay 100", "#10b981"), ("replay 500", "#8b5cf6")]
REGIME_LABEL = {"stable": "Stable", "data_drift": "Data", "relational_drift": "Relational", "both": "Both"}


def color_for(name):
    low = name.lower()
    if "transfer" in low and "rollback" in low:
        return "#0ea5e9"
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html { font-size: 18px; }
html, body, .stApp { font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp :is(p, label, button, input, textarea, h1, h2, h3, h4, li, a, td, th, div),
.stApp span:not([data-testid="stIconMaterial"]) { font-family: inherit; }
/* keep Streamlit's icon font intact (fixes the "keyboard_double_" text in the sidebar) */
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {
    font-family: "Material Symbols Rounded", "Material Icons" !important; }
.stApp { background: #f5f6fb; color: #0f172a; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: #0f172a; }

.hero { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
        padding: 28px 32px; border-radius: 20px; color: #fff;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #3b82f6 100%);
        box-shadow: 0 10px 30px rgba(79,70,229,.25); }
.hero-title { font-size: 2.2rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; color: #fff; }
.hero-sub { font-size: .95rem; color: rgba(255,255,255,.85); margin-top: .35rem; max-width: 62ch; }
.hero-badge { font-size: .8rem; font-weight: 500; color: #fff; padding: 7px 14px; border-radius: 999px;
              background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); }

.verdict { display: flex; align-items: center; gap: .7rem; font-size: 1rem; font-weight: 500;
           margin: 1.3rem 0 0; padding: 14px 18px; border-radius: 14px; border: 1px solid; }
.verdict.ok  { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
.verdict.bad { background: #fff1f2; border-color: #fecdd3; color: #9f1239; }
.verdict svg { flex: none; }

.kpi-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 1.1rem 0 1.4rem; }
.kpi { background: #fff; border: 1px solid #e5e8f0; border-radius: 16px; padding: 16px 18px;
       box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.kpi-label { font-size: .74rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }
.kpi-value { font-size: 1.9rem; font-weight: 700; line-height: 1.3; font-variant-numeric: tabular-nums; color: #0f172a; }
.kpi-value.long { font-size: 1.2rem; line-height: 1.7; }
.kpi-note { font-size: .78rem; color: #94a3b8; }

.sec { font-size: 1.08rem; font-weight: 600; margin: 2rem 0 .15rem; color: #0f172a; }
.sec-note { font-size: .86rem; color: #64748b; margin-bottom: .7rem; max-width: 72ch; }
.takeaway { font-size: .93rem; color: #0f172a; margin: .5rem 0; padding: 12px 14px; background: #fff;
            border: 1px solid #e5e8f0; border-left: 3px solid #4f46e5; border-radius: 10px; max-width: 75ch; }

/* tabs as a pill switcher (selectors cover old and new Streamlit versions) */
[role="tablist"] { gap: 4px; background: #e9ebf4; padding: 4px; border-radius: 12px; width: fit-content; border-bottom: none !important; }
button[role="tab"] { height: auto !important; padding: 8px 18px !important; margin: 0 !important; border-radius: 9px !important; }
button[role="tab"], button[role="tab"] * { font-size: .95rem; font-weight: 500; color: #475569 !important; }
button[role="tab"][aria-selected="true"] { background: #fff; box-shadow: 0 1px 3px rgba(15,23,42,.12); }
button[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] * { color: #4f46e5 !important; }
button[role="tab"][aria-selected="true"] { background: #fff !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [data-testid="stTabHighlight"], [role="tablist"] > div:not([role="tab"]) { display: none !important; }

/* charts and tables as cards */
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e5e8f0; border-radius: 16px; padding: 10px 8px;
                                box-shadow: 0 1px 2px rgba(15,23,42,.04); }
[data-testid="stDataFrame"] { border: 1px solid #e5e8f0; border-radius: 14px; overflow: hidden; }
[data-testid="stExpander"] { background: #fff; border: 1px solid #e5e8f0 !important; border-radius: 14px; margin-top: .8rem; }

[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e5e8f0; }
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed); border: none; border-radius: 10px; color: #fff; font-weight: 600; }
button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
@media (max-width: 640px) { .hero { padding: 20px; } .hero-title { font-size: 1.5rem; } }
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
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value{" long" if len(str(v)) > 13 else ""}">{v}</div>'
                    f'<div class="kpi-note">{n}</div></div>' for l, v, n in items)
    st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)


def style(fig, height=531, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, showlegend=legend,
        font=dict(family=FONT, size=14, color=INK2), margin=dict(l=56, r=24, t=40, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=14, color=INK2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(family=FONT, size=13, color=INK)))
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


def run_script(label, script_name, flags):
    """Run one project script and return its captured output and success status."""
    script = ROOT / script_name
    if not script.is_file():
        return f"Missing {script_name}. Keep it beside app.py.", False
    command = [sys.executable, str(script), *flags]
    try:
        with st.spinner(f"Running {label}. This may take several minutes..."):
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=7200,
            )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return output.strip()[-12000:], result.returncode == 0
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return f"{label} exceeded the two-hour limit.\n{partial}"[-12000:], False
    except Exception as exc:
        return f"Could not start {label}: {exc}", False


with st.sidebar:
    st.markdown("**Experiment pipeline**")
    st.caption(
        "Run the GRU streaming experiment and then evaluate adaptation strategies. "
        "The model review is saved separately in results/review/."
    )
    full_training = st.checkbox(
        "Full training (slower)",
        value=False,
        help="Unchecked runs the quick smoke test. Checked runs the full settings and three-seed review.",
    )
    flags = [] if full_training else ["--fast"]

    if wide(st.button, "Run complete pipeline", type="primary"):
        logs = []
        output, ok = run_script("main streaming experiment", "experiment.py", flags)
        logs.append("===== MAIN STREAMING EXPERIMENT =====\n" + output)
        if ok:
            output, ok = run_script("multi-seed model review", "review_run.py", flags)
            logs.append("===== MULTI-SEED MODEL REVIEW =====\n" + output)
        st.session_state["pipeline_log"] = "\n\n".join(logs)
        st.session_state["pipeline_success"] = ok
        if ok:
            st.session_state.pop("run_error", None)
            st.rerun()
        else:
            st.session_state["run_error"] = st.session_state["pipeline_log"]
            st.error("Pipeline failed. Open the error details on the main page.")

    if wide(st.button, "Run model review only"):
        output, ok = run_script("multi-seed model review", "review_run.py", flags)
        st.session_state["pipeline_log"] = output
        st.session_state["pipeline_success"] = ok
        if ok:
            st.session_state.pop("run_error", None)
            st.rerun()
        else:
            st.session_state["run_error"] = output
            st.error("Model review failed. Open the error details on the main page.")

meta = ""
if cfg:
    meta = f'{"Fast" if cfg.get("fast") else "Full"} run, seed {cfg.get("seed", 0)}'
badge = f'<div class="hero-badge">{meta}</div>' if meta else ""
st.markdown(
    f'<div class="hero"><div><div class="hero-title">Drift monitor</div>'
    f'<div class="hero-sub">GRU regression on a 60,000-row industrial sensor stream, with drift detection and model adaptation</div></div>'
    f'{badge}</div>', unsafe_allow_html=True)

if st.session_state.get("pipeline_success"):
    st.success("Pipeline completed. The latest experiment and model-review results are ready.")
    _pipeline_log = st.session_state.get("pipeline_log", "")
    if _pipeline_log:
        with st.expander("View pipeline log"):
            st.code(_pipeline_log, language="text")

if st.session_state.get("run_error"):
    _msg = st.session_state["run_error"]
    _lines = [l for l in _msg.splitlines() if l.strip()]
    st.error("The experiment failed: " + (_lines[-1] if _lines else "no output was captured"))
    with st.expander("Full error output", expanded=True):
        try:
            st.code(_msg, language="text", wrap_lines=True)
        except TypeError:
            st.code(_msg, language="text")
    if st.button("Dismiss error"):
        st.session_state.pop("run_error", None)
        st.rerun()

if not has_results:
    section("No results yet", "Open the sidebar and select <b>Run GRU experiment</b>, or run <code>python run_experiment.py</code> "
                              "in a terminal. The fast setting takes a few minutes.")
    st.stop()

# ------------------------------------------------------------------ load + derive
score, metrics, types = read("detector_scorecard.csv"), read("strategy_metrics.csv"), read("drift_type_evaluation.csv")
summary, recovery, events = read("performance_summary.csv"), read("episode_recovery.csv"), read("adaptation_events.csv")

log["timestamp"] = pd.to_datetime(log["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])
strategy_cols = [c for c in pred.columns if c not in ("row", "timestamp", "actual")]
summary_is_fallback = False
if summary is None and strategy_cols:
    _ss_tot = float(((pred["actual"] - pred["actual"].mean()) ** 2).sum())
    _rows = []
    for _c in strategy_cols:
        _e = pred[_c] - pred["actual"]
        _rows.append({"Strategy": _c, "MAE": float(_e.abs().mean()), "RMSE": float(np.sqrt((_e ** 2).mean())),
                      "R2": 1 - float((_e ** 2).sum()) / _ss_tot if _ss_tot else np.nan})
    summary = pd.DataFrame(_rows).sort_values("MAE").reset_index(drop=True)
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    summary_is_fallback = True
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
                               showarrow=False, font=dict(size=12.5, color="#b45309"))


def alarm_lines(fig, row=None):
    kw = {} if row is None else {"row": row, "col": 1}
    for t in alarms["timestamp"]:
        fig.add_vline(x=ts(t), line_width=1, line_dash="dot", line_color=CRITICAL, opacity=0.7, **kw)


def window_mae_cols():
    return [c for c in log.columns if c.startswith("mae[")]


# ------------------------------------------------------------------ tabs
# Model review tab is commented out (see the block at the bottom). To bring it back, use the 7-tab line below.
# tab_over, tab_model, tab_err, tab_det, tab_pred, tab_ev, tab_rev = st.tabs(
#     ["Overview", "Model performance", "Error inspector", "Detector", "Predictions", "Events", "Model review"])
tab_over, tab_model, tab_err, tab_det, tab_pred, tab_ev = st.tabs(
    ["Overview", "Model performance", "Error inspector", "Detector", "Predictions", "Events"])

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
        st.markdown(f'<div class="verdict {"ok" if ok else "bad"}">{icon}<span>{verdict}</span></div>', unsafe_allow_html=True)
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

    mids = segments("True drift", [(a, b, f"{REGIME_LABEL.get(r, r)} drift") for a, b, r in BLOCKS], "rgba(245,158,11,0.55)", 16)
    fig.add_trace(go.Scatter(x=mids, y=["True drift"] * len(mids), mode="text",
                             text=[REGIME_LABEL.get(r, r) for _, _, r in BLOCKS], textposition="top center",
                             textfont=dict(size=13, color=INK2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[ts(t) for t in alarms["timestamp"]], y=["Detector alarm"] * len(alarms), mode="markers",
                             marker=dict(symbol="triangle-up", size=12, color=CRITICAL, line=dict(width=2, color=SURFACE)),
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
        segments("Models adapting", items, "#6366f1", 12)
    style(fig, height=271, legend=False)
    fig.update_layout(margin=dict(l=150, r=24, t=16, b=40))
    fig.update_yaxes(categoryorder="array", categoryarray=rows_order, showgrid=False, tickfont=dict(color=INK2, size=14))
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
            style(fig, height=max(295, 52 * len(order) + 70), legend=False)
            fig.update_layout(bargap=0.55, margin=dict(l=250, r=50, t=8, b=36))
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
        if summary_is_fallback:
            st.info("performance_summary.csv is missing, so basic scores (MAE, RMSE, R squared) are calculated "
                    "directly from predictions.csv. Run the experiment to get the full leaderboard.")
        chosen = st.selectbox("Model", summary["Strategy"].tolist(), index=0)
        r0 = summary[summary["Strategy"] == chosen].iloc[0]
        gain = r0.get("Gain vs Static, drift (%)", np.nan)
        kpi_strip([
            ("MAE", f'{r0["MAE"]:.2f}', "average absolute error"),
            ("RMSE", f'{r0["RMSE"]:.2f}', "penalises large misses"),
            ("R squared", f'{r0["R2"]:.3f}', "share of variance explained"),
            ("Error during drift", "n/a" if pd.isna(r0.get("Drift error (x normal)", np.nan)) else f'{r0["Drift error (x normal)"]:.2f}x',
             "relative to pre-drift error"),
            ("Gain vs Static in drift", "n/a" if pd.isna(gain) else f"{gain:+.0f}%", "negative means worse"),
        ])

        section("Leaderboard", "Accuracy, how each model behaves around drift, and what its updates cost.")
        cols = ["Rank", "Strategy", "MAE", "RMSE", "R2", "MAE normal", "MAE stable", "MAE drift", "Gain vs Static, drift (%)",
                "Updates", "Training samples", "Training time (s)"]
        core = ["Rank", "Strategy", "MAE", "RMSE", "R2", "Gain vs Static, drift (%)", "Training time (s)"]
        show_df(summary[[c for c in core if c in summary]].round(2))
        with st.expander("Show all columns"):
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
                    colorscale=[[0, "#eef2ff"], [0.5, "#818cf8"], [1, "#3730a3"]], showscale=False,
                    text=np.round(piv.values, 1), texttemplate="%{text}", xgap=2, ygap=2,
                    textfont=dict(size=14), hovertemplate="%{y}, %{x}: %{z:.2f}<extra></extra>"))
                style(fig, height=max(342, 56 * len(piv) + 90), legend=False)
                fig.update_layout(margin=dict(l=240, r=12, t=8, b=36))
                fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK2))
                fig.update_xaxes(side="top", tickfont=dict(color=INK2))
                show_fig(fig)
        with right:
            section("Peak error per drift episode", "Highest window error reached during each episode.")
            if recovery is not None and len(recovery):
                fig = go.Figure()
                for name in summary["Strategy"]:
                    d = recovery[recovery["Strategy"] == name]
                    fig.add_trace(go.Bar(x=[f"Ep {int(e)}" for e in d["episode"]],
                                         y=d["peak_mae"], name=name, marker_color=color_for(name),
                                         hovertemplate="%{x}: %{y:.2f}<extra>" + name + "</extra>"))
                style(fig, height=342)
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
        style(fig, height=448)
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title="Window MAE", rangemode="tozero")
        show_fig(fig)

        bw = read("blend_weights.csv")
        if bw is not None and len(bw):
            section("What the blend trusted", "Share of the blend's weight given to each model, window by window. "
                                              "It shifts toward whichever model has been most accurate recently.")
            xs = log.set_index("window")["timestamp"].reindex(bw["window"]).values
            fig = go.Figure()
            for c in [c for c in bw.columns if c != "window"]:
                col = color_for(c)
                fig.add_trace(go.Scatter(x=xs, y=bw[c] * 100, name=c, mode="lines", stackgroup="w",
                                         line=dict(width=0.6, color=col, shape="hv"), fillcolor=rgba(col, 0.78),
                                         hovertemplate="%{y:.0f}%<extra>" + c + "</extra>"))
            bands(fig)
            style(fig, height=400)
            fig.update_layout(hovermode="x unified")
            fig.update_yaxes(title="Weight (%)", range=[0, 100])
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
            style(fig, height=425)
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
                                         fillcolor=rgba(color_for(chosen).replace("#94a3b8", "#64748b"), 0.18)))
            fig.add_hline(y=0, line_width=1, line_color=INK2)
            style(fig, height=425, legend=False)
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

# ================================================================== Error inspector
with tab_err:
    section("Where does the model miss?",
            "Pick a model and an error limit. Every row where predicted and actual differ by more than the limit is flagged in red.")
    c1, c2, c3 = st.columns([1.1, 1.4, 2])
    insp = c1.selectbox("Model", strategy_cols, key="insp_model")
    mae_all = float((pred[insp] - pred["actual"]).abs().mean())
    mode = c2.radio("Limit", ["Multiple of MAE", "Fixed value"], horizontal=True, key="insp_mode")
    if mode == "Multiple of MAE":
        k = c2.slider("Times the MAE", 0.5, 6.0, 1.0, 0.25, key="insp_k")
        limit = k * mae_all
    else:
        limit = float(c2.number_input("Error limit", min_value=0.0, value=round(mae_all, 2), step=0.5, key="insp_fixed"))
    t0, t1 = pred["timestamp"].min().to_pydatetime(), pred["timestamp"].max().to_pydatetime()
    lo_t, hi_t = c3.slider("Time range", min_value=t0, max_value=t1, value=(t0, t1), format="YYYY-MM-DD", key="insp_range")

    df = pred[["row", "timestamp", "actual", insp]].rename(columns={insp: "predicted"})
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    if truth is not None:
        df = df.merge(truth[["row", "regime"]], on="row", how="left")
    else:
        df["regime"] = "stable"
    df = df[(df["timestamp"] >= lo_t) & (df["timestamp"] <= hi_t)]
    df["over"] = df["abs_error"] > limit
    n, n_over = len(df), int(df["over"].sum())

    kpi_strip([
        ("Overall MAE", f"{mae_all:.2f}", f"{insp}, whole stream"),
        ("Error limit", f"{limit:.2f}", "flag rows above this"),
        ("Rows over the limit", f"{n_over:,}", f"of {n:,} in range"),
        ("Share over the limit", f"{(100 * n_over / n if n else 0):.1f}%", "of rows in range"),
        ("Largest miss", f"{(df['abs_error'].max() if n else 0):.2f}", "absolute error"),
    ])

    section("Actual against predicted over time", "Red dots are rows that miss by more than the limit. Shaded bands are true drift periods.")
    step = max(1, n // 6000)
    line_d = df.iloc[::step]
    flagged = df[df["over"]]
    if len(flagged) > 3000:
        flagged = flagged.nlargest(3000, "abs_error")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=line_d["timestamp"], y=line_d["actual"], name="Actual", mode="lines",
                             line=dict(width=2, color=INK), hovertemplate="%{y:.1f}<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=line_d["timestamp"], y=line_d["predicted"], name=f"Predicted ({insp})", mode="lines",
                             line=dict(width=1.6, color=color_for(insp)), hovertemplate="%{y:.1f}<extra>Predicted</extra>"))
    fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["predicted"], name="Over the limit", mode="markers",
                             marker=dict(size=6, color=CRITICAL, opacity=0.7),
                             customdata=flagged[["actual", "error"]].values,
                             hovertemplate="Predicted %{y:.1f}, actual %{customdata[0]:.1f}, error %{customdata[1]:+.1f}<extra>Over limit</extra>"))
    bands(fig)
    style(fig, height=500)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Health deterioration")
    show_fig(fig)
    if len(flagged) < n_over:
        st.caption(f"Showing the {len(flagged):,} largest of {n_over:,} flagged rows to keep the chart responsive.")

    left, right = st.columns(2)
    with left:
        section("Share of rows over the limit, by regime", "Which kinds of period the model struggles in.")
        order = [r for r in ["stable", "data_drift", "relational_drift", "both"] if r in set(df["regime"].dropna())]
        if order:
            share = [100 * df.loc[df["regime"] == r, "over"].mean() for r in order]
            fig = go.Figure(go.Bar(x=[REGIME_LABEL.get(r, r) for r in order], y=share,
                                   marker_color=[MUTED if r == "stable" else DRIFT for r in order],
                                   text=[f"{v:.0f}%" for v in share], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            style(fig, height=380, legend=False)
            fig.update_layout(bargap=0.5)
            fig.update_yaxes(title="% of rows over limit", rangemode="tozero")
            show_fig(fig)
    with right:
        section("How big are the errors?", "Distribution of absolute error. The red line is your limit.")
        fig = go.Figure(go.Histogram(x=df["abs_error"], nbinsx=60, marker_color=color_for(insp), opacity=0.85,
                                     hovertemplate="error %{x:.1f}: %{y} rows<extra></extra>"))
        fig.add_vline(x=limit, line_width=2, line_dash="dash", line_color=CRITICAL)
        style(fig, height=380, legend=False)
        fig.update_xaxes(title="Absolute error")
        fig.update_yaxes(title="Rows")
        show_fig(fig)

    section("Largest misses", "Worst rows in the selected range. Download the full list of flagged rows as CSV.")
    worst = df[df["over"]].nlargest(200, "abs_error")[["timestamp", "regime", "actual", "predicted", "error"]].copy()
    worst["regime"] = worst["regime"].map(lambda r: REGIME_LABEL.get(r, r))
    worst = worst.rename(columns={"timestamp": "Time", "regime": "Regime", "actual": "Actual", "predicted": "Predicted", "error": "Error"})
    if len(worst):
        show_df(worst.round(2))
        wide(st.download_button, "Download all flagged rows (CSV)", df[df["over"]].round(3).to_csv(index=False).encode(),
             file_name=f"rows_over_limit_{insp.replace(' ', '_')}.csv", mime="text/csv")
    else:
        st.success("No rows exceed this limit in the selected range.")

# ================================================================== Detector
with tab_det:
    section("How the detector decides",
            f"Data drift needs at least {DATA_MIN_FEATURES} sensors to move for two windows in a row. Relational drift is flagged when "
            "the Page-Hinkley statistic of the frozen-rule residual crosses its limit. Thin lines mark each alarm.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Sensors that moved (KS or PSI)", "Page-Hinkley statistic",
                                        "Mean residual of the frozen rule"))
    line = dict(width=1.8, color="#6366f1")
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["n_moved"], mode="lines", line=dict(width=1.8, color="#6366f1", shape="hv"),
                             name="Sensors moved", hovertemplate="%{y}<extra>Sensors moved</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["ph_stat"], mode="lines", line=line, name="Page-Hinkley",
                             hovertemplate="%{y:.1f}<extra>Page-Hinkley</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=log["timestamp"], y=log["mean_resid"], mode="lines", line=line, name="Mean residual",
                             hovertemplate="%{y:.2f}<extra>Mean residual</extra>"), row=3, col=1)
    for r, (level, label) in enumerate([(DATA_MIN_FEATURES, "alarm level"), (PH_PARAMS["threshold"], "alarm level"),
                                        (RULE_BACK, "old rule works again below this")], start=1):
        fig.add_hline(y=level, line_width=1, line_dash="dash", line_color=MUTED, row=r, col=1,
                      annotation_text=label, annotation_position="top left", annotation_font=dict(size=12.5, color=MUTED))
        bands(fig, row=r, labels=False)
        alarm_lines(fig, row=r)
    style(fig, height=755, legend=False)
    fig.update_layout(margin=dict(l=56, r=24, t=44, b=40))
    for a in fig.layout.annotations:
        if a.text in ("Sensors that moved (KS or PSI)", "Page-Hinkley statistic", "Mean residual of the frozen rule"):
            a.update(x=0, xanchor="left", font=dict(size=15, color=INK))
    show_fig(fig)

    with st.expander("Window-by-window detector log"):
      show_df(log[["window", "timestamp", "diagnosis", "alarm", "recovery", "n_moved", "features_moved", "worst_ks",
                 "worst_psi", "ph_stat", "mean_resid"]].round(3))

# ================================================================== Predictions
with tab_pred:
    c1, c2 = st.columns([1, 3])
    res = c1.radio("Resolution", ["Daily mean", "Hourly"], horizontal=True)
    chosen_cols = c2.multiselect("Models", strategy_cols, default=[c for c in strategy_cols if "rollback" not in c.lower()])
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
    style(fig, height=614)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Health deterioration")
    show_fig(fig)
    st.caption("Shaded bands are the true drift periods (offline ground truth). Thin vertical lines are detector alarms.")

# ================================================================== Events
with tab_ev:
    if types is not None:
        section("Drift episodes and what the detector said", "Compared against the ground-truth labels, which the live detector never sees.")
        tt = types.copy()
        tt["type_correct"] = np.where(tt["type_correct"], "\u2713 Yes", "\u2717 No")
        show_df(tt.rename(columns={"episode": "Episode", "true_type": "True type", "detected_type": "Detected type",
                                   "type_correct": "Type matched", "alarm_window": "Alarm window",
                                   "episode_start_window": "Episode start", "episode_end_window": "Episode end",
                                   "delay_windows": "Delay (windows)"}))
    if events is not None:
        section("Alarms, recoveries and adaptation windows", "One row per window where something happened.")
        view = st.radio("Show", ["All", "Alarms", "Recoveries", "Adapting"], horizontal=True, label_visibility="collapsed")
        ev = events.copy()
        for c in ("alarm", "recovery", "adapted"):
            ev[c] = ev[c].astype(str).str.lower().isin(["true", "1"])
        keep = {"All": ev["alarm"] | ev["recovery"] | ev["adapted"], "Alarms": ev["alarm"],
                "Recoveries": ev["recovery"], "Adapting": ev["adapted"]}[view]
        ev = ev[keep]

        def status(r):
            parts = [n for n, f in (("Alarm", r["alarm"]), ("Recovery", r["recovery"]), ("Adapting", r["adapted"])) if f]
            return " + ".join(parts) if parts else "-"

        ev["Status"] = ev.apply(status, axis=1)
        ev["features_moved"] = ev["features_moved"].fillna("-").replace("", "-")
        out = ev[["window", "timestamp", "diagnosis", "Status", "features_moved", "ph_stat"]].rename(columns={
            "window": "Window", "timestamp": "Date", "diagnosis": "Diagnosis",
            "features_moved": "Sensors moved", "ph_stat": "PH statistic"}).round(2)
        show_df(out, column_config={"Sensors moved": st.column_config.TextColumn(width="large")})

# # ================================================================== Model review
# with tab_rev:
#     import review_tab
#     review_tab.render(RESULTS / "review", dict(
#         section=section, show_fig=show_fig, show_df=show_df, style=style, color_for=color_for, kpi_strip=kpi_strip,
#         INK=INK, INK2=INK2, MUTED=MUTED, BORDER=BORDER, SURFACE=SURFACE, GOOD_TEXT=GOOD_TEXT, CRITICAL=CRITICAL, DRIFT=DRIFT))
