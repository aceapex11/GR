"""'Model review' tab: which adaptation strategy works, why, when to use it and what it costs.

Every number comes from results/review/*.csv (written by review_run.py). Nothing here is a fixed
example value. The analysis lives in review_metrics.py (pure pandas); this file only draws it.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import review_metrics as rm

LEVEL_STYLE = {   # level -> (label, colour key)
    "supported": ("Supported", "good"), "preliminary": ("Preliminary", "warn"),
    "insufficient": ("Insufficient evidence", "muted"), "noise": ("Within noise", "warn"),
    "untestable": ("Cannot test", "muted"), "failure": ("Failure", "bad"), "note": ("Note", "muted")}
VERDICT_ORDER = {"Supported": 0, "Preliminary": 1, "No clear difference": 2, "Not supported": 3,
                 "Insufficient evidence": 4, "Excluded": 5, "Reference": 6}


def _load(folder):
    def rd(name):
        p = folder / name
        return pd.read_csv(p) if p.is_file() and p.stat().st_size else None
    meta = {}
    if (folder / "meta.json").is_file():
        try:
            meta = json.loads((folder / "meta.json").read_text())
        except Exception:
            meta = {}
    return rd("metrics.csv"), rd("episodes.csv"), rd("cost.csv"), meta


def _pill(ctx, level, text=None):
    label, key = LEVEL_STYLE.get(level, (level, "muted"))
    col = {"good": ctx["GOOD_TEXT"], "bad": ctx["CRITICAL"], "warn": ctx["DRIFT"], "muted": ctx["MUTED"]}[key]
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;'
            f'color:{col};border:1px solid {col};white-space:nowrap">{text or label}</span>')


def _card(ctx, level, title, body):
    label, key = LEVEL_STYLE[level]
    col = {"good": ctx["GOOD_TEXT"], "bad": ctx["CRITICAL"], "warn": ctx["DRIFT"], "muted": ctx["MUTED"]}[key]
    st.markdown(
        f'<div style="border:1px solid {ctx["BORDER"]};border-left:4px solid {col};background:{ctx["SURFACE"]};'
        f'border-radius:10px;padding:14px 18px;margin:8px 0 14px 0"><div style="margin-bottom:6px">'
        f'{_pill(ctx, level)}</div><div style="font-size:18px;font-weight:600;color:{ctx["INK"]}">{title}</div>'
        f'<div style="color:{ctx["INK2"]};margin-top:4px">{body}</div></div>', unsafe_allow_html=True)


def _short(name, width=26):
    return name if len(name) <= width else name[:width - 1] + "\u2026"


def render(folder, ctx):
    M, EP, C, meta = _load(folder)
    if M is None or EP is None or C is None:
        st.info("This tab needs the multi-seed review results. Run `python review_run.py --fast` for a quick check "
                "or `python review_run.py` for full training (three seeds). It writes results/review/ and does not "
                "change any other results file.")
        return

    section, show_fig, show_df, style, color_for = (ctx[k] for k in ("section", "show_fig", "show_df", "style", "color_for"))
    strategies = list(M["Strategy"].unique())
    static = next((s for s in strategies if "static" in s.lower()), None)
    if static is None:
        st.warning("No Static baseline in the review results, so no strategy can be judged against it.")
        return
    seeds = sorted(M["seed"].unique())
    n_seeds = len(seeds)
    phases = [p for p in ["overall"] + rm.REGIMES if p in set(M["Phase"])]
    n_ep = int(EP["episode"].nunique()) if len(EP) else 0

    st.markdown(f"**{n_seeds} seed{'s' if n_seeds != 1 else ''}**, {n_ep} drift episodes, {len(strategies)} strategies"
                + (" - fast run (few epochs)" if meta.get("fast") else ""))
    if meta.get("fast"):
        st.warning("This was a fast run with few training epochs. Treat every conclusion below as a smoke test, "
                   "not a research result; rerun `python review_run.py` for full training.")
    if n_seeds < rm.MIN_SEEDS:
        st.warning(f"Only {n_seeds} seed{'s' if n_seeds != 1 else ''}. Nothing can be called Supported until "
                   f"{rm.MIN_SEEDS} seeds agree, so results below are at most Preliminary.")
    st.caption("Seeds change GRU initialisation and sampling. The stream and detector are the same for every seed.")

    metric = st.radio("Error measure", ["MAE", "RMSE"], horizontal=True, key="rev_metric")

    # ------------------------------------------------------------------ 1. best model per drift type
    section("1. Best model for each drift type",
            "Every strategy ranked separately for stable, data, relational and combined drift. Mean over seeds, with the spread.")
    cols = []
    for ph in phases:
        la = rm.lead_assessment(M, ph, metric)
        if la:
            cols.append((rm.LABEL[ph], la["best"], f'{la["verdict"]}, {la["gap_pct"]:.0f}% ahead of runner-up'))
    ctx["kpi_strip"]([(l, _short(b, 22), n) for l, b, n in cols])

    ph_sel = st.selectbox("Phase", phases, format_func=lambda p: rm.LABEL[p], key="rev_phase")
    rt = rm.rank_table(M, ph_sel, metric)
    if len(rt):
        fig = go.Figure(go.Bar(
            y=[_short(s, 34) for s in rt["Strategy"]][::-1], x=rt["mean"][::-1], orientation="h",
            marker_color=[color_for(s) for s in rt["Strategy"]][::-1],
            error_x=dict(type="data", array=rt["std"].fillna(0)[::-1], color=ctx["INK2"], thickness=1.2),
            hovertemplate="%{y}<br>" + metric + " %{x:.2f}<extra></extra>"))
        fig.update_xaxes(title=f"{metric} (lower is better); bars show +-1 std over seeds")
        show_fig(style(fig, height=max(320, 46 * len(rt)), legend=False))
        show = rt.rename(columns={"mean": metric, "std": "Std over seeds", "best_in_seeds": "Seeds where best"})
        show_df(show[["Rank", "Strategy", metric, "Std over seeds", "Seeds where best"]].round(3))
        la = rm.lead_assessment(M, ph_sel, metric)
        if la:
            level = {"clear lead": "supported", "within noise": "noise"}.get(la["verdict"], "untestable")
            _card(ctx, level, f"{rm.LABEL[ph_sel]}: {la['best']} leads by {la['gap_pct']:.1f}% over {la['second']}",
                  f"Verdict: <b>{la['verdict']}</b> ({la['n_seeds']} seed{'s' if la['n_seeds'] != 1 else ''}). "
                  + ("The lead is larger than seed-to-seed noise." if la["verdict"] == "clear lead" else
                     "The gap is not larger than seed-to-seed noise, so do not rank these two." if la["verdict"] == "within noise" else
                     f"At least {rm.MIN_SEEDS} seeds are needed to test whether this gap is real."))

    # ------------------------------------------------------------------ 2. adaptation effectiveness
    section("2. Adaptation effectiveness",
            "Gain over Static (positive is better), whether the gain lasts through the episode, and how fast error recovers afterwards.")
    gt = rm.gain_table(M, static, metric)
    if len(gt):
        piv = gt.pivot(index="Strategy", columns="Phase", values="gain_pct").reindex(
            [s for s in strategies if s != static])[[p for p in phases if p in set(gt["Phase"])]]
        lim = max(10.0, float(np.nanmax(np.abs(piv.to_numpy()))))
        fig = go.Figure(go.Heatmap(
            z=piv.to_numpy(), x=[rm.LABEL[c] for c in piv.columns], y=[_short(s, 34) for s in piv.index],
            zmin=-lim, zmax=lim, colorscale=[[0, ctx["CRITICAL"]], [0.5, "#f8fafc"], [1, ctx["GOOD_TEXT"]]],
            text=np.round(piv.to_numpy(), 0), texttemplate="%{text:+.0f}%", hovertemplate="%{y}<br>%{x}: %{z:+.1f}%<extra></extra>",
            colorbar=dict(title=f"% {metric} vs Static")))
        fig.update_yaxes(autorange="reversed")
        show_fig(style(fig, height=max(340, 52 * len(piv)), legend=False))
        st.caption("A cell is only a real difference if it exceeds the " + f"{rm.MIN_GAIN_PCT:.0f}% threshold and holds in every seed; "
                   "see section 5 for where it does not.")

    pt = rm.persistence_table(EP, static)
    if len(pt):
        out = pt.rename(columns={
            "episodes": "Episodes", "held_pct": "Error fell during episode (%)", "vs_static_late": "Late-episode gain vs Static (%)",
            "post_ratio": "Error after episode (x own baseline)", "recovered": "Episodes recovered (share)",
            "n_judged": "Judgeable", "recovery_med": "Median windows to recover"})
        out["Episodes recovered (share)"] = (out["Episodes recovered (share)"] * 100).round(0)
        show_df(out.round(2))
        st.caption("'Judgeable' counts episodes with enough later windows to tell whether error recovered; the rest are left out, "
                   "not counted as failures. Recovery means error within 1.25x the strategy's own pre-drift level for 2 windows in a row.")

    # ------------------------------------------------------------------ 3. accuracy vs cost
    section("3. Accuracy versus cost",
            "Prediction error against what each strategy spends. Strategies on the frontier are not beaten on both axes by another.")
    c1, c2 = st.columns(2)
    cost_label = {"train_seconds": "Training time (s)", "updates": "Updates", "train_samples": "Training samples",
                  "memory_mb": "Estimated state memory (MB)"}
    cost_col = c1.selectbox("Cost", list(cost_label), format_func=lambda k: cost_label[k], key="rev_cost")
    cph = c2.selectbox("Error in", phases, format_func=lambda p: rm.LABEL[p], key="rev_cost_phase")
    cs = rm.cost_summary(C, M, static, cph, metric)
    cf = cs.sort_values(cost_col)
    fig = go.Figure()
    fr = cf[cf["on_frontier"]] if cost_col == "train_seconds" else pd.DataFrame()
    if cost_col != "train_seconds":
        keep = rm.pareto_frontier(cs.set_index("Strategy"), cost_col, "error")
        fr = cf[cf["Strategy"].isin(keep)]
    fig.add_trace(go.Scatter(x=fr[cost_col], y=fr["error"], mode="lines", line=dict(color=ctx["MUTED"], dash="dot"),
                             hoverinfo="skip", name="Frontier"))
    for r in cs.itertuples():
        fig.add_trace(go.Scatter(x=[getattr(r, cost_col)], y=[r.error], mode="markers+text", text=[_short(r.Strategy, 22)],
                                 textposition="top center", name=r.Strategy, showlegend=False,
                                 marker=dict(size=14, color=color_for(r.Strategy), line=dict(width=2 if r.on_frontier else 0, color=ctx["INK"])),
                                 hovertemplate=f"{r.Strategy}<br>{cost_label[cost_col]}: %{{x:.2f}}<br>{metric}: %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title=cost_label[cost_col])
    fig.update_yaxes(title=f"{metric} ({rm.LABEL[cph]}, mean over seeds)")
    show_fig(style(fig, height=520, legend=False))
    tbl = cs.rename(columns={"error": metric, "gain_pct": "Gain vs Static (%)", "train_seconds": "Training time (s)",
                             "updates": "Updates", "train_samples": "Training samples", "memory_mb": "Est. memory (MB)",
                             "on_frontier": "On frontier"}).sort_values(metric)
    tbl["On frontier"] = np.where(tbl["On frontier"], "Yes", "")
    show_df(tbl[["Strategy", metric, "Gain vs Static (%)", "Training time (s)", "Updates", "Training samples",
                 "Est. memory (MB)", "On frontier"]].round(2))
    st.caption("Time is wall-clock inside update(); memory is an estimate of the state a deployment must hold (weights, rollback copy, "
               "stored rows), not process memory. The blend is charged for every member it runs. Times depend on the machine and are only "
               "comparable within one run.")
    for n in rm.tradeoff_notes(cs, static, "train_seconds"):
        if n["expensive"] or n["negligible"]:
            cost_txt = "a very large increase in" if np.isinf(n["cost_x"]) else f"{n['cost_x']:.1f}x the"
            _card(ctx, "failure" if n["expensive"] else "noise", f"{n['from']} \u2192 {n['to']}",
                  f"{n['gain_pct']:+.1f}% error for {cost_txt} training time. "
                  + ("The gain is below the noise threshold." if n["negligible"] else "A small accuracy gain for a large jump in compute."))

    # ------------------------------------------------------------------ 4. decision engine
    section("4. Use-case decision engine",
            "Pick a scenario and constraints. Each strategy gets a verdict backed by episodes and seeds, or 'insufficient evidence'.")
    name = st.selectbox("Scenario", list(rm.SCENARIOS), key="rev_scn")
    scn = rm.SCENARIOS[name]
    st.caption(scn["about"])
    if scn["proxy"]:
        st.info("Approximation: " + scn["proxy"] + " Results for this scenario are capped at Preliminary.")
    cmean = C.groupby("Strategy").agg(sec=("train_seconds", "mean"),
                                      rows=("stored_rows", "max") if "stored_rows" in C else ("updates", "max"))
    k1, k2 = st.columns(2)
    hi = float(cmean["sec"].max())
    budget = k1.slider("Max training time per run (s)", 0.0, max(hi, 1.0), max(hi, 1.0), key="rev_budget") if hi > 0 else hi
    no_store = k2.checkbox("Cannot keep raw history (rules out replay and full retraining)", value=False, key="rev_nostore",
                           disabled="stored_rows" not in C)
    excluded = {s: f"Needs {cmean.loc[s, 'sec']:.0f}s of training, over the {budget:.0f}s budget."
                for s in strategies if s != static and cmean.loc[s, "sec"] > budget + 1e-9}
    if no_store and "stored_rows" in C:
        for s in strategies:
            if s != static and C.groupby("Strategy")["stored_rows"].max().get(s, 0) > 0:
                excluded.setdefault(s, "Must store labelled rows to update, which this scenario forbids.")
    per_seed = M.groupby("Strategy")["seed"].nunique()
    untested = {s: f"Evaluated in {int(per_seed[s])} of {n_seeds} seeds only." for s in strategies
                if per_seed.get(s, 0) < n_seeds and s != static}

    res, E = rm.scenario_eval(M, EP, scn["weights"], static, late_only=scn["late_only"], proxy_note=scn["proxy"],
                              untested=untested, excluded=excluded)
    head = rm.scenario_headline(res, E, static)
    if head["pick"] is None:
        _card(ctx, "insufficient", "No strategy can be recommended for this scenario",
              "Nothing met the evidence rules (enough episodes, seeds that agree, a gain above the threshold). "
              "That is a result, not an error. See the reasons below.")
    else:
        _card(ctx, head["level"], f"{head['pick']}",
              (f"Runner-up: {head['runner']}. {head['note'].capitalize()}." if head["runner"] else "No other strategy qualified.")
              + (" Preliminary results should not be written up as findings." if head["level"] == "preliminary" else ""))
    r = res.copy()
    r["_o"] = r["verdict"].map(VERDICT_ORDER)
    r = r.sort_values(["_o", "error"])
    r = r.rename(columns={"error": "Error", "static_error": "Static error", "gain_pct": "Gain vs Static (%)",
                          "verdict": "Verdict", "reason": "Evidence"})
    show_df(r[["Strategy", "Verdict", "Error", "Gain vs Static (%)", "Evidence"]].round(2),
            column_config={"Evidence": st.column_config.TextColumn(width="large")})

    # ------------------------------------------------------------------ 5. research conclusions
    section("5. Research conclusions",
            "Generated from the tables, never typed in. Shows consistency across seeds, where strategies fail, and what could not be tested.")
    concl = rm.research_conclusions(M, EP, C, static)
    order = {"failure": 0, "untestable": 1, "noise": 2, "supported": 3, "note": 4}
    for c in sorted(concl, key=lambda c: order.get(c["level"], 9)):
        st.markdown(f'{_pill(ctx, c["level"])} &nbsp; **{c["topic"]}** - {c["text"]}', unsafe_allow_html=True)
