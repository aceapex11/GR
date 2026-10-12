"""Analysis helpers for the "Model Review & Use Cases" section.

Pure numpy / pandas (no torch, no streamlit) so the same code is used by the experiment, which
records per-seed tables, and by the dashboard, which interprets them.

Evidence rules (the advisor may answer "insufficient evidence"):
    MIN_EPISODES    a drift type needs at least this many separate episodes before any claim about it
    MIN_SEEDS       "Supported" needs this many seeds, all agreeing; fewer seeds can only be "Preliminary"
    MIN_GAIN_PCT    a difference from Static smaller than this (in %) is treated as no difference
    CONSISTENCY     share of episodes that must agree for a result to count as consistent
"""
import numpy as np
import pandas as pd

REGIMES = ["stable", "data_drift", "relational_drift", "both"]
DRIFT_REGIMES = ["data_drift", "relational_drift", "both"]
LABEL = {"overall": "Overall", "stable": "Stable", "data_drift": "Data drift",
         "relational_drift": "Relational drift", "both": "Combined drift"}

MIN_EPISODES = 2
MIN_SEEDS = 3
MIN_GAIN_PCT = 3.0
CONSISTENCY = 0.75

EP_COLUMNS = ["seed", "Strategy", "episode", "regime", "n_windows", "mae", "rmse", "early_mae", "late_mae",
              "peak_mae", "baseline_mae", "post_mae", "recovery_windows", "recovered"]


# ----------------------------------------------------------------------------- per-seed tables
def metrics_long(pred, actual, regime, seed=0):
    """MAE / RMSE for every strategy overall and in each regime. `pred` is indexed by row number."""
    idx = pred.index
    reg = regime.reindex(idx).fillna("stable").to_numpy()
    y = actual.reindex(idx).to_numpy(float)
    phases = [("overall", np.ones(len(idx), bool))] + [(r, reg == r) for r in REGIMES if (reg == r).any()]
    rows = []
    for c in pred.columns:
        e = y - pred[c].to_numpy(float)
        for ph, m in phases:
            if m.sum() < 2:
                continue
            rows.append({"seed": seed, "Strategy": c, "Phase": ph, "Rows": int(m.sum()),
                         "MAE": float(np.abs(e[m]).mean()), "RMSE": float(np.sqrt((e[m] ** 2).mean()))})
    return pd.DataFrame(rows)


def episode_table(win, pred, actual, data_flag, concept_flag, seed=0, tol=1.25, horizon=8, confirm=2):
    """One row per (strategy, true drift episode).

    win          stream log with `start` / `end` row numbers of every window
    pred         strategy predictions indexed by row number
    actual       true target indexed by row number
    data_flag / concept_flag   full-length ground-truth flags (row i = flag of row i)

    early_mae / late_mae   error in the first / last third of the episode's windows
    baseline_mae           the strategy's own median window error before the first episode
    post_mae               mean window error in the (up to `horizon`) windows after the episode
    recovery_windows       windows after the episode until error is <= tol x baseline for `confirm` windows in a row
    recovered              whether that happened (NaN when too few windows follow to tell)
    """
    dd, cc = np.asarray(data_flag, bool), np.asarray(concept_flag, bool)
    n = len(dd)
    st, en = win["start"].to_numpy(int), win["end"].to_numpy(int)
    W = len(win)
    wdd = np.array([dd[s:e].any() for s, e in zip(st, en)])
    wcc = np.array([cc[s:e].any() for s, e in zip(st, en)])
    wf = wdd | wcc
    eps, i = [], 0
    while i < W:
        if not wf[i]:
            i += 1
            continue
        j = i
        while j + 1 < W and wf[j + 1]:
            j += 1
        eps.append((i, j))
        i = j + 1
    if not eps:
        return pd.DataFrame(columns=EP_COLUMNS)

    y = actual.reindex(pred.index).to_numpy(float)
    rows_idx = np.asarray(pred.index, int)
    out = []
    for c in pred.columns:
        err = np.full(n, np.nan)
        err[rows_idx] = np.abs(y - pred[c].to_numpy(float))
        wm = np.array([np.nanmean(err[s:e]) if np.isfinite(err[s:e]).any() else np.nan for s, e in zip(st, en)])
        pre = wm[:eps[0][0]]
        pre = pre[np.isfinite(pre)]
        if len(pre):
            base = float(np.median(pre))
        else:
            calm = wm[~wf]
            base = float(np.nanmedian(calm)) if np.isfinite(calm).any() else np.nan
        for k, (a, b) in enumerate(eps):
            L = b - a + 1
            third = max(1, L // 3)
            seg = err[st[a]:en[b]]
            nxt = eps[k + 1][0] if k + 1 < len(eps) else W
            post = wm[np.arange(b + 1, min(b + 1 + horizon, nxt))]
            rec, recovered = np.nan, np.nan
            if len(post) >= confirm and np.isfinite(base):
                ok = post <= tol * base
                recovered = False
                for t in range(len(post) - confirm + 1):
                    if ok[t:t + confirm].all():
                        rec, recovered = t + 1, True
                        break
            has_d, has_c = wdd[a:b + 1].any(), wcc[a:b + 1].any()
            out.append({
                "seed": seed, "Strategy": c, "episode": k + 1,
                "regime": "both" if has_d and has_c else "data_drift" if has_d else "relational_drift",
                "n_windows": L, "mae": float(np.nanmean(seg)), "rmse": float(np.sqrt(np.nanmean(seg ** 2))),
                "early_mae": float(np.nanmean(wm[a:a + third])), "late_mae": float(np.nanmean(wm[b - third + 1:b + 1])),
                "peak_mae": float(np.nanmax(wm[a:b + 1])), "baseline_mae": base,
                "post_mae": float(np.nanmean(post)) if len(post) and np.isfinite(post).any() else np.nan,
                "recovery_windows": rec, "recovered": recovered})
    return pd.DataFrame(out, columns=EP_COLUMNS)


# ----------------------------------------------------------------------------- comparisons
def seed_matrix(M, phase, metric="MAE"):
    """seed x strategy table of one metric in one phase."""
    d = M[M["Phase"] == phase]
    return d.pivot(index="seed", columns="Strategy", values=metric)


def lead_assessment(M, phase, metric="MAE"):
    """Is the best strategy's lead over the runner-up larger than seed-to-seed noise?"""
    mat = seed_matrix(M, phase, metric)
    if mat.shape[1] < 2:
        return None
    mean = mat.mean().sort_values()
    best, second = mean.index[0], mean.index[1]
    n = len(mat)
    gap = 100 * (mean[second] - mean[best]) / mean[best] if mean[best] else np.nan
    out = {"best": best, "second": second, "best_value": float(mean[best]), "second_value": float(mean[second]),
           "gap_pct": float(gap), "n_seeds": n}
    if n < MIN_SEEDS:
        out["verdict"] = f"untestable with {n} seed{'s' if n != 1 else ''}"
    else:
        d = (mat[second] - mat[best]).to_numpy(float)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        clear = bool((d > 0).all() and d.mean() > 2 * se)
        out["verdict"] = "clear lead" if clear else "within noise"
    return out


def pareto_frontier(df, cost_col, err_col):
    """Names of strategies not beaten on both cost and error by another strategy (cheapest first)."""
    d = df.sort_values([cost_col, err_col])
    best, keep = np.inf, []
    for name, row in d.iterrows():
        if row[err_col] < best - 1e-12:
            keep.append(name)
            best = row[err_col]
    return keep


# ----------------------------------------------------------------------------- scenario engine
def scenario_eval(M, EP, weights, static, *, late_only=False, proxy_note=None, untested=None, excluded=None):
    """Judge every strategy for one scenario against Static, with an explicit evidence verdict.

    weights      {regime: weight} describing the scenario
    late_only    judge drift regimes by the last third of each episode (used for 'permanent drift')
    proxy_note   text explaining why the experiment can only approximate this scenario (caps verdict at Preliminary)
    untested     {strategy: reason}: behaviour in this scenario was never measured
    excluded     {strategy: reason}: ruled out by a constraint (for example a compute budget)
    """
    untested, excluded = untested or {}, excluded or {}
    present = set(M["Phase"].unique())
    w = {k: v for k, v in weights.items() if v > 0 and k in present}
    drift_w = [k for k in w if k != "stable"]
    seeds = sorted(M["seed"].unique())
    strategies = list(M["Strategy"].unique())
    tot = sum(w.values()) or 1.0
    w = {k: v / tot for k, v in w.items()}

    def err_for(seed, c):
        s = 0.0
        for k, wt in w.items():
            if late_only and k != "stable":
                e = EP[(EP["seed"] == seed) & (EP["Strategy"] == c) & (EP["regime"] == k)]["late_mae"]
                v = float(e.mean()) if len(e) else np.nan
            else:
                e = M[(M["seed"] == seed) & (M["Strategy"] == c) & (M["Phase"] == k)]["MAE"]
                v = float(e.iloc[0]) if len(e) else np.nan
            s += wt * v
        return s

    E = pd.DataFrame({c: [err_for(s_, c) for s_ in seeds] for c in strategies}, index=seeds)
    gain = 100 * (E[static].to_numpy()[:, None] - E.to_numpy()) / E[static].to_numpy()[:, None]
    gain = pd.DataFrame(gain, index=seeds, columns=strategies)

    # episode-level comparison with Static (seed-averaged, so each separate episode counts once)
    ep_sel = EP[EP["regime"].isin(drift_w)]
    n_eps = int(ep_sel["episode"].nunique()) if len(ep_sel) else 0
    by_reg = ep_sel.drop_duplicates("episode").groupby("regime").size().to_dict() if len(ep_sel) else {}
    # every drift type the scenario relies on needs enough separate episodes, not just the total
    thin = [f"{LABEL[k].lower()} ({by_reg.get(k, 0)})" for k in drift_w if by_reg.get(k, 0) < MIN_EPISODES]
    col = "late_mae" if late_only else "mae"
    epm = ep_sel.groupby(["Strategy", "episode"])[col].mean().unstack(0) if len(ep_sel) else pd.DataFrame()

    rows = []
    for c in strategies:
        g_mean = float(gain[c].mean())
        base = {"Strategy": c, "error": float(E[c].mean()), "static_error": float(E[static].mean()),
                "gain_pct": g_mean, "gain_std": float(gain[c].std(ddof=1)) if len(seeds) > 1 else np.nan,
                "n_episodes": n_eps, "n_seeds": len(seeds)}
        if c == static:
            rows.append({**base, "verdict": "Reference", "reason": "The baseline every other strategy is compared with.",
                         "ep_better": np.nan, "ep_worse": np.nan, "seeds_positive": np.nan})
            continue
        if c in excluded:
            rows.append({**base, "verdict": "Excluded", "reason": excluded[c], "ep_better": np.nan,
                         "ep_worse": np.nan, "seeds_positive": np.nan})
            continue
        if c in untested:
            rows.append({**base, "verdict": "Insufficient evidence", "reason": untested[c], "ep_better": np.nan,
                         "ep_worse": np.nan, "seeds_positive": np.nan})
            continue
        if thin:
            rows.append({**base, "verdict": "Insufficient evidence",
                         "reason": f"Too few separate drift episodes to judge: {', '.join(thin)}; at least {MIN_EPISODES} of each type are needed.",
                         "ep_better": np.nan, "ep_worse": np.nan, "seeds_positive": np.nan})
            continue
        better = worse = tot_e = 0
        if len(epm) and c in epm and static in epm:
            ratio = (epm[static] - epm[c]) / epm[static] * 100
            better, worse, tot_e = int((ratio >= MIN_GAIN_PCT).sum()), int((ratio <= -MIN_GAIN_PCT).sum()), int(ratio.notna().sum())
        seeds_pos = int((gain[c] >= MIN_GAIN_PCT).sum())
        pos = g_mean >= MIN_GAIN_PCT and (tot_e == 0 or better / tot_e >= CONSISTENCY)
        neg = g_mean <= -MIN_GAIN_PCT and (tot_e == 0 or worse / tot_e >= CONSISTENCY)
        ev = f"{g_mean:+.0f}% vs Static"
        if tot_e:
            ev += f", better in {better} of {tot_e} episodes"
        ev += f", positive in {seeds_pos} of {len(seeds)} seed{'s' if len(seeds) != 1 else ''}"
        if pos:
            full = len(seeds) >= MIN_SEEDS and seeds_pos == len(seeds) and not proxy_note
            verdict = "Supported" if full else "Preliminary"
            if verdict == "Preliminary":
                why = proxy_note if proxy_note else f"only {len(seeds)} seed{'s' if len(seeds) != 1 else ''}; {MIN_SEEDS} agreeing seeds are needed"
                ev += f". Preliminary: {why}"
        elif neg:
            verdict = "Not supported"
            ev += ". Consistently worse than Static"
        else:
            verdict = "No clear difference"
            ev += ". Mixed or smaller than the noise threshold"
        rows.append({**base, "verdict": verdict, "reason": ev, "ep_better": better, "ep_worse": worse,
                     "seeds_positive": seeds_pos})
    res = pd.DataFrame(rows)
    return res, E


def scenario_headline(res, E, static):
    """One honest sentence-level conclusion for a scenario."""
    ok = res[res["verdict"].isin(["Supported", "Preliminary"])].sort_values("error")
    if ok.empty:
        return {"level": "insufficient", "pick": None, "runner": None, "note": None}
    top = ok.iloc[0]
    level = "supported" if (ok["verdict"] == "Supported").any() and top["verdict"] == "Supported" else "preliminary"
    runner, note = None, None
    if len(ok) > 1:
        runner = ok.iloc[1]["Strategy"]
        n = len(E)
        if n >= MIN_SEEDS:
            d = (E[runner] - E[top["Strategy"]]).to_numpy(float)
            se = d.std(ddof=1) / np.sqrt(n)
            note = "clear lead over the runner-up" if (d > 0).all() and d.mean() > 2 * se else \
                   "the runner-up is statistically indistinguishable, so choose between them on cost"
        else:
            gap = 100 * (ok.iloc[1]["error"] - top["error"]) / top["error"]
            note = f"{gap:.0f}% ahead of the runner-up, but with {n} seed{'s' if n != 1 else ''} this cannot be tested"
    return {"level": level, "pick": top["Strategy"], "runner": runner, "note": note}


# ----------------------------------------------------------------------------- dashboard views
def rank_table(M, phase, metric="MAE"):
    """Per-strategy mean +- std over seeds in one phase, ranked, with the seed-to-seed rank stability."""
    mat = seed_matrix(M, phase, metric)
    if mat.empty:
        return pd.DataFrame()
    ranks = mat.rank(axis=1, method="min")
    out = pd.DataFrame({"Strategy": mat.columns, "mean": mat.mean().to_numpy(),
                        "std": (mat.std(ddof=1) if len(mat) > 1 else mat.mean() * np.nan).to_numpy(),
                        "best_in_seeds": (ranks == 1).sum().to_numpy(), "n_seeds": len(mat)})
    out = out.sort_values("mean").reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def gain_table(M, static, metric="MAE"):
    """Strategy x phase: mean % improvement over Static (positive = better), its std, and seeds that agree."""
    rows = []
    for ph in M["Phase"].unique():
        mat = seed_matrix(M, ph, metric)
        if static not in mat:
            continue
        g = 100 * (mat[static].to_numpy()[:, None] - mat.to_numpy()) / mat[static].to_numpy()[:, None]
        g = pd.DataFrame(g, index=mat.index, columns=mat.columns)
        for c in mat.columns:
            if c == static:
                continue
            rows.append({"Strategy": c, "Phase": ph, "gain_pct": float(g[c].mean()),
                         "gain_std": float(g[c].std(ddof=1)) if len(g) > 1 else np.nan,
                         "seeds_better": int((g[c] >= MIN_GAIN_PCT).sum()),
                         "seeds_worse": int((g[c] <= -MIN_GAIN_PCT).sum()), "n_seeds": len(g)})
    return pd.DataFrame(rows)


def persistence_table(EP, static):
    """Does the improvement last, and how fast does error come back? One row per strategy (drift episodes).

    held_pct       how much of the early-episode error is still gone by the end (late vs early MAE; positive =
                   error fell during the episode, negative = it got worse while the drift continued)
    vs_static_late late-episode MAE relative to Static's (percent better; positive = better)
    post_ratio     error in the windows after the episode, as a multiple of the strategy's own pre-drift baseline
    recovered      share of episodes (with enough following windows to tell) that returned to baseline
    recovery_med   median windows until error was back to baseline, over episodes that recovered
    """
    if EP is None or EP.empty:
        return pd.DataFrame()
    g = EP.groupby(["Strategy", "episode"], as_index=False).mean(numeric_only=True)
    s_late = g[g["Strategy"] == static].set_index("episode")["late_mae"]
    rows = []
    for c, d in g.groupby("Strategy"):
        d = d.set_index("episode")
        known = d["recovered"].dropna()
        rec_w = d.loc[d["recovered"] >= 0.5, "recovery_windows"].dropna()
        late_vs = 100 * (s_late.reindex(d.index) - d["late_mae"]) / s_late.reindex(d.index)
        rows.append({"Strategy": c, "episodes": len(d),
                     "held_pct": float((100 * (d["early_mae"] - d["late_mae"]) / d["early_mae"]).mean()),
                     "vs_static_late": float(late_vs.mean()) if c != static else np.nan,
                     "post_ratio": float((d["post_mae"] / d["baseline_mae"]).mean(skipna=True)),
                     "recovered": float((known >= 0.5).mean()) if len(known) else np.nan,
                     "n_judged": int(len(known)),
                     "recovery_med": float(rec_w.median()) if len(rec_w) else np.nan})
    return pd.DataFrame(rows)


def cost_summary(C, M, static, phase="overall", metric="MAE"):
    """One row per strategy: mean error, mean cost, gain over Static, Pareto flag and cost multiple vs Static."""
    err = seed_matrix(M, phase, metric).mean().rename("error")
    cost = C.groupby("Strategy").agg(updates=("updates", "mean"), train_samples=("train_samples", "mean"),
                                    train_seconds=("train_seconds", "mean"), memory_mb=("memory_mb", "mean"),
                                    includes_members=("includes_members", "max"))
    d = cost.join(err).reset_index().rename(columns={"index": "Strategy"})
    d["gain_pct"] = 100 * (d.loc[d["Strategy"] == static, "error"].iloc[0] - d["error"]) / \
        d.loc[d["Strategy"] == static, "error"].iloc[0]
    d = d.set_index("Strategy")
    d["on_frontier"] = d.index.isin(pareto_frontier(d, "train_seconds", "error"))
    return d.reset_index()


def tradeoff_notes(cs, static, cost_col="train_seconds"):
    """Plain-language steps along the cost/error frontier; flags steps where cost grows much faster than accuracy."""
    f = cs[cs["on_frontier"]].sort_values(cost_col)
    notes = []
    for a, b in zip(f.iloc[:-1].itertuples(), f.iloc[1:].itertuples()):
        gain = 100 * (a.error - b.error) / a.error if a.error else np.nan
        mult = getattr(b, cost_col) / getattr(a, cost_col) if getattr(a, cost_col) > 0 else np.inf
        flag = gain < 10 and mult >= 3          # small accuracy gain for a large jump in compute
        notes.append({"from": a.Strategy, "to": b.Strategy, "gain_pct": float(gain), "cost_x": float(mult),
                      "negligible": bool(gain < MIN_GAIN_PCT), "expensive": bool(flag)})
    return notes


SCENARIOS = {
    "Permanent concept drift": dict(
        weights={"relational_drift": 1, "both": 1}, late_only=True,
        about="The relationship between sensors and target changes and stays changed. Judged on the last third of each drift episode, "
              "after the model has had time to adapt.",
        proxy="No episode in this stream is a pure permanent change that is also repeated; late-episode error is used as a proxy."),
    "Temporary drift with recovery": dict(
        weights={"stable": 1, "data_drift": 1, "relational_drift": 1, "both": 1}, late_only=False,
        about="Drift arrives, then the original behaviour returns. Judged on the whole stream: drift periods and the stable periods around them.",
        proxy=None),
    "Recurring concept": dict(
        weights={"relational_drift": 1}, late_only=False,
        about="The same changed relationship comes back later. Judged on relational-drift episodes.",
        proxy="The stream repeats one concept only once, so 'recurring' is approximated by relational-drift episodes."),
    "Sensor (data) drift only": dict(
        weights={"data_drift": 1}, late_only=False,
        about="Inputs shift but the sensor-to-target rule is unchanged.", proxy=None),
    "Mostly stable, rare drift": dict(
        weights={"stable": 0.85, "data_drift": 0.05, "relational_drift": 0.05, "both": 0.05}, late_only=False,
        about="Most of the time nothing changes. Penalises strategies that hurt stable-period accuracy.", proxy=None),
}


def research_conclusions(M, EP, cost, static):
    """Evidence-tagged statements generated from the tables. Each: {level, topic, text}."""
    out = []
    seeds = sorted(M["seed"].unique())
    n = len(seeds)
    phases = [p for p in REGIMES if p in set(M["Phase"])]
    for ph in phases:
        la = lead_assessment(M, ph)
        if not la:
            continue
        lv = {"clear lead": "supported", "within noise": "noise"}.get(la["verdict"], "untestable")
        txt = (f"{LABEL[ph]}: {la['best']} has the lowest MAE ({la['best_value']:.2f}), {la['gap_pct']:.0f}% below "
               f"{la['second']}; this is a {la['verdict']} across {n} seed{'s' if n != 1 else ''}.")
        out.append({"level": lv, "topic": "Consistency", "text": txt})
    gt = gain_table(M, static)
    for r in gt[(gt["Phase"].isin(DRIFT_REGIMES)) & (gt["seeds_worse"] == gt["n_seeds"]) & (gt["gain_pct"] <= -MIN_GAIN_PCT)].itertuples():
        out.append({"level": "failure", "topic": "Failure",
                    "text": f"{r.Strategy} was worse than Static in {LABEL[r.Phase].lower()} in all {r.n_seeds} seeds "
                            f"({r.gain_pct:+.0f}%)."})
    for r in gt[(gt["Phase"] == "stable") & (gt["gain_pct"] <= -MIN_GAIN_PCT)].itertuples():
        out.append({"level": "failure", "topic": "Failure",
                    "text": f"{r.Strategy} degrades stable-period accuracy by {-r.gain_pct:.0f}% relative to Static."})
    pt = persistence_table(EP, static)
    if len(pt):
        slow = pt[(pt["Strategy"] != static) & (pt["recovered"] < 0.5) & (pt["n_judged"] > 0)]
        for r in slow.itertuples():
            out.append({"level": "failure", "topic": "Failure",
                        "text": f"{r.Strategy} returned to its pre-drift error in only {r.recovered * 100:.0f}% of "
                                f"{r.n_judged} judgeable episodes."})
    ep_by = EP.drop_duplicates("episode").groupby("regime").size().to_dict() if len(EP) else {}
    for k in DRIFT_REGIMES:
        if ep_by.get(k, 0) < MIN_EPISODES:
            out.append({"level": "untestable", "topic": "Coverage",
                        "text": f"{LABEL[k]}: {ep_by.get(k, 0)} episode(s); at least {MIN_EPISODES} are needed before any claim."})
    if n < MIN_SEEDS:
        out.append({"level": "untestable", "topic": "Coverage",
                    "text": f"Only {n} seed{'s' if n != 1 else ''} recorded; no result can be called Supported until {MIN_SEEDS} agree."})
    out.append({"level": "note", "topic": "Scope",
                "text": "Seeds vary GRU initialisation and sampling only. The stream and detector are identical across seeds, "
                        "so agreement across seeds shows training stability, not generalisation to other streams."})
    return out
