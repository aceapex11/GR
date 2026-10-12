"""Checks the review analysis on a synthetic stream (numpy/pandas only). Run:  python test_review_metrics.py"""
import numpy as np
import pandas as pd

import review_metrics as rm

STATIC, W, WIN = "Static (no adaptation)", 10, 60


def build(seeds, drift_types=("data_drift", "relational_drift", "both") * 2):
    n_train, n_win = 100, WIN
    n = n_train + n_win * W
    # one episode of 4 windows every 8 windows, cycling through the requested drift types
    regime = np.array(["stable"] * n, dtype=object)
    dd, cc = np.zeros(n, int), np.zeros(n, int)
    for k, t in enumerate(drift_types):
        a = n_train + (3 + 8 * k) * W
        b = a + 4 * W
        regime[a:b] = t
        dd[a:b] = int(t in ("data_drift", "both"))
        cc[a:b] = int(t in ("relational_drift", "both"))
    df = pd.DataFrame({"y": 50 + np.arange(n) * 0.0, "regime": regime, "dd": dd, "cc": cc})
    start = np.arange(n_train, n, W)
    win = pd.DataFrame({"start": start, "end": start + W})
    M, E = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = np.arange(n_train, n)
        drift = (dd | cc)[idx].astype(bool)
        mk = lambda stable, drifted: 50 + np.where(drift, drifted, stable) * rng.choice([-1, 1], len(idx)) \
            * (1 + 0.05 * rng.normal(size=len(idx)))
        pred = pd.DataFrame({STATIC: mk(1.0, 8.0), "Good": mk(1.0, 3.0), "Costly": mk(1.0, 2.9), "Harmful": mk(1.0, 10.0)}, index=idx)
        M.append(rm.metrics_long(pred, df["y"], df["regime"], s))
        E.append(rm.episode_table(win, pred, df["y"], df["dd"], df["cc"], s))
    return pd.concat(M), pd.concat(E)


def cost(seeds):
    rows = []
    for s in seeds:
        for name, t in [(STATIC, 0.0), ("Good", 20.0), ("Costly", 400.0), ("Harmful", 25.0)]:
            rows.append({"seed": s, "Strategy": name, "updates": 0 if t == 0 else 30, "train_samples": 0 if t == 0 else 9000,
                         "train_seconds": t, "memory_mb": 0.05, "rollbacks": 0, "includes_members": False})
    return pd.DataFrame(rows)


def verdicts(res):
    return dict(zip(res["Strategy"], res["verdict"]))


def main():
    sc = rm.SCENARIOS["Temporary drift with recovery"]
    # 3 seeds, 6 episodes (2 of each type): the evidence rules should allow a real verdict
    M, E = build([0, 1, 2])
    assert E.groupby("regime")["episode"].nunique().min() == 2
    res, Emat = rm.scenario_eval(M, E, sc["weights"], STATIC)
    v = verdicts(res)
    assert v[STATIC] == "Reference"
    assert v["Good"] == "Supported" and v["Costly"] == "Supported", v
    assert v["Harmful"] in ("Not supported", "No clear difference"), v
    print("3 seeds ->", v)

    # one seed can never be Supported
    M1, E1 = build([0])
    v1 = verdicts(rm.scenario_eval(M1, E1, sc["weights"], STATIC)[0])
    assert v1["Good"] == "Preliminary", v1

    # a scenario that depends on a drift type with <2 episodes must say "insufficient evidence"
    M2, E2 = build([0, 1, 2], drift_types=("data_drift", "data_drift", "relational_drift", "both", "both"))
    v2 = verdicts(rm.scenario_eval(M2, E2, rm.SCENARIOS["Recurring concept"]["weights"], STATIC)[0])
    assert v2["Good"] == "Insufficient evidence", v2

    # proxy scenarios are capped at Preliminary; excluded / untested strategies are reported as such
    pr = rm.SCENARIOS["Permanent concept drift"]
    v3 = verdicts(rm.scenario_eval(M, E, pr["weights"], STATIC, late_only=True, proxy_note=pr["proxy"],
                                   excluded={"Costly": "over budget"}, untested={"Harmful": "never run"})[0])
    assert v3["Good"] == "Preliminary" and v3["Costly"] == "Excluded" and v3["Harmful"] == "Insufficient evidence", v3

    head = rm.scenario_headline(res, Emat, STATIC)
    assert head["level"] == "supported" and head["pick"] in ("Good", "Costly"), head
    assert "indistinguishable" in head["note"] or "clear lead" in head["note"], head

    # dashboard views run and make sense
    rt = rm.rank_table(M, "both")
    assert rt.iloc[0]["Strategy"] in ("Good", "Costly") and rt["n_seeds"].iloc[0] == 3
    gt = rm.gain_table(M, STATIC)
    assert gt[(gt.Strategy == "Good") & (gt.Phase == "overall")]["gain_pct"].iloc[0] > 3
    assert rm.persistence_table(E, STATIC).shape[0] == 4
    cs = rm.cost_summary(cost([0, 1, 2]), M, STATIC)
    assert set(cs.loc[cs.on_frontier, "Strategy"]) == {STATIC, "Good", "Costly"}, cs
    notes = rm.tradeoff_notes(cs, STATIC)
    assert any(nn["to"] == "Costly" and nn["expensive"] for nn in notes), notes
    concl = rm.research_conclusions(M, E, cost([0, 1, 2]), STATIC)
    topics = {c["topic"] for c in concl}
    assert {"Consistency", "Failure", "Scope"} <= topics, concl
    c1 = rm.research_conclusions(M1, E1, cost([0]), STATIC)
    assert any(c["level"] == "untestable" and "seed" in c["text"] for c in c1)
    print("OK")


if __name__ == "__main__":
    main()
