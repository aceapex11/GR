# Transfer vs Static: findings

**Evidence level.** PyTorch was not available when this was written, so the numbers below come from
`surrogate_study.py`: a numpy MLP stand-in (encoder layers = GRU, head = head) run through the REAL data,
detector and streaming engine, 3 seeds. The mechanisms transfer to the GRU; the exact numbers will not.
Confirm with `python diagnose_transfer.py` (real GRU).

Overall MAE, mean of 3 seeds (lower is better; stable-period noise floor is about 2.0):

| strategy | overall | stable | data drift | relational | both |
|---|---|---|---|---|---|
| Static | 6.03 | 2.31 | 3.25 | 10.66 | 10.71 |
| Original Transfer (head only, newest window, no rollback) | 6.08 | 6.56 | 5.16 | 5.17 | 6.62 |
| Replay 500 (as designed) | 5.15 | 3.96 | 4.82 | 6.61 | 6.25 |
| Full retraining | 5.64 | 2.70 | 3.88 | 10.04 | 8.56 |
| **New Transfer (top unfrozen, episode buffer, no old replay, rollback)** | **3.47** | 3.06 | 2.65 | 4.19 | 4.24 |

## Causes
1. **Never rolled back.** Without rollback the adapted model is far worse than Static in stable periods and in
   data drift (overall 7.7 in the stand-in). Biggest single effect.
2. **Old-regime replay hurts during relational drift.** Old labels contradict the new relation; replay
   variants reached about 6.6 in relational drift versus about 4.1 without it.
3. **Fitting one narrow window.** Windows are autocorrelated (AR(1), phi 0.85-0.95); a head fitted to one of them
   extrapolates badly. Training on the whole episode so far fixes this.
4. **Head-only is a small capacity limit.** Unfreezing the top layer is about 6% better than head-only.
   More layers per se are not needed for accuracy; a second layer just gives transfer something to freeze/unfreeze.

## What cannot be fixed by tuning
The first window after each episode starts and ends is predicted before the detector can know anything
(errors 5-11 vs 2.3). That is why stable MAE stays above Static (3.06 vs 2.31). A label-based rollback guard was
tested and changed nothing, because it fires in the same window as the detector's recovery flag.

## Other observations
* The target depends only on the current row (no lag terms), so the 12-step GRU memory is not needed;
  a smaller `seq_len` would train faster. Not changed, since it was not tested on the real GRU.
* Replay 100/500 and Full retraining are left as designed baselines, so they still lose to Static in some regimes.
