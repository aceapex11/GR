# GRU drift detection and adaptation

Streaming experiment on a synthetic industrial-sensor dataset (60,000 hourly rows, 8 sensors,
6 labelled drift episodes + one harmless sensor glitch). A PyTorch GRU predicts
`health_deterioration`; a **classic-ML drift detector** decides when to adapt it.

```
python data_generator.py        # optional - run_experiment.py generates the data if missing
python run_experiment.py --fast # quick check (few epochs)
python run_experiment.py        # full training
streamlit run app.py            # dashboard; the sidebar button also runs the experiment
python tests/test_engine.py     # detector + streaming loop test (no PyTorch needed)
```

## Pipeline (per 600-row window): predict → detect → diagnose → adapt

* **Data drift** - per-sensor KS test on the window vs the normal history. Each sensor's alarm
  level is calibrated from normal blocks of the *same* window size (`max(0.15, 1.2 × worst normal KS)`);
  PSI is a second trigger. At least 3 sensors must move for 2 consecutive windows.
* **Relational drift** - Page-Hinkley on the clipped, normalised residual of a frozen poly-2 Ridge
  "old rule".
* **Diagnosis** - none / data / relational / both. One drift episode = one alarm; "recovery" when the
  diagnosis returns to none.
* **Adaptation** - a new alarm adapts every strategy for 3 windows (labels are used only after the
  window has been predicted).

The detector never uses the GRU's error, so a weak GRU cannot cause false alarms and all five
strategies receive identical alarms.

## Strategies

Static · Transfer (frozen GRU, retrain head) · Full retraining · Continual + replay 100 · Continual + replay 500

## Files

| file | purpose |
|---|---|
| `detectors.py` | KS / PSI / Page-Hinkley detector (model-independent) |
| `engine.py` | streaming loop |
| `gru_models.py` | GRU + the five strategies |
| `evaluation.py` | offline scoring against ground truth (episode precision/recall/delay, per-regime MAE) |
| `run_experiment.py` | runs everything, writes `results/*.csv` |
| `app.py` | Streamlit dashboard |
| `tests/test_engine.py` | detector test with stand-in models |

Ground-truth columns (`regime`, `data_drift`, `concept_drift`, `true_drift_point`) are used only for
scoring and for shading charts. The data is synthetic - do not present it as plant data.
