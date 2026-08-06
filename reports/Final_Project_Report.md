# BiteTracker — Final Project Report

> All performance figures in this report were measured on the **synthetic development
> dataset** (`src/synthetic_data.py`). They demonstrate that the full pipeline — data
> generation, cleaning, feature engineering, training, tuning, forecasting, and the
> Streamlit app — works correctly end to end. They are not a claim about real-world
> forecasting accuracy; see [Known Limitations](#known-limitations).

## Project Summary

BiteTracker is a machine-learning decision-support platform that forecasts school meal
demand, so schools, county governments, NGOs, and donors can plan procurement and budgets
proactively instead of reactively. It compares four regression models (Linear Regression,
Decision Tree, Random Forest, XGBoost), selects the best one by measured test-set error,
and exposes next-day/7-day/30-day forecasts through a Streamlit dashboard.

## Final Architecture

```
synthetic_data.py / real CSV
        │
        ▼
data_loader.py (schema validation)
        │
        ▼
preprocessing.py (clean, impute, cap outliers, encode)
        │
        ▼
feature_engineering.py (lag/rolling features, target)
        │
        ├──▶ model_training.py (4 baseline models)
        │
        ├──▶ model_tuning.py (GridSearchCV/RandomizedSearchCV + TimeSeriesSplit)
        │            │
        │            ▼
        │      final_best_model.joblib
        │            │
        └──▶ forecasting.py (recursive multi-horizon prediction)
                     │
                     ▼
        Streamlit app (Home, Dashboard, Forecast, Model Performance,
                        Data Explorer, About)
```

Every stage is a standalone, testable module in `src/`, orchestrated by thin CLI scripts
(`run_pipeline.py`, `run_eda.py`, `run_training.py`, `run_tuning.py`) so each phase can be
re-run independently and its output inspected before moving to the next.

## Machine Learning Workflow

1. Generate/validate data against a fixed schema.
2. Clean: parse dates, sort chronologically per school, drop duplicates, enforce
   non-negative values, impute missing values per school, cap outliers per school (IQR),
   one-hot encode categoricals with `drop_first=True`.
3. Engineer features: lag_1/7/14_meals, rolling_mean_7/14, rolling_std_7, attendance_rate,
   previous_attendance, calendar/holiday/exam/event flags — all computed per school with
   `.shift(1)` before any rolling window, so no feature ever sees its own day's target.
4. Split chronologically 80/20 — no shuffling.
5. Train four baseline models with library defaults.
6. Tune Decision Tree/Random Forest/XGBoost with `TimeSeriesSplit` (5 folds) cross-validated
   `GridSearchCV`/`RandomizedSearchCV` on the training set only; evaluate once on the
   untouched test set.
7. Select the final model by lowest test MAE (RMSE tiebreak, R² supporting evidence).

## Final Selected Model

**XGBoost** (tuned) — selected by lowest test MAE among all baseline and tuned candidates.

Best hyperparameters: `max_depth=2, n_estimators=200, learning_rate=0.05, subsample=0.7,
colsample_bytree=1.0, min_child_weight=1, reg_alpha=0, reg_lambda=10` (see
`models/best_hyperparameters.json`).

## Performance Metrics

Tuned models (test set, synthetic data):

| Model | Test MAE | Test RMSE | Test R² | Generalization |
|---|---|---|---|---|
| Decision Tree | 32.81 | 77.10 | 0.863 | Good Generalization |
| Random Forest | 33.95 | 74.62 | 0.872 | Overfitting (MAE gap) |
| **XGBoost** | **31.24** | **68.41** | **0.892** | Good Generalization |
| Linear Regression (untuned) | 41.43 | 82.49 | 0.843 | — |

Baseline → tuned improvement (Test MAE): Decision Tree −27.68, Random Forest −2.97,
XGBoost −13.80 — every tuned model beat its baseline; no result was reported as improved
when it wasn't (see `reports/metrics/baseline_vs_tuned_comparison.csv`).

## Forecasting Workflow

`src/forecasting.py` extends the one-step-ahead model to multi-day horizons recursively:
each day's prediction is fed back into the lag/rolling history as if observed, so day 2's
features are computed from day 1's *prediction*. Attendance/weather inputs are either
user-provided (day 1 only) or fall back to a recent 14-day average — every forecast row is
labelled with exactly which basis was used. Full detail in
`reports/Forecasting_Report.md`.

## Dashboard Overview

Six Streamlit pages, all reading live from `data/processed/` and `models/`:

- **Home** — project overview, stakeholders, pipeline-readiness status.
- **Dashboard** — KPI tiles (schools, enrolment, attendance, latest demand, waste rate)
  and historical demand/leftover charts.
- **Forecast** — school + horizon selection, optional attendance/weather assumptions,
  generates a labelled multi-day forecast with a chart against recent actuals.
- **Model Performance** — baseline vs. tuned tables and charts, actual-vs-predicted,
  residuals, feature importance.
- **Data Explorer** — filterable dataset preview, summary statistics, missing-value
  report, CSV download.
- **About** — methodology, ethics, synthetic-data disclosure, limitations.

## Testing Summary

**76 tests, 100% passing**, covering:

- Preprocessing (7 tests) — missing values, duplicates, outliers, dtype coercion, sorting.
- Feature engineering (9 tests) — lag/rolling correctness, cross-school isolation, target
  alignment, no-leakage guarantees.
- Evaluation (6 tests) — metric calculation, chronological split correctness.
- Model training (9 tests) — splitting, fitting, persistence, reproducibility.
- Model tuning (17 tests) — `TimeSeriesSplit` chronology, parameter grids, search
  execution, artifact persistence, baseline-vs-tuned comparison logic, best-model
  selection, reproducibility.
- Forecasting (12 tests) — 1/7/30-day horizons, recursive feature updates, assumption
  labelling, invalid-school handling, non-negative outputs, reproducibility.
- Streamlit app (16 tests, via `AppTest`) — every page loads exception-free, Forecast
  button produces real results, Model Performance loads saved metrics, Data Explorer
  filters correctly.

## Deployment Readiness

- `.streamlit/config.toml` — green/white/gold theme configured.
- `.streamlit/secrets.toml.example` — template provided; real secrets are gitignored.
- `requirements.txt` / `pyproject.toml` — every dependency pinned to a tested version.
- No hardcoded paths — all paths resolved via `pathlib` through `config/settings.py`.
- No circular imports (verified by importing every module fresh in isolation).
- Zero `pyflakes` warnings across `src/`, `app/`, `config/`, and `tests/`.
- App entry point confirmed to run with exactly `streamlit run app/Home.py`.

**Not yet production-ready for real deployment** — see Known Limitations below;
principally, the model has only ever seen synthetic data.

## Known Limitations

- Synthetic data only — no real-world accuracy claim.
- Forecast error compounds over longer horizons (recursive multi-step prediction).
- No true future weather/attendance forecast — day 2+ always uses a historical average.
- No safety-buffer/recommended-preparation-quantity logic on top of the raw point forecast.
- Only 3 synthetic schools, ~2 years of history — limited data for the longer lag
  features and cross-school generalization.
- Kenyan school calendar in the synthetic generator is approximated, not sourced from an
  authoritative gazette.
- PostgreSQL support is prepared in `config/settings.py` but not wired into the pipeline —
  the app runs entirely on CSV files.

## Future Improvements

- Replace synthetic data with real school records and re-run baseline + tuning from
  scratch — hyperparameters found here are fit to synthetic noise, not real patterns.
- Add genuine weather/attendance forecast inputs instead of historical-average fallbacks.
- Calibrate a safety-stock recommendation from each model's measured residual spread.
- Wire in the prepared PostgreSQL schema for multi-user, real-time data.
- Add authentication and role-based views (school / county / NGO).
