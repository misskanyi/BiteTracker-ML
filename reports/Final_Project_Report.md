# BiteTracker — Final Project Report

> All performance figures in this report were measured on the **anonymised Food4Education
> research dataset**, read live from its Google Sheet (5 schools, 722 usable school-day
> records, 24 January to 3 November 2025). Weather, cost and exam/event fields are not in
> that sheet and were held constant. See [Known Limitations](#known-limitations).

## Project Summary

BiteTracker is a machine-learning decision-support platform that forecasts school meal
demand, so schools, county governments, NGOs, and donors can plan procurement and budgets
proactively instead of reactively. It compares four regression models (Linear Regression,
Decision Tree, Random Forest, XGBoost), selects one by TimeSeriesSplit validation MAE on the
training period, checks it against two naive forecasting baselines on the held-out test period,
and exposes next-day/7-day/30-day forecasts through a Streamlit dashboard.

## Final Architecture

```
google_sheet_loader.py (Food4Education) / synthetic_data.py
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
        ├──▶ naive_baselines.py (persistence, 7-day mean)
        │
        ├──▶ model_training.py (4 default-hyperparameter ML models)
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
5. Score two naive forecasting baselines (Persistence / Last Observation, 7-Day Historical
   Mean) on the test period.
6. Train the four ML models with library defaults (reference only).
7. Tune Decision Tree/Random Forest/XGBoost with `TimeSeriesSplit` (5 folds) cross-validated
   `GridSearchCV`/`RandomizedSearchCV` on the training set only, and cross-validate Linear
   Regression on the same folds.
8. Select the final model by lowest TimeSeriesSplit validation MAE (validation RMSE
   tiebreak). The test period plays no part in selection.
9. Evaluate the selected model once on the untouched test period.

## Final Selected Model

> FOOD4EDUCATION RESEARCH RESULTS.

**XGBoost** (tuned) — selected by lowest TimeSeriesSplit validation MAE (75.92 ± 13.49)
among the four candidates. Linear Regression (78.49 ± 13.26) and Random Forest
(79.23 ± 18.57) are within one fold standard deviation, so the ranking is fragile. The current validation ranking, hyperparameters and all metrics are
regenerated on every run in `reports/Hyperparameter_Tuning_Report.md`,
`reports/metrics/validation_model_selection.csv` and `models/final_best_model_metadata.json`;
treat those files as the source of truth.

## Performance Metrics

Selected model vs naive baselines (test period 10 September to 3 November 2025, 148 rows):

| Method | Type | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Persistence / Last Observation | Naive baseline | 39.07 | 69.10 | 0.994 |
| 7-Day Historical Mean | Naive baseline | 58.10 | 119.79 | 0.981 |
| **XGBoost (selected)** | ML model | **51.64** | **88.80** | **0.989** |

**The selected model does not beat the persistence baseline**: its test MAE is 12.57 meals
(32.2%) worse than repeating each school's previous day. It beats the 7-day mean by 6.46
meals (11.1%). No ML candidate beat persistence on the test period (best: Random Forest,
test MAE 49.54). The other candidates' test metrics are in
`reports/metrics/model_comparison.csv` for research comparison; they were not used to select
the model. R² is above 0.98 for every method, including persistence, so it does not
separate the models.

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
principally, on the Food4Education data the model does not yet beat the persistence baseline.

## Decision-Support Capabilities (Phase 2)

The original 6-page ML demonstration was extended into a 13-page decision-support product
without altering the selected model, its features, or the pipeline that trains it. New
capabilities, each backed by a dedicated `src/` module and covered by unit tests:

- **Specific Date Prediction** (`src/forecasting.predict_specific_date`) — a single-point
  estimate for one exact future date, with user-supplied context (attendance, weather, term,
  holiday/exam/event) explicitly separated from auto-derived values (day-of-week, month,
  lag/rolling history from the school's own past records).
- **Safety buffer & decision-support bundle** (`src/decision_support.py`) — recommended
  preparation quantity = predicted demand + a transparent buffer driven by recent demand
  volatility (`rolling_std_7`) and forecast horizon, with qualitative risk (Low/Moderate/High)
  and uncertainty (Lower/Moderate/Higher) labels — never a fabricated confidence percentage.
- **Scenario Simulator** — baseline-vs-what-if comparison for one school, explaining the result
  in decision-support language without claiming causality from correlation.
- **Forecast Calendar** — a 14/30-school-day planning table with per-day risk and cost, plus a
  day-level decision-support detail view.
- **School Comparison & Resource Allocation** (`src/resource_allocation.py`) — cross-school
  metrics (demand, attendance, waste, cost, risk) and a demand-vs-planned-capacity gap analysis,
  framed as a recommendation for review, never an automatic reallocation instruction.
- **Budget Planning** (`src/budgeting.py`) — daily/weekly/monthly KSh cost projections from
  predicted demand and average cost-per-meal, plus historical-vs-forecast expenditure charting.
- **Upload School Data & validation** (`src/data_validation.py`) — CSV/Excel upload, a
  transparent 0–100 data-quality score (completeness, duplicates, date validity, coverage,
  column richness, numeric validity), and a forecast-readiness verdict (READY / LIMITED /
  NOT READY) based on how much history a school has. Uploads never overwrite the synthetic
  dataset and never trigger automatic retraining — the workflow is strictly
  upload → validate → process → assess readiness → forecast where compatible.

All of the above are decision-support heuristics layered on top of the existing model's raw
prediction — they do not change what the model was trained on or how it predicts.

## Known Limitations

- On the Food4Education test period no ML model beats the persistence baseline.
- Forecast error compounds over longer horizons (recursive multi-step prediction).
- No true future weather/attendance forecast — day 2+ always uses a historical average.
- The safety buffer (`src/decision_support.py`) is a documented heuristic based on recent
  volatility and horizon, not a statistically fitted prediction interval.
- Only 5 Food4Education schools and under one school year of history — limited data for
  the longer lag features and cross-school generalization.
- The Food4Education sheet has no weather, cost or exam/event fields; they are held constant.
- PostgreSQL support is prepared in `config/settings.py` but not wired into the pipeline —
  the app runs entirely on CSV files.
- Uploaded schools need 14+ daily records before any prediction is possible (lag-14 feature
  requirement) and 60+ to be marked forecast-ready; a real school with a very different feature
  distribution than the synthetic training schools may still produce misleading predictions even
  once "ready" — `src/data_validation.check_feature_compatibility` flags the detectable cases only.

## Future Improvements

- Add a real school calendar, weather source and exam/event dates for the Food4Education
  schools, collect a longer history, and re-run baseline + tuning against persistence.
- Add genuine weather/attendance forecast inputs instead of historical-average fallbacks.
- Fit the safety buffer to each model's measured residual spread instead of the current
  volatility/horizon heuristic.
- Add a controlled retraining workflow for uploaded school data.
- Evaluate SHAP for local (per-prediction) explainability if it can be added without
  destabilizing the app.
- Wire in the prepared PostgreSQL schema for multi-user, real-time data.
- Add authentication and role-based views (school / county / NGO).
