# 🍽️ BiteTracker

**Proactive School Meal Planning: Integrating Time Series Forecasting to Provide Decision Support for Food Demand Prediction**

A machine-learning decision-support platform that forecasts school meal demand, helping schools, county governments, NGOs, and donors reduce food waste, prevent shortages, and plan procurement and budgets proactively rather than reactively.

> ⚠️ **Synthetic data notice:** the dataset shipped with this repository is synthetically generated (`src/synthetic_data.py`) because a real-world dataset was not available during development. All metrics in this README and the app are genuine measurements *on that synthetic data* — they validate that the pipeline works correctly, not real-world model accuracy. See [Limitations](#limitations).

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Decision-Support Features](#decision-support-features)
- [ML Pipeline](#ml-pipeline)
- [Models Compared](#models-compared)
- [Evaluation Metrics](#evaluation-metrics)
- [Screenshots](#screenshots)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)

---

## Motivation

School feeding programmes routinely swing between two costly failures: **food waste** (preparing more meals than attendance justifies) and **meal shortages** (under-preparing on days attendance spikes). Both stem from the same root cause — meal preparation decisions are made reactively, based on yesterday's numbers or intuition, not a forecast of tomorrow's actual demand. BiteTracker addresses this by turning historical attendance, weather, and calendar data into a next-day/next-week demand forecast that decision-makers can act on *before* the fact.

## Problem Statement

Kitchen and procurement staff need to decide, every school day, how many meals to prepare — without knowing tomorrow's attendance in advance. Rainfall, school holidays, exam periods, and special events all shift attendance in ways that are hard to anticipate by eye. BiteTracker frames this as a supervised regression problem: predict next-day `meals_served` per school from lagged historical demand, attendance patterns, weather, and calendar features.

## Objectives

- Build a reusable, testable data pipeline (clean → engineer features → train → tune → forecast) that works identically on synthetic or real data.
- Compare four regression approaches — Linear Regression, Decision Tree, Random Forest, XGBoost — on identical, leakage-safe, chronologically-split data.
- Select the best model by **measured** test-set error (MAE primary, RMSE tiebreak, R² supporting evidence), never by assumption.
- Provide multi-horizon forecasts (next day / 7 days / 30 days) with an honest accounting of what's assumed vs. measured.
- Ship a working, professional Streamlit dashboard for non-technical stakeholders.

## Features

- **Synthetic data generator** with believable domain relationships (rainfall depressing attendance, holidays halting service, exam-period effects) — clearly labelled, swappable for real data with zero code changes.
- **Leakage-safe feature engineering**: per-school lag (1/7/14-day) and rolling (7/14-day) meal-count features, computed strictly from past data.
- **Chronological 80/20 train-test split** — no shuffling, ever.
- **Baseline → tuned model comparison** with `GridSearchCV`/`RandomizedSearchCV` + `TimeSeriesSplit`, and an explicit overfitting classifier applied consistently to both stages.
- **Multi-horizon recursive forecasting** with transparent assumption labelling (user-provided vs. recent-average) for attendance/weather.
- **13-page Streamlit decision-support platform**: Home, Dashboard, Forecast, Specific Date Prediction, Scenario Simulator, Forecast Calendar, School Comparison, Resource Allocation, Budget Planning, Upload School Data, Data Explorer, Model Performance, About.
- **School data upload** with CSV/Excel support, transparent data-quality scoring, and forecast-readiness assessment — without retraining the model or overwriting the development dataset.
- **What-if scenario simulation**, cross-school comparison, resource-allocation gap analysis, and cost/budget projections in KSh.
- **128 automated tests** covering preprocessing, feature engineering, evaluation, model training, hyperparameter tuning, forecasting, decision-support logic, data validation, resource allocation, budgeting, and the Streamlit app itself (via `streamlit.testing.v1.AppTest`).

## Technology Stack

| Layer | Tools |
|---|---|
| Data & ML | Python 3.9+, Pandas, NumPy, scikit-learn, XGBoost, statsmodels |
| Visualization / App | Plotly, Streamlit |
| Persistence | Joblib (models), CSV (data), PostgreSQL (prepared, not yet wired in) |
| Testing | Pytest, `streamlit.testing.v1.AppTest` |
| Config | `python-dotenv`, `pathlib`-based settings module |

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ synthetic_data.py│───▶│  data_loader.py   │───▶│  preprocessing.py   │
│ (or real CSV)    │    │ (schema validate) │    │ (clean, impute, cap)│
└─────────────────┘    └──────────────────┘    └──────────┬─────────┘
                                                            ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  forecasting.py  │◀───│  model_tuning.py  │◀───│feature_engineering.py│
│ (recursive multi- │    │ (GridSearchCV /   │    │ (lag/rolling, target)│
│  horizon predict) │    │  RandomizedSearchCV│    └──────────┬─────────┘
└────────┬─────────┘    │  + TimeSeriesSplit)│               ▼
         │              └──────────────────┘    ┌────────────────────┐
         │                        ▲              │ model_training.py   │
         │                        └──────────────│ (baseline 4 models) │
         ▼                                       └────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                    Streamlit App (app/Home.py + pages/)              │
│  Home · Dashboard · Forecast · Model Performance · Data Explorer ·   │
│  About — all reading from data/processed/ and models/                │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
BiteTracker/
├── app/                          # Streamlit application
│   ├── Home.py                   # Entry point — streamlit run app/Home.py
│   ├── theme.py                  # Shared styling, page headers, chart theming
│   ├── data_access.py            # Cached data/model loaders
│   ├── assets/
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Specific_Date_Prediction.py
│       ├── 4_Scenario_Simulator.py
│       ├── 5_Forecast_Calendar.py
│       ├── 6_School_Comparison.py
│       ├── 7_Resource_Allocation.py
│       ├── 8_Budget_Planning.py
│       ├── 9_Upload_School_Data.py
│       ├── 10_Data_Explorer.py
│       ├── 11_Model_Performance.py
│       └── 12_About.py
├── config/
│   └── settings.py               # Central pathlib-based config, seeds, schema
├── data/
│   ├── raw/                      # Real data goes here (gitignored)
│   ├── processed/                # Pipeline output (gitignored)
│   ├── uploads/                  # User-uploaded datasets (gitignored)
│   └── sample/                   # Synthetic dataset (committed)
├── models/                       # Trained models + metadata (gitignored)
├── notebooks/                    # Exploratory notebooks (empty by default)
├── reports/
│   ├── figures/                  # Generated Plotly HTML charts (gitignored)
│   ├── metrics/                  # Generated CSV/JSON metrics (gitignored)
│   ├── EDA_Report.md
│   ├── Baseline_Model_Training_Report.md
│   ├── Hyperparameter_Tuning_Report.md
│   ├── Forecasting_Report.md
│   └── Final_Project_Report.md
├── src/
│   ├── synthetic_data.py         # Synthetic dataset generator
│   ├── data_loader.py            # Schema validation + flexible loading
│   ├── preprocessing.py          # Cleaning pipeline
│   ├── feature_engineering.py    # Lag/rolling features, target construction
│   ├── pipeline.py                # load → clean → engineer → save
│   ├── model_training.py         # Baseline model training
│   ├── model_tuning.py           # GridSearchCV/RandomizedSearchCV + TimeSeriesSplit
│   ├── model_evaluation.py       # Metrics, chronological split, overfitting classifier
│   ├── forecasting.py            # Multi-horizon recursive forecasting + specific-date prediction
│   ├── decision_support.py       # Safety buffer, risk level, uncertainty labelling
│   ├── data_validation.py        # Upload parsing, validation, quality score, forecast readiness
│   ├── resource_allocation.py    # School comparison + demand-vs-capacity gap logic
│   ├── budgeting.py               # Cost projection from predicted demand + cost-per-meal
│   ├── visualization.py          # Reusable Plotly chart builders
│   ├── eda.py                    # Statistical summaries, data-quality report
│   └── utils.py                  # Logging, seeding
├── tests/                        # 128 automated tests
├── run_pipeline.py                # CLI: data pipeline
├── run_eda.py                     # CLI: EDA charts + reports
├── run_training.py                # CLI: baseline model training
├── run_tuning.py                  # CLI: hyperparameter tuning
├── .streamlit/
│   ├── config.toml                # App theme
│   └── secrets.toml.example       # Template — copy to secrets.toml, never commit real secrets
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Installation

```bash
git clone <this-repo-url>
cd BiteTracker

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running Locally

Run the pipeline stages in order (each is idempotent and safe to re-run):

```bash
python run_pipeline.py     # generate synthetic data (if missing) → clean → engineer features
python run_eda.py          # EDA charts + reports/EDA_Report.md
python run_training.py     # baseline models (Linear/Tree/Forest/XGBoost) → models/, reports/
python run_tuning.py       # GridSearchCV/RandomizedSearchCV tuning → models/*_tuned.joblib
python -m pytest -q        # run the full test suite (76 tests)
```

## Running the Streamlit App

```bash
streamlit run app/Home.py
```

Open the printed local URL (typically `http://localhost:8501`) in your browser. See the **Troubleshooting** table below if it fails to start.

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` for streamlit/pandas/etc. | venv not activated, or deps not installed — see [Installation](#installation). |
| Pages show "Missing ❌" status | Run the pipeline scripts above in order first. |
| `FileNotFoundError: final_best_model.joblib` | Run `python run_tuning.py`. |
| Port 8501 already in use | `streamlit run app/Home.py --server.port 8502`. |

## Decision-Support Features

Beyond the raw ML pipeline, the app provides a set of practical planning tools built on top of
the trained model — for schools, county governments, NGOs, and donors, not just ML practitioners:

- **Specific Date Prediction** (`app/pages/3_Specific_Date_Prediction.py`) — an estimate for one
  exact future date and school, with explicit user-provided assumptions (attendance, rainfall,
  temperature, term, holiday/exam/event status) clearly distinguished from auto-derived values
  (day of week, month, lag/rolling history) — plus a decision-support summary (predicted demand,
  recommended preparation, safety buffer, risk level) and a model-wide feature-importance chart.
- **Scenario Simulator** (`app/pages/4_Scenario_Simulator.py`) — compares a baseline next-day
  forecast against a user-defined what-if scenario (attendance change %, rainfall, temperature,
  holiday/exam/event), with a plain-language explanation that stops short of claiming causality.
- **Forecast Calendar** (`app/pages/5_Forecast_Calendar.py`) — a 14/30-school-day planning table
  (predicted demand, recommended preparation, risk, estimated cost) with a day-selector for
  full decision-support detail on any date.
- **School Comparison** (`app/pages/6_School_Comparison.py`) — side-by-side demand, attendance,
  waste, and cost metrics across selected schools, aimed at NGOs/donors/county governments.
- **Resource Allocation** (`app/pages/7_Resource_Allocation.py`) — forecast demand vs
  manually-entered planned capacity per school, surfacing shortage/surplus gaps as a
  **decision-support recommendation** (never an automatic reallocation command).
- **Budget Planning** (`app/pages/8_Budget_Planning.py`) — translates predicted demand and
  average cost-per-meal into daily/weekly/monthly cost estimates (KSh), plus a
  historical-vs-forecast expenditure chart.
- **Upload School Data** (`app/pages/9_Upload_School_Data.py`) — lets a school/NGO bring its own
  historical CSV/Excel dataset. Runs a full validation pass (schema, dates, duplicates, missing
  values, numeric ranges, coverage) and a transparent 0–100 **Data Quality Score**, then a
  **forecast readiness** verdict (READY / LIMITED / NOT READY, based on record count — see
  `src/data_validation.py`). Uploads are stored under `data/uploads/` (git-ignored), never
  overwrite the synthetic dataset, and never trigger automatic retraining.

### Forecast Assumptions

Every prediction distinguishes three kinds of input, shown consistently across pages:

- **Auto-derived** — day of week, month, and lag/rolling demand history (computed from the
  school's own past records).
- **User-provided** — attendance, rainfall, temperature, term, holiday/exam/event status.
- **Assumed default** — when a page falls back to a recent historical average because the user
  hasn't supplied a value (e.g. beyond day 1 of a multi-day forecast).

The **safety buffer** and **recommended preparation quantity** (`src/decision_support.py`) are a
transparent planning heuristic — *not* a statistical prediction interval — driven by the school's
recent demand volatility (`rolling_std_7`) and the forecast horizon (later days in a recursive
multi-day forecast carry more compounding error). Risk level (Low/Moderate/High) and uncertainty
labels (Lower/Moderate/Higher) are derived from that same buffer logic; no confidence percentage
is fabricated. See the module docstring for the exact formula.

### Data Privacy

The upload workflow only accepts aggregated, school-level daily records — never student names,
admission numbers, phone numbers, addresses, or other personally-identifying information. This is
stated explicitly on the Upload School Data page.

### Synthetic vs. Real Data

Every page shows a badge indicating whether it is looking at **Synthetic Development Data** or
**User-Uploaded Data** (`app/theme.py::page_header`'s `badge_text` parameter) — the two are never
mixed silently.

## ML Pipeline

1. **Data generation / loading** (`src/synthetic_data.py`, `src/data_loader.py`) — schema-validated CSV, synthetic by default, swappable for real data via `config.settings.ACTIVE_DATA_PATH`.
2. **Preprocessing** (`src/preprocessing.py`) — date parsing, chronological sort, duplicate removal, per-school missing-value imputation, per-school IQR outlier capping, one-hot encoding (with `drop_first=True` to avoid the dummy-variable trap).
3. **Feature engineering** (`src/feature_engineering.py`) — lag (1/7/14-day) and rolling (7/14-day mean, 7-day std) meal-count features computed per-school with `.shift(1)` before rolling, so today never sees itself; target is `meals_served.shift(-1)` per school.
4. **Chronological split** — last 20% of time-ordered rows held out; never shuffled.
5. **Baseline training** (`src/model_training.py`) — four regressors with library defaults, to establish an honest starting point.
6. **Hyperparameter tuning** (`src/model_tuning.py`) — `TimeSeriesSplit` (5 folds) cross-validation on the training set only; the test set is never exposed during tuning. Decision Tree uses exhaustive `GridSearchCV`; Random Forest and XGBoost use `RandomizedSearchCV` (documented in `reports/Hyperparameter_Tuning_Report.md` — the full grids were too expensive for a development machine).
7. **Forecasting** (`src/forecasting.py`) — recursive one-step-ahead prediction extended to multi-day horizons, with attendance/weather assumptions labelled `user-provided` or `recent 14-day average`.

## Models Compared

| Model | Role |
|---|---|
| Linear Regression | Baseline, unchanged/untuned reference |
| Decision Tree Regressor | Captures non-linear splits (e.g. holiday thresholds) |
| Random Forest Regressor | Ensemble, generally more robust than a single tree |
| XGBoost Regressor | Gradient boosting, strong on tabular data with mixed feature types |

**Final selected model (on synthetic data): XGBoost** — Test MAE 31.24, RMSE 68.41, R² 0.892, selected by lowest test MAE (not R² alone). See `reports/Hyperparameter_Tuning_Report.md` for the full baseline-vs-tuned comparison and overfitting analysis.

## Evaluation Metrics

- **MAE** (Mean Absolute Error) — primary metric; directly interpretable as "meals mispredicted on average".
- **RMSE** (Root Mean Squared Error) — secondary tiebreak; penalizes large errors more.
- **R²** — supporting evidence only; never used alone to select a model.
- **Generalization gap classification** — a shared heuristic (`src/model_evaluation.classify_generalization_gap`) flags "Overfitting" / "Moderate Gap" / "Good Generalization" consistently across baseline and tuned reports.

## Screenshots

> _Add screenshots here after running the app locally:_
> - `docs/screenshots/home.png`
> - `docs/screenshots/dashboard.png`
> - `docs/screenshots/forecast.png`
> - `docs/screenshots/model_performance.png`
> - `docs/screenshots/data_explorer.png`

## Ethical Considerations

- No student-level personally identifiable information is collected or required — no names, admission numbers, or phone numbers.
- Data is aggregated at the school-day level (enrolment, attendance counts, meal totals).
- Any future real dataset should use anonymized school identifiers, aggregated daily attendance/meal totals, secure storage, and comply with Kenya's **Data Protection Act, 2019**.

## Limitations

- **Synthetic data only** — no real-world accuracy claim is made; see the in-app "Synthetic Development Data" badge on every page.
- **Compounding forecast error** over longer horizons (recursive multi-step forecasting).
- **No true future weather/attendance forecast** — beyond day 1 of a forecast, attendance/weather always fall back to a recent historical average.
- **Safety buffer is a heuristic, not a statistical prediction interval** — it is derived from recent demand volatility and forecast horizon (`src/decision_support.py`), transparently documented, but not fitted to held-out residuals.
- **Only 3 synthetic schools, ~2 years of history** — limited data for lag_14/rolling_14 features and cross-school generalization.
- **Approximated Kenyan school calendar** in the synthetic generator, not sourced from an authoritative gazette.
- **Uploaded-school forecasting is gated by history length** — a newly-uploaded school needs at least 14 daily records before lag/rolling features (and therefore any prediction) are possible, and 60+ before being marked forecast-ready; the model was trained on the synthetic schools' feature distribution, so predictions for a very different real school should be treated cautiously (see `src/data_validation.py::check_feature_compatibility`).

## Future Work

- Swap in real school-level data and re-run baseline + tuning from scratch (hyperparameters found here were fit to synthetic noise and should not be reused as-is).
- Add true weather/attendance forecast inputs instead of historical-average fallbacks.
- Fit the safety buffer to each model's measured residual spread instead of the current volatility/horizon heuristic.
- Add a controlled retraining workflow for uploaded school data (currently upload → validate → process → assess readiness, with no automatic retraining).
- Consider SHAP for local (per-prediction) explainability if it doesn't destabilize the app — the current feature-importance chart is model-wide, not prediction-specific.
- Move from CSV to the prepared PostgreSQL schema for multi-user, real-time updates.
- Add authentication/role-based views for schools vs. county vs. NGO stakeholders.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Sharon Kanyi** — Final-year project, BiteTracker: Proactive School Meal Planning.
