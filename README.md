# BiteTracker

**A Machine Learning-Based School Meal Demand Forecasting and Resource Allocation Decision Support System**

Final-year project, BSc Informatics and Computer Science, Strathmore University.

> **Important: current dataset.** Every chart, model, metric and forecast in this version of the repository comes from a **synthetic (generated) development dataset**, `data/sample/synthetic_school_meals.csv`, created by `src/synthetic_data.py`. The anonymised Food4Education research dataset is **not** included in this repository, and this version of the code does not load it. See [Dataset](#dataset).

---

## Table of Contents

- [Overview](#overview)
- [Problem Being Addressed](#problem-being-addressed)
- [Dataset](#dataset)
- [Machine Learning](#machine-learning)
- [Forecasting](#forecasting)
- [Application Functionality](#application-functionality)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Local Setup](#installation-and-local-setup)
- [Configuration](#configuration)
- [Running the Machine-Learning Pipeline](#running-the-machine-learning-pipeline)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Deployment](#deployment)
- [Privacy and Data Use](#privacy-and-data-use)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Acknowledgement](#acknowledgement)
- [License / Academic Use](#license--academic-use)

---

## Overview

BiteTracker is a machine-learning decision-support system that forecasts how many meals a school is likely to need, and helps with planning the resources those meals require.

It learns from historical school-day meal records and shows its forecasts in a Django web application. The aim is to turn past meal-demand records into forward-looking information that helps staff plan meal preparation, capacity and budgets.

BiteTracker only **supports** decisions. It does not make final meal-preparation or resource-allocation decisions automatically. Every forecast in the app is labelled as an estimate, and all allocation outputs are worded as suggestions for staff to review.

## Problem Being Addressed

The number of learners who eat a school meal changes from day to day, so it is hard to know in advance how many meals will be needed.

- **Underestimating** demand can lead to meal shortages.
- **Overestimating** demand can waste food and money.

BiteTracker tests whether historical realised meal-demand data can be used to forecast future demand and support better planning.

## Dataset

### What the code uses today

The pipeline and app currently use a **synthetic dataset** that `src/synthetic_data.py` generates. It was built because a real dataset was not available during development. The committed file `data/sample/synthetic_school_meals.csv` contains:

| Property | Value |
|---|---|
| Rows | 1,545 school-day records |
| Schools | 3 fictional schools (`SCH001`, `SCH002`, `SCH003`) |
| Date range | 2023-01-02 to 2024-12-20 (school days, Mon–Fri) |

The generator creates the relationships between fields (for example, rainfall lowering attendance, or holidays stopping meal service) so that they look plausible. They are **not** fitted to real records. The generator also adds some data-quality problems on purpose (`_inject_data_quality_issues`) so that the cleaning steps have something to fix.

### Raw schema

`src/synthetic_data.py` defines the schema in `SCHEMA_COLUMNS`, and `src/data_loader.py` checks incoming data against it:

| Field | Meaning |
|---|---|
| `date` | School day of the record |
| `school_id`, `school_name` | School identifier and display name |
| `county`, `sub_county` | Location labels |
| `enrolled_students` | Enrolment for that school |
| `students_present`, `students_absent` | Daily attendance counts |
| `meals_prepared`, `meals_served`, `leftover_meals` | Daily meal counts |
| `average_cost_per_meal` | Cost per meal (KSh) |
| `rainfall_mm`, `temperature_c` | Daily weather values (stored in the dataset; no weather API is called) |
| `is_holiday`, `is_exam_period`, `is_school_event` | Calendar flags (0/1) |
| `school_term`, `day_of_week`, `month` | Calendar descriptors |

**Realised meal demand** is `meals_served`. Each processed row represents one school on one **target date**, and the **forecasting target** is `target_meals_served`: the `meals_served` recorded on that date (`config/settings.py` → `TARGET_COLUMN`). The features only use information available before that day's meals are served (see below).

### How enrolment is used

`enrolled_students` is **not** a direct model feature. It is used in two ways:

- To calculate the `previous_attendance_rate` feature (the previous school day's `students_present / enrolled_students`). The same-day `attendance_rate` is also computed for dashboards, but it is **not** a model feature.
- As reference information: the Dashboard lists each school's enrolment as recorded in the dataset (labelled as dataset enrolment, not verified enrolment).

The app does **not** compare predicted demand against enrolment to produce a "planning difference". `School.verified_enrolment` exists in the database for verified figures entered by an administrator, but it is not yet used in any calculation and is never used as a model feature.

### Engineered features

`src/feature_engineering.py` (`FEATURE_COLUMNS`) defines the 20 model inputs, computed per school. For a target date T, `src/forecasting.py` builds exactly the same features at prediction time, and `tests/test_feature_alignment.py` checks that the two match:

| Group | Features |
|---|---|
| Lagged demand | `lag_1_meals`, `lag_7_meals`, `lag_14_meals`: meals served 1, 7 and 14 school days before T |
| Rolling demand | `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`: over the school days before T (calculated after `.shift(1)`) |
| Attendance | `previous_attendance`, `previous_attendance_rate`: the school day before T. T's own attendance is excluded, because it isn't known in advance |
| Calendar (of T) | `month`, `is_holiday`, `is_exam_period`, `is_school_event`: known in advance from the school calendar |
| Encoded categories (of T) | `school_term_Term 2`, `school_term_Term 3`, `day_of_week_Tuesday` … `day_of_week_Friday` (one-hot, with Term 1 and Monday as the reference) |
| Weather (of T) | `rainfall_mm`, `temperature_c`: training uses the observed values as a stand-in for a weather forecast (an optimistic assumption); at prediction time these are user-entered or recent averages |

Each school's first 14 rows (without a full lag history) are dropped.

### Food4Education research dataset

The project was designed for use with anonymised historical school-feeding data that Food4Education provided for academic research. That dataset is **not committed to this repository**, and the current code does not load it or reference its field names. The pipeline expects the column schema above. To use a different dataset, set `BITETRACKER_DATA_PATH` to a CSV with the same columns (see [Configuration](#configuration)). A dataset with different column names would need code changes first.

**Integration status:** no Food4Education workbook has been placed in `data/raw/` yet, so it has not been audited or used. `data/raw/*` is git-ignored. The web application's data label is derived from the active model metadata and dataset path (`services/status_service.py`), so pages switch from **SYNTHETIC DEVELOPMENT DATA** only when a model trained on non-synthetic data is active. A 2025 Kenyan primary-school calendar (terms, half-terms, holidays) is prepared in `services/calendar_service.py` for the real-data pipeline; it is not used by the current synthetic model.

## Machine Learning

BiteTracker treats forecasting as a supervised regression problem: predict the meals served by a school on a target school day, using that day's calendar and information from earlier days.

### Models

`src/model_training.py` and `src/model_tuning.py` implement all four candidate ML models:

| Model | Library | Default-hyperparameter fit | Tuned |
|---|---|---|---|
| Linear Regression | scikit-learn | Yes | No tuning grid; cross-validated on the same `TimeSeriesSplit` folds |
| Decision Tree Regression | scikit-learn | Yes | `GridSearchCV` |
| Random Forest Regression | scikit-learn | Yes | `RandomizedSearchCV` (25 iterations) |
| XGBoost Regression | xgboost | Yes | `RandomizedSearchCV` (40 iterations) |

Linear Regression is a candidate ML model. It is **not** the naive forecasting baseline; see [Naive forecasting baselines](#naive-forecasting-baselines).

### Chronological evaluation

- **Train/test split:** chronological 80/20 split, with no shuffling (`TRAIN_TEST_SPLIT_RATIO = 0.8`). Earlier records are used for training and later records for testing.
- **Tuning and validation:** `TimeSeriesSplit` with 5 folds (`CV_N_SPLITS = 5`), run on the training period only. Every search scores MAE and RMSE on each validation fold and picks hyperparameters by MAE. Linear Regression is cross-validated on the same folds with the same scorers, so all four candidates get a comparable validation score.
- **Model selection** (`src/model_selection.py::select_model_by_validation`, called by `run_tuning.py::select_final_model`): the lowest **TimeSeriesSplit validation MAE** wins, with validation RMSE as the tiebreak. The selector only accepts `ValidationScore` records, which have no test-metric fields, so the test period cannot influence the choice.
- **Final assessment:** only after selection is the chosen model evaluated once on the held-out test period (MAE, RMSE, R²). Test metrics for the other models are still reported, for research comparison only.
- **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R²), in `src/model_evaluation.py`.

### Naive forecasting baselines

`src/naive_baselines.py` implements two simple rules that answer "is ML better than a simple forecasting rule?". Both use only each school's own realised meal demand from earlier observations (grouped by school, sorted by date, shifted by one day before any window), are not tuned, and are scored on the same test period as the ML models:

- **Persistence / Last Observation:** prediction for date T = meals served on the school's most recent available observation before T.
- **7-Day Historical Mean:** prediction for date T = mean meals served over the school's previous 7 available observations (T excluded).

`projected_students` is not used as a baseline. The naive baselines are not evaluated on the validation folds, so their validation MAE is reported as empty rather than estimated.

### Selected model (SYNTHETIC DEVELOPMENT RESULTS)

In the most recent local run, the validation ranking was (TimeSeriesSplit, training period only):

| Model | Validation MAE | Selection rank |
|---|---|---|
| Random Forest (tuned) | 55.43 | 1 |
| Linear Regression | 60.54 | 2 |
| XGBoost (tuned) | 71.31 | 3 |
| Decision Tree (tuned) | 90.89 | 4 |

`models/final_best_model_metadata.json` therefore records **tuned Random Forest** as the selected model. Its single evaluation on the test period, next to the naive baselines:

| Method | Type | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Persistence / Last Observation | Naive baseline | 31.43 | 81.19 | 0.849 |
| 7-Day Historical Mean | Naive baseline | 58.91 | 122.92 | 0.653 |
| Random Forest (selected) | ML model | 19.37 | 47.07 | 0.949 |

On this synthetic test period the selected model's MAE is 12.06 meals (38.4%) lower than persistence and 39.54 meals (67.1%) lower than the 7-day mean. The other ML candidates had lower test MAE than Random Forest (see `reports/metrics/model_comparison.csv`), but test results were not used to choose the model. Validation MAE varies strongly between folds (the first fold, trained on only ~3 months, is much worse than the rest), so the ranking is fragile.

Train period: 2023-01-20 to 2024-08-01 (1,200 rows). Test period: 2024-08-02 to 2024-12-20 (303 rows). The split never puts the same date in both periods.

> These numbers come from the **synthetic** dataset. They show that the pipeline works end to end, not how accurate the model would be on real school data. `models/` and `reports/metrics/` are git-ignored, so a fresh clone has no trained model. You will get new results after running the pipeline yourself (see [Running the Machine-Learning Pipeline](#running-the-machine-learning-pipeline)).

## Forecasting

`src/forecasting.py` loads the saved model (`models/final_best_model.joblib`) and makes forecasts per school. It does not retrain anything.

- **Horizons:** next school day, next 7 school days, and next 30 school days (Monday–Friday).
- **Multi-day method:** forecasts are recursive. Each predicted day is fed back in as if it had been observed, to build the next day's lag and rolling features. Error can therefore build up over longer horizons, and the app says so.
- **Unknown future inputs:** future attendance, rainfall and temperature are not known. A forecast day's own attendance is never a model input: the attendance features use the previous school day (the last recorded day for day 1, then an assumed value). For weather, the user can enter expected values; otherwise the school's recent 14-day averages are used. Each row of the results says which assumption was used.
- **Calendar flags:** term, holiday and exam status for future dates come from the calendar rules in `src/synthetic_data.py`, the same rules the synthetic model was trained with.
- **Forecast origin:** forecasts start from the last recorded date in the dataset (20 December 2024 for the synthetic data), not from today. The web pages state this on every forecast. Because the synthetic calendar treats late December as a school holiday, short-horizon forecasts from this origin are low.

## Application Functionality

BiteTracker is a Django web application. Every page except the login page requires a signed-in user.

| Page | URL | What it does |
|---|---|---|
| Login / logout | `/accounts/login/` | Django authentication. Logout is a POST form (CSRF-protected). |
| Dashboard | `/` | Data status (synthetic/real), number of schools, latest recorded day, recent demand per school, the selected model (read from its metadata), a quick-forecast form, a meals-served chart for all schools, and recent forecast runs |
| Historical demand | `/history/` | Choose a school and date range; summary figures, a trend chart and a paginated table of daily records |
| Forecast & planning | `/forecast/` | Choose a school and horizon (next school day, next 7 school days, next 30 school days) and optional assumptions; generates the forecast and planning recommendation |
| Forecast result | `/forecast/runs/<id>/` | Predicted demand and recommended preparation (shown separately), safety buffer, risk level, estimated cost, optional capacity comparison, chart, day-by-day table, assumptions and forecast origin date |
| Forecast history | `/forecast/runs/` | Every forecast generated through the app, filterable by school |
| Model performance | `/performance/` | Selected model, how it was selected, validation and held-out test metrics, and comparison with the two naive baselines |
| Data upload | `/data/upload/` | Upload a CSV/XLSX file, validate it and record the result (users with the upload permission only) |
| Admin | `/admin/` | Django Admin for schools, upload records, forecast records and users (staff only) |

### Authentication and access

- Django's built-in authentication (`django.contrib.auth`). `LoginRequiredMiddleware` protects every page except the login page.
- There is no self-registration. An administrator creates accounts in Django Admin (or with `createsuperuser`).
- Dataset upload requires the `schools.add_datasetupload` permission. Django Admin requires staff status.
- The project-design roles (School Manager, Resource Planning Stakeholder, System Administrator) are **not** set up as Django groups yet; access is controlled with Django's user, staff and permission flags.

### Forecast & planning

1. Choose a school and a horizon. Optionally enter expected rainfall, temperature, expected day-1 attendance (it only affects day 2 onwards, because a day's own attendance is not a model input) and the meals currently planned for day 1. Blank fields use the school's recent 14-day averages, which the form shows.
2. Select **Generate forecast**. The request goes through `services/forecast_service.py`, which calls the existing `src/forecasting.py`, the saved model, `src/decision_support.py`, `src/budgeting.py` and `src/resource_allocation.py`. No model is trained during a request.
3. The result is saved as a `ForecastRun` record and shown on its own page.

**Safety buffer and risk** (`src/decision_support.py`): a documented rule of thumb, not a statistical confidence interval. The buffer grows with recent demand volatility (coefficient of variation) and with how far ahead a day is, and is capped at 20% of predicted demand. Risk is Low if the buffer is under 5%, Moderate if under 12%, and High otherwise. The model name shown on results comes from `models/final_best_model_metadata.json`.

### Data upload and validation

- Accepts `.csv` and `.xlsx` files up to 10 MB. Required columns are `date`, `school_id`, `school_name` and `meals_served`.
- Validation (`src/data_validation.py`, via `services/upload_service.py`) reports record and school counts, columns, date range, missing values, duplicates, invalid values, a 0–100 quality score, forecast readiness (READY / LIMITED / NOT READY) and model-compatibility warnings.
- Every validated file is recorded as a `DatasetUpload`. Valid files are stored under `data/uploads/` (git-ignored and never served over HTTP). Invalid or unreadable files are recorded without being stored.
- Uploading does **not** replace the active dataset, retrain the model or change any saved artifacts.

### Data status label

Every page shows a banner describing the data behind its results. While the synthetic artifacts are active it reads **SYNTHETIC DEVELOPMENT DATA**, and the Model Performance page shows the stored **SYNTHETIC DEVELOPMENT RESULTS** label. The label comes from the active model metadata and dataset path; if either is synthetic the label is synthetic, and if the metadata is missing the status is shown as unknown.

### Not carried over from the earlier Streamlit prototype

The Streamlit interface was replaced by Django. These prototype pages were deliberately not rebuilt for Sprint II (their `src/` functions remain and are still tested): specific-date "Predict & Compare", School Comparison, multi-school Resource Allocation, Budget Planning charts, Data Explorer CSV download, the per-model baseline-vs-tuned tables, and the About page. Details are in `MIGRATION_NOTES.md`.

## Architecture

```text
Browser
  │  HTML forms (CSRF-protected), Bootstrap 5, Plotly charts
  ▼
Django views and forms            web/core, web/schools, web/accounts
  │  thin: validate input, call services, render templates
  ▼
Service layer (no Django imports) services/
  │  data_service · model_service · forecast_service · performance_service
  │  upload_service · status_service · calendar_service
  ▼
Existing ML and decision-support code   src/
  │  forecasting · feature building · decision_support · budgeting · resource_allocation · data_validation
  ▼
Saved artifacts (generated by scripts/train_models.py, git-ignored)
     models/final_best_model.joblib + metadata · data/processed/*.csv · reports/metrics/*.csv
```

- **Django database** (SQLite by default) stores application state only: users, `School` metadata, `DatasetUpload` records and `ForecastRun` records. Historical meal-demand records stay in the processed CSV that the ML pipeline produces.
- **Model loading:** `services/model_service.py` loads the saved model on first use and caches it per process. Before use it checks that the model, its metadata and `models/feature_columns.json` all match the 20 features inference builds, and that the model type and hyperparameters match the metadata. A mismatch stops the forecast with a clear message.
- **Reuse:** the services layer returns plain Python objects, so a REST API or another frontend could use it without changing the ML code.

## Technology Stack

Versions below come from `requirements.txt`.

| Area | Technology | Version | Used for |
|---|---|---|---|
| Language | Python | ≥ 3.10 (`pyproject.toml`); developed and tested on 3.11 | Everything |
| Web framework | Django | 5.2.17 (LTS) | Web application, authentication, admin, ORM |
| Web serving | WhiteNoise | 6.12.0 | Serving static files when `DJANGO_DEBUG=False` |
| Web serving | Gunicorn | 26.2.0 | Production WSGI server (Linux/macOS) |
| Frontend | Bootstrap, Bootstrap Icons, plotly.js | 5.3.3, 1.11.3, 3.7.0 (CDN) | Layout, icons, interactive charts |
| Data | pandas | 2.3.3 | Data handling |
| Data | NumPy | 2.0.2 | Numerical work, synthetic data |
| ML | scikit-learn | 1.6.1 | Linear Regression, Decision Tree, Random Forest, CV, metrics |
| ML | XGBoost | 2.1.4 | XGBoost regressor |
| Statistics | statsmodels | 0.14.6 | OLS trendlines in EDA scatter charts (via Plotly) |
| Visualisation | Plotly (Python) | 6.9.0 | Chart specifications |
| Persistence | joblib | 1.5.3 | Saving and loading models |
| Config | python-dotenv | 1.2.1 | Loading `.env` |
| Reporting | tabulate | 0.9.0 | Markdown tables in generated reports |
| Uploads | openpyxl | 3.1.5 | Reading Excel uploads |
| Testing | pytest, pytest-django | 8.4.2, 4.14.0 | Test suite |
| Database (optional) | psycopg2-binary | 2.9.12 | PostgreSQL driver if `DJANGO_DB_ENGINE=postgresql` (not tested against a PostgreSQL server) |

## Project Structure

```text
BiteTracker-ML/
├── manage.py                       # Django management entry point
├── bitetracker_web/                # Django project: settings, urls, wsgi, asgi
├── web/
│   ├── accounts/                   # Login/logout URLs (Django auth views)
│   ├── core/                       # Dashboard, history, forecast, performance views; forms; charts; error messages
│   └── schools/                    # School, DatasetUpload, ForecastRun models; admin; upload views;
│                                   #   management/commands/import_schools.py
├── services/                       # Framework-independent bridge between the web layer and src/
├── templates/                      # Django templates (base layout, pages, error pages)
├── static/css/bitetracker.css      # Site styles
├── config/settings.py              # ML pipeline paths, target column, split ratio, seed
├── src/                            # Data pipeline, ML, forecasting and decision-support code
│   ├── synthetic_data.py           # Synthetic dataset generator + calendar rules
│   ├── data_loader.py              # Load dataset and validate schema
│   ├── preprocessing.py            # Cleaning, imputation, outlier capping, encoding
│   ├── feature_engineering.py      # Lag/rolling/attendance features and target
│   ├── pipeline.py                 # load → preprocess → feature-engineer
│   ├── eda.py, visualization.py    # EDA statistics and Plotly charts
│   ├── model_training.py           # Default-hyperparameter ML models, chronological split
│   ├── model_tuning.py             # TimeSeriesSplit tuning (MAE + RMSE scoring)
│   ├── model_selection.py          # Final model selection from validation scores only
│   ├── naive_baselines.py          # Persistence and 7-day-mean forecasting baselines
│   ├── model_evaluation.py         # MAE / RMSE / R², generalisation gap
│   ├── forecasting.py              # Multi-horizon and specific-date forecasts
│   ├── decision_support.py         # Safety buffer, risk level, uncertainty labels, model label
│   ├── resource_allocation.py      # School summaries, capacity gap
│   ├── budgeting.py                # Cost estimates
│   ├── data_validation.py          # Upload validation and quality score
│   └── utils.py                    # Logging, random seed
├── scripts/train_models.py         # One command: data → features → train → tune → select → save
├── run_pipeline.py, run_eda.py, run_training.py, run_tuning.py   # Step-by-step scripts
├── data/
│   ├── sample/synthetic_school_meals.csv   # Committed synthetic dataset
│   ├── raw/                        # git-ignored (place restricted datasets here)
│   ├── processed/                  # git-ignored (pipeline output)
│   └── uploads/                    # git-ignored (validated uploads)
├── models/                         # git-ignored (trained models + metadata)
├── reports/                        # Written reports; figures/ and metrics/ are git-ignored
├── tests/                          # pytest suite (ML, services, Django)
├── .github/workflows/ci.yml        # GitHub Actions: install → pipeline → checks → tests
├── MIGRATION_NOTES.md              # Streamlit → Django migration log
├── .env.example, requirements.txt, pyproject.toml, LICENSE, README.md
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/misskanyi/BiteTracker-ML.git
cd BiteTracker-ML
```

### 2. Create a virtual environment (Python 3.10 or newer; 3.11 recommended)

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the environment (optional for local development)

```bash
cp .env.example .env
```

With the defaults (`DJANGO_DEBUG=True`, no secret key) Django uses a development-only key. Never use those defaults on a server; see [Configuration](#configuration).

### 5. Generate the development model artifacts (first run only)

A fresh clone has **no trained model**, because `models/`, `data/processed/` and `reports/metrics/` are git-ignored. Build them from the committed synthetic dataset:

```bash
python scripts/train_models.py
```

If you skip this step the app still starts, but pages show a message asking an administrator to run the pipeline.

### 6. Set up the database

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py import_schools
```

`import_schools` creates or updates `School` records from the active dataset. It is safe to run repeatedly: schools are matched on their ID, and the command reports how many were created, updated or left unchanged. It never sets verified enrolment.

### 7. Run BiteTracker

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and log in with the superuser account. To let a non-staff user upload datasets, give them the "Can add dataset upload" and "Can view dataset upload" permissions in Django Admin.

## Configuration

| Setting | Where | Purpose |
|---|---|---|
| `DJANGO_DEBUG` | environment / `.env` | `True` for local development (default). Set `False` on any server. |
| `DJANGO_SECRET_KEY` | environment / `.env` | Required when `DJANGO_DEBUG=False` (startup fails without it). Generate with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`. |
| `DJANGO_ALLOWED_HOSTS` | environment / `.env` | Comma-separated host names. Defaults to `localhost,127.0.0.1` in debug mode; required when `DJANGO_DEBUG=False`. |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | environment / `.env` | Comma-separated origins (e.g. `https://bitetracker.example.org`) when served behind HTTPS |
| `DJANGO_SECURE_COOKIES` | environment / `.env` | Secure session/CSRF cookies when not in debug mode (default `True`; set `False` only to try a non-debug build over plain HTTP locally) |
| `DJANGO_SECURE_SSL_REDIRECT`, `DJANGO_SECURE_HSTS_SECONDS`, `DJANGO_BEHIND_HTTPS_PROXY` | environment / `.env` | HTTPS redirect, HSTS and proxy-header settings for a deployment; off by default because they depend on the hosting setup |
| `DJANGO_DB_ENGINE` | environment / `.env` | `sqlite` (default) or `postgresql` (then uses `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) |
| `DJANGO_SQLITE_PATH` | environment / `.env` | Alternative location for the SQLite file (default `db.sqlite3`, git-ignored) |
| `BITETRACKER_DATA_PATH` | environment / `.env` | Path to a different input CSV with the same schema for the training pipeline. Defaults to `data/sample/synthetic_school_meals.csv`. |
| `RANDOM_SEED` | environment / `.env` | Seed used for reproducibility (default `42`) |

Never commit `.env`; it is listed in `.gitignore`. The app does not call any weather or other external API; weather values come from the dataset or from user input.

## Running the Machine-Learning Pipeline

Starting or using the Django app does **not** retrain anything. The app loads the saved model from `models/final_best_model.joblib` on the first forecast and keeps it in memory. Rerun the pipeline only when the data or the modelling code changes.

### One-Command ML Pipeline

```bash
python scripts/train_models.py
```

This runs the whole training process in one go, on the **synthetic** development dataset by default:

| Stage | What happens | Code reused |
|---|---|---|
| `[1/9]` Loading data | Load the dataset and check its schema | `src/data_loader.py::load_dataset` |
| `[2/9]` Preprocessing | Clean the dataset | `src/preprocessing.py::run_preprocessing` |
| `[3/9]` Engineering features | Build lag, rolling, attendance and calendar features and the target, save the processed CSV | `src/feature_engineering.py::engineer_features`, `src/pipeline.py::save_processed_dataset` |
| `[4/9]` Chronological split | Split 80/20 **chronologically** (no shuffling, no shared dates) | `run_training.py::split_train_test` |
| `[5/9]` Naive baselines | Score Persistence and 7-Day Historical Mean on the test period | `src/naive_baselines.py::evaluate_naive_baselines` |
| `[6/9]` Training ML candidates | Default-hyperparameter fits of all four models; tune Decision Tree, Random Forest and XGBoost and cross-validate Linear Regression with `TimeSeriesSplit` (5 folds) on the training period only | `run_training.py::train_and_evaluate_baselines`, `run_tuning.py::tune_models`, `run_tuning.py::collect_validation_scores` |
| `[7/9]` Selecting model | Pick the lowest TimeSeriesSplit validation MAE (validation RMSE tiebreak); no test data involved | `run_tuning.py::select_final_model` |
| `[8/9]` Final test evaluation | Evaluate the selected model once on the held-out test period (MAE, RMSE, R²); other models are scored afterwards for the research comparison tables | `run_tuning.py::evaluate_selected_model`, `run_tuning.py::evaluate_tuned_models` |
| `[9/9]` Saving artifacts | Save all models, `final_best_model.joblib`, metadata, metrics (incl. `validation_model_selection.csv`, `naive_baseline_metrics.csv`, `model_comparison.csv`, `selected_model_vs_naive_baselines.csv`), charts and reports | `run_training.py::save_baseline_outputs`, `run_tuning.py::save_tuning_outputs` |

It uses the same functions as the step scripts, so the results are identical. To train on a different CSV with the same schema:

```bash
python scripts/train_models.py --data-path path/to/dataset.csv
```

Without `--data-path`, the script uses `BITETRACKER_DATA_PATH` if it is set, otherwise `data/sample/synthetic_school_meals.csv`. When you use a non-default dataset the script prints a warning, because the generated reports, model metadata and app badges still describe the data as synthetic.

If any stage fails, the script says which stage failed, prints the full error and exits with a non-zero status, so CI can detect the failure. On the synthetic dataset the full run takes about 20 seconds.

### Step-by-step scripts

The individual scripts are still available:

| Step | Command | What it does | Main outputs |
|---|---|---|---|
| Preprocess | `python run_pipeline.py` | Loads the dataset (and generates the synthetic one if it is missing), then cleans it and builds features | `data/processed/school_meals_processed.csv` |
| EDA (optional) | `python run_eda.py` | Descriptive statistics, data-quality report, charts | `reports/metrics/*`, `reports/figures/*`, `reports/EDA_Report.md` |
| Train default models | `python run_training.py` | Chronological 80/20 split; trains the 4 ML models with default settings (not the naive baselines) | `models/*.joblib`, `reports/metrics/baseline_*`, `reports/Baseline_Model_Training_Report.md` |
| Tune and select | `python run_tuning.py` | Tunes Decision Tree, Random Forest and XGBoost and cross-validates Linear Regression with `TimeSeriesSplit`; selects the final model by validation MAE; evaluates it once on the test period; scores the naive baselines | `models/final_best_model.joblib`, `models/final_best_model_metadata.json`, `reports/metrics/tuned_*`, `reports/metrics/naive_baseline_*`, `reports/metrics/model_comparison.csv`, `reports/Hyperparameter_Tuning_Report.md` |
| Run app | `python manage.py runserver` | Loads the processed data and the final model for prediction | — |

## Testing

```bash
python manage.py check
python manage.py makemigrations --check --dry-run
pytest
```

`pytest` runs everything (pytest-django reads `DJANGO_SETTINGS_MODULE` from `pyproject.toml`):

- **ML tests** (`tests/test_*.py`): preprocessing, feature engineering, training/inference feature alignment, evaluation, training, tuning, model selection, naive baselines, the one-command pipeline, forecasting, decision support (including the model label), data validation, resource allocation and budgeting.
- **Service tests** (`tests/services/`): model loading and caching, feature-alignment and metadata checks, forecast parity with `src/forecasting.py`, horizons, error handling, data/performance/upload/status/calendar services.
- **Django tests** (`tests/web/`): authentication and page protection, models, admin, `import_schools`, every page, all three forecast horizons, forecast persistence, uploads, permissions, data labels, error messages and production settings.

Tests that need the saved development model use it if `scripts/train_models.py` has been run and skip otherwise; most service and Django tests use a small synthetic stub dataset and model created in a temporary folder. No test uses real Food4Education data. The Sprint II testing record is in `reports/SPRINT_II_TESTING.md`.

## Continuous Integration

GitHub Actions checks the project automatically (`.github/workflows/ci.yml`). It runs on every push to `main` or to a `feature/**` branch, and on every pull request into `main`.

Each run, on Ubuntu with Python 3.11:

1. Installs the dependencies from `requirements.txt` (pip downloads are cached between runs).
2. Runs `python scripts/train_models.py` end to end on the committed **synthetic** dataset, then checks that the processed data, final model, model metadata and metrics files were created. No real or restricted data is used, and no secrets are needed.
3. Runs `python manage.py check` and `python manage.py makemigrations --check --dry-run`.
4. Runs the test suite with `pytest -q`. Because the pipeline has just produced a trained model, the tests that use it run instead of being skipped.

The run fails if the dependencies cannot be installed, if any pipeline stage or check fails, or if any test fails. CI does not deploy anything.

## Deployment

**BiteTracker is not deployed.** There is no live URL. The project is prepared for hosting:

- settings come from environment variables, and startup fails without a secret key and allowed hosts when `DJANGO_DEBUG=False`;
- secure cookies are on by default outside debug mode, with HTTPS redirect, HSTS and proxy headers available as settings;
- static files are collected with `python manage.py collectstatic` and served by WhiteNoise;
- `bitetracker_web/wsgi.py` (e.g. `gunicorn bitetracker_web.wsgi:application`) and `bitetracker_web/asgi.py` are provided;
- custom 403/404/500 pages never show error details.

A hosted deployment would also need: the model artifacts generated on (or copied to) the server, a persistent database (PostgreSQL recommended over SQLite for multi-user hosting), and persistent storage for `data/uploads/`.

## Privacy and Data Use

- The project is designed around anonymised, school-level school-feeding data. It never needs individual learner identities; records are daily totals per school.
- The Food4Education dataset was provided for academic research only, and it is not included in this repository.
- The committed dataset is fully synthetic. Its school names and locations are made up.
- The Upload page warns users not to include student names, admission numbers, phone numbers or addresses. Upload records store aggregate validation results (counts, column names, scores), not data rows.
- Raw, processed and uploaded data, trained models, generated metrics, the SQLite database and `.env` are git-ignored (`data/raw/*`, `data/processed/*`, `data/uploads/*`, `models/*`, `reports/figures/*`, `reports/metrics/*`, `db.sqlite3`, `.env`). Keep restricted datasets out of version control.

## Limitations

- **Synthetic data only.** The current results are SYNTHETIC DEVELOPMENT RESULTS and say nothing about real-world or Food4Education accuracy. The model needs to be retrained and re-evaluated on real data before anyone relies on it.
- **Forecast origin.** Forecasts start after the last recorded date (20 December 2024), not today, and the synthetic calendar places the next school days in a holiday, so short-horizon forecasts from this origin are low.
- **Small dataset.** Three synthetic schools over about two school years.
- **Preprocessing before the split.** Missing-value filling (forward then backward fill, then the per-school median) and IQR outlier capping (including `meals_served`) run on each school's full history before the chronological split, so some test-period information reaches training rows and test targets are capped. This is existing pipeline behaviour, documented in `MIGRATION_NOTES.md`; it was not changed during the web migration because changing it would change the reported results.
- **Weather.** Training uses observed weather as if it were a perfect forecast; at prediction time users enter expected values or recent averages are used.
- **Compounding error.** Multi-day forecasts are recursive, and days after day 1 use historical averages for attendance and weather.
- **Heuristic uncertainty.** The safety buffer and risk levels are rules of thumb, not statistical confidence intervals.
- **Estimates, not guarantees.** Every output is meant to support human decisions, not replace them.
- **Access control** uses Django staff/permission flags; the design's named roles are not yet set up as groups. Uploaded data is validated and recorded but not reviewed or merged into the main dataset.

## Future Improvements

- Audit and integrate the anonymised Food4Education dataset (schema adapter, real-data target `daily_unique_students_tapped`, real school calendar, weather source), then retrain and re-evaluate.
- Fit preprocessing statistics on the training period only.
- Add Django groups for the School Manager, Resource Planning Stakeholder and System Administrator roles.
- Add an administrator review and approval step so that verified uploads can join the training data.
- Compare forecasts with verified enrolment as a planning reference.
- Expose the services through a REST API.

## Author

**Kanyi Sharon Wambui**
Bachelor of Science in Informatics and Computer Science
Strathmore University

## Acknowledgement

- **Madam Salome Chemiat**, for supervising this project.
- **Food4Education**, for providing the anonymised historical school-feeding dataset for academic research. Food4Education did not develop BiteTracker and has not endorsed it.

## License / Academic Use

This is an academic final-year project. The repository contains a `LICENSE` file with the **MIT License** (Copyright (c) 2026 Sharon Kanyi), which `pyproject.toml` also declares. The Food4Education dataset is not covered by this license and is not distributed here.
