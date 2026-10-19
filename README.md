# BiteTracker

**A Machine Learning-Based School Meal Demand Forecasting and Resource Allocation Decision Support System**

Final-year project, BSc Informatics and Computer Science, Strathmore University.

> **Important: current dataset.** The reported model results come from the **anonymised Food4Education research dataset**, read live and read-only from its Google Sheet when `BITETRACKER_GOOGLE_SHEET_URL` is set. That dataset is **not** included in this repository. Without the link (for example in CI and the test suite), the pipeline falls back to the **synthetic** development dataset, `data/sample/synthetic_school_meals.csv`. See [Dataset](#dataset).

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
- [Google Colab and Google Drive Workflow](#google-colab-and-google-drive-workflow)
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

The pipeline uses one of two sources:

1. **Food4Education Google Sheet** (used for the reported results) when `BITETRACKER_GOOGLE_SHEET_URL` is set in `.env`. See [Food4Education research dataset](#food4education-research-dataset).
2. **Synthetic dataset** otherwise, for example in CI, the test suite and any fresh clone without the link. `src/synthetic_data.py` generates it; it was built because a real dataset was not available early in development. The committed file `data/sample/synthetic_school_meals.csv` contains:

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

The project was designed for use with anonymised historical school-feeding data that Food4Education provided for academic research. That dataset is **not committed to this repository**. `src/google_sheet_loader.py` reads it live from its Google Sheets link (set `BITETRACKER_GOOGLE_SHEET_URL` in `.env`; the sheet must be shared as *Anyone with the link → Viewer*) and maps it onto the schema above. The prediction target is `daily_unique_students_tapped`. Weather, cost and exam/event fields are not in the sheet and are held constant.

The current sheet gives **722 usable school-day records for 5 schools, 24 January to 3 November 2025** (after each school's first 14 rows are dropped for the lag window). The processed copy is written to `data/processed/`, which is git-ignored.

**Integration status:** the Google Sheet is integrated and is the source of the reported results. `data/raw/*` is git-ignored. The web application's data label is derived from the active model metadata and dataset path (`services/status_service.py`), so pages switch from **SYNTHETIC DEVELOPMENT DATA** only when a model trained on non-synthetic data is active. A 2025 Kenyan primary-school calendar (terms, half-terms, holidays) is prepared in `services/calendar_service.py` for the real-data pipeline; it is not used by the current synthetic model.

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

### Selected model (FOOD4EDUCATION RESEARCH RESULTS)

In the most recent local run on the Food4Education Google Sheet, the validation ranking was (TimeSeriesSplit, training period only):

| Model | Validation MAE | Selection rank |
|---|---|---|
| XGBoost (tuned) | 75.92 | 1 |
| Linear Regression | 78.49 | 2 |
| Random Forest (tuned) | 79.23 | 3 |
| Decision Tree (tuned) | 82.86 | 4 |

`models/final_best_model_metadata.json` therefore records **tuned XGBoost** as the selected model. Its single evaluation on the test period, next to the naive baselines:

| Method | Type | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Persistence / Last Observation | Naive baseline | 39.07 | 69.10 | 0.994 |
| 7-Day Historical Mean | Naive baseline | 58.10 | 119.79 | 0.981 |
| XGBoost (selected) | ML model | 51.64 | 88.80 | 0.989 |

**The selected model does not beat the persistence baseline**: its test MAE is 12.57 meals (32.2%) worse than simply repeating each school's previous day. It beats the 7-day mean by 6.46 meals (11.1%). None of the four ML candidates beat persistence on the test period (best: Random Forest, test MAE 49.54; see `reports/metrics/model_comparison.csv`). The top three models are within one fold standard deviation of each other on validation MAE (XGBoost 75.92 ± 13.49, Linear Regression 78.49 ± 13.26, Random Forest 79.23 ± 18.57), so the ranking is fragile. R² is above 0.98 for every method, including persistence, so it does not separate the models; compare MAE instead.

Train period: 2025-01-24 to 2025-09-09 (574 rows). Test period: 2025-09-10 to 2025-11-03 (148 rows). The split never puts the same date in both periods.

> These are **FOOD4EDUCATION RESEARCH RESULTS** from a small dataset (5 schools, under a year), using only lag, rolling-average, month and weekday features (the sheet has no weather, cost, attendance or exam/event fields). `models/` and `reports/metrics/` are git-ignored, so a fresh clone has no trained model. Without the sheet link the pipeline trains on the synthetic dataset and produces different numbers (see [Running the Machine-Learning Pipeline](#running-the-machine-learning-pipeline)).

## Forecasting

`src/forecasting.py` loads the saved model (`models/final_best_model.joblib`) and makes forecasts per school. It does not retrain anything.

- **Horizons:** next school day, next 7 school days, and next 30 school days (Monday–Friday).
- **Multi-day method:** forecasts are recursive. Each predicted day is fed back in as if it had been observed, to build the next day's lag and rolling features. Error can therefore build up over longer horizons, and the app says so.
- **Unknown future inputs:** future attendance, rainfall and temperature are not known. A forecast day's own attendance is never a model input: the attendance features use the previous school day (the last recorded day for day 1, then an assumed value). For weather, the user can enter expected values; otherwise the school's recent 14-day averages are used. Each row of the results says which assumption was used.
- **Inputs the model actually uses:** the Food4Education model uses only lag, rolling-average, month and weekday features, so attendance and weather assumptions do not change its forecasts. For a synthetic-data model, term, holiday and exam status for future dates come from the calendar rules in `src/synthetic_data.py`, the same rules it was trained with.
- **Forecast origin:** forecasts start from the last recorded date in the dataset (3 November 2025 for the Food4Education sheet; 20 December 2024 for the synthetic data), not from today. The web pages state this on every forecast.

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

Every page shows a banner describing the data behind its results. With the Food4Education model active it reads **FOOD4EDUCATION RESEARCH RESULTS** ("Model trained on Food4Education Google Sheet"). While synthetic artifacts are active it reads **SYNTHETIC DEVELOPMENT DATA**, and the Model Performance page shows the stored **SYNTHETIC DEVELOPMENT RESULTS** label. The label comes from the active model metadata and dataset path; if either is synthetic the label is synthetic, and if the metadata is missing the status is shown as unknown.

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
│   ├── experiment_tracking.py      # Copies one run's outputs into its own experiment folder (Colab → Drive)
│   └── utils.py                    # Logging, random seed
├── scripts/train_models.py         # One command: data → features → train → tune → select → save
├── scripts/summarize_experiments.py  # Lists Colab runs saved to Google Drive (local review)
├── notebooks/bitetracker_training.ipynb  # Google Colab training workflow (GitHub → Colab → Drive)
├── requirements-colab.txt          # Pinned ML packages for Colab (no Django)
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

A fresh clone has **no trained model**, because `models/`, `data/processed/` and `reports/metrics/` are git-ignored. Build them with the command below. With `BITETRACKER_GOOGLE_SHEET_URL` set in `.env` it trains on the Food4Education sheet; without it, on the committed synthetic dataset:

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
| `BITETRACKER_GOOGLE_SHEET_URL` | environment / `.env` | Share link of the Food4Education Google Sheet (*Anyone with the link → Viewer*). When set, the pipeline and app read it live and it takes priority over `BITETRACKER_DATA_PATH`. Never commit the link. |
| `BITETRACKER_DATA_PATH` | environment / `.env` | Path to a different input CSV with the same schema for the training pipeline. Defaults to `data/sample/synthetic_school_meals.csv`. |
| `RANDOM_SEED` | environment / `.env` | Seed used for reproducibility (default `42`) |

Never commit `.env`; it is listed in `.gitignore`. The app does not call any weather or other external API; weather values come from the dataset or from user input.

## Running the Machine-Learning Pipeline

Starting or using the Django app does **not** retrain anything. The app loads the saved model from `models/final_best_model.joblib` on the first forecast and keeps it in memory. Rerun the pipeline only when the data or the modelling code changes.

### One-Command ML Pipeline

```bash
python scripts/train_models.py
```

This runs the whole training process in one go, on the Food4Education Google Sheet when `BITETRACKER_GOOGLE_SHEET_URL` is set, otherwise on the **synthetic** development dataset:

| Stage | What happens | Code reused |
|---|---|---|
| `[1/9]` Loading data | Load the dataset and check its schema | `src/data_loader.py::load_dataset` |
| `[2/9]` Preprocessing | Clean the dataset (stateless steps only) | `src/preprocessing.py::clean_raw_data` |
| `[3/9]` Engineering features | Fit imputation and outlier statistics on the training period only, apply them to every row, build lag, rolling, attendance and calendar features and the target, save the processed CSV | `src/pipeline.py::build_training_data`, `src/pipeline.py::save_processed_dataset` |
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

Without `--data-path`, the script uses the Food4Education Google Sheet if `BITETRACKER_GOOGLE_SHEET_URL` is set, then `BITETRACKER_DATA_PATH` if it is set, otherwise `data/sample/synthetic_school_meals.csv`. `--data-path` always overrides the sheet, so `python scripts/train_models.py --data-path data/sample/synthetic_school_meals.csv` gives a synthetic run even with the link in `.env`. Reports and model metadata are labelled **FOOD4EDUCATION RESEARCH RESULTS** or **SYNTHETIC DEVELOPMENT RESULTS** to match. Any other CSV prints a warning, because its reports and app badges are still labelled synthetic.

> Running the pipeline overwrites `reports/Baseline_Model_Training_Report.md` and `reports/Hyperparameter_Tuning_Report.md`. The committed copies are from the Food4Education sheet; do not commit copies regenerated from synthetic data by mistake.

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

## Google Colab and Google Drive Workflow

The notebook [`notebooks/bitetracker_training.ipynb`](notebooks/bitetracker_training.ipynb) shows the Sprint II development workflow:

**Local IDE → GitHub → Google Colab → Google Drive → Local IDE + AI agent**

1. Write and test code locally.
2. Commit and push it to GitHub.
3. Open the notebook in Colab. It clones an exact commit and runs the existing pipeline, `scripts/train_models.py`. No training code is copied into the notebook.
4. The notebook saves the model, metrics, plots and logs to a new folder in Google Drive.
5. Google Drive for desktop syncs that folder to your Mac, where you (or the AI agent) review the results in the IDE.

The default data source is the **synthetic** dataset, a safe demonstration with no real records. The real Food4Education source can be selected for later experiments. Its results are **PROVISIONAL** (see [Current methodological limitations](#current-methodological-limitations)).

### Before you start: commit and push

Colab can only run code that is on GitHub. Local changes that haven't been committed and pushed **are not included** in a Colab run. The notebook stops with a clear message if the commit it clones is missing files it needs, such as `src/experiment_tracking.py` or `src/google_sheet_loader.py`. Every run records the exact Git commit SHA in `metadata.json`.

### 1. Open the notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and choose **File → Open notebook → GitHub**.
2. Because the repository is private, tick **Include private repos** and allow Colab to access your GitHub account when asked.
3. Choose `misskanyi/BiteTracker-ML`, branch `main`, and open `notebooks/bitetracker_training.ipynb`.

Alternatively, once the notebook is pushed, use this direct link: `https://colab.research.google.com/github/misskanyi/BiteTracker-ML/blob/main/notebooks/bitetracker_training.ipynb`

### 2. Connect the private GitHub repository (Colab Secret `GITHUB_TOKEN`)

Opening the notebook (step 1) and cloning the code inside it (this step) are separate permissions. The clone needs a read-only token:

1. On GitHub, go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. Under *Repository access*, choose **Only select repositories** → `misskanyi/BiteTracker-ML`. Under *Permissions → Repository permissions*, set **Contents: Read-only**. Pick an expiry date.
3. Copy the token. In Colab, click the **key icon (Secrets)** in the left sidebar → **Add new secret**. Set the name to `GITHUB_TOKEN`, paste the token as the value, and switch on **Notebook access**.

The token is never typed into a cell or printed. It's sent to `git` as a one-off request header, so it isn't stored in `.git/config` and isn't saved with the experiment.

### 3. Mount Google Drive

The notebook runs `drive.mount("/content/drive")`. Colab opens a Google sign-in window: choose your account and allow access. Experiments are saved to:

```text
MyDrive/BiteTracker/experiments/
```

The folder is created automatically on the first run.

### 4. Configure the dataset

Edit `DATA_SOURCE` in section 1 of the notebook:

| `DATA_SOURCE` | Data | Setup |
|---|---|---|
| `"synthetic"` (default) | Committed synthetic dataset `data/sample/synthetic_school_meals.csv` | None |
| `"food4education"` | Real Food4Education workbook, read live and read-only from its Google Sheets link by `src/google_sheet_loader.py` | Add the share link as the Colab Secret **`BITETRACKER_GOOGLE_SHEET_URL`**. If it isn't set, the notebook asks for the link in a hidden input box. The sheet must be shared as *Anyone with the link → Viewer*. |

No Excel file needs to be downloaded. The notebook shows only summary information about the data (row and school counts, date range, quality score), never individual records.

### 5. Run the training workflow

1. Check the settings in section 1 (`BRANCH`, optional `COMMIT`, `DATA_SOURCE`, `RANDOM_SEED`).
2. Choose **Runtime → Run all**.
3. If the dependency cell asks you to restart, choose **Runtime → Restart session**, then **Run all** again. This happens when Colab had already loaded different numpy/pandas versions.

The notebook installs the pinned package versions from `requirements-colab.txt`, which match `requirements.txt`. That way a model trained in Colab loads locally with the same library versions. On the synthetic dataset, training takes about one minute.

The pipeline writes its outputs inside the temporary Colab copy of the repository. Your local `models/` folder, including the model the Django app uses, is never changed.

### 6. Where experiment outputs are saved

Each run gets its own new folder. The timestamp is in **UTC**. Earlier runs are never overwritten. If two runs start in the same second, the second folder gets a `_2` suffix.

```text
MyDrive/BiteTracker/experiments/
└── run_YYYYMMDD_HHMMSS/
    ├── metadata.json        # status, Git commit SHA, branch, data source, random seed, command,
    │                        #   timestamps, Python/package versions, dataset summary, artifact list
    ├── metrics.json         # selected model, validation/test metrics, baseline and model comparisons
    ├── model.joblib         # the selected final model
    ├── models/              # every trained model + feature list, split and tuning metadata
    ├── metrics/             # the pipeline's metric CSV files
    ├── plots/               # interactive Plotly charts (.html)
    ├── reports/             # generated training and tuning reports (.md)
    ├── environment/pip_freeze.txt
    └── logs/                # training.log, pip_install.log
```

- If the pipeline didn't produce an expected file, it's listed under `artifacts.missing` in `metadata.json` and printed by the notebook. Nothing is generated to fill the gap.
- If training fails, the folder still holds `metadata.json` with `"status": "failed"` and the full `logs/training.log`.
- Dataset rows (`data/processed/`) are never saved.
- For Food4Education runs, row-level outputs are not saved by default: per-day test predictions and the actual-vs-predicted and residual charts. Set `SAVE_ROW_LEVEL_OUTPUTS = True` only if you have permission to store them.

### 7. Access the results locally (Google Drive for desktop)

1. Download and install **Google Drive for desktop** from [google.com/drive/download](https://www.google.com/drive/download/). Sign in with the same Google account you used in Colab.
2. Drive appears in Finder under **Locations → Google Drive**. The experiments folder is at:

   ```text
   ~/Library/CloudStorage/GoogleDrive-<your-email>/My Drive/BiteTracker/experiments/
   ```

3. By default Drive **streams** files, which means they download when opened. To let the IDE and AI agent read them without delay, right-click the `experiments` folder in Finder and choose **Make available offline**. You can also switch to *Mirror files* in Drive's preferences.
4. List all runs from the project folder:

   ```bash
   python scripts/summarize_experiments.py
   # or, if Drive is somewhere else:
   python scripts/summarize_experiments.py --experiments-dir "/path/to/BiteTracker/experiments"
   ```

5. Open `metadata.json`, `metrics.json`, `logs/training.log` and the charts in `plots/` in VS Code. You can also ask the AI agent to review a run folder by its path.

Treat Drive results as experiment records. **Don't copy a Colab `model.joblib` into the local `models/` folder**, because that would replace the model the Django app uses.

### Current methodological limitations

The real-data pipeline still has unresolved issues: target outlier capping and validation leakage. Until they are fixed:

- Food4Education results from this workflow are labelled **PROVISIONAL** in the notebook, `metadata.json` and `metrics.json`. They are not final validated research findings.
- The model selected on real data should not be presented as the definitive best model.
- The notebook's purpose is to demonstrate a reproducible Colab workflow, not to claim the methodology is final.

### Verifying the notebook locally (without Colab)

The notebook also runs outside Colab. Two environment variables redirect it: `BITETRACKER_NOTEBOOK_REPO_URL` sets the clone source (for example a local test repository) and `BITETRACKER_EXPERIMENTS_DIR` sets the output folder, which replaces Google Drive. It always clones into a new temporary folder, so it never writes into your working copy.

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

Tests that need a saved model trained on synthetic data use it if `scripts/train_models.py` has been run on that data and skip otherwise, so with the Food4Education model in `models/` about 29 tests skip locally; most service and Django tests use a small synthetic stub dataset and model created in a temporary folder. No test uses real Food4Education data. The Sprint II testing record is in `reports/SPRINT_II_TESTING.md`.

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
- The only committed dataset is fully synthetic. Its school names and locations are made up. The Food4Education sheet is read live from its link, which lives only in `.env`; the processed copy in `data/processed/` is git-ignored.
- The committed Food4Education reports contain only aggregate metrics, row counts and date ranges. Row-level outputs (per-day predictions and actual-vs-predicted charts in `reports/metrics/` and `reports/figures/`) are git-ignored.
- The Upload page warns users not to include student names, admission numbers, phone numbers or addresses. Upload records store aggregate validation results (counts, column names, scores), not data rows.
- Raw, processed and uploaded data, trained models, generated metrics, the SQLite database and `.env` are git-ignored (`data/raw/*`, `data/processed/*`, `data/uploads/*`, non-synthetic files in `data/sample/`, `models/*`, `reports/figures/*`, `reports/metrics/*`, `db.sqlite3`, `.env`, `*.xlsx`, Google Drive shortcut files such as `*.gsheet`). Keep restricted datasets out of version control.

## Limitations

- **ML does not yet beat persistence.** On the Food4Education test period no ML model beats the persistence baseline (repeat the previous day). The dataset is small (5 schools, 722 usable rows, January–November 2025) and lacks weather, cost and exam/event fields.
- **Forecast origin.** Forecasts start after the last recorded date (3 November 2025 for the Food4Education sheet), not today.
- **Small dataset.** Five Food4Education schools over less than one school year.
- **Test-period capping.** Imputation and IQR outlier statistics are fitted on the training period only, but they are then applied to every row, so test-period `meals_served` values outside the training-period fences are capped before evaluation. The pipeline logs how many.
- **Missing real-data inputs.** The Food4Education sheet has no weather, cost, attendance or exam/event fields, so the model uses only past demand, month and weekday. (On synthetic data, training uses observed weather as if it were a perfect forecast.)
- **Compounding error.** Multi-day forecasts are recursive: each predicted day feeds the next day's lag and rolling features.
- **Heuristic uncertainty.** The safety buffer and risk levels are rules of thumb, not statistical confidence intervals.
- **Estimates, not guarantees.** Every output is meant to support human decisions, not replace them.
- **Access control** uses Django staff/permission flags; the design's named roles are not yet set up as groups. Uploaded data is validated and recorded but not reviewed or merged into the main dataset.

## Future Improvements

- Add the missing real-data inputs (real school calendar, weather source, exam/event dates) for the Food4Education dataset, then retrain and re-evaluate against the persistence baseline.
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
