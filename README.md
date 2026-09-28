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
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Local Setup](#installation-and-local-setup)
- [Configuration](#configuration)
- [Running the Machine-Learning Pipeline](#running-the-machine-learning-pipeline)
- [Testing](#testing)
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

It learns from historical school-day meal records and shows its forecasts in a Streamlit web application. The aim is to turn past meal-demand records into forward-looking information that helps staff plan meal preparation, capacity and budgets.

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

**Realised meal demand** is `meals_served`. The **forecasting target** is `meals_served_next_day`, which is the next school day's `meals_served` for the same school (`config/settings.py` → `TARGET_COLUMN`).

### How enrolment is used

`enrolled_students` is **not** a direct model feature. It is used in two ways:

- To calculate the `attendance_rate` feature (`students_present / enrolled_students`).
- As reference information after forecasting: the Dashboard shows total enrolment, and the School Comparison page shows each school's attendance rate.

The app does **not** compare predicted demand against enrolment to produce a "planning difference". See [Planning and Resource Analysis](#planning-and-resource-analysis).

### Engineered features

`src/feature_engineering.py` (`FEATURE_COLUMNS`) defines the 20 model inputs. It computes all of them per school, using only information from past days:

| Group | Features |
|---|---|
| Lagged demand | `lag_1_meals`, `lag_7_meals`, `lag_14_meals` |
| Rolling demand | `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7` (calculated after `.shift(1)`, so the current day is excluded) |
| Attendance | `attendance_rate`, `previous_attendance` |
| Calendar | `month`, `is_holiday`, `is_exam_period`, `is_school_event` |
| Encoded categories | `school_term_Term 2`, `school_term_Term 3`, `day_of_week_Tuesday` … `day_of_week_Friday` (one-hot, with Term 1 and Monday as the reference) |
| Weather | `rainfall_mm`, `temperature_c` |

Rows that do not have a full 14-day lag history, or a known next-day target, are dropped.

### Food4Education research dataset

The project was designed for use with anonymised historical school-feeding data that Food4Education provided for academic research. That dataset is **not committed to this repository**, and the current code does not load it or reference its field names. The pipeline expects the column schema above. To use a different dataset, set `BITETRACKER_DATA_PATH` to a CSV with the same columns (see [Configuration](#configuration)). A dataset with different column names would need code changes first.

## Machine Learning

BiteTracker treats forecasting as a supervised regression problem: predict tomorrow's meals served for a school, using today's and earlier information.

### Models

`src/model_training.py` and `src/model_tuning.py` implement all four candidate models:

| Model | Library | Baseline | Tuned |
|---|---|---|---|
| Linear Regression | scikit-learn | Yes | No (kept as the reference) |
| Decision Tree Regression | scikit-learn | Yes | `GridSearchCV` |
| Random Forest Regression | scikit-learn | Yes | `RandomizedSearchCV` (25 iterations) |
| XGBoost Regression | xgboost | Yes | `RandomizedSearchCV` (40 iterations) |

### Chronological evaluation

- **Train/test split:** chronological 80/20 split, with no shuffling (`TRAIN_TEST_SPLIT_RATIO = 0.8`). Earlier records are used for training and later records for testing.
- **Tuning:** `TimeSeriesSplit` with 5 folds (`CV_N_SPLITS = 5`), run on the training set only. The test set is used once, after tuning.
- **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R²), in `src/model_evaluation.py`.
- **Selection rule** (`select_best_model`): lowest test MAE first, then RMSE, then R². The model is also checked for a train/test generalisation gap.

### Selected model (synthetic data)

In the most recent local run, `models/final_best_model_metadata.json` records **tuned XGBoost** as the selected model:

| Metric (test set) | Value |
|---|---|
| MAE | 31.24 meals |
| RMSE | 68.41 meals |
| R² | 0.892 |

Train period: 2023-01-20 to 2024-08-01 (1,200 rows). Test period: 2024-08-02 to 2024-12-19 (300 rows).

> These numbers come from the **synthetic** dataset. They show that the pipeline works end to end, not how accurate the model would be on real school data. `models/` and `reports/metrics/` are git-ignored, so a fresh clone has no trained model. You will get new results after running the pipeline yourself (see [Running the Machine-Learning Pipeline](#running-the-machine-learning-pipeline)).

## Forecasting

`src/forecasting.py` loads the saved model (`models/final_best_model.joblib`) and makes forecasts per school. It does not retrain anything.

- **Horizons:** next school day, next 7 school days, and next 30 days (school days only).
- **Multi-day method:** forecasts are recursive. Each predicted day is fed back in as if it had been observed, to build the next day's lag and rolling features. Error can therefore build up over longer horizons, and the app says so.
- **Unknown future inputs:** future attendance, rainfall and temperature are not known. For day 1, the user can enter their own values. Otherwise, and for every later day, the forecast uses the school's recent 14-day averages. Each row of the results says which assumption was used.
- **Calendar flags:** term, holiday and exam status for future dates come from the calendar rules in `src/synthetic_data.py`.

## Application Functionality

The Streamlit app is started from `app/Home.py` and has the pages below, in the order they appear in the sidebar.

| Page | File |
|---|---|
| Home | `app/Home.py` |
| Dashboard | `app/pages/1_Dashboard.py` |
| Forecast | `app/pages/2_Forecast.py` |
| Planning | `app/pages/3_Planning.py` |
| School Comparison | `app/pages/6_School_Comparison.py` |
| Resource Allocation | `app/pages/7_Resource_Allocation.py` |
| Budget Planning | `app/pages/8_Budget_Planning.py` |
| Upload School Data | `app/pages/9_Upload_School_Data.py` |
| Data Explorer | `app/pages/10_Data_Explorer.py` |
| Model Performance | `app/pages/11_Model_Performance.py` |
| About | `app/pages/12_About.py` |

### Authentication and Role-Based Access

**Not implemented.** The app has no login, user accounts or role-based permissions. Anyone who can open the app can use every page. The roles in the project design (School Manager, Resource Planning Stakeholder, System Administrator) do not appear in the code.

### Home

The Home page introduces the project, its stakeholders and its purpose. It also shows a **Pipeline Status** panel that checks whether the processed dataset, baseline metrics, tuned metrics, final model and model metadata exist. If any are missing, it lists the commands you need to run.

### Historical Meal-Demand Dashboard

The Dashboard page shows:

- **Summary metrics:** total schools, total enrolment, average attendance rate, latest-day meals served, average leftover meals per school per day, and the leftover/prepared rate.
- **Current model:** the name of the selected model with its test MAE and R².
- **Charts** (Plotly, with a school multi-select filter): daily meals served per school, average meals served by school, and leftover meals over time.

### Meal-Demand Forecasting

On the **Forecast** page, the user:

1. Selects a school and a horizon (next school day, next 7 school days, or next 30 days).
2. Optionally enters expected attendance, rainfall and temperature for day 1. Otherwise the recent 14-day averages are used and shown.
3. Clicks **Generate Forecast**.

The page then shows:

- A decision-support summary for day 1: predicted meals, a recommended preparation quantity with a safety buffer, a risk level, and the assumptions used.
- The estimated daily, weekly and monthly cost.
- A forecast table with the assumption used for each day.
- A chart of recent actual demand against the forecast.

**Safety buffer and risk** (`src/decision_support.py`): this is a documented rule of thumb, not a statistical prediction interval. The buffer grows with recent demand volatility (coefficient of variation) and with the forecast horizon, and is capped at 20% of predicted demand. Risk is Low if the buffer is under 5%, Moderate if under 12%, and High otherwise.

### Planning and Resource Analysis

**Planning page** (two tabs, for a selected school):

- **Predict & Compare.** Choose a future date and set attendance, rainfall, temperature, term, holiday, exam-period and school-event values. The page compares your prediction with an automatic baseline forecast for the same date, and shows the difference, a decision-support card, the model's top-5 feature importances (for tree-based models), and recent demand next to the prediction.
- **Forecast Calendar.** A 14- or 30-school-day table showing predicted demand, recommended preparation, risk level and estimated cost for each day. You can open any single day to see its full decision-support details.

**Resource Allocation page:** for each selected school, the user enters the *available capacity* (meals). The page compares this with the next-day forecast and labels the gap as a **Shortage**, **Surplus** or **Balanced**. If some schools show a shortage and others a surplus, it suggests reviewing allocation between them, and states that this is not an automatic reallocation.

**Budget Planning page:** multiplies the next-day forecast by an editable cost per meal (the default is the recent average). This gives daily, weekly (×5 days) and monthly (×22 days) estimates, and a chart of recent historical spending against the forecast spending.

> **Note:** the app does not compare predicted demand with verified total school enrolment. Its planning comparisons are forecast vs. user-entered capacity (Resource Allocation) and forecast vs. cost (Budget Planning). "Surplus" on the Resource Allocation page means capacity above forecast demand. It is not a measurement of food waste. The "waste" figures on the Dashboard and School Comparison pages come from the synthetic `leftover_meals` field.

### School Comparison

The user selects two or more schools. For each school, the page shows a table with predicted next-day demand, the averages over the last 30 days (attendance, attendance rate, meals served, leftovers, estimated waste %, cost per meal), the expected daily cost and the forecast risk. Bar charts show predicted demand, expected daily cost, and attendance rate against estimated waste.

### Data Submission and Verification

The **Upload School Data** page supports:

- **Uploading** a CSV or Excel (`.xlsx`/`.xls`) file. Required columns are `date`, `school_id`, `school_name` and `meals_served`; the other columns are optional.
- **Automatic validation** (`src/data_validation.py`): record count, duplicates, missing values, date range, months of coverage, a data-quality score out of 100 (six weighted factors: completeness, duplicates, date validity, coverage, column richness, numeric validity), a forecast-readiness label (READY / LIMITED / NOT READY), and warnings where the file is not compatible with the trained model.
- **Processing:** runs the same cleaning and feature-engineering steps and saves the upload to `data/uploads/` with a timestamp in the filename.

Not implemented: uploading supporting documents, administrator review or approval, and adding uploaded data to the main dataset. Uploads do **not** replace the development dataset, do **not** retrain the model, and are not used by the other pages.

### Data Explorer

Filter the raw dataset by school and date range, preview rows, view summary statistics and missing-value counts, and download the filtered data as CSV.

### Model Performance

This page is available to all users (there are no roles). It shows:

- The selected model with its test MAE, RMSE and R².
- Tabs for baseline metrics, tuned metrics, and baseline vs. tuned improvement, each with tables and bar charts.
- Embedded charts: actual vs. predicted and residuals for the final model, and feature importance for tuned Random Forest and XGBoost.

### About

Methodology, the models compared, ethical considerations, the synthetic-data disclosure, and limitations.

## Technology Stack

Versions below come from `requirements.txt`.

| Area | Technology | Version | Used for |
|---|---|---|---|
| Language | Python | ≥ 3.9 (`pyproject.toml`) | Everything |
| Data | pandas | 2.3.3 | Data handling |
| Data | NumPy | 2.0.2 | Numerical work, synthetic data |
| ML | scikit-learn | 1.6.1 | Linear Regression, Decision Tree, Random Forest, CV, metrics |
| ML | XGBoost | 2.1.4 | XGBoost regressor |
| Statistics | statsmodels | 0.14.6 | OLS trendlines in EDA scatter charts (via Plotly) |
| Visualisation | Plotly | 6.9.0 | All charts (Matplotlib is not used) |
| Web app | Streamlit | 1.50.0 | User interface |
| Persistence | joblib | 1.5.3 | Saving and loading models |
| Config | python-dotenv | 1.2.1 | Loading `.env` |
| Reporting | tabulate | 0.9.0 | Markdown tables in generated reports |
| Uploads | openpyxl | 3.1.5 | Reading Excel uploads |
| Testing | pytest | 8.4.2 | Test suite |
| Not used yet | psycopg2-binary | 2.9.12 | Installed for possible future PostgreSQL use; the app uses CSV files |
| Version control | Git, GitHub | — | Source control |

## Project Structure

```text
BiteTracker-ML/
├── app/
│   ├── Home.py                     # Streamlit entry point
│   ├── data_access.py              # Cached loading of data, model, metrics; upload saving
│   ├── theme.py                    # Page styling and shared UI components
│   ├── assets/                     # (empty)
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Planning.py
│       ├── 6_School_Comparison.py
│       ├── 7_Resource_Allocation.py
│       ├── 8_Budget_Planning.py
│       ├── 9_Upload_School_Data.py
│       ├── 10_Data_Explorer.py
│       ├── 11_Model_Performance.py
│       └── 12_About.py
├── config/
│   └── settings.py                 # Paths, target column, split ratio, seed, env vars
├── src/
│   ├── synthetic_data.py           # Synthetic dataset generator + calendar rules
│   ├── data_loader.py              # Load dataset and validate schema
│   ├── preprocessing.py            # Cleaning, imputation, outlier capping, encoding
│   ├── feature_engineering.py      # Lag/rolling/attendance features and target
│   ├── pipeline.py                 # load → preprocess → feature-engineer
│   ├── eda.py, visualization.py    # EDA statistics and Plotly charts
│   ├── model_training.py           # Baseline models, chronological split
│   ├── model_tuning.py             # TimeSeriesSplit tuning, model selection
│   ├── model_evaluation.py         # MAE / RMSE / R², generalisation gap
│   ├── forecasting.py              # Multi-horizon and specific-date forecasts
│   ├── decision_support.py         # Safety buffer, risk level, uncertainty labels
│   ├── resource_allocation.py      # School summaries, capacity gap
│   ├── budgeting.py                # Cost estimates
│   ├── data_validation.py          # Upload validation and quality score
│   └── utils.py                    # Logging, random seed
├── data/
│   ├── sample/synthetic_school_meals.csv   # Committed synthetic dataset
│   ├── raw/                        # git-ignored
│   ├── processed/                  # git-ignored (pipeline output)
│   └── uploads/                    # git-ignored (user uploads)
├── models/                         # git-ignored (trained models + metadata)
├── reports/
│   ├── *.md                        # EDA, training, tuning, forecasting, final reports
│   ├── figures/                    # git-ignored (generated charts)
│   └── metrics/                    # git-ignored (generated metrics)
├── notebooks/                      # (empty)
├── tests/                          # pytest suite
├── scripts/
│   └── train_models.py             # One command: data → features → train → tune → select → save
├── run_pipeline.py                 # Step 1: build processed dataset
├── run_eda.py                      # Optional: EDA charts and report
├── run_training.py                 # Step 2: baseline models
├── run_tuning.py                   # Step 3: tuning + final model selection
├── .streamlit/config.toml          # Theme and server settings
├── .streamlit/secrets.toml.example
├── .env.example
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

- **`app/`** — the Streamlit interface. Pages only display results; the logic lives in `src/`.
- **`src/`** — the data pipeline, ML code and decision-support logic, which both the scripts and the app use.
- **`config/settings.py`** — the single place for paths and key settings.
- **`scripts/train_models.py`** — runs the full ML pipeline with one command.
- **`run_*.py`** — the individual step scripts that produce the processed data, models, metrics and reports.
- **`reports/*.md`** — written reports produced by the scripts on the synthetic data.

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/misskanyi/BiteTracker-ML.git
cd BiteTracker-ML
```

### 2. Create a virtual environment

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

### 4. Build the model (first run only)

A fresh clone has **no trained model**, because `models/`, `data/processed/` and `reports/metrics/` are git-ignored. Build everything with one command (see [One-Command ML Pipeline](#one-command-ml-pipeline)):

```bash
python scripts/train_models.py
```

Running the three step scripts in order (`python run_pipeline.py`, `python run_training.py`, `python run_tuning.py`) produces the same result.

If you skip this step, the app still opens, but the Home page shows missing artifacts and most pages show an error with these same commands.

### 5. Run BiteTracker

```bash
source .venv/bin/activate
streamlit run app/Home.py
```

Streamlit prints a local URL in the terminal, usually something like `http://localhost:8501`. If that port is already in use, it picks a different one, so use the URL it prints.

## Configuration

You do not need any configuration to run the app locally with the synthetic dataset. The optional settings are:

| Setting | Where | Purpose |
|---|---|---|
| `BITETRACKER_DATA_PATH` | environment / `.env` | Path to a different input CSV with the same schema. Defaults to `data/sample/synthetic_school_meals.csv`. |
| `RANDOM_SEED` | environment / `.env` | Seed used for reproducibility (default `42`) |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | `.env` | Read by `config/settings.py` for future PostgreSQL use. **Not used** by the current app. |
| `[postgres]` block | `.streamlit/secrets.toml` | Template only (`secrets.toml.example`). **Not read** by the current code. |
| Theme / server | `.streamlit/config.toml` | Colours, headless mode, XSRF protection, telemetry turned off |

To create a local `.env`:

```bash
cp .env.example .env
```

Then replace the placeholder values with your own. Never commit `.env` or `.streamlit/secrets.toml`; both are already listed in `.gitignore`.

The app does not use any weather API, external API or API keys. Weather values come from the dataset columns.

## Running the Machine-Learning Pipeline

Starting the Streamlit app does **not** retrain anything. The app loads the saved model from `models/final_best_model.joblib`. Rerun the pipeline only when the data or the modelling code changes.

### One-Command ML Pipeline

```bash
python scripts/train_models.py
```

This runs the whole training process in one go, on the **synthetic** development dataset by default:

| Stage | What happens | Code reused |
|---|---|---|
| `[1/7]` Preparing data | Load the dataset and check its schema, then clean it | `src/data_loader.py::load_dataset`, `src/preprocessing.py::run_preprocessing` |
| `[2/7]` Engineering features | Build lag, rolling, attendance and calendar features and the next-day target, save the processed CSV, then split 80/20 **chronologically** (no shuffling) | `src/feature_engineering.py::engineer_features`, `src/pipeline.py::save_processed_dataset`, `run_training.py::split_train_test` |
| `[3/7]` Training models | Train Linear Regression, Decision Tree, Random Forest and XGBoost with default settings | `run_training.py::train_and_evaluate_baselines` |
| `[4/7]` Tuning models | Tune Decision Tree, Random Forest and XGBoost with `TimeSeriesSplit` (5 folds) on the training period only | `run_tuning.py::tune_models` |
| `[5/7]` Evaluating models | Score the tuned models on the held-out test period (MAE, RMSE, R²) | `run_tuning.py::evaluate_tuned_models` |
| `[6/7]` Selecting model | Pick the lowest test MAE (then RMSE, then R²) among the tuned models and the Linear Regression baseline | `run_tuning.py::select_final_model` |
| `[7/7]` Saving artifacts | Save all models, `final_best_model.joblib`, metadata, metrics, charts and reports | `run_training.py::save_baseline_outputs`, `run_tuning.py::save_tuning_outputs` |

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
| Train baselines | `python run_training.py` | Chronological 80/20 split; trains the 4 models with default settings | `models/*.joblib`, `reports/metrics/baseline_*`, `reports/Baseline_Model_Training_Report.md` |
| Tune and select | `python run_tuning.py` | Tunes Decision Tree, Random Forest and XGBoost with `TimeSeriesSplit`; selects the final model by test MAE | `models/final_best_model.joblib`, `models/final_best_model_metadata.json`, `reports/metrics/tuned_*`, `reports/Hyperparameter_Tuning_Report.md` |
| Run app | `streamlit run app/Home.py` | Loads the processed data and the final model for prediction | — |

## Testing

```bash
pytest
```

The `tests/` folder covers preprocessing, feature engineering, evaluation, training, tuning, the one-command pipeline (on a small dataset with cheap tuning, writing to a temporary folder), forecasting, decision support, data validation, resource allocation, budgeting, and loading the Streamlit pages with `streamlit.testing.v1.AppTest`.

## Deployment

The repository does not show a deployment. It has no deployment configuration and no live application URL. `.streamlit/config.toml` sets `headless = true`, which suits hosted environments, but that alone does not show that the app has been deployed anywhere.

## Privacy and Data Use

- The project is designed around anonymised, school-level school-feeding data. It never needs individual learner identities; records are daily totals per school.
- The Food4Education dataset was provided for academic research only, and it is not included in this repository.
- The committed dataset is fully synthetic. Its school names and locations are made up.
- The Upload page warns users not to include student names, admission numbers, phone numbers or addresses.
- Raw, processed and uploaded data, trained models and generated metrics are git-ignored (`data/raw/*`, `data/processed/*`, `data/uploads/*`, `models/*`, `reports/figures/*`, `reports/metrics/*`). Keep restricted datasets out of version control.

## Limitations

- **Synthetic data only.** The current results say nothing about real-world accuracy. The model needs to be retrained and re-evaluated on real data before anyone relies on it.
- **Small dataset.** Three synthetic schools over about two school years. A real deployment needs more schools and a longer history.
- **Data quality.** Forecast quality depends on how complete and accurate the historical records are.
- **Unusual events** (strikes, disease outbreaks, sudden policy changes) may not appear in past patterns.
- **Compounding error.** Multi-day forecasts are recursive, and days after day 1 use historical averages for attendance and weather.
- **Heuristic uncertainty.** The safety buffer and risk levels are rules of thumb, not statistical confidence intervals.
- **Estimates, not guarantees.** Every output is meant to support human decisions, not replace them.
- **No authentication or roles**, and uploaded data is not reviewed or merged into the main dataset.

## Future Improvements

- Load and retrain on the anonymised Food4Education dataset, with a schema mapping for its field names.
- Add authentication and role-based access (School Manager, Resource Planning Stakeholder, System Administrator).
- Add an administrator review and approval step so that verified uploads can join the training data.
- Compare forecasts with verified enrolment as a planning reference.
- Move from CSV files to the planned PostgreSQL setup.

## Author

**Kanyi Sharon Wambui**
Bachelor of Science in Informatics and Computer Science
Strathmore University

## Acknowledgement

- **Madam Salome Chemiat**, for supervising this project.
- **Food4Education**, for providing the anonymised historical school-feeding dataset for academic research. Food4Education did not develop BiteTracker and has not endorsed it.

## License / Academic Use

This is an academic final-year project. The repository contains a `LICENSE` file with the **MIT License** (Copyright (c) 2026 Sharon Kanyi), which `pyproject.toml` also declares. The Food4Education dataset is not covered by this license and is not distributed here.
