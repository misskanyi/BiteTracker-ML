# BiteTracker: Streamlit → Django Migration Notes

Working log for the migration of the web layer from Streamlit to Django.
All results and forecasts use **SYNTHETIC DEVELOPMENT DATA**. Real Food4Education
data has not been integrated. Nothing has been deployed.

## Agreed decisions

| Topic | Decision |
|---|---|
| Historical demand data | The processed CSV stays the single source of truth for the ML pipeline and forecasting. It is not migrated into the database for Sprint II. |
| Django database | Holds users/authentication, School metadata, DatasetUpload records and ForecastRun audit records. |
| Sprint II scope | Login/Logout, Dashboard, Historical Demand, Forecast & Planning, Model Performance, Dataset Upload/Admin. |
| Deferred | School Comparison, standalone Resource Allocation, standalone Budget Planning, Data Explorer CSV download, specific-date Predict & Compare. |
| Streamlit | Kept until Django reached parity for the required workflow, then removed in Stage 8 after a side-by-side comparison. |
| Model label | Read from `final_best_model_metadata.json`. The `"Tuned XGBoost"` default in `src/decision_support.py` was fixed in Stage 2. |
| Horizons | Next school day / Next 7 school days / Next 30 school days. |
| Forecast context | Pages must state that forecasts start from the end of the development data (20 December 2024), not from today. |
| Uploads | Validated and recorded only. They never replace the active dataset and never trigger retraining. |
| Frontend libraries | Bootstrap and Plotly are loaded from CDN for Sprint II. |
| ML methodology | Unchanged: chronological split, TimeSeriesSplit validation, validation-MAE selection with RMSE tiebreak, naive baselines, test isolation, 20-feature logic. |

## Architecture

```
Browser → Django views/forms (web/) → service layer (services/, no Django imports)
        → existing src/ modules → saved model (models/final_best_model.joblib)
        → decision support / budgeting / allocation → ForecastRun record → template
```

- `bitetracker_web/`: Django project configuration (settings, URLs, WSGI/ASGI).
- `web/accounts/`: login and logout using Django's built-in auth views.
- `web/core/`: dashboard, history, forecast, forecast history and performance views; forms; charts; error messages; context processor.
- `web/schools/`: `School`, `DatasetUpload`, `ForecastRun` models; admin; upload views; `import_schools`.
- `services/`: data, model, forecast, performance, upload, status and calendar services.
- `templates/`, `static/`: Bootstrap 5 templates and site CSS.
- `src/`, `config/`, `scripts/`: the ML pipeline, unchanged apart from the model-label fix and docstring wording.

## Django technologies used

- Django 5.2.17 (LTS), with `django.contrib.auth`, `admin`, `sessions`, `messages` and `staticfiles`.
- `LoginRequiredMiddleware` (Django 5.1+): every page requires login unless explicitly exempted.
- Built-in `LoginView` and `LogoutView`. Logout is POST-only and CSRF-protected.
- SQLite for development (`db.sqlite3`, gitignored); optional PostgreSQL via environment (not tested against a server).
- Django ORM + migrations (`schools.0001_initial`), Django Admin, a custom management command (`import_schools`), permissions (`schools.add_datasetupload`), `django.contrib.humanize`.
- WhiteNoise (static files when DEBUG is off) and Gunicorn (WSGI server).
- pytest-django 4.14.0, so a single `pytest` command runs both the ML and the Django tests.

## Stage log

### Stage 0: Baseline (before any change)

- `pytest -q` → **164 passed, 19 warnings in 10.70s** (Python 3.11.16).

### Stage 1: Django project, authentication, base layout

**Created**
- `manage.py`
- `bitetracker_web/__init__.py`, `settings.py`, `urls.py`, `wsgi.py`, `asgi.py`
- `web/__init__.py`
- `web/accounts/__init__.py`, `apps.py`, `urls.py`
- `web/core/__init__.py`, `apps.py`, `urls.py`, `views.py`, `context_processors.py`
- `templates/base.html`, `templates/accounts/login.html`, `templates/core/dashboard.html` (placeholder)
- `static/css/bitetracker.css`
- `tests/web/test_django_auth.py`
- `MIGRATION_NOTES.md`

**Modified**
- `requirements.txt`, `pyproject.toml`: added Django and pytest-django. `requires-python` raised from `>=3.9` to `>=3.10` because Django 5.2 requires it. `DJANGO_SETTINGS_MODULE` set for pytest.
- `.env.example`: added `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS` (placeholders only).
- `.gitignore`: added `staticfiles/`.
- `.github/workflows/ci.yml`: added a `python manage.py check` step.

**Security**
- `SECRET_KEY` comes from the environment. A development-only fallback is used only when `DJANGO_DEBUG` is true. With DEBUG off, startup fails if no key is set (tested).
- No credentials are committed. User accounts are created with `python manage.py createsuperuser`.

**Results**
- `python manage.py check` → `System check identified no issues (0 silenced).`
- `python manage.py makemigrations --check --dry-run` → `No changes detected`
- `python manage.py migrate` → all built-in migrations applied OK (contenttypes, auth, admin, sessions).
- `pytest -q` → **176 passed, 19 warnings in 11.33s**.
- Original tests only (`--ignore=tests/web`) → **164 passed**.
- New Django tests (`tests/web`) → **12 passed**.
- All 19 warnings were already present before the migration (sklearn matmul RuntimeWarnings and a pandas date-parsing UserWarning). None come from Django.
- `runserver` smoke check:
  - `/` → 302 to `/accounts/login/?next=/`
  - `/accounts/login/` → 200, with the synthetic banner
  - `/static/css/bitetracker.css` → 200
  - `/admin/` → 302 to the admin login

### Stage 2: Framework-independent service layer

**Services added** (`services/`, no Django or Streamlit imports; checked by a test)

| Module | Purpose |
|---|---|
| `__init__.py` | `ServiceError` base class. Every service error carries a message that can be shown to users. |
| `model_service.py` | Loads `final_best_model.joblib` and its metadata through the existing `src.forecasting.load_forecasting_model`, checks the model against its metadata and the 20-feature list, and caches it. Never trains. |
| `data_service.py` | Read-only access to the processed CSV: school list, school history (via `src.forecasting.get_school_history`), date filtering, date range, latest date. Cached by file modification time; callers get copies. |
| `forecast_service.py` | `generate_forecast(school_id, horizon_days, expected_attendance, expected_rainfall_mm, expected_temperature_c, available_capacity)`. Validates input, then calls `src.forecasting.forecast_school_demand`, `src.decision_support.build_decision_support_result`, `src.budgeting.estimate_meal_cost` and `src.resource_allocation.compute_allocation`. Returns a plain `ForecastResult` (with `to_dict()` for JSON). |
| `performance_service.py` | Selected model, results label, validation/test metrics and the model-vs-baseline comparison, read from the metadata JSON and `reports/metrics/naive_baseline_metrics.csv`. Nothing is recomputed. |

**Model-loading approach**
- Loading is lazy: the model is deserialised on the first forecast, not at server start.
- The cache is per process and keyed on the modification times of the model and metadata files. Repeated forecasts reuse one in-memory model (measured: first forecast 0.81 s, next 0.02 s). Re-running the training pipeline is picked up automatically.
- The cache is protected by a lock for threaded servers.
- Errors are reported as typed service errors: `ModelNotFoundError`, `ModelLoadError` (corrupt file or version mismatch), `ModelMetadataError` and `FeatureAlignmentError`. Each message tells the user to run `scripts/train_models.py`.

**Feature-alignment approach**
- Inference features are built only by the existing `src.forecasting._build_feature_row`, which emits `src.feature_engineering.FEATURE_COLUMNS`. Nothing is duplicated.
- On load, `model_service.verify_model` requires all of these to equal `FEATURE_COLUMNS` (20 columns, same order):
  - metadata `features_used`
  - `models/feature_columns.json`
  - `model.n_features_in_`
  - `model.feature_names_in_`
- It also checks that the target variable matches, that the estimator class matches the metadata `selected_model`, and that the scalar `best_parameters` match the model's parameters.
- Any mismatch raises before any prediction is made (tested).

**Model-label correction (`src/decision_support.py`)**
- `model_label` now defaults to `None`.
- When no label is passed, the new `selected_model_label()` reads `selected_model` from the metadata JSON. If the metadata can't be read, it returns the neutral "Selected model (metadata unavailable)" rather than any model name.
- The services always pass the label from metadata explicitly. Because the default changed, the legacy Streamlit pages now also show "Random Forest" without being edited.
- `tests/test_decision_support.py` line 59 asserted the old wrong value (`== "Tuned XGBoost"`). It now asserts `== selected_model_label()`. The test still exists, so the original count stays 164.

**Input validation added in the service** (UI-level checks only; the ML logic is unchanged)
- Horizon must be 1, 7 or 30.
- Attendance, rainfall and capacity must be finite and ≥ 0.
- Temperature must be finite and between −10 and 50 °C (the Streamlit form's minimum was −10).
- At least 14 history records are required (the longest lag/rolling window). Below that, the lag features would be empty and the prediction meaningless.

**Tests added** (107, in `tests/services/`)
- `conftest.py`: fixtures. Stub artifacts are a small synthetic dataset plus a LinearRegression whose weights are set directly, with no fitting. A separate fixture uses the real artifacts and skips if they are missing.
- `test_services_model.py` (24): loading, caching, reload on change, missing/corrupt model, missing/invalid/incomplete metadata, five feature-mismatch cases, mismatch blocks prediction, class/hyperparameter/target mismatch, real-model alignment (20 features), no `.fit` during forecasting, services import without Django/Streamlit/training modules.
- `test_services_forecast.py` (58): **parity** (18 real-model cases plus 6 stub cases), day-1 decision support matches the existing Forecast page logic, planning rows match the existing calendar logic, horizon labels, consecutive school days, origin date and note, JSON output, cost and capacity, model label from metadata, invalid horizon (9 cases), invalid inputs (8 cases), unknown school, insufficient history, missing data, missing model.
- `test_services_data_performance.py` (15): school list, history filtering, date range, cache returns copies, missing data and columns, real CSV; performance values equal the saved artifacts exactly, the model is not loaded, missing metrics/baseline/comparison/metadata.
- `test_model_label.py` (10): label comes from metadata, never "Tuned XGBoost" when Random Forest is selected, follows a changed selection, neutral fallback (5 bad-metadata cases), explicit label still wins, real metadata.

**Parity-test result**
- Service output equals the direct `src.forecasting.forecast_school_demand` output exactly (`assert_frame_equal(check_exact=True)`):
  - all 3 schools × horizons 1/7/30 × {recent-average assumptions, user-provided inputs}: 18/18 with the real saved Random Forest model
  - 6/6 with the stub model
- Mutation check: changing the service to ignore the rainfall input made 12 parity cases fail. With the file restored, all passed again.

**Results**
- `python manage.py check` → `System check identified no issues (0 silenced).`
- `pytest -q` → **283 passed, 19 warnings in 13.13s**.
- Original tests (`--ignore=tests/web --ignore=tests/services`) → **164 passed**.
- Stage 1 Django tests (`tests/web`) → **12 passed**.
- Stage 2 tests (`tests/services`) → **107 passed**, none skipped.
- Warnings: the same 19 as before the migration.
- No retraining or retuning was done. The existing artifacts from 2026-09-28 were used as they are.

### Stage 2.5: Real-data audit (not performed)

- `data/raw/` contains only `.gitkeep`. The Food4Education workbook has not been provided yet, so no audit was run and no real data was used. `data/raw/*` is still gitignored (`.gitignore` line 228).
- The migration continued on the verified synthetic pipeline, as instructed.

### Stage 3: Application data layer

**Added**
- `web/schools/` app containing:
  - models `School`, `DatasetUpload`, `ForecastRun`, with migration `0001_initial`
  - Django Admin registrations
  - `import_schools` management command
- `services/status_service.py`: works out the data label from the active model metadata and the configured dataset path. The status is "synthetic" if either source is synthetic, "unknown" if the metadata is missing, and "real" only if both say real.
- `services/calendar_service.py` with two calendars:
  - `SyntheticDevelopmentCalendar`: wraps `src.synthetic_data`. It is the calendar the current model was trained with, so it is the one used while synthetic artifacts are active.
  - `KenyaPrimaryCalendar`: the supplied 2025 calendar (terms, half-terms, April and August holidays). Dates outside it are classed "unknown". It is not used by the synthetic model.
  - `get_active_calendar()` picks between them.
- `data_service.recent_demand_summary()` for per-school recent demand.
- `forecast_service` now takes its holiday *warning* from `calendar_service` instead of importing `src.synthetic_data._is_holiday` directly. Predictions and forecast dates are unchanged (parity tests pass).
- The banner text now comes from `status_service` through the context processor.
- Settings:
  - `web.schools` and `django.contrib.humanize` added
  - `MEDIA_ROOT = data/uploads/` (gitignored and not served over HTTP), 10 MB upload limit
  - production security options driven by environment variables
  - WhiteNoise (production only)
  - optional PostgreSQL through `DJANGO_DB_ENGINE=postgresql` (not tested against a real Postgres server)
  - console logging
- Dependencies: `whitenoise==6.12.0` and `gunicorn==26.2.0`.
- Shared test fixtures moved to `tests/support/service_fixtures.py` and re-exported by `tests/services/conftest.py` and `tests/web/conftest.py`.

**Database decisions**
- `School.verified_enrolment` is left blank unless entered by an admin. `import_schools` never writes it, because dataset enrolment is not verified enrolment.
- `ForecastRun` stores `school_code` and `school_name` as text, plus an optional link to `School`, so forecasts are recorded even before `import_schools` has been run.
- `ForecastRun.result_summary` holds the per-day results (at most 30 rows), not raw data.
- `DatasetUpload` stores the file only for valid uploads. Admins can view and delete upload and forecast records but cannot add or edit them.

**Results**
- `makemigrations --check` → `No changes detected`.
- `migrate` → `schools.0001_initial... OK`.
- `import_schools` on the development DB: first run created 3; second run left 3 unchanged (skipped).
- `pytest -q` → **335 passed, 19 warnings**. Original tests: 164 passed.
- New tests: `tests/web/test_django_schools.py` (19) and `tests/services/test_calendar_status.py` (33).

**Issue fixed during this stage:** moving `tests/services/conftest.py` with `git mv -k` silently did nothing (the file was untracked), and the next command overwrote it. It was recreated from the identical source, and all 107 Stage 2 tests passed again afterwards.

### Stages 4–7: Django pages

**Pages** (all require login; logout is POST-only)

| URL | Page | Service used |
|---|---|---|
| `/` | Dashboard: data status, school count, latest recorded day, recent demand per school, model card (from metadata), quick-forecast form, all-schools chart, recent forecast runs | `data_service`, `model_service.load_model_metadata`, `ForecastRun` |
| `/history/` | Historical demand: school + date filters, KPI summary, Plotly trend chart, paginated table (25/page) | `data_service` |
| `/forecast/` | Forecast form: school, horizon (Next school day / Next 7 school days / Next 30 school days), optional attendance, rainfall, temperature and planned capacity. Hints show each school's recent averages and data end date | `forecast_service.default_assumptions` |
| `/forecast/runs/<id>/` | Forecast result: origin note, warnings, day-1 **predicted demand** and **recommended preparation** shown separately, buffer described as a rule-based heuristic, risk, cost, capacity comparison, chart, day-by-day table, assumptions, model version | `ForecastRun.result_summary` |
| `/forecast/runs/` | Forecast history: all runs, filterable by school | `ForecastRun` |
| `/performance/` | Model performance: results label and synthetic warning, selected model and selection rationale (mean validation MAE across TimeSeriesSplit folds, with a fold-variance caveat), validation and test metrics, R² explained as *not* accuracy, both naive baselines, comparison table, two charts, candidate ranking | `performance_service` |
| `/data/upload/`, `/data/uploads/<id>/` | Dataset upload and validation (requires the `schools.add_datasetupload` / `view_datasetupload` permissions) | `upload_service` → `src.data_validation` |
| `/admin/` | Django Admin: schools, uploads, forecast runs, users | — |

**Forecast flow:** POST → `ForecastForm` → `forecast_service.generate_forecast` → `record_forecast_run` saves a `ForecastRun` → redirect to the result page. This is POST/redirect/GET, so refreshing the result page does not create a duplicate run.

**Error handling:** `web/core/errors.py` maps configuration and artifact errors to fixed messages with no file paths. Errors about the user's own request (invalid horizon or input, unknown school, insufficient history) show their own message. Custom `403`, `404` and `500` templates are included; `500.html` is standalone and shows no details.

**Upload rules**
- Accepted types are `.csv` and `.xlsx`. `.xls` was dropped because reading it needs `xlrd`, which is not installed. The size limit is 10 MB.
- Every file that reaches validation is recorded. Valid files are stored under `data/uploads/datasets/`; invalid and unreadable files are recorded without storing the file.
- Files rejected by the form for type or size are not recorded.
- Uploading never changes the active dataset, the model or the artifacts (tested by checking file modification times).

**Also added**
- `web/core/charts.py` (Plotly figure specs rendered by plotly.js 3.7.0 from the jsdelivr CDN)
- `web/core/forms.py`, `web/core/forecast_records.py`
- `web/core/templatetags/bitetracker.py`
- `services/upload_service.py`
- `forecast_service.default_assumptions()`, and `ForecastResult.model_trained_at` (the model version stored in `ForecastRun`)
- Bootstrap Icons from CDN; reworked `static/css/bitetracker.css`

**Bug found by tests and fixed:** the upload view built its form with `request.POST or None`. An empty POST therefore produced an unbound form and no "This field is required" message. It now binds whenever the method is POST.

**Visual check:** headless Chrome screenshots of the dashboard, history, forecast form, forecast result, performance and upload pages at 1400px, and of the result page at 500px. All rendered with Bootstrap and Plotly.
- Chart legends overlapped long titles and were moved below the charts.
- The risk card now explains that risk comes from the buffer size and labels the horizon-uncertainty text separately, because "High" risk next to "Lower uncertainty" read as a contradiction. Both values are unchanged outputs of `src.decision_support`.

**Results**
- `pytest -q` → **407 passed, 19 warnings**.
- New tests: `tests/web/test_django_pages.py` (65) and `tests/services/test_upload_service.py` (7).

### Stage 8: Streamlit comparison and removal

**Dependency check before removal.** No file in `bitetracker_web/`, `web/`, `services/`, `src/`, `config/` or `scripts/` imports `app/`, `data_access`, `theme` or `streamlit`. The only remaining references were docstrings and comments, which were reworded.

**Side-by-side comparison (performed before deletion).** Streamlit's Forecast page was run with `streamlit.testing.v1.AppTest` and Django's forecast view with the Django test client, both for Greenfield Primary (SCH001) with default assumptions:

| Horizon | Rows (Streamlit / Django) | Rounded predictions equal | Dates equal |
|---|---|---|---|
| 1 | 1 / 1 | True | True |
| 7 | 7 / 7 | True | True |
| 30 | 30 / 30 | True | True |

The Streamlit decision card showed "Random Forest" (no "XGBoost"), confirming the Stage 2 label fix.

**Parity for the required Sprint II workflow**

| Workflow | Streamlit | Django |
|---|---|---|
| Authentication | None | Django auth, login/logout, every page protected |
| Dashboard | Home + Dashboard pages | `/` |
| Historical demand | Dashboard chart + Data Explorer table/filters | `/history/` |
| Forecast (1/7/30) + planning recommendation | Forecast page + Planning calendar | `/forecast/` + result page + history |
| Model performance | Model Performance page | `/performance/` |
| Upload + validation | Upload School Data page | `/data/upload/` |
| Admin | None | Django Admin |

**Streamlit functionality NOT ported (deferred by agreement; the underlying `src/` functions are kept and still tested)**
- Planning "Predict & Compare" for a specific date (`src.forecasting.predict_specific_date`) and the 14-day calendar option.
- School Comparison page (`src.resource_allocation.compare_schools`) and the multi-school Resource Allocation page with its reallocation note (`build_reallocation_note`). Django keeps single-school capacity comparison through `compute_allocation`.
- Budget Planning page (`src.budgeting.historical_expenditure` chart). Django keeps cost estimates through `estimate_meal_cost`.
- Data Explorer CSV download, summary statistics and missing-value table.
- Dashboard attendance-rate, leftover and food-waste KPIs and charts.
- Model Performance per-model "baseline vs tuned" test tables and the embedded HTML figures from `reports/figures/`.
- The upload page's "Process Dataset" preview (preprocessing + feature engineering of an uploaded file).
- The About page (content is covered by the README).

**Removed**
- `app/`: `Home.py`, `data_access.py`, `theme.py`, `pages/` (10 pages)
- `.streamlit/`: `config.toml`, `secrets.toml.example`
- `tests/test_streamlit_app.py`: 24 test functions, 34 collected tests
- `streamlit==1.50.0` from `requirements.txt` and `pyproject.toml`

**Expected test-count change:** 407 → 373 (−34), and the original ML/legacy group 164 → 130. The drop is entirely the intentionally removed Streamlit UI tests. No ML test was removed.

**Also updated:** docstrings and comments in `src/forecasting.py`, `src/decision_support.py`, `src/data_validation.py`, `src/resource_allocation.py`, `src/visualization.py`, `src/pipeline.py`, `services/*` and the CI comment no longer describe Streamlit as current. These are wording changes only. The Streamlit lines in `.gitignore` were removed. `openpyxl` was added to `pyproject.toml` (it was already in `requirements.txt` and is needed for `.xlsx` uploads).

**Results after removal**
- `streamlit` was uninstalled from `.venv`. An orphaned pytest bytecode file left in `site-packages/streamlit/` made `import streamlit` still resolve as an empty namespace package; it was deleted, after which `import streamlit` fails as expected.
- `manage.py check`: no issues. `makemigrations --check`: no changes.
- `pytest -q` → **373 passed, 19 warnings**: 130 original ML tests + 96 Django + 147 service, 0 skipped.

### Stage 9: CI, documentation and final verification

- **CI** (`.github/workflows/ci.yml`): added `python manage.py makemigrations --check --dry-run` after `manage.py check`. The pipeline-artifact checks and `pytest -q` are unchanged. The workflow has **not** run on these changes, because nothing has been committed or pushed.
- **README.md**: Application Functionality, Architecture, Technology Stack, Project Structure, Installation (migrate, createsuperuser, import_schools, runserver), Configuration (Django environment variables), Testing, CI, Deployment (not deployed), Privacy, Limitations and Future Improvements were rewritten. The stale "Authentication: Not implemented" and Streamlit instructions were removed.
- **`reports/SPRINT_II_TESTING.md`**: test record. **`reports/DJANGO_DOCUMENTATION_NOTES.md`**: facts for Chapters 4 and 5.
- **Deployment preparation:**
  - startup fails without `DJANGO_SECRET_KEY` or `DJANGO_ALLOWED_HOSTS` when DEBUG is off
  - secure cookies on by default outside debug mode
  - optional HTTPS redirect, HSTS, proxy header and `CSRF_TRUSTED_ORIGINS` through environment variables
  - WhiteNoise with hashed static files
  - custom error pages
  - `check --deploy` leaves only W004 (HSTS) and W008 (SSL redirect), which are opt-in by design
- **Production-mode smoke test** (gunicorn, DEBUG off): hashed CSS served (200), real CSRF login, all pages 200, custom 404, 30-day forecast saved, POST without CSRF returned 403.
- **Fresh-install test:**
  - setup: copied the working tree without ignored files and created a new venv with `pip install -r requirements.txt`
  - before the pipeline ran: pages showed friendly messages and `import_schools` failed cleanly
  - `scripts/train_models.py` reproduced the same metrics (Random Forest; validation MAE 55.43; test MAE 19.37 / RMSE 47.07 / R² 0.949)
  - `migrate`, `createsuperuser`, `import_schools` (3 created) all succeeded
  - `runserver`: login, all pages, all three horizons and logout worked
  - 373 tests passed

### Leakage review (documented, not changed)

Reviewed while preparing for real data. Changing any of this would alter the reported synthetic metrics and requires retraining, so it is recorded here for the real-data pipeline rather than fixed during the web migration.
- `src/preprocessing.py::handle_missing_values` fills forward, then **backward**, then with the per-school **median of the full history**. All of this runs before the chronological split, so later (test-period) values can fill earlier gaps and influence training rows.
- `src/preprocessing.py::handle_outliers` caps `meals_served`, `leftover_meals` and `enrolled_students` at per-school IQR fences computed on the **full history** before the split. Test-period values shape the fences, and **test targets themselves are capped**, which can flatter test error.
- `chronological_split` keeps each date entirely in train or test. Inside the training period, however, `TimeSeriesSplit` splits by row position, so a fold boundary can fall in the middle of a date across schools. This is minor.
- Lag/rolling features use `.shift(1)` per school, and the attendance features use the previous day. Both are covered by the existing alignment tests and look correct.
- Training uses observed same-day weather as a perfect forecast; operational forecasts use user input or recent averages.

**Recommendation for the real-data pipeline:** fit imputation and outlier statistics on the training period only, never back-fill across the split, keep evaluation targets uncapped, and use date-grouped folds.

## Issues encountered

## Issues encountered

- None blocking in Stage 1.
- Found during the audit: `build_decision_support_result` defaulted to `model_label="Tuned XGBoost"`, and the Streamlit pages never overrode it, so the live Streamlit app labelled decision cards with the wrong model. An original test asserted that wrong value. Fixed in Stage 2.
- Stage 2: the development data ends on Friday 20 Dec 2024. The next school days (23 Dec 2024 onward) fall in the synthetic calendar's school holiday, so short-horizon forecasts are low (for example, SCH001 day 1 ≈ 4 meals). This is existing forecasting behaviour and was not changed. The service adds a warning naming how many forecast days fall in a holiday.
- Found during the audit: the Streamlit Forecast page labels its 30-day horizon "30 calendar days (school days only)", but the code forecasts 30 school days (weekdays). The Django version uses "Next 30 school days".

- Stage 3: a fixture move with `git mv -k` silently skipped the untracked file and the next command overwrote it. The file was recreated from identical content; all tests passed again.
- Stages 4–7: an empty upload POST showed no error (form built with `request.POST or None`). Fixed and tested.
- Stage 3: WhiteNoise warned about the missing `staticfiles/` during tests. WhiteNoise is now enabled only when DEBUG is off.
- Real Food4Education workbook not provided, so the real-data audit and integration have not been done (see Stage 2.5).

## Final functionality implemented

- Django 5.2.17 application with authentication (login, POST logout, all pages protected), a dashboard, historical demand (filters, chart, paginated table), forecast & planning for the next school day / 7 / 30 school days (saved model, no training, predicted demand vs recommended preparation, heuristic buffer, risk, cost, capacity comparison, origin date, assumptions, warnings, chart, table), forecast history, model performance (validation, test, both naive baselines, synthetic label), dataset upload and validation (permission-restricted, recorded, never replaces data or retrains), and Django Admin for schools, uploads, forecasts and users.
- Framework-independent service layer with verified, cached model loading, exact parity with `src.forecasting`, and a data-status label derived from the active model metadata.
- Database models `School`, `DatasetUpload`, `ForecastRun`, and the idempotent `import_schools` command.
- Streamlit removed after a side-by-side comparison.
- Tests: 373 passing (130 ML, 147 service, 96 Django).
- Not done: real Food4Education integration (no workbook provided), role groups, deployment, and the deferred prototype pages listed in Stage 8.
