# BiteTracker: Sprint II Testing Record

Record of tests and checks actually run during the Streamlit → Django migration (6 October 2026).

- **Environment:** macOS (Darwin 25.6.0), Python 3.11.16, Django 5.2.17, pytest 8.4.2, pytest-django 4.14.0.
- **Data:** all results use the **SYNTHETIC DEVELOPMENT DATA** (`data/sample/synthetic_school_meals.csv`) and the saved development model (tuned Random Forest, artifacts dated 2026-09-28). No Food4Education data was used.
- **CI:** at the time of this record (6 October 2026) nothing had been committed, so every result below is from local runs. The changes have since been committed and pushed; GitHub Actions result for commit `7baa9d9` on `feature/ui-redesign` (pushed 19 October 2026): **TODO — fill in from the Actions tab**. Every CI step was also replicated locally on a clean checkout of `7baa9d9` (pipeline, artifact check, `manage.py check`, `makemigrations --check`, `pytest -q`: 508 passed, 19 warnings).
- **Model numbers below** (Random Forest, validation MAE 55.43, test MAE 19.37) are from the 6 October code on the synthetic dataset. The reported results now come from the Food4Education Google Sheet (19 October 2026: tuned XGBoost, validation MAE 75.92, test MAE 51.64, which does not beat the persistence baseline's 39.07); see `reports/Hyperparameter_Tuning_Report.md`.

## 1. Automated test suite

| Run | Command | Result |
|---|---|---|
| Baseline before migration | `pytest -q` | 164 passed, 19 warnings |
| After Stage 1 (Django + auth) | `pytest -q` | 176 passed, 19 warnings |
| After Stage 2 (service layer) | `pytest -q` | 283 passed, 19 warnings |
| After Stage 3 (models, admin, import) | `pytest -q` | 335 passed, 19 warnings |
| After Stages 4–7 (pages, upload) | `pytest -q` | 407 passed, 19 warnings |
| **Final, after Streamlit removal** | `pytest -q` | **373 passed, 19 warnings, 0 skipped, 0 failed** |
| Final, clean copy with a new virtual environment | `pytest -q` | 373 passed, 19 warnings |

**Why the count fell from 407 to 373:** `tests/test_streamlit_app.py` (34 collected tests) was removed together with the Streamlit interface it tested. No ML test was removed. The original group therefore went from 164 to 130 (164 − 34).

**Final breakdown**

| Group | Files | Tests | Result |
|---|---|---|---|
| ML pipeline (original) | `tests/test_*.py` (14 files) | 130 | Passed |
| Service layer | `tests/services/` (6 files) | 147 | Passed |
| Django web application | `tests/web/` (3 files) | 96 | Passed |
| **Total** | | **373** | **Passed** |

**Warnings:** the same 19 warnings were present before the migration. They are sklearn `RuntimeWarning`s (matmul overflow) in the Linear Regression tests and one pandas date-parsing `UserWarning` from `src/preprocessing.py:51`. The migration added no warnings.

## 2. Workflow test record

| # | Test | Expected result | Actual result | Status | Correction / notes |
|---|---|---|---|---|---|
| 1 | Unauthenticated access to every page | Redirect to login | All 6 app pages return 302 to `/accounts/login/?next=…` | Pass | |
| 2 | Login with valid credentials | Redirect to dashboard, session created | 302 to `/`; session set | Pass | Also checked against a live gunicorn server with a real CSRF token |
| 3 | Login with wrong password | Error shown, no session | Error shown, user stays anonymous | Pass | |
| 4 | Logout | POST logs out; GET rejected | POST → 302 to login, pages protected again; GET → 405 | Pass | |
| 5 | Forecast POST without CSRF token | Rejected | 403, no `ForecastRun` created | Pass | |
| 6 | Dashboard | Schools, latest date, model name from metadata, chart, recent runs | All shown; model name "Random Forest" read from metadata | Pass | |
| 7 | Historical demand: filter by school and date | Only that school and range | Filtered table and chart; reversed range rejected; empty range shows empty state | Pass | |
| 8 | Forecast: next school day | 1 row, saved run, predicted vs recommended shown | 302 to result page; 1 row; run saved | Pass | |
| 9 | Forecast: next 7 school days | 7 consecutive weekdays | 7 rows (23–31 Dec 2024) | Pass | |
| 10 | Forecast: next 30 school days | 30 consecutive weekdays | 30 rows (23 Dec 2024 – 31 Jan 2025) | Pass | Label is "Next 30 school days", not "calendar days" |
| 11 | Forecast origin shown | States data end date, not today | "Forecast generated from available development data ending 20 December 2024." | Pass | |
| 12 | Saved model loaded, not trained | No `fit` during requests | Tests that make every `.fit` raise still pass | Pass | Model loaded once per process (0.81 s first, 0.02 s next) |
| 13 | Service ↔ `src.forecasting` parity | Identical predictions | 18/18 real-model cases and 6/6 stub cases exactly equal | Pass | A deliberately broken service made 12 parity tests fail, showing the test can detect changes |
| 14 | Django ↔ Streamlit comparison | Same forecasts for the same input | SCH001, horizons 1/7/30: rounded predictions and dates identical | Pass | Run before Streamlit was removed |
| 15 | Training/inference feature alignment | 20 identical features | Model, metadata, `feature_columns.json` and inference all list the same 20 columns in order | Pass | |
| 16 | Feature mismatch | Clear error, no prediction | Friendly message; `predict` never called; no run saved | Pass | |
| 17 | Missing model / metadata / processed data | Friendly message, no file paths | Fixed messages shown; no paths or tracebacks in HTML | Pass | Checked in a fresh copy before the pipeline was run |
| 18 | Invalid input (horizon 5, negative rainfall, 80 °C, text in numbers, unknown school) | Form errors, nothing saved | Errors shown; 0 runs saved | Pass | |
| 19 | Insufficient history (< 14 records) | Clear message | "…at least 14 are needed…" | Pass | |
| 20 | Planning recommendation | Recommended ≥ predicted; risk, cost, capacity | Shown per day; capacity comparison via `compute_allocation` | Pass | Buffer is labelled as a heuristic, not a confidence interval |
| 21 | Forecast history | Lists runs; filter by school | Lists and filters correctly | Pass | |
| 22 | Model performance | Values equal saved artifacts; synthetic label | Validation MAE 55.43, test MAE 19.37, RMSE 47.07, R² 0.949, both baselines; "SYNTHETIC DEVELOPMENT RESULTS" | Pass | R² is explained as not an accuracy percentage; "94.9%" does not appear |
| 23 | Upload: valid CSV / XLSX | Validated, recorded, stored | `DatasetUpload` valid, file stored under `data/uploads/` | Pass | Model and processed-data file modification times unchanged |
| 24 | Upload: missing columns / corrupt Excel | Invalid / unreadable, not stored | Recorded without the file; readable error, no traceback | Pass | |
| 25 | Upload: .txt / .xls / .exe / over 10 MB / no file | Rejected by the form | Error shown; nothing recorded | Pass | Bug found and fixed: an empty POST showed no error (see section 4) |
| 26 | Upload permission | Users without permission blocked | 403 page | Pass | |
| 27 | Admin | Staff can manage schools/users; audit records read-only | Change lists load; adding uploads or runs returns 403; non-staff redirected to admin login | Pass | |
| 28 | `import_schools` | Idempotent | Run 1: 3 created; run 2: 3 unchanged; dry run saves nothing | Pass | |
| 29 | Data labels | Synthetic banner while synthetic; real label only for real data; unknown if metadata missing | All three cases tested | Pass | |
| 30 | Production settings | Startup fails without secret key or allowed hosts; secure cookies on | Tested in subprocesses | Pass | |

## 3. Manual and system checks

| Check | Command / method | Result |
|---|---|---|
| Django system check | `python manage.py check` | `System check identified no issues (0 silenced).` |
| Migration consistency | `python manage.py makemigrations --check --dry-run` | `No changes detected` |
| Deployment check (DEBUG off) | `python manage.py check --deploy` | 2 warnings: W004 (HSTS) and W008 (SSL redirect). Both are deliberately left off by default and can be enabled with environment variables once hosting is known. |
| Static files | `python manage.py collectstatic` (DEBUG off) | Hashed files collected; served by WhiteNoise (200) |
| Production-mode server | `gunicorn bitetracker_web.wsgi:application` with `DJANGO_DEBUG=False` | Login, all pages 200, custom 404, 30-day forecast saved, POST without CSRF 403, no errors in log |
| Visual check | Headless Chrome screenshots at 1400 px and 500 px | Pages render with Bootstrap and Plotly; legend overlap and an ambiguous risk/uncertainty label were found and fixed; layout responsive |
| Fresh setup from README | Clean copy + new venv → `pip install -r requirements.txt` → `train_models.py` → `migrate` → `createsuperuser` → `import_schools` → `runserver` | Same metrics regenerated (Random Forest, test MAE 19.37); login, all pages, 3 forecasts and logout worked; 373 tests passed |
| Streamlit independence | `pip uninstall streamlit` then full suite | `import streamlit` fails; 373 passed |

## 4. Defects found and corrected

| Defect | Found by | Correction |
|---|---|---|
| Decision cards labelled the model "Tuned XGBoost" while Random Forest was selected | Repository audit | Label now read from model metadata; regression tests added |
| Empty upload POST showed no "This field is required" message | `test_upload_without_file_shows_error` | Form now binds on every POST |
| WhiteNoise warned about a missing `staticfiles/` directory during tests | Test warnings | WhiteNoise enabled only when `DJANGO_DEBUG=False` |
| Chart legends overlapped titles | Visual check | Legends moved below charts |
| "High" risk shown next to "Lower uncertainty" without explanation | Visual check | Risk basis and horizon uncertainty labelled separately (logic unchanged) |
