# BiteTracker — Forecasting Report (Phase 6)

> Results below use **synthetic development data** (`data/sample/synthetic_school_meals.csv`)
> and the Phase 5 final selected model. They validate that the forecasting pipeline works
> correctly — they are not a claim about real-world forecast accuracy.

## Forecasting Workflow

`src/forecasting.py` turns the Phase 5 final model (`models/final_best_model.joblib`) into
per-school, multi-horizon predictions. The workflow for any forecast request:

1. Load that school's full processed history (`get_school_history`).
2. Determine the next N school days (`_next_school_days` — weekdays only, matching how
   the dataset itself only contains school days).
3. For each forecast day, build a feature row (`_build_feature_row`) using:
   - lag/rolling meal-count features computed from history **plus any already-forecast
     days earlier in this same horizon**,
   - calendar features (month, day-of-week, term, holiday/exam flags) computed directly
     for that forecast date using the same calendar logic as `src/synthetic_data.py`,
   - attendance/weather features from either a user-supplied assumption or a recent
     historical average (see "Assumptions" below).
4. Run the loaded model on that feature row, floor the prediction at zero (meals served
   can't be negative), and record it.
5. Append the prediction to the running meals-served history so the *next* day's
   lag/rolling features reflect it, then repeat.

This is exposed as one function, `forecast_school_demand(df, school_id, horizon_days, ...)`,
used identically by the CLI/tests and the Streamlit Forecast page.

## Next-Day, 7-Day, and 30-Day Forecasting

All three horizons use the same function — `horizon_days=1`, `7`, or `30` — because the
underlying mechanism (recursive one-step-ahead prediction) doesn't change with horizon
length, only how many times it repeats. "30 calendar days" in the UI means 30 *school*
days (weekdays), consistent with the dataset only containing school days; this is stated
in the Streamlit Forecast page label to avoid implying 30 calendar days including weekends.

## Recursive Forecasting

The model is trained to predict **one day ahead** using **known-past** lag/rolling
features. It has no native concept of "day 2 ahead" or "day 30 ahead". To produce a
multi-day forecast, each predicted day's `meals_served` is fed back into the history list
as if it had actually happened, so the model computes day 2's lag/rolling features from
day 1's *prediction*, day 3's from day 2's prediction, and so on. This is the standard way
to extend a one-step model to a multi-step forecast, but it means **prediction errors
compound**: an error on day 1 skews day 2's lag_1_meals input, which skews day 2's
prediction, and so on. The further out the horizon, the less trustworthy the forecast —
this is disclosed directly in the Streamlit Forecast page, not just in this report.

## How Lag Features Are Updated

For a forecast date `t`, using the running `meals_history` list (actual history + any
predictions already made earlier in this call):

- `lag_1_meals` = `meals_history[-1]` (the immediately preceding school day)
- `lag_7_meals` = `meals_history[-7]`
- `lag_14_meals` = `meals_history[-14]`
- `rolling_mean_7` / `rolling_mean_14` = mean of the last 7 / 14 entries in `meals_history`
- `rolling_std_7` = sample standard deviation of the last 7 entries (0.0 if fewer than 2
  entries are available, matching how a single observation has no variance to report)

This mirrors exactly how `src/feature_engineering.py` computes these features during
training (past-only, per-school) — the forecasting code is a second implementation of the
same logic operating on a live-growing list instead of a static DataFrame column, so
training and inference stay consistent.

## Assumptions When Future Attendance or Weather Is Unknown

Real future attendance and weather are not knowable in advance. BiteTracker handles this
explicitly rather than silently:

- **If the user supplies a value** (via the Forecast page's "Provide expected attendance /
  rainfall / temperature" checkboxes), that value is used for the **first day only** of the
  forecast horizon, and the output is labelled `"user-provided"` in the
  `attendance_assumption` / `rainfall_assumption` / `temperature_assumption` columns.
- **Otherwise**, the school's trailing 14-day historical average is used, and the output is
  labelled `"recent 14-day average"`.
- **Every day after day 1** always falls back to the recent-average assumption for
  attendance, since the user is only asked for a single day-1 estimate — there is no
  mechanism (yet) for the user to specify a multi-day attendance/weather plan.

The returned DataFrame always includes these three assumption-source columns so any
consumer (the Streamlit UI, a report, an API caller) can see exactly which basis was used
for which day, rather than presenting every number as equally certain.

## Recommendation and Buffer Logic

BiteTracker's Phase 6/7 scope stops at producing a point forecast (`predicted_meals_served`)
per day — it does **not** yet compute a recommended preparation quantity or safety-stock
buffer on top of that number. This is intentionally left as a **future improvement**: a
sensible buffer would need to be sized using each model's *measured* residual spread (e.g.
`reports/figures/final_best_model_residuals.html` shows the selected model's residual
distribution), not an arbitrary percentage, and that calibration work has not been done.
Any current use of these forecasts for procurement should add a manual safety margin based
on the reported final Test MAE for the selected model (see `final_test_mae` in
`models/final_best_model_metadata.json`; synthetic development result) rather than treating
the point forecast as exact.

## Model Metadata

The forecasting function loads whichever model is saved at
`models/final_best_model.joblib`. Its metadata (`models/final_best_model_metadata.json`)
records the selected model name, the selection criterion (lowest TimeSeriesSplit validation
MAE) and its validation MAE, its hyperparameters, the train/test date ranges, the
exact feature list (`src.feature_engineering.FEATURE_COLUMNS`), the target variable name,
measured final Test MAE/RMSE/R², the naive-baseline comparison, a synthetic-data warning, a timestamp, and the software
versions (Python/pandas/NumPy/scikit-learn/XGBoost) used to produce it — so any forecast
can be traced back to exactly which model, trained on exactly which data split, produced it.

## Limitations

- **Compounding error** over longer horizons (see "Recursive Forecasting" above).
- **No true future weather/attendance forecast** — day 2+ always uses a historical average,
  even if the user provided a day-1 estimate.
- **No safety-buffer/recommended-quantity logic yet** — only a raw point forecast.
- **Only 3 synthetic schools, ~2 years of history** — lag_14/rolling_14 features have
  limited data to learn from, and the model has seen only 3 schools' worth of
  enrolment/attendance patterns.
- **Calendar logic is approximated** (see `src/synthetic_data.py`'s holiday/term windows),
  not sourced from an authoritative Kenyan school calendar — a real deployment should
  replace this with an actual gazetted calendar per school/county.

## Synthetic Data Disclosure

Every number in this report — the model's measured error, the example forecasts, the
residual spread — comes from the **synthetic** dataset generated by `src/synthetic_data.py`
for development, because a real dataset was not available when this was built. The
forecasting *mechanism* (recursive lag-feature updates, assumption labelling, calendar
feature computation) is dataset-agnostic and requires no code changes to work with real
data — only pointing `config.settings.ACTIVE_DATA_PATH` at a real CSV with the same schema
and re-running the full pipeline (`run_pipeline.py` → `run_training.py` → `run_tuning.py`)
so the model itself is retrained on real patterns before its forecasts are trusted
operationally.
