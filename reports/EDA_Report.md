# BiteTracker — Exploratory Data Analysis Report

> Dataset: **synthetic** school-meal data (`data/sample/synthetic_school_meals.csv`), for development/testing until a real dataset is available.

## Data Quality Summary

- Raw records: **1545**
- Processed records (post feature engineering): **1500**
- Schools: **3**
- Date range: **2023-01-02 to 2024-12-20**
- Feature count (processed dataset): **35**
- Target variable: **meals_served_next_day**
- Missing values (raw): **99**
- Missing values (processed, post-cleaning): **0**
- Duplicate rows (raw): **0**
- Duplicate (school, date) pairs (raw): **0**
- Outliers detected (raw, IQR method): `{'meals_served': 3, 'leftover_meals': 18, 'enrolled_students': 5}`

### Preprocessing steps applied

1. schema validation
1. date parsing
1. dtype coercion
1. chronological sort (per school)
1. duplicate removal
1. non-negative value enforcement
1. per-school missing-value imputation (ffill/bfill + median fallback)
1. per-school IQR outlier capping
1. one-hot encoding of school_term and day_of_week
1. lag/rolling feature engineering (leakage-safe, per school)
1. next-day target construction
1. incomplete-row removal (insufficient history / no target)

## Charts

All charts are interactive Plotly HTML files in `reports/figures/`. Open any `.html` file directly in a browser.

### Daily Meal Demand Over Time

File: `reports/figures/daily_meal_demand.html`

**Insight:** Meal demand tracks attendance closely, with sharp drops to near-zero during holiday windows and visible weekly sawtooth patterns during term time.

### Attendance Rate Trends Over Time

File: `reports/figures/attendance_trends.html`

**Insight:** Attendance oscillates around 85-95% during term time and collapses during holidays; short dips align with heavy-rainfall periods.

### Rainfall vs Attendance Rate

File: `reports/figures/rainfall_vs_attendance.html`

**Insight:** Negative relationship — attendance rate drops as rainfall increases, most visible above ~15mm/day, consistent with the generator's design assumption.

### Rainfall vs Meals Served

File: `reports/figures/rainfall_vs_meals_served.html`

**Insight:** Mirrors the rainfall-attendance relationship since meals served is driven by how many students actually show up.

### Temperature vs Attendance Rate

File: `reports/figures/temperature_vs_attendance.html`

**Insight:** Weak relationship in this synthetic data — temperature swings in the modelled counties are mild and not a strong attendance driver.

### Leftover Meals Over Time

File: `reports/figures/leftover_meals_over_time.html`

**Insight:** Leftovers spike when caterers over-prepare relative to actual attendance, particularly right after holidays when attendance is hard to predict.

### Distribution of Meals Served

File: `reports/figures/meals_served_distribution.html`

**Insight:** Roughly bell-shaped per-school, reflecting stable typical demand with a long left tail from holiday/low-attendance days.

### Distribution of Attendance Rate

File: `reports/figures/attendance_rate_distribution.html`

**Insight:** Bimodal — a large cluster near 85-95% (normal school days) and a small cluster near 0% (holidays).

### Correlation Heatmap of Numerical Variables

File: `reports/figures/correlation_heatmap.html`

**Insight:** meals_served correlates strongly with students_present, enrolled_students, and its own lag/rolling features; rainfall correlates negatively with attendance.

### Outlier Detection Boxplots

File: `reports/figures/outlier_boxplots.html`

**Insight:** A handful of extreme points beyond the whiskers confirm the injected outliers in meals_served, leftover_meals, and enrolled_students that preprocessing caps.

### Average Meal Demand by School

File: `reports/figures/school_comparison.html`

**Insight:** Average demand differs by school roughly in proportion to enrolment size, as expected.

### Monthly Meal Demand Trends

File: `reports/figures/monthly_trends.html`

**Insight:** Demand dips in holiday months (April, August, December) and is stable across term months, consistent with the academic calendar.

### Day-of-Week Meal Demand Pattern

File: `reports/figures/day_of_week_pattern.html`

**Insight:** Demand is fairly flat across weekdays with minor variation; no strong Monday/Friday effect in this synthetic dataset.

### School Term Comparison

File: `reports/figures/school_term_comparison.html`

**Insight:** Terms show similar average demand per school, since enrolment (the main driver) doesn't change much term to term.

### Rolling 7-Day Average Meal Demand

File: `reports/figures/rolling_average.html`

**Insight:** Smooths out day-to-day noise, making the underlying trend and the holiday dips much easier to see than in the raw daily series.

## Engineered Feature Analysis

| Feature | Why it exists | How it helps prediction | Possible weaknesses |
|---|---|---|---|
| `lag_1_meals` | Yesterday's meals served for the same school. | Strongest single predictor — demand is highly autocorrelated day-to-day. | Breaks down right after a holiday or disruption, when yesterday isn't representative. |
| `lag_7_meals` | Meals served exactly one school week earlier. | Captures weekly seasonality (e.g. recurring weekday effects) lag_1 can't see. | Needs 7 school days of history; noisy for schools with irregular calendars. |
| `lag_14_meals` | Meals served two school weeks earlier. | Smooths over one-off weekly anomalies, captures slower biweekly patterns. | Furthest lag — most exposed to drift if enrolment or capacity changed since then. |
| `rolling_mean_7` | Trailing 7-day average of meals served (excluding today). | Filters day-to-day noise that raw lags carry, giving a stable trend estimate. | Lags behind sudden shifts (e.g. a new term starting) by design. |
| `rolling_mean_14` | Trailing 14-day average of meals served (excluding today). | Longer-horizon trend baseline, complements the 7-day average. | Slower to react; can understate a genuine recent change in demand. |
| `rolling_std_7` | Trailing 7-day standard deviation of meals served. | Signals volatility — schools with erratic attendance need bigger safety buffers. | Undefined/NaN with too few observations; sensitive to one large recent outlier. |
| `attendance_rate` | students_present / enrolled_students for the current day. | Direct, interpretable driver — meals needed track who actually shows up. | Today's rate must be forecast/estimated by the user for real future predictions. |
| `previous_attendance` | Yesterday's raw headcount of students present. | Adds absolute-scale momentum information the rate alone doesn't carry. | Conflates enrolment size with attendance behaviour across schools. |
| `month` | Calendar month of the record. | Captures seasonal/term-boundary effects (e.g. April/August/December dips). | Coarse — doesn't distinguish early vs late in a month. |
| `rainfall_mm` | Daily rainfall. | Heavy rain suppresses attendance (and thus demand) the same/next day. | Requires a rainfall forecast or historical average for future predictions. |
| `temperature_c` | Daily temperature. | Minor comfort/travel-tolerance effect on attendance. | Weak signal in mild climates; easily dominated by other features. |
| `is_holiday` | Binary flag for school holidays. | Single strongest categorical signal — demand collapses to near zero. | Depends on an accurate, up-to-date school calendar being supplied. |
| `is_exam_period` | Binary flag for exam weeks. | Attendance rises slightly; helps the model avoid under-predicting during exams. | Effect size is modest and may not generalize to every school's exam culture. |
| `is_school_event` | Binary flag for special school events. | Captures attendance spikes the calendar-based features can't anticipate. | Rare and irregular — hard for the model to learn a stable effect from few examples. |
| `school_term_*` | One-hot encoded school term (Term 1/2/3). | Captures term-level baseline differences beyond what month alone shows. | Assumes a 3-term calendar; won't transfer directly to a different school system. |
| `day_of_week_*` | One-hot encoded weekday (Monday-Friday). | Captures any consistent start/end-of-week attendance pattern. | Weak signal in this dataset; risk of overfitting to noise on small samples. |

Full feature list used for modelling: `lag_1_meals, lag_7_meals, lag_14_meals, rolling_mean_7, rolling_mean_14, rolling_std_7, attendance_rate, previous_attendance, month, rainfall_mm, temperature_c, is_holiday, is_exam_period, is_school_event, school_term_Term 2, school_term_Term 3, day_of_week_Tuesday, day_of_week_Wednesday, day_of_week_Thursday, day_of_week_Friday`.
