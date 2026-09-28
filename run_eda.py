"""CLI entry point: run EDA — generate charts, statistical reports, and
reports/EDA_Report.md from the processed dataset.

Usage:
    python run_eda.py
"""

from __future__ import annotations

from config import settings
from src.data_loader import load_dataset
from src.eda import save_reports
from src.pipeline import run_data_pipeline
from src.utils import get_logger
from src.visualization import CHART_REGISTRY

logger = get_logger(__name__)

CHART_TITLES = {
    "daily_meal_demand": "Daily Meal Demand Over Time",
    "attendance_trends": "Attendance Rate Trends Over Time",
    "rainfall_vs_attendance": "Rainfall vs Attendance Rate",
    "rainfall_vs_meals_served": "Rainfall vs Meals Served",
    "temperature_vs_attendance": "Temperature vs Attendance Rate",
    "leftover_meals_over_time": "Leftover Meals Over Time",
    "meals_served_distribution": "Distribution of Meals Served",
    "attendance_rate_distribution": "Distribution of Attendance Rate",
    "correlation_heatmap": "Correlation Heatmap of Numerical Variables",
    "outlier_boxplots": "Outlier Detection Boxplots",
    "school_comparison": "Average Meal Demand by School",
    "monthly_trends": "Monthly Meal Demand Trends",
    "day_of_week_pattern": "Day-of-Week Meal Demand Pattern",
    "school_term_comparison": "School Term Comparison",
    "rolling_average": "Rolling 7-Day Average Meal Demand",
}

CHART_INSIGHTS = {
    "daily_meal_demand": (
        "Meal demand tracks attendance closely, with sharp drops to near-zero during "
        "holiday windows and visible weekly sawtooth patterns during term time."
    ),
    "attendance_trends": (
        "Attendance oscillates around 85-95% during term time and collapses during "
        "holidays; short dips align with heavy-rainfall periods."
    ),
    "rainfall_vs_attendance": (
        "Negative relationship — attendance rate drops as rainfall increases, most "
        "visible above ~15mm/day, consistent with the generator's design assumption."
    ),
    "rainfall_vs_meals_served": (
        "Mirrors the rainfall-attendance relationship since meals served is driven "
        "by how many students actually show up."
    ),
    "temperature_vs_attendance": (
        "Weak relationship in this synthetic data — temperature swings in the "
        "modelled counties are mild and not a strong attendance driver."
    ),
    "leftover_meals_over_time": (
        "Leftovers spike when caterers over-prepare relative to actual attendance, "
        "particularly right after holidays when attendance is hard to predict."
    ),
    "meals_served_distribution": (
        "Roughly bell-shaped per-school, reflecting stable typical demand with a "
        "long left tail from holiday/low-attendance days."
    ),
    "attendance_rate_distribution": (
        "Bimodal — a large cluster near 85-95% (normal school days) and a small "
        "cluster near 0% (holidays)."
    ),
    "correlation_heatmap": (
        "meals_served correlates strongly with students_present, enrolled_students, "
        "and its own lag/rolling features; rainfall correlates negatively with attendance."
    ),
    "outlier_boxplots": (
        "A handful of extreme points beyond the whiskers confirm the injected outliers "
        "in meals_served, leftover_meals, and enrolled_students that preprocessing caps."
    ),
    "school_comparison": (
        "Average demand differs by school roughly in proportion to enrolment size, as expected."
    ),
    "monthly_trends": (
        "Demand dips in holiday months (April, August, December) and is stable across "
        "term months, consistent with the academic calendar."
    ),
    "day_of_week_pattern": (
        "Demand is fairly flat across weekdays with minor variation; no strong "
        "Monday/Friday effect in this synthetic dataset."
    ),
    "school_term_comparison": (
        "Terms show similar average demand per school, since enrolment (the main driver) "
        "doesn't change much term to term."
    ),
    "rolling_average": (
        "Smooths out day-to-day noise, making the underlying trend and the holiday "
        "dips much easier to see than in the raw daily series."
    ),
}


def main() -> None:
    raw_df = load_dataset()
    processed_df = run_data_pipeline(save=True)

    figures_dir = settings.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d charts", len(CHART_REGISTRY))
    for name, plot_fn in CHART_REGISTRY.items():
        fig = plot_fn(processed_df)
        out_path = figures_dir / f"{name}.html"
        fig.write_html(out_path, include_plotlyjs="cdn")
        logger.info("Saved %s", out_path)

    save_reports(raw_df, processed_df)
    write_eda_markdown_report(processed_df)
    logger.info("EDA complete.")


def write_eda_markdown_report(processed_df) -> None:
    from src.eda import build_data_quality_report
    from src.feature_engineering import FEATURE_COLUMNS

    raw_df = load_dataset()
    quality = build_data_quality_report(raw_df, processed_df)

    lines = [
        "# BiteTracker — Exploratory Data Analysis Report",
        "",
        "> Dataset: **synthetic** school-meal data "
        "(`data/sample/synthetic_school_meals.csv`), for development/testing "
        "until a real dataset is available.",
        "",
        "## Data Quality Summary",
        "",
        f"- Raw records: **{quality['raw_record_count']}**",
        f"- Processed records (post feature engineering): **{quality['processed_record_count']}**",
        f"- Schools: **{quality['num_schools']}**",
        f"- Date range: **{quality['date_range']['start']} to {quality['date_range']['end']}**",
        f"- Feature count (processed dataset): **{quality['feature_count']}**",
        f"- Target variable: **{quality['target_variable']}**",
        f"- Missing values (raw): **{quality['missing_values_raw']}**",
        f"- Missing values (processed, post-cleaning): **{quality['missing_values_processed']}**",
        f"- Duplicate rows (raw): **{quality['duplicates_raw']['exact_duplicate_rows']}**",
        f"- Duplicate (school, date) pairs (raw): **{quality['duplicates_raw']['duplicate_school_date_pairs']}**",
        f"- Outliers detected (raw, IQR method): `{quality['outliers_detected_raw']}`",
        "",
        "### Preprocessing steps applied",
        "",
        *[f"1. {step}" for step in quality["preprocessing_steps_applied"]],
        "",
        "## Charts",
        "",
        "All charts are interactive Plotly HTML files in `reports/figures/`. "
        "Open any `.html` file directly in a browser.",
        "",
    ]

    for name in CHART_TITLES:
        lines.append(f"### {CHART_TITLES[name]}")
        lines.append("")
        lines.append(f"File: `reports/figures/{name}.html`")
        lines.append("")
        lines.append(f"**Insight:** {CHART_INSIGHTS[name]}")
        lines.append("")

    lines.append("## Engineered Feature Analysis")
    lines.append("")
    lines.append("| Feature | Why it exists | How it helps prediction | Possible weaknesses |")
    lines.append("|---|---|---|---|")

    feature_analysis = {
        "lag_1_meals": (
            "Yesterday's meals served for the same school.",
            "Strongest single predictor — demand is highly autocorrelated day-to-day.",
            "Breaks down right after a holiday or disruption, when yesterday isn't representative.",
        ),
        "lag_7_meals": (
            "Meals served exactly one school week earlier.",
            "Captures weekly seasonality (e.g. recurring weekday effects) lag_1 can't see.",
            "Needs 7 school days of history; noisy for schools with irregular calendars.",
        ),
        "lag_14_meals": (
            "Meals served two school weeks earlier.",
            "Smooths over one-off weekly anomalies, captures slower biweekly patterns.",
            "Furthest lag — most exposed to drift if enrolment or capacity changed since then.",
        ),
        "rolling_mean_7": (
            "Trailing 7-day average of meals served (excluding today).",
            "Filters day-to-day noise that raw lags carry, giving a stable trend estimate.",
            "Lags behind sudden shifts (e.g. a new term starting) by design.",
        ),
        "rolling_mean_14": (
            "Trailing 14-day average of meals served (excluding today).",
            "Longer-horizon trend baseline, complements the 7-day average.",
            "Slower to react; can understate a genuine recent change in demand.",
        ),
        "rolling_std_7": (
            "Trailing 7-day standard deviation of meals served.",
            "Signals volatility — schools with erratic attendance need bigger safety buffers.",
            "Undefined/NaN with too few observations; sensitive to one large recent outlier.",
        ),
        "previous_attendance_rate": (
            "students_present / enrolled_students on the previous school day.",
            "Interpretable attendance signal that is known before the target day's meals.",
            "Lags reality by a day; the target day's own attendance is excluded to avoid leakage.",
        ),
        "previous_attendance": (
            "Previous school day's raw headcount of students present.",
            "Adds absolute-scale momentum information the rate alone doesn't carry.",
            "Conflates enrolment size with attendance behaviour across schools.",
        ),
        "month": (
            "Calendar month of the record.",
            "Captures seasonal/term-boundary effects (e.g. April/August/December dips).",
            "Coarse — doesn't distinguish early vs late in a month.",
        ),
        "rainfall_mm": (
            "Daily rainfall.",
            "Heavy rain suppresses attendance (and thus demand) the same/next day.",
            "Requires a rainfall forecast or historical average for future predictions.",
        ),
        "temperature_c": (
            "Daily temperature.",
            "Minor comfort/travel-tolerance effect on attendance.",
            "Weak signal in mild climates; easily dominated by other features.",
        ),
        "is_holiday": (
            "Binary flag for school holidays.",
            "Single strongest categorical signal — demand collapses to near zero.",
            "Depends on an accurate, up-to-date school calendar being supplied.",
        ),
        "is_exam_period": (
            "Binary flag for exam weeks.",
            "Attendance rises slightly; helps the model avoid under-predicting during exams.",
            "Effect size is modest and may not generalize to every school's exam culture.",
        ),
        "is_school_event": (
            "Binary flag for special school events.",
            "Captures attendance spikes the calendar-based features can't anticipate.",
            "Rare and irregular — hard for the model to learn a stable effect from few examples.",
        ),
        "school_term_*": (
            "One-hot encoded school term (Term 1/2/3).",
            "Captures term-level baseline differences beyond what month alone shows.",
            "Assumes a 3-term calendar; won't transfer directly to a different school system.",
        ),
        "day_of_week_*": (
            "One-hot encoded weekday (Monday-Friday).",
            "Captures any consistent start/end-of-week attendance pattern.",
            "Weak signal in this dataset; risk of overfitting to noise on small samples.",
        ),
    }

    for feat, (why, how, weakness) in feature_analysis.items():
        lines.append(f"| `{feat}` | {why} | {how} | {weakness} |")

    lines.append("")
    lines.append(f"Full feature list used for modelling: `{', '.join(FEATURE_COLUMNS)}`.")
    lines.append("")

    out_path = settings.REPORTS_DIR / "EDA_Report.md"
    out_path.write_text("\n".join(lines))
    logger.info("EDA markdown report written to %s", out_path)


if __name__ == "__main__":
    main()
