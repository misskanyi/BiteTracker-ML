"""BiteTracker — Model Performance page: baseline vs tuned comparison."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import plotly.express as px
import streamlit as st
from theme import configure_page, page_header, style_chart

from config import settings
from data_access import artifacts_ready, load_csv_if_exists, load_json

configure_page("Model Performance", icon="📈")
page_header(
    "Model Performance",
    subtitle="Baseline vs tuned comparison across all four regressors, measured on the held-out test set.",
    icon="📈",
)

status = artifacts_ready()
if not status["tuned_metrics"]:
    st.error("Tuned model metrics not found. Run `python run_tuning.py` first.")
    st.stop()

baseline_df = load_csv_if_exists(settings.METRICS_DIR / "baseline_model_metrics.csv")
tuned_df = load_csv_if_exists(settings.METRICS_DIR / "tuned_model_metrics.csv")
comparison_df = load_csv_if_exists(settings.METRICS_DIR / "baseline_vs_tuned_comparison.csv")

if status["final_model_metadata"]:
    meta = load_json(settings.MODELS_DIR / "final_best_model_metadata.json")
    st.success(
        f"**Selected model: {meta['selected_model']}** — "
        f"Test MAE={meta['final_test_mae']:.2f}, Test RMSE={meta['final_test_rmse']:.2f}, "
        f"Test R²={meta['final_test_r2']:.3f}. Selected by lowest TimeSeriesSplit validation MAE "
        "(training period); the test period was used only for this final assessment."
    )

tab1, tab2, tab3 = st.tabs(["Baseline Models", "Tuned Models", "Baseline vs Tuned"])

with tab1:
    st.markdown("#### Baseline Model Comparison (default hyperparameters)")
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)
    long_df = baseline_df.melt(id_vars="Model", value_vars=["Test MAE", "Test RMSE"], var_name="Metric", value_name="Value")
    fig = px.bar(long_df, x="Model", y="Value", color="Metric", barmode="group")
    style_chart(fig, "Baseline Test Error by Model")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Tuned Model Comparison (GridSearchCV / RandomizedSearchCV + TimeSeriesSplit)")
    st.dataframe(tuned_df, use_container_width=True, hide_index=True)
    long_df = tuned_df.melt(id_vars="Model", value_vars=["Test MAE", "Test RMSE"], var_name="Metric", value_name="Value")
    fig = px.bar(long_df, x="Model", y="Value", color="Metric", barmode="group")
    style_chart(fig, "Tuned Test Error by Model")
    st.plotly_chart(fig, use_container_width=True)

    best_row = tuned_df.loc[tuned_df["Test MAE"].idxmin()]
    st.info(
        f"Lowest tuned Test MAE: **{best_row['Model']}** ({best_row['Test MAE']:.2f}) — "
        "research comparison only; the final model is chosen by validation MAE."
    )

with tab3:
    st.markdown("#### Did Tuning Help?")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    for metric in ["MAE", "RMSE", "R2"]:
        col = f"{metric} Improvement"
        if col in comparison_df.columns:
            fig = px.bar(
                comparison_df, x="Model", y=col,
                color=comparison_df[col] > 0, color_discrete_map={True: "#2E7D32", False: "#B3261E"},
            )
            style_chart(fig, f"{metric} Improvement (Baseline − Tuned; positive = better)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("### Actual vs Predicted / Feature Importance")
figures_dir = settings.FIGURES_DIR
chart_files = {
    "Final Best Model — Actual vs Predicted": "final_best_model_actual_vs_predicted.html",
    "Final Best Model — Residuals": "final_best_model_residuals.html",
    "Feature Importance — Random Forest (tuned)": "feature_importance_tuned_random_forest.html",
    "Feature Importance — XGBoost (tuned)": "feature_importance_tuned_xgboost.html",
}
cols = st.columns(2)
for i, (label, filename) in enumerate(chart_files.items()):
    path = figures_dir / filename
    with cols[i % 2]:
        st.markdown(f"**{label}**")
        if path.exists():
            st.components.v1.html(path.read_text(), height=450, scrolling=True)
        else:
            st.caption("Chart not generated yet.")
