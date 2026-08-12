"""Tests for src.decision_support: safety buffer, risk level, decision-support bundle."""

from __future__ import annotations

from src.decision_support import (
    build_decision_support_result,
    classify_risk,
    compute_safety_buffer,
    uncertainty_label,
)


def test_compute_safety_buffer_zero_volatility_gives_small_buffer():
    result = compute_safety_buffer(predicted_demand=500, recent_std=0, horizon_days=1)
    assert result["buffer_meals"] == 0
    assert result["buffer_pct"] == 0.0


def test_compute_safety_buffer_scales_with_volatility():
    low_vol = compute_safety_buffer(predicted_demand=500, recent_std=5, horizon_days=1)
    high_vol = compute_safety_buffer(predicted_demand=500, recent_std=100, horizon_days=1)
    assert high_vol["buffer_pct"] >= low_vol["buffer_pct"]


def test_compute_safety_buffer_scales_with_horizon():
    day1 = compute_safety_buffer(predicted_demand=500, recent_std=50, horizon_days=1)
    day30 = compute_safety_buffer(predicted_demand=500, recent_std=50, horizon_days=30)
    assert day30["buffer_pct"] >= day1["buffer_pct"]


def test_compute_safety_buffer_never_negative_demand():
    result = compute_safety_buffer(predicted_demand=-10, recent_std=5, horizon_days=1)
    assert result["buffer_meals"] >= 0


def test_compute_safety_buffer_capped():
    result = compute_safety_buffer(predicted_demand=500, recent_std=10_000, horizon_days=30)
    assert result["buffer_pct"] <= 20.0 + 1e-6


def test_classify_risk_thresholds():
    assert classify_risk(2) == "Low"
    assert classify_risk(8) == "Moderate"
    assert classify_risk(15) == "High"


def test_uncertainty_label_varies_by_horizon():
    assert "Lower" in uncertainty_label(1)
    assert "Moderate" in uncertainty_label(7)
    assert "Higher" in uncertainty_label(30)


def test_build_decision_support_result_recommended_ge_predicted():
    result = build_decision_support_result(
        predicted_meals=480, recent_std=40, horizon_days=1, horizon_label="Next school day",
    )
    assert result.recommended_meals >= result.predicted_meals
    assert result.risk_level in {"Low", "Moderate", "High"}
    assert result.model_label == "Tuned XGBoost"
