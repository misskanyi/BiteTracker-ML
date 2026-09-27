"""Reusable decision-support logic layered on top of raw model predictions.

Deliberately kept separate from the ML pipeline (`src/forecasting.py`,
`src/model_training.py`, etc.): the safety buffer, risk level, and
recommended-preparation figures below are *planning heuristics*, not model
outputs, and must stay auditable independently of the model.

Safety buffer logic
--------------------
`recommended = predicted_demand + safety_buffer`

The buffer is driven by two observable, explainable factors:

1. **Demand volatility** — the school's recent day-to-day standard deviation
   in meals served (`rolling_std_7`, already computed by
   `src.feature_engineering`). A school with volatile attendance needs a
   bigger cushion than one with a very stable pattern.
2. **Forecast horizon** — multi-day forecasts feed predictions back into
   their own lag features (see `src.forecasting`), so uncertainty compounds
   the further out the horizon goes. The buffer scales up with horizon.

`buffer = predicted_demand * base_rate * horizon_multiplier`, where
`base_rate` is derived from the volatility-to-demand ratio (coefficient of
variation), capped so the buffer never exceeds a sane share of demand. This
is a heuristic, not a statistically fitted prediction interval — labelled as
such everywhere it's shown.
"""

# In simple terms: takes the model's raw prediction and adds a safety
# cushion (extra meals) so schools don't run short if demand is shaky.

from __future__ import annotations

from dataclasses import dataclass, field

# Horizon multipliers: day-1 predictions carry the least compounding error;
# day-30 predictions (recursive, self-fed) carry the most. Chosen to reflect
# `src.forecasting`'s autoregressive design, not fitted to held-out error.
_HORIZON_MULTIPLIERS = {1: 1.0, 7: 1.4, 30: 2.0}

# Buffer is capped at this share of predicted demand, however volatile the
# school's history, so the heuristic can't recommend absurd over-preparation.
_MAX_BUFFER_SHARE = 0.20

# Below this coefficient of variation, volatility contributes ~0 buffer;
# scaling is linear above it up to _MAX_BUFFER_SHARE.
_CV_FLOOR = 0.02
_CV_CEILING = 0.25


def _horizon_multiplier(horizon_days: int) -> float:
    """Nearest defined multiplier for horizons that don't exactly match 1/7/30."""
    if horizon_days <= 1:
        return _HORIZON_MULTIPLIERS[1]
    if horizon_days <= 7:
        return _HORIZON_MULTIPLIERS[7]
    return _HORIZON_MULTIPLIERS[30]


def compute_safety_buffer(predicted_demand: float, recent_std: float, horizon_days: int = 1) -> dict:
    """Compute a transparent planning buffer for one prediction.

    `recent_std`: the school's recent rolling standard deviation of
    meals_served (e.g. `rolling_std_7`). Pass 0 if unavailable — the buffer
    then falls back to a small flat share of predicted demand.

    Returns a dict with `buffer_meals`, `buffer_pct`, and the intermediate
    `coefficient_of_variation` so the UI can show its working, not just the
    final number.
    """
    predicted_demand = max(predicted_demand, 0.0)
    cv = (recent_std / predicted_demand) if predicted_demand > 0 else 0.0

    cv_clamped = min(max(cv, _CV_FLOOR), _CV_CEILING)
    volatility_share = (cv_clamped - _CV_FLOOR) / (_CV_CEILING - _CV_FLOOR) * _MAX_BUFFER_SHARE

    buffer_share = min(volatility_share * _horizon_multiplier(horizon_days), _MAX_BUFFER_SHARE)
    buffer_meals = round(predicted_demand * buffer_share)

    return {
        "buffer_meals": buffer_meals,
        "buffer_pct": round(buffer_share * 100, 1),
        "coefficient_of_variation": round(cv, 3),
    }


def classify_risk(buffer_pct: float) -> str:
    """Map buffer percentage (a proxy for demand volatility + horizon) to a qualitative risk level."""
    if buffer_pct < 5:
        return "Low"
    if buffer_pct < 12:
        return "Moderate"
    return "High"


def uncertainty_label(horizon_days: int) -> str:
    """Qualitative uncertainty description — no fabricated confidence percentage.

    Grounded in `src.forecasting`'s documented design: multi-day forecasts
    are recursive (each day's prediction feeds the next), so error compounds
    with horizon. This labels that fact rather than inventing a confidence
    interval the model doesn't actually produce.
    """
    if horizon_days <= 1:
        return "Lower uncertainty — same-day-ahead prediction uses only observed history."
    if horizon_days <= 7:
        return "Moderate uncertainty — later days in this horizon reuse earlier predictions as if observed."
    return "Higher uncertainty — long-horizon forecasts compound recursive prediction error; treat as directional."


@dataclass
class DecisionSupportResult:
    """Reusable decision-support bundle for any single-point meal-demand prediction.

    Rendered consistently by every prediction-producing page (Forecast,
    Specific Date Prediction, Scenario Simulator) via
    `app/theme.py::decision_support_card` (kept in the UI layer since it's
    pure presentation of these already-computed values).
    """

    predicted_meals: float
    recommended_meals: int
    buffer_meals: int
    buffer_pct: float
    risk_level: str
    uncertainty: str
    horizon_label: str
    model_label: str
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_decision_support_result(
    predicted_meals: float,
    recent_std: float,
    horizon_days: int,
    horizon_label: str,
    model_label: str = "Tuned XGBoost",
    assumptions: list[str] | None = None,
    warnings: list[str] | None = None,
) -> DecisionSupportResult:
    """Single entry point pages should call to turn a raw prediction into the full decision-support bundle."""
    buffer = compute_safety_buffer(predicted_meals, recent_std, horizon_days)
    risk = classify_risk(buffer["buffer_pct"])
    recommended = round(predicted_meals) + buffer["buffer_meals"]

    return DecisionSupportResult(
        predicted_meals=round(predicted_meals),
        recommended_meals=recommended,
        buffer_meals=buffer["buffer_meals"],
        buffer_pct=buffer["buffer_pct"],
        risk_level=risk,
        uncertainty=uncertainty_label(horizon_days),
        horizon_label=horizon_label,
        model_label=model_label,
        assumptions=assumptions or [],
        warnings=warnings or [],
    )
