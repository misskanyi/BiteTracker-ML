"""Tests for src.resource_allocation: school summaries, comparison, allocation gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.resource_allocation import build_reallocation_note, compare_schools, compute_allocation, summarize_school


class _StubModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["lag_1_meals"].fillna(300) + 1).to_numpy()


def _fake_processed_df(n_days: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for school, base, cost in [("SCH001", 300, 20.0), ("SCH002", 500, 25.0)]:
        for d in dates:
            rows.append(
                {
                    "date": d, "school_id": school, "school_name": f"{school} Primary",
                    "enrolled_students": 400 if school == "SCH001" else 600,
                    "students_present": base * 0.9 + rng.normal(0, 5),
                    "meals_served": base + rng.normal(0, 10),
                    "meals_prepared": base + 10,
                    "leftover_meals": 10 + rng.normal(0, 2),
                    "average_cost_per_meal": cost,
                    "rainfall_mm": max(0.0, rng.normal(3, 2)),
                    "temperature_c": rng.normal(22, 1),
                    "is_holiday": 0, "is_exam_period": 0, "is_school_event": 0, "month": d.month,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_school_returns_expected_keys():
    df = _fake_processed_df()
    summary = summarize_school(df, "SCH001", model=_StubModel())
    assert summary["school_id"] == "SCH001"
    assert summary["predicted_demand"] >= 0
    assert summary["risk_level"] in {"Low", "Moderate", "High"}
    assert summary["expected_daily_cost"] >= 0


def test_summarize_school_invalid_id_raises():
    df = _fake_processed_df()
    with pytest.raises(ValueError):
        summarize_school(df, "NOPE", model=_StubModel())


def test_compare_schools_returns_row_per_school():
    df = _fake_processed_df()
    comparison = compare_schools(df, ["SCH001", "SCH002"], model=_StubModel())
    assert len(comparison) == 2
    assert set(comparison["school_id"]) == {"SCH001", "SCH002"}


def test_compute_allocation_shortage():
    result = compute_allocation(predicted_demand=520, available_capacity=470)
    assert result["status"] == "Shortage"
    assert result["gap"] == 50


def test_compute_allocation_surplus():
    result = compute_allocation(predicted_demand=390, available_capacity=450)
    assert result["status"] == "Surplus"
    assert result["gap"] == -60


def test_compute_allocation_balanced():
    result = compute_allocation(predicted_demand=400, available_capacity=400)
    assert result["status"] == "Balanced"


def test_build_reallocation_note_present_when_mixed():
    allocations = [
        {"gap": 50, "school_name": "A"},
        {"gap": -60, "school_name": "B"},
    ]
    assert build_reallocation_note(allocations) is not None


def test_build_reallocation_note_absent_when_all_shortage():
    allocations = [{"gap": 50, "school_name": "A"}, {"gap": 20, "school_name": "B"}]
    assert build_reallocation_note(allocations) is None
