"""Checks for the explicitly retrospective secondary comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_retrospective_baselines.py"
SPEC = importlib.util.spec_from_file_location("retrospective_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(factual: float, reference: float, score: float) -> dict:
    return {
        "factual_return": factual,
        "reference_return": reference,
        "factual_event_features": {"ego_blocked_move_rate": score},
    }


def test_zero_return_priority_precedes_feature_score() -> None:
    rows = [_row(20, 40, 1.0), _row(0, 20, 0.1)]
    assert MODULE.expected_yield(rows, "ego_blocked_move_rate", 1) == 1.0


def test_ties_have_fractional_expected_selection() -> None:
    rows = [_row(0, 20, 0.5), _row(0, 0, 0.5)]
    assert MODULE.expected_yield(rows, "ego_blocked_move_rate", 1) == 0.5


RESULTS = SCRIPT.parents[1] / "results/confirmation"


@pytest.mark.skipif(
    any(
        not (RESULTS / name).is_file()
        for name in (
            "diagnostic_baseline_confirmation_swap_lock.json",
            "diagnostic_baseline_confirmation_initial_soup.json",
        )
    ),
    reason="confirmation results are not distributed with the code",
)
def test_branch_steps_reproduce_canonical_count() -> None:
    root = SCRIPT.parents[1]
    swap = root / "results/confirmation/diagnostic_baseline_confirmation_swap_lock.json"
    initial = root / "results/confirmation/diagnostic_baseline_confirmation_initial_soup.json"
    assert MODULE.branch_steps(swap)[0] + MODULE.branch_steps(initial)[0] == 2786
