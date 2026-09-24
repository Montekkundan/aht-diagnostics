"""Regression checks for the interpretation of frozen confirmation decisions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_diagnostic_claim_scope.py"
RESULTS = SCRIPT.parents[1] / "results/confirmation"
NEEDED_RESULTS = (
    "diagnostic_baseline_confirmation_rows.json",
    "diagnostic_baseline_confirmation_suite.json",
    "diagnostic_baseline_confirmation_comparison.json",
)
requires_saved_results = pytest.mark.skipif(
    any(not (RESULTS / name).is_file() for name in NEEDED_RESULTS),
    reason="confirmation results are not distributed with the code",
)
SPEC = importlib.util.spec_from_file_location("claim_scope_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@requires_saved_results
def test_control_downgrades_are_not_specific_movement_repairs() -> None:
    result = MODULE.audit()
    changed = result["full_control_downgrades"]
    assert len(changed) == 4
    assert all(item["specificity"] == "wait_also_improves" for item in changed)
    assert all(item["duration_matched_wait_effect"] > 0 for item in changed)


@requires_saved_results
def test_swap_triggers_include_successful_factual_episodes() -> None:
    result = MODULE.audit()
    assert result["swap_trigger_by_factual_return"] == {0: 255, 20: 140, 40: 8}
    assert result["swap_trigger_cells_with_positive_factual_return"] == 148
