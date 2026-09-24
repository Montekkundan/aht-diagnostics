import pytest

from aht_diagnostics.baseline_evaluation import (
    acquisition_summary,
    compare_candidate_decisions,
    index_unique_rows,
)


def test_index_unique_rows_rejects_duplicate_pairings():
    rows = [
        {"train_seed": 1, "episode_seed": 2},
        {"train_seed": 1, "episode_seed": 2},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        index_unique_rows(rows, label="canonical")


def test_acquisition_summary_uses_tie_aware_expected_counts():
    rows = {
        (1, 1): {
            "factual_return": 0.0,
            "initial_state_competence_witness": True,
        },
        (1, 2): {
            "factual_return": 0.0,
            "initial_state_competence_witness": False,
        },
        (1, 3): {
            "factual_return": 20.0,
            "initial_state_competence_witness": True,
        },
        (1, 4): {
            "factual_return": 20.0,
            "initial_state_competence_witness": True,
        },
    }
    scores = {
        (1, 1): 10.0,
        (1, 2): 8.0,
        (1, 3): 8.0,
        (1, 4): 1.0,
    }

    summary = acquisition_summary(
        rows=rows,
        scores=scores,
        budget=2,
        operational_candidate_keys={(1, 1), (1, 2)},
        intervention_supported_keys={(1, 1)},
    )

    assert summary["expected_candidate_acquisitions"] == 1.5
    assert summary["expected_intervention_supported_acquisitions"] == 1.0
    assert summary["expected_zero_return_acquisitions"] == 1.5
    assert summary["expected_competence_witness_acquisitions"] == 1.5


def test_acquisition_summary_rejects_score_key_mismatch():
    rows = {
        (1, 1): {
            "factual_return": 0.0,
            "initial_state_competence_witness": True,
        }
    }

    with pytest.raises(ValueError, match="score keys"):
        acquisition_summary(
            rows=rows,
            scores={(1, 2): 1.0},
            budget=1,
            operational_candidate_keys=set(),
            intervention_supported_keys=set(),
        )


def test_candidate_decision_comparison_preserves_distinct_output_semantics():
    baseline = {
        (1, 2, "swap"): "supported_by_primary_only",
        (1, 3, "initial"): "supported_by_primary_only",
    }
    full = {
        (1, 2, "swap"): "supported",
        (1, 3, "initial"): "ambiguous",
    }

    result = compare_candidate_decisions(baseline, full)

    assert result["comparison_counts"] == {
        "supported_by_primary_only -> ambiguous": 1,
        "supported_by_primary_only -> supported": 1,
    }
    assert result["full_supported"] == 1
    assert result["baseline_positive_full_not_supported"] == 1
