import pytest

from aht_diagnostics.diagnosis import (
    classify_intervention_evidence,
    interpret_contextual_diagnosis,
)


def classify(
    *,
    exact=True,
    no_change=0.0,
    primary=20.0,
    wait=0.0,
    alternatives=None,
):
    return classify_intervention_evidence(
        exact_replay=exact,
        no_change_return=no_change,
        primary_return=primary,
        wait_return=wait,
        alternative_returns=alternatives or {"movement": 0.0},
        success_return=20.0,
    )


def test_primary_repair_separates_from_controls():
    result = classify()

    assert result["status"] == "supported_behavioral_limitation"
    assert result["specificity"] == "primary_separates_from_controls"


def test_multiple_movement_repairs_preserve_broad_limitation():
    result = classify(alternatives={"left_down": 20.0, "down_stay": 0.0})

    assert result["status"] == "supported_behavioral_limitation"
    assert result["specificity"] == "multiple_movement_repairs"


def test_successful_no_change_continuation_is_self_resolving_event():
    result = classify(
        no_change=20.0,
        primary=20.0,
        wait=20.0,
        alternatives={"movement": 20.0},
    )

    assert result["status"] == "observed_event_self_resolved"
    assert result["primary_effect"] == 0.0


def test_wait_effect_is_not_specific_evidence():
    result = classify(primary=20.0, wait=20.0)

    assert result["status"] == "non_specific_interruption_effect"


def test_exact_replay_failure_invalidates_comparison():
    result = classify(exact=False)

    assert result["status"] == "invalid_exact_replay"


def test_no_improving_branch_supports_no_limitation():
    result = classify(primary=0.0, wait=0.0)

    assert result["status"] == "no_supported_limitation"


def test_swap_lock_evidence_maps_to_bounded_motion_diagnosis():
    evidence = classify()

    result = interpret_contextual_diagnosis(
        candidate_kind="soup_carrier_swap_lock",
        intervention_evidence=evidence,
    )

    assert result["decision"] == "supported"
    assert result["diagnosis"] == "reciprocal_motion_conflict"


def test_wait_sensitive_initial_soup_case_remains_ambiguous():
    evidence = classify(primary=20.0, wait=20.0)

    result = interpret_contextual_diagnosis(
        candidate_kind="initial_completed_soup_drop",
        intervention_evidence=evidence,
    )

    assert result["decision"] == "ambiguous"
    assert result["diagnosis"] == "timing_sensitive_completion"


def test_self_resolving_trigger_is_not_diagnosed():
    evidence = classify(
        no_change=20.0,
        primary=20.0,
        wait=20.0,
        alternatives={"movement": 20.0},
    )

    result = interpret_contextual_diagnosis(
        candidate_kind="soup_carrier_swap_lock",
        intervention_evidence=evidence,
    )

    assert result["decision"] == "no_supported_limitation"
    assert result["diagnosis"] is None


def test_unknown_candidate_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown candidate_kind"):
        interpret_contextual_diagnosis(
            candidate_kind="unknown",
            intervention_evidence=classify(),
        )
