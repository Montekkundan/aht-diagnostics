from types import SimpleNamespace

import pytest

from aht_diagnostics.baselines import (
    approximate_reference_gap,
    event_rule_decision,
    factual_event_features,
    primary_only_unit_test_decision,
    tie_aware_top_budget_weights,
    weighted_event_burden,
)


def step(
    ego_event,
    partner_event,
    *,
    ego_inventory="empty",
    partner_inventory="empty",
):
    return SimpleNamespace(
        ego_event=ego_event,
        partner_event=partner_event,
        ego_inventory=ego_inventory,
        partner_inventory=partner_inventory,
    )


def episode(*steps):
    return SimpleNamespace(steps=steps)


def test_reference_gap_preserves_negative_values():
    assert approximate_reference_gap(40.0, 20.0) == -20.0
    assert approximate_reference_gap(0.0, 20.0) == 20.0


def test_factual_event_features_use_only_observed_step_records():
    observed = episode(
        step("blocked_move", "move", ego_inventory="onion"),
        step("wait", "wait", ego_inventory="dish", partner_inventory="empty"),
        step("fetch_onion", "fetch_onion"),
        step("interact_no_effect", "blocked_move"),
    )

    features = factual_event_features(observed)

    assert features["steps"] == 4
    assert features["ego_blocked_move_rate"] == 0.25
    assert features["ego_interact_no_effect_rate"] == 0.25
    assert features["ego_wait_rate"] == 0.25
    assert features["ego_nonempty_wait_rate"] == 0.25
    assert features["partner_nonempty_wait_rate"] == 0.0
    assert features["duplicate_cooperative_event_rate"] == 0.25


def test_factual_event_features_reject_empty_episode():
    with pytest.raises(ValueError, match="at least one step"):
        factual_event_features(episode())


def test_event_burden_requires_explicit_weights():
    features = {
        "ego_blocked_move_rate": 0.5,
        "ego_interact_no_effect_rate": 0.25,
    }

    assert weighted_event_burden(
        features,
        {
            "ego_blocked_move_rate": 2.0,
            "ego_interact_no_effect_rate": 4.0,
        },
    ) == 2.0

    with pytest.raises(KeyError, match="unknown_feature"):
        weighted_event_burden(features, {"unknown_feature": 1.0})


def test_event_rule_trigger_is_a_nomination_not_causal_support():
    result = event_rule_decision(
        candidate_kind="soup_carrier_swap_lock",
        triggered=True,
        interval=(19, 26),
        score=8.0,
    )

    assert result["decision"] == "nominated"
    assert result["diagnosis"] == "reciprocal_motion_conflict"
    assert result["candidate_interval"] == [19, 26]
    assert result["uses_counterfactual_outcome"] is False


def test_primary_only_unit_test_does_not_claim_specificity_controls():
    result = primary_only_unit_test_decision(
        candidate_kind="soup_carrier_swap_lock",
        exact_replay=True,
        no_change_return=0.0,
        primary_return=20.0,
    )

    assert result["decision"] == "supported_by_primary_only"
    assert result["diagnosis"] == "reciprocal_motion_conflict"
    assert result["effect"] == 20.0
    assert result["alternative_mechanisms_tested"] is False


def test_primary_only_unit_test_handles_invalid_and_null_effects():
    invalid = primary_only_unit_test_decision(
        candidate_kind="initial_completed_soup_drop",
        exact_replay=False,
        no_change_return=0.0,
        primary_return=20.0,
    )
    null = primary_only_unit_test_decision(
        candidate_kind="initial_completed_soup_drop",
        exact_replay=True,
        no_change_return=20.0,
        primary_return=20.0,
    )

    assert invalid["decision"] == "invalid"
    assert null["decision"] == "no_primary_effect"


def test_tie_aware_budget_does_not_use_row_order_as_a_tiebreaker():
    weights = tie_aware_top_budget_weights([10.0, 8.0, 8.0, 1.0], budget=2)

    assert weights == [1.0, 0.5, 0.5, 0.0]
    assert sum(weights) == 2.0


def test_tie_aware_budget_validates_budget():
    with pytest.raises(ValueError, match="budget"):
        tie_aware_top_budget_weights([1.0], budget=2)
