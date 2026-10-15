"""Transparent baselines for frozen-ego diagnostic evaluation.

The functions in this module deliberately separate three information levels:

* score-only ranking from an approximate-reference return gap;
* factual-trajectory rules derived only from observed semantic events; and
* a fixed behavioural unit test that compares one intervention with exact
  no-change replay, but does not use matched alternative-mechanism controls.

These baselines do not treat the full diagnostic suite's own adjudication as
independent ground truth. They provide matched acquisition or evidence outputs
that can be compared under a common, frozen downstream protocol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

from .semantics import COOPERATIVE_EVENTS


SEMANTIC_EVENT_NAMES = (
    "blocked_move",
    "deliver_soup",
    "drop_onion",
    "drop_plate",
    "drop_soup",
    "fetch_onion",
    "fetch_plate",
    "interact_no_effect",
    "move",
    "orientation_change",
    "pick_up_soup",
    "place_onion_in_pot",
    "terminal_state_unobserved",
    "wait",
)

_EVENT_RULE_DIAGNOSES = {
    "initial_completed_soup_drop": "initial_soup_completion_failure",
    "soup_carrier_swap_lock": "reciprocal_motion_conflict",
}

_PRIMARY_ONLY_DIAGNOSES = {
    "initial_completed_soup_drop": "initial_soup_movement_sensitive_completion",
    "soup_carrier_swap_lock": "reciprocal_motion_conflict",
}


def approximate_reference_gap(
    factual_return: float,
    reference_return: float,
) -> float:
    """Return the approximate-reference score gap without clipping.

    A positive value means the evaluated ego scored lower than the learned
    reference with the same teammate. A negative value is retained because an
    approximate reference is not an oracle and can underperform the ego.
    """

    return float(reference_return) - float(factual_return)


def factual_event_features(episode: Any) -> dict[str, float | int]:
    """Summarize only factual semantic events from one completed episode.

    Every event count is divided by the observed episode length so episodes of
    different durations remain comparable. Inventory values in ``StepRecord``
    describe the state before the recorded action; they therefore support the
    explicit non-empty-wait feature without reading a counterfactual branch.
    """

    records = tuple(episode.steps)
    if not records:
        raise ValueError("factual_event_features requires at least one step")

    allowed = set(SEMANTIC_EVENT_NAMES)
    ego_counts = Counter(str(record.ego_event) for record in records)
    partner_counts = Counter(str(record.partner_event) for record in records)
    unexpected = (set(ego_counts) | set(partner_counts)) - allowed
    if unexpected:
        raise ValueError(f"unknown semantic events: {sorted(unexpected)!r}")

    count = len(records)
    features: dict[str, float | int] = {"steps": count}
    for event in SEMANTIC_EVENT_NAMES:
        features[f"ego_{event}_rate"] = ego_counts[event] / count
        features[f"partner_{event}_rate"] = partner_counts[event] / count

    features["ego_nonempty_wait_rate"] = sum(
        record.ego_event == "wait" and record.ego_inventory != "empty"
        for record in records
    ) / count
    features["partner_nonempty_wait_rate"] = sum(
        record.partner_event == "wait" and record.partner_inventory != "empty"
        for record in records
    ) / count
    features["duplicate_cooperative_event_rate"] = sum(
        record.ego_event == record.partner_event
        and record.ego_event in COOPERATIVE_EVENTS
        for record in records
    ) / count
    return features


def weighted_event_burden(
    features: Mapping[str, float | int],
    weights: Mapping[str, float],
) -> float:
    """Combine factual event rates using an explicit, externally frozen rule."""

    score = 0.0
    for name, weight in weights.items():
        if name not in features:
            raise KeyError(f"event-burden weight references unknown feature {name!r}")
        value = float(features[name])
        coefficient = float(weight)
        if not math.isfinite(value) or not math.isfinite(coefficient):
            raise ValueError("event-burden features and weights must be finite")
        score += coefficient * value
    return score


def _diagnosis_for(candidate_kind: str, *, primary_only: bool) -> str:
    mapping = _PRIMARY_ONLY_DIAGNOSES if primary_only else _EVENT_RULE_DIAGNOSES
    try:
        return mapping[candidate_kind]
    except KeyError as error:
        raise ValueError(
            f"unknown candidate_kind {candidate_kind!r}; expected one of "
            f"{sorted(mapping)!r}"
        ) from error


def event_rule_decision(
    *,
    candidate_kind: str,
    triggered: bool,
    interval: tuple[int, int] | None,
    score: float,
) -> dict[str, Any]:
    """Convert a factual trigger into a bounded event-rule nomination.

    The output intentionally says ``nominated`` rather than ``supported``.
    Factual event rules do not run a counterfactual and therefore cannot show
    that changing the nominated behaviour would improve the outcome.
    """

    diagnosis = _diagnosis_for(candidate_kind, primary_only=False)
    if interval is not None:
        start, end = map(int, interval)
        if start < 0 or end < start:
            raise ValueError("candidate interval must satisfy 0 <= start <= end")
        candidate_interval: list[int] | None = [start, end]
    else:
        candidate_interval = None
    if triggered and candidate_interval is None:
        raise ValueError("a triggered event rule requires a candidate interval")

    return {
        "candidate_kind": candidate_kind if triggered else None,
        "candidate_interval": candidate_interval if triggered else None,
        "proposed_intervention": None,
        "score": float(score),
        "decision": "nominated" if triggered else "abstain",
        "diagnosis": diagnosis if triggered else None,
        "uses_counterfactual_outcome": False,
    }


def primary_only_unit_test_decision(
    *,
    candidate_kind: str,
    exact_replay: bool,
    no_change_return: float,
    primary_return: float,
    minimum_effect: float = 1e-9,
) -> dict[str, Any]:
    """Interpret a fixed primary intervention without specificity controls.

    This is the strongest simple behavioural-unit-test baseline: it checks that
    a predeclared intervention improves on exact no-change replay. It does not
    compare duration-matched waiting or alternative movement mechanisms, so its
    output must not be presented as evidence that the primary mechanism is
    uniquely responsible.
    """

    diagnosis = _diagnosis_for(candidate_kind, primary_only=True)
    baseline = float(no_change_return)
    effect = float(primary_return) - baseline
    if not exact_replay:
        decision = "invalid"
        reported_diagnosis = None
    elif effect > minimum_effect:
        decision = "supported_by_primary_only"
        reported_diagnosis = diagnosis
    else:
        decision = "no_primary_effect"
        reported_diagnosis = None
    return {
        "candidate_kind": candidate_kind,
        "decision": decision,
        "diagnosis": reported_diagnosis,
        "no_change_return": baseline,
        "primary_return": float(primary_return),
        "effect": effect,
        "exact_replay": bool(exact_replay),
        "alternative_mechanisms_tested": False,
    }


def tie_aware_top_budget_weights(
    scores: Sequence[float],
    *,
    budget: int,
) -> list[float]:
    """Return expected inclusion weights for a highest-score acquisition budget.

    Rows strictly above the boundary receive weight one. Rows tied at the
    boundary share the remaining budget uniformly. This avoids treating input
    order as scientific evidence when a score-only baseline produces ties.
    """

    values = [float(score) for score in scores]
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError("budget must be an integer")
    if budget < 0 or budget > len(values):
        raise ValueError("budget must be between zero and the number of scores")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    if budget == 0:
        return [0.0] * len(values)
    if budget == len(values):
        return [1.0] * len(values)

    boundary = sorted(values, reverse=True)[budget - 1]
    above = sum(value > boundary for value in values)
    tied = sum(value == boundary for value in values)
    tie_weight = (budget - above) / tied
    return [
        1.0 if value > boundary else tie_weight if value == boundary else 0.0
        for value in values
    ]
