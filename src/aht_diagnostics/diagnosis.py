"""Transparent evidence categories for option-level diagnostic tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_CANDIDATE_KINDS = {
    "initial_completed_soup_drop",
    "soup_carrier_swap_lock",
}


def classify_intervention_evidence(
    *,
    exact_replay: bool,
    no_change_return: float,
    primary_return: float,
    wait_return: float,
    alternative_returns: Mapping[str, float],
    success_return: float,
    minimum_effect: float = 1e-9,
) -> dict[str, Any]:
    """Classify one candidate event from paired continuation outcomes.

    The function does not discover an event or infer policy intent. It only
    separates common evidential outcomes after an observable candidate event
    has been selected and replayed from exact saved state.
    """

    baseline = float(no_change_return)
    primary_effect = float(primary_return) - baseline
    wait_effect = float(wait_return) - baseline
    alternative_effects = {
        name: float(value) - baseline
        for name, value in sorted(alternative_returns.items())
    }
    alternative_works = any(
        effect > minimum_effect for effect in alternative_effects.values()
    )

    if not exact_replay:
        status = "invalid_exact_replay"
        specificity = "not_assessed"
    elif (
        baseline >= success_return
        and primary_effect <= minimum_effect
        and wait_effect <= minimum_effect
        and not alternative_works
    ):
        status = "observed_event_self_resolved"
        specificity = "no_intervention_effect"
    elif wait_effect > minimum_effect:
        status = "non_specific_interruption_effect"
        specificity = "wait_also_improves"
    elif primary_effect > minimum_effect:
        status = "supported_behavioral_limitation"
        specificity = (
            "multiple_movement_repairs"
            if alternative_works
            else "primary_separates_from_controls"
        )
    elif alternative_works:
        status = "alternative_repair_supported"
        specificity = "primary_not_supported"
    else:
        status = "no_supported_limitation"
        specificity = "no_tested_repair_improves"

    return {
        "status": status,
        "specificity": specificity,
        "no_change_return": baseline,
        "primary_effect": primary_effect,
        "wait_effect": wait_effect,
        "alternative_effects": alternative_effects,
        "success_return": float(success_return),
    }


def interpret_contextual_diagnosis(
    *,
    candidate_kind: str,
    intervention_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Map paired intervention evidence to a bounded diagnostic statement.

    This function deliberately separates an observable candidate context from
    the conclusion supported by its counterfactual controls. It does not infer
    a policy's internal intent, and it never converts a trigger alone into a
    diagnosed limitation.
    """

    if candidate_kind not in _CANDIDATE_KINDS:
        raise ValueError(
            f"unknown candidate_kind {candidate_kind!r}; expected one of "
            f"{sorted(_CANDIDATE_KINDS)!r}"
        )

    evidence_status = str(intervention_evidence["status"])
    specificity = str(intervention_evidence["specificity"])

    if evidence_status == "invalid_exact_replay":
        decision = "invalid"
        diagnosis = None
        explanation = "Exact replay failed, so branch effects are not comparable."
    elif evidence_status == "observed_event_self_resolved":
        decision = "no_supported_limitation"
        diagnosis = None
        explanation = "The unmodified continuation succeeds; the event is a lookalike."
    elif evidence_status == "non_specific_interruption_effect":
        decision = "ambiguous"
        diagnosis = "timing_sensitive_completion"
        explanation = (
            "A duration-matched pause also improves return, so the tested route "
            "does not identify a specific cooperative behaviour."
        )
    elif evidence_status == "supported_behavioral_limitation":
        decision = "supported"
        if candidate_kind == "soup_carrier_swap_lock":
            diagnosis = "reciprocal_motion_conflict"
            explanation = (
                "A short movement response resolves a persistent conflict that "
                "matched waiting does not resolve."
            )
        else:
            diagnosis = "initial_soup_movement_sensitive_completion"
            explanation = (
                "A movement intervention restores completion while matched "
                "waiting does not; the final delivery interaction is not implied."
            )
    elif evidence_status == "alternative_repair_supported":
        decision = "ambiguous"
        diagnosis = "alternative_movement_repair"
        explanation = (
            "A tested alternative improves return but the primary hypothesis does "
            "not, so the proposed mechanism is unsupported."
        )
    elif evidence_status == "no_supported_limitation":
        decision = "no_supported_limitation"
        diagnosis = None
        explanation = "No tested intervention improves the factual continuation."
    else:
        raise ValueError(f"unknown intervention evidence status {evidence_status!r}")

    return {
        "candidate_kind": candidate_kind,
        "decision": decision,
        "diagnosis": diagnosis,
        "specificity": specificity,
        "explanation": explanation,
        "intervention_evidence": dict(intervention_evidence),
    }
