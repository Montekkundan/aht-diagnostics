"""Combine frozen candidate tests into an ego-specific diagnostic summary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .diagnosis import (
    classify_intervention_evidence,
    interpret_contextual_diagnosis,
)


def _remaining_return(case: Mapping[str, Any], branch: str) -> float:
    return float(case["branches"][branch]["remaining_return"])


def diagnose_swap_lock_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Interpret one frozen soup-carrier swap-lock intervention test."""

    evidence = classify_intervention_evidence(
        exact_replay=bool(case["no_change_exact_replay"]),
        no_change_return=_remaining_return(case, "no_change"),
        primary_return=_remaining_return(case, "detour_down_left_2"),
        wait_return=_remaining_return(case, "control_wait_2"),
        alternative_returns={
            "down_stay": _remaining_return(case, "control_down_stay_2"),
            "left_down": _remaining_return(case, "control_left_down_2"),
        },
        success_return=20.0,
    )
    return {
        "train_seed": int(case["train_seed"]),
        "episode_seed": int(case["episode_seed"]),
        **interpret_contextual_diagnosis(
            candidate_kind="soup_carrier_swap_lock",
            intervention_evidence=evidence,
        ),
    }


def diagnose_initial_soup_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Interpret one frozen initial-completed-soup intervention test."""

    one_step = {
        name.removeprefix("one_step_"): _remaining_return(case, name)
        for name in (
            "one_step_up",
            "one_step_down",
            "one_step_left",
            "one_step_right",
            "one_step_stay",
            "one_step_interact",
        )
    }
    evidence = classify_intervention_evidence(
        exact_replay=bool(case["no_change_exact_replay"]),
        no_change_return=_remaining_return(case, "no_change"),
        primary_return=_remaining_return(case, "reference_actions_no_interact"),
        wait_return=_remaining_return(case, "reference_actions_wait"),
        alternative_returns=one_step,
        success_return=20.0,
    )
    return {
        "train_seed": int(case["train_seed"]),
        "episode_seed": int(case["episode_seed"]),
        **interpret_contextual_diagnosis(
            candidate_kind="initial_completed_soup_drop",
            intervention_evidence=evidence,
        ),
    }


def summarize_diagnostic_suite(
    *,
    swap_result: Mapping[str, Any],
    initial_soup_result: Mapping[str, Any],
    capable_control_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize two frozen candidate tests over the same policy-start split.

    The initial-soup analysis records every factual pairing, which supplies the
    denominator. The swap-lock result records only detected candidates. Both
    inputs must name identical policies, checkpoints, and source results.
    """

    swap_provenance = swap_result["provenance"]
    initial_provenance = initial_soup_result["provenance"]
    for key in (
        "checkpoint_dataset_revision",
        "fixed_checkpoint_sha256",
        "saved_policy_sha256",
        "source_results",
    ):
        if swap_provenance[key] != initial_provenance[key]:
            raise ValueError(f"diagnostic inputs disagree on provenance field {key}")

    exposures = list(initial_soup_result["pairing_exposures"])
    exposure_keys = {
        (int(row["train_seed"]), int(row["episode_seed"])) for row in exposures
    }
    if len(exposure_keys) != len(exposures):
        raise ValueError("initial-soup pairing exposures contain duplicate keys")
    evaluated_pairings = int(initial_soup_result["summary"]["evaluated_pairings"])
    if len(exposures) != evaluated_pairings:
        raise ValueError("pairing exposure count does not match evaluated pairings")
    if int(swap_result["summary"]["evaluated_pairings"]) != evaluated_pairings:
        raise ValueError("diagnostic inputs cover different numbers of pairings")

    swap_diagnoses = [
        diagnose_swap_lock_case(case) for case in swap_result.get("cases", [])
    ]
    initial_diagnoses = [
        diagnose_initial_soup_case(case)
        for case in initial_soup_result.get("cases", [])
    ]
    diagnoses = sorted(
        [*swap_diagnoses, *initial_diagnoses],
        key=lambda row: (
            row["train_seed"],
            row["episode_seed"],
            row["candidate_kind"],
        ),
    )
    if any(
        (row["train_seed"], row["episode_seed"]) not in exposure_keys
        for row in diagnoses
    ):
        raise ValueError("a candidate diagnosis is outside the factual pairing split")

    capable_control_summary: dict[str, Any] | None = None
    if capable_control_result is not None:
        control_provenance = capable_control_result["provenance"]
        if control_provenance["source_results"] != initial_provenance["source_results"]:
            raise ValueError("capable control and diagnostic inputs use different source results")
        if (
            control_provenance["source_implementation_reference"]
            != initial_provenance["source_implementation_reference"]
        ):
            raise ValueError(
                "capable control and diagnostic inputs use different implementations"
            )

        control_rows = list(capable_control_result["rows"])
        control_by_key = {
            (int(row["train_seed"]), int(row["episode_seed"])): row
            for row in control_rows
        }
        if len(control_by_key) != len(control_rows):
            raise ValueError("capable-control rows contain duplicate keys")
        if set(control_by_key) != exposure_keys:
            raise ValueError("capable control and diagnostic inputs cover different pairings")

        exposure_by_key = {
            (int(row["train_seed"]), int(row["episode_seed"])): row
            for row in exposures
        }
        if any(
            float(control_by_key[key]["fixed_return"])
            != float(exposure_by_key[key]["factual_return"])
            for key in exposure_keys
        ):
            raise ValueError("capable-control fixed returns disagree with factual returns")

        swap_case_keys = {
            (int(case["train_seed"]), int(case["episode_seed"]))
            for case in swap_result.get("cases", [])
        }
        control_trigger_keys = {
            key for key, row in control_by_key.items() if bool(row["fixed_trigger"])
        }
        if control_trigger_keys != swap_case_keys:
            raise ValueError(
                "capable-control fixed triggers disagree with swap-lock candidates"
            )

        threshold = float(capable_control_result["summary"]["competence_threshold"])
        for row in diagnoses:
            key = (row["train_seed"], row["episode_seed"])
            control = control_by_key[key]
            reference_return = float(control["reference_return"])
            row["initial_state_reference_return"] = reference_return
            row["initial_state_competence_witness"] = reference_return >= threshold
            row["reference_swap_lock_trigger"] = bool(control["reference_trigger"])

        competent_reference_rows = [
            row
            for row in control_rows
            if float(row["reference_return"]) >= threshold
        ]
        capable_control_summary = {
            "competence_threshold": threshold,
            "competent_reference_pairings": len(competent_reference_rows),
            "reference_swap_lock_triggers": sum(
                bool(row["reference_trigger"]) for row in control_rows
            ),
            "reference_swap_lock_triggers_among_competent_pairings": sum(
                bool(row["reference_trigger"]) for row in competent_reference_rows
            ),
        }

    candidate_keys = {
        (row["train_seed"], row["episode_seed"]) for row in diagnoses
    }
    supported_keys = {
        (row["train_seed"], row["episode_seed"])
        for row in diagnoses
        if row["decision"] == "supported"
    }
    ambiguous_keys = {
        (row["train_seed"], row["episode_seed"])
        for row in diagnoses
        if row["decision"] == "ambiguous"
    } - supported_keys
    zero_return_keys = {
        (int(row["train_seed"]), int(row["episode_seed"]))
        for row in exposures
        if float(row["factual_return"]) == 0.0
    }
    candidate_counts = Counter(row["candidate_kind"] for row in diagnoses)
    decision_counts = Counter(row["decision"] for row in diagnoses)
    diagnosis_counts = Counter(
        row["diagnosis"] for row in diagnoses if row["diagnosis"] is not None
    )
    supported_diagnoses = [
        row for row in diagnoses if row["decision"] == "supported"
    ]

    summary = {
        "evaluated_pairings": evaluated_pairings,
        "zero_return_pairings": len(zero_return_keys),
        "candidate_tests": len(diagnoses),
        "candidate_pairings": len(candidate_keys),
        "candidate_pairing_overlap": len(diagnoses) - len(candidate_keys),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "supported_pairings": len(supported_keys),
        "ambiguous_pairings": len(ambiguous_keys),
        "abstained_zero_return_pairings": len(zero_return_keys - candidate_keys),
        "supported_coverage_over_zero_return": (
            len(supported_keys & zero_return_keys) / len(zero_return_keys)
            if zero_return_keys
            else None
        ),
        "candidate_coverage_over_zero_return": (
            len(candidate_keys & zero_return_keys) / len(zero_return_keys)
            if zero_return_keys
            else None
        ),
    }
    if capable_control_summary is not None:
        summary.update(
            {
                "supported_diagnoses_with_initial_state_competence_witness": sum(
                    bool(row["initial_state_competence_witness"])
                    for row in supported_diagnoses
                ),
                "supported_diagnoses_reference_inconclusive": sum(
                    not bool(row["initial_state_competence_witness"])
                    for row in supported_diagnoses
                ),
                "swap_lock_capable_control": capable_control_summary,
            }
        )

    return {
        "diagnoses": diagnoses,
        "summary": summary,
        "provenance": {
            key: initial_provenance[key]
            for key in (
                "checkpoint_dataset_revision",
                "fixed_checkpoint_sha256",
                "saved_policy_sha256",
                "source_results",
            )
        },
    }
