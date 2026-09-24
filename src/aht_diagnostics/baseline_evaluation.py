"""Shared evaluation utilities for diagnostic baseline comparisons."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .baselines import tie_aware_top_budget_weights


PairingKey = tuple[int, int]
CandidateKey = tuple[int, int, str]


def pairing_key(row: Mapping[str, Any]) -> PairingKey:
    """Return the canonical teammate-policy and environment-start key."""

    return int(row["train_seed"]), int(row["episode_seed"])


def index_unique_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> dict[PairingKey, Mapping[str, Any]]:
    """Index rows and reject accidental duplicate evaluation cells."""

    indexed: dict[PairingKey, Mapping[str, Any]] = {}
    for row in rows:
        key = pairing_key(row)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate pairing key {key!r}")
        indexed[key] = row
    return indexed


def require_matching_pairing_keys(
    reference: Mapping[PairingKey, Any],
    comparison: Mapping[PairingKey, Any],
    *,
    reference_label: str,
    comparison_label: str,
) -> None:
    """Reject joins with missing or extra policy-start cells."""

    reference_keys = set(reference)
    comparison_keys = set(comparison)
    if reference_keys != comparison_keys:
        missing = sorted(reference_keys - comparison_keys)
        extra = sorted(comparison_keys - reference_keys)
        raise ValueError(
            f"{comparison_label} pairing keys do not match {reference_label}; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )


def acquisition_summary(
    *,
    rows: Mapping[PairingKey, Mapping[str, Any]],
    scores: Mapping[PairingKey, float],
    budget: int,
    operational_candidate_keys: set[PairingKey],
    intervention_supported_keys: set[PairingKey],
) -> dict[str, float | int]:
    """Evaluate a score-only selector under a fixed expected query budget.

    ``intervention_supported_keys`` are outcomes from the frozen operational
    adjudicator, not independent ground truth. The returned field names retain
    that distinction.
    """

    row_keys = set(rows)
    if set(scores) != row_keys:
        raise ValueError("score keys must exactly match canonical row keys")
    if not operational_candidate_keys <= row_keys:
        raise ValueError("operational candidate keys fall outside canonical rows")
    if not intervention_supported_keys <= operational_candidate_keys:
        raise ValueError(
            "intervention-supported keys must be a subset of candidate keys"
        )

    ordered_keys = sorted(row_keys)
    weights = tie_aware_top_budget_weights(
        [float(scores[key]) for key in ordered_keys],
        budget=budget,
    )
    weighted = dict(zip(ordered_keys, weights, strict=True))

    def expected(predicate: Any) -> float:
        return sum(
            weight for key, weight in weighted.items() if bool(predicate(key))
        )

    return {
        "pairings": len(ordered_keys),
        "query_budget": budget,
        "expected_selected": float(sum(weights)),
        "expected_candidate_acquisitions": expected(
            lambda key: key in operational_candidate_keys
        ),
        "expected_intervention_supported_acquisitions": expected(
            lambda key: key in intervention_supported_keys
        ),
        "expected_zero_return_acquisitions": expected(
            lambda key: float(rows[key]["factual_return"]) == 0.0
        ),
        "expected_competence_witness_acquisitions": expected(
            lambda key: bool(rows[key]["initial_state_competence_witness"])
        ),
        "score_boundary": (
            float(sorted(scores.values(), reverse=True)[budget - 1])
            if budget > 0
            else 0.0
        ),
        "boundary_ties": (
            sum(
                float(value)
                == float(sorted(scores.values(), reverse=True)[budget - 1])
                for value in scores.values()
            )
            if budget > 0
            else 0
        ),
    }


def compare_candidate_decisions(
    baseline: Mapping[CandidateKey, str],
    full_adjudication: Mapping[CandidateKey, str],
) -> dict[str, Any]:
    """Compare candidate-level decisions without collapsing their semantics."""

    if set(baseline) != set(full_adjudication):
        raise ValueError("baseline and full adjudication candidate keys must match")
    counts = Counter(
        f"{baseline[key]} -> {full_adjudication[key]}"
        for key in baseline
    )
    positive = {
        "nominated",
        "supported_by_primary_only",
    }
    return {
        "candidate_tests": len(baseline),
        "comparison_counts": dict(sorted(counts.items())),
        "full_supported": sum(
            decision == "supported" for decision in full_adjudication.values()
        ),
        "baseline_positive_full_not_supported": sum(
            baseline[key] in positive and full_adjudication[key] != "supported"
            for key in baseline
        ),
    }


def candidate_key(row: Mapping[str, Any]) -> CandidateKey:
    """Return the canonical key for one candidate test."""

    return (
        int(row["train_seed"]),
        int(row["episode_seed"]),
        str(row["candidate_kind"]),
    )


def require_unique_candidate_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[CandidateKey, Mapping[str, Any]]:
    """Index candidate tests and reject duplicate mechanisms per pairing."""

    indexed: dict[CandidateKey, Mapping[str, Any]] = {}
    for row in rows:
        key = candidate_key(row)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate candidate key {key!r}")
        indexed[key] = row
    return indexed
