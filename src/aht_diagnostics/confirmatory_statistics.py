"""Frozen clustered statistics for confirmatory diagnostic comparisons.

The policy and environment-start identifiers are the sampling units.  A
counterfactual branch or a candidate test is an outcome nested inside one
policy-start cell, not an independent replication.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import itertools
import math
import random
from typing import Any


PairingKey = tuple[int, int]
CandidateKey = tuple[int, int, str]

_EVALUABLE_FULL_DECISIONS = frozenset(
    {"supported", "ambiguous", "no_supported_limitation"}
)
_FULL_DOWNGRADES = frozenset({"ambiguous", "no_supported_limitation"})
_FULL_DECISIONS = _EVALUABLE_FULL_DECISIONS | {"invalid"}
_PRIMARY_ONLY_DECISIONS = frozenset(
    {"supported_by_primary_only", "no_primary_effect", "invalid"}
)

_ACQUISITION_SCORE_FIELDS = {
    "diagnosis": "diagnosis_score",
    "event": "event_burden_score",
    "raw_gap": "raw_gap_score",
    "random": None,
}
_MAX_EXACT_SIGN_FLIP_POLICIES = 20


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("a percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_bootstrap_arguments(replicates: int, seed: int) -> None:
    if not isinstance(replicates, int) or isinstance(replicates, bool):
        raise ValueError("replicates must be an integer")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _index_acquisition_cells(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    dict[PairingKey, tuple[float, bool, bool]],
]:
    indexed: dict[PairingKey, tuple[float, bool, bool]] = {}
    for row in rows:
        key = int(row["train_seed"]), int(row["episode_seed"])
        if key in indexed:
            raise ValueError(f"duplicate acquisition cell {key!r}")
        score = float(row["event_burden_score"])
        if not math.isfinite(score):
            raise ValueError("event-burden scores must be finite")
        nominated = row["suite_nominated"]
        supported = row["suite_supported"]
        if not isinstance(nominated, bool) or not isinstance(supported, bool):
            raise ValueError("suite nomination and support fields must be booleans")
        if supported and not nominated:
            raise ValueError("a supported cell must also be nominated")
        indexed[key] = score, nominated, supported

    if not indexed:
        raise ValueError("at least one acquisition cell is required")
    policies = tuple(sorted({key[0] for key in indexed}))
    starts = tuple(sorted({key[1] for key in indexed}))
    if len(policies) < 2 or len(starts) < 2:
        raise ValueError(
            "clustered acquisition analysis requires at least two policies "
            "and two starts"
        )
    expected = {(policy, start) for policy in policies for start in starts}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(
            "acquisition rows must form a complete policy-by-start rectangle; "
            f"missing={missing[:5]!r}"
        )
    return policies, starts, indexed


def _score_groups(
    indexed: Mapping[PairingKey, tuple[float, bool, bool]],
) -> tuple[tuple[float, tuple[PairingKey, ...]], ...]:
    grouped: dict[float, list[PairingKey]] = defaultdict(list)
    for key, (score, _, _) in indexed.items():
        grouped[score].append(key)
    return tuple(
        (score, tuple(sorted(keys)))
        for score, keys in sorted(grouped.items(), reverse=True)
    )


def _acquisition_estimate(
    indexed: Mapping[PairingKey, tuple[float, bool, bool]],
    groups: Sequence[tuple[float, Sequence[PairingKey]]],
    multiplicities: Mapping[PairingKey, int],
) -> dict[str, float | int | None]:
    cells = sum(int(multiplicities.get(key, 0)) for key in indexed)
    if cells <= 0:
        raise ValueError("acquisition estimate requires positive cell multiplicity")

    budget = sum(
        int(multiplicities.get(key, 0))
        for key, (_, nominated, _) in indexed.items()
        if nominated
    )
    suite_supported = sum(
        int(multiplicities.get(key, 0))
        for key, (_, _, supported) in indexed.items()
        if supported
    )

    remaining = budget
    event_supported = 0.0
    score_boundary: float | None = None
    boundary_ties = 0
    for score, keys in groups:
        group_size = sum(int(multiplicities.get(key, 0)) for key in keys)
        if group_size == 0:
            continue
        if remaining >= group_size:
            selection_weight = 1.0
            remaining -= group_size
            if remaining == 0:
                score_boundary = float(score)
                boundary_ties = group_size
        elif remaining > 0:
            selection_weight = remaining / group_size
            score_boundary = float(score)
            boundary_ties = group_size
            remaining = 0
        else:
            selection_weight = 0.0
        if selection_weight:
            event_supported += selection_weight * sum(
                int(multiplicities.get(key, 0))
                for key in keys
                if indexed[key][2]
            )

    if budget == 0:
        score_boundary = None
        boundary_ties = 0
    if remaining != 0:
        raise AssertionError("tie-aware selection did not exhaust the query budget")

    suite_rate = suite_supported / cells
    event_rate = event_supported / cells
    return {
        "cells": cells,
        "query_budget": budget,
        "suite_supported": suite_supported,
        "event_expected_supported": event_supported,
        "suite_supported_rate": suite_rate,
        "event_expected_supported_rate": event_rate,
        "acquisition_delta": suite_rate - event_rate,
        "additional_supported_cases": suite_supported - event_supported,
        "score_boundary": score_boundary,
        "boundary_ties": boundary_ties,
    }


def _sample_counts(
    rng: random.Random,
    values: Sequence[int],
) -> Counter[int]:
    return Counter(rng.choice(values) for _ in values)


def _interval_summary(
    estimates: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "replicates": replicates,
        "valid_replicates": len(estimates),
        "invalid_replicates": replicates - len(estimates),
        "seed": seed,
        "ci95": (
            [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]
            if estimates
            else None
        ),
        "estimates": list(estimates),
    }


def _index_external_binary_cells(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    dict[PairingKey, dict[str, float | bool]],
]:
    indexed: dict[PairingKey, dict[str, float | bool]] = {}
    for row in rows:
        key = int(row["train_seed"]), int(row["episode_seed"])
        if key in indexed:
            raise ValueError(f"duplicate external-outcome cell {key!r}")

        target_positive = row["target_positive"]
        factual_zero = row["factual_zero"]
        if not isinstance(target_positive, bool):
            raise ValueError("target_positive must be a boolean")
        if not isinstance(factual_zero, bool):
            raise ValueError("factual_zero must be a boolean")
        if target_positive and not factual_zero:
            raise ValueError("a target-positive cell must have factual_zero=True")

        values: dict[str, float | bool] = {
            "target_positive": target_positive,
            "factual_zero": factual_zero,
        }
        for selector, field in _ACQUISITION_SCORE_FIELDS.items():
            if field is None:
                continue
            score = float(row[field])
            if not math.isfinite(score):
                raise ValueError(f"{selector} scores must be finite")
            values[field] = score
        indexed[key] = values

    if not indexed:
        raise ValueError("at least one external-outcome cell is required")
    policies = tuple(sorted({key[0] for key in indexed}))
    starts = tuple(sorted({key[1] for key in indexed}))
    if len(policies) < 2:
        raise ValueError("policy-cluster analysis requires at least two policies")
    expected = {(policy, start) for policy in policies for start in starts}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            "external-outcome rows must form a complete policy-by-start rectangle; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    return policies, starts, indexed


def _validate_queries_per_policy(queries_per_policy: int, start_count: int) -> None:
    if (
        not isinstance(queries_per_policy, int)
        or isinstance(queries_per_policy, bool)
    ):
        raise ValueError("queries_per_policy must be an integer")
    if not 1 <= queries_per_policy <= start_count:
        raise ValueError(
            "queries_per_policy must be between one and the number of starts"
        )


def _within_policy_selector_estimate(
    *,
    policy: int,
    starts: Sequence[int],
    indexed: Mapping[PairingKey, Mapping[str, float | bool]],
    score_field: str | None,
    queries_per_policy: int,
    start_multiplicities: Mapping[int, int],
) -> dict[str, float | int | None]:
    score_groups: dict[tuple[bool, float], list[int]] = defaultdict(list)
    for start in starts:
        score = (
            0.0
            if score_field is None
            else float(indexed[(policy, start)][score_field])
        )
        factual_zero = bool(indexed[(policy, start)]["factual_zero"])
        score_groups[(factual_zero, score)].append(start)

    remaining = queries_per_policy
    expected_positives = 0.0
    boundary_score: float | None = None
    boundary_factual_zero: bool | None = None
    boundary_ties = 0
    for factual_zero, score in sorted(score_groups, reverse=True):
        group_starts = score_groups[(factual_zero, score)]
        group_size = sum(
            int(start_multiplicities.get(start, 0)) for start in group_starts
        )
        if group_size == 0:
            continue
        if remaining >= group_size:
            selection_weight = 1.0
            remaining -= group_size
            if remaining == 0:
                boundary_score = score
                boundary_factual_zero = factual_zero
                boundary_ties = group_size
        elif remaining > 0:
            selection_weight = remaining / group_size
            boundary_score = score
            boundary_factual_zero = factual_zero
            boundary_ties = group_size
            remaining = 0
        else:
            selection_weight = 0.0
        if selection_weight:
            expected_positives += selection_weight * sum(
                int(start_multiplicities.get(start, 0))
                * int(bool(indexed[(policy, start)]["target_positive"]))
                for start in group_starts
            )

    if remaining != 0:
        raise AssertionError("within-policy selection did not exhaust its query budget")
    return {
        "queries": queries_per_policy,
        "expected_positives": expected_positives,
        "yield": expected_positives / queries_per_policy,
        "score_boundary": boundary_score,
        "factual_zero_boundary": boundary_factual_zero,
        "boundary_ties": boundary_ties,
    }


def _policy_selector_results(
    *,
    policy: int,
    starts: Sequence[int],
    indexed: Mapping[PairingKey, Mapping[str, float | bool]],
    queries_per_policy: int,
    start_multiplicities: Mapping[int, int],
) -> dict[str, Any]:
    selectors = {
        selector: _within_policy_selector_estimate(
            policy=policy,
            starts=starts,
            indexed=indexed,
            score_field=score_field,
            queries_per_policy=queries_per_policy,
            start_multiplicities=start_multiplicities,
        )
        for selector, score_field in _ACQUISITION_SCORE_FIELDS.items()
    }
    diagnosis_minus_event = (
        float(selectors["diagnosis"]["yield"])
        - float(selectors["event"]["yield"])
    )
    return {
        "train_seed": policy,
        "selectors": selectors,
        "diagnosis_minus_event_expected_positives": (
            float(selectors["diagnosis"]["expected_positives"])
            - float(selectors["event"]["expected_positives"])
        ),
        "diagnosis_minus_event_yield": diagnosis_minus_event,
    }


def _aggregate_policy_results(policy_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not policy_results:
        raise ValueError("at least one policy result is required")
    count = len(policy_results)
    selector_means = {
        selector: {
            "mean_expected_positives": sum(
                float(row["selectors"][selector]["expected_positives"])
                for row in policy_results
            )
            / count,
            "mean_yield": sum(
                float(row["selectors"][selector]["yield"])
                for row in policy_results
            )
            / count,
        }
        for selector in _ACQUISITION_SCORE_FIELDS
    }
    return {
        "policies": count,
        "selectors": selector_means,
        "diagnosis_minus_event_expected_positives": (
            selector_means["diagnosis"]["mean_expected_positives"]
            - selector_means["event"]["mean_expected_positives"]
        ),
        "diagnosis_minus_event_yield": (
            selector_means["diagnosis"]["mean_yield"]
            - selector_means["event"]["mean_yield"]
        ),
    }


def _exact_paired_sign_flip(differences: Sequence[float]) -> dict[str, Any]:
    nonzero = [float(value) for value in differences if float(value) != 0.0]
    observed = sum(float(value) for value in differences) / len(differences)
    if len(nonzero) > _MAX_EXACT_SIGN_FLIP_POLICIES:
        return {
            "feasible": False,
            "alternative": "two_sided",
            "observed_mean_difference": observed,
            "policy_count": len(differences),
            "nonzero_policy_count": len(nonzero),
            "permutations": None,
            "p_value": None,
            "reason": (
                "more than "
                f"{_MAX_EXACT_SIGN_FLIP_POLICIES} policies have nonzero paired differences"
            ),
        }

    permutations = 1 << len(nonzero)
    observed_sum = abs(sum(nonzero))
    extreme = 0
    tolerance = 1e-12
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        permuted_sum = abs(
            sum(sign * value for sign, value in zip(signs, nonzero))
        )
        if permuted_sum + tolerance >= observed_sum:
            extreme += 1
    return {
        "feasible": True,
        "alternative": "two_sided",
        "observed_mean_difference": observed,
        "policy_count": len(differences),
        "nonzero_policy_count": len(nonzero),
        "permutations": permutations,
        "p_value": extreme / permutations,
        "reason": None,
    }


def external_binary_acquisition_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    queries_per_policy: int,
    replicates: int = 10_000,
    seed: int = 20_260_922,
) -> dict[str, Any]:
    """Compare equal-budget selectors against an external binary outcome.

    Required row fields are ``train_seed``, ``episode_seed``,
    ``diagnosis_score``, ``event_burden_score``, ``raw_gap_score``, the shared
    observed-failure indicator ``factual_zero`` and the independently observed
    boolean ``target_positive``. Every selector first ranks zero-return cells
    ahead of positive-return cells, then uses its own score within each stratum.
    Each selector receives the same top-``queries_per_policy`` budget separately
    within every policy. If a policy has fewer zero-return cells than the fixed
    budget, every selector fills the remainder from the positive-return stratum.
    Score ties receive fractional expected inclusion.

    The primary interval resamples policies, the independent training unit. A
    policy-by-start pigeonhole bootstrap is reported as a sensitivity analysis;
    it resamples both axes and recomputes every within-policy tie-aware selection.
    """

    _validate_bootstrap_arguments(replicates, seed)
    policies, starts, indexed = _index_external_binary_cells(rows)
    _validate_queries_per_policy(queries_per_policy, len(starts))
    unit_start_multiplicities = {start: 1 for start in starts}
    per_policy = [
        _policy_selector_results(
            policy=policy,
            starts=starts,
            indexed=indexed,
            queries_per_policy=queries_per_policy,
            start_multiplicities=unit_start_multiplicities,
        )
        for policy in policies
    ]
    point = _aggregate_policy_results(per_policy)
    difference_by_policy = {
        int(row["train_seed"]): float(row["diagnosis_minus_event_yield"])
        for row in per_policy
    }

    policy_rng = random.Random(seed)
    policy_bootstrap: list[float] = []
    for _ in range(replicates):
        sampled_policies = [policy_rng.choice(policies) for _ in policies]
        policy_bootstrap.append(
            sum(difference_by_policy[policy] for policy in sampled_policies)
            / len(sampled_policies)
        )

    two_way_rng = random.Random(seed + 1)
    two_way_bootstrap: list[float] = []
    for _ in range(replicates):
        policy_counts = _sample_counts(two_way_rng, policies)
        start_counts = _sample_counts(two_way_rng, starts)
        resampled_policy_results = {
            policy: _policy_selector_results(
                policy=policy,
                starts=starts,
                indexed=indexed,
                queries_per_policy=queries_per_policy,
                start_multiplicities=start_counts,
            )
            for policy in policy_counts
        }
        two_way_bootstrap.append(
            sum(
                policy_counts[policy]
                * float(result["diagnosis_minus_event_yield"])
                for policy, result in resampled_policy_results.items()
            )
            / len(policies)
        )

    leave_one_policy_out = []
    for omitted in policies:
        retained = [row for row in per_policy if int(row["train_seed"]) != omitted]
        leave_one_policy_out.append(
            {
                "omitted_train_seed": omitted,
                **_aggregate_policy_results(retained),
            }
        )

    leave_one_start_out = []
    for omitted in starts:
        retained_starts = [start for start in starts if start != omitted]
        retained_multiplicities = {start: 1 for start in retained_starts}
        retained_policy_results = [
            _policy_selector_results(
                policy=policy,
                starts=retained_starts,
                indexed=indexed,
                queries_per_policy=queries_per_policy,
                start_multiplicities=retained_multiplicities,
            )
            for policy in policies
        ]
        leave_one_start_out.append(
            {
                "omitted_episode_seed": omitted,
                **_aggregate_policy_results(retained_policy_results),
            }
        )

    return {
        "method": (
            "within_policy_equal_budget_shared_failure_priority_"
            "external_binary_outcome"
        ),
        "shared_selection_priority": (
            "factual_zero descending, then selector score descending"
        ),
        "score_fields": dict(_ACQUISITION_SCORE_FIELDS),
        "policies": list(policies),
        "starts": list(starts),
        "queries_per_policy": queries_per_policy,
        "point": point,
        "per_policy": per_policy,
        "policy_cluster_bootstrap": _interval_summary(
            policy_bootstrap,
            replicates=replicates,
            seed=seed,
        ),
        "two_way_policy_start_pigeonhole_bootstrap": _interval_summary(
            two_way_bootstrap,
            replicates=replicates,
            seed=seed + 1,
        ),
        "exact_paired_sign_flip": _exact_paired_sign_flip(
            [difference_by_policy[policy] for policy in policies]
        ),
        "leave_one_policy_out": leave_one_policy_out,
        "leave_one_start_out": leave_one_start_out,
    }


def acquisition_delta_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_922,
) -> dict[str, Any]:
    """Estimate acquisition delta with a policy-by-start pigeonhole bootstrap.

    Required row fields are ``train_seed``, ``episode_seed``,
    ``event_burden_score``, ``suite_nominated`` and ``suite_supported``.  The
    query budget is recomputed in every bootstrap replicate from the replicated
    full-suite nominations.  Boundary ties receive fractional expected weights.
    """

    _validate_bootstrap_arguments(replicates, seed)
    policies, starts, indexed = _index_acquisition_cells(rows)
    groups = _score_groups(indexed)
    unit_multiplicity = {key: 1 for key in indexed}
    point = _acquisition_estimate(indexed, groups, unit_multiplicity)

    rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    start_rng = random.Random(seed + 2)
    estimates: list[float] = []
    policy_only_estimates: list[float] = []
    start_only_estimates: list[float] = []
    for _ in range(replicates):
        policy_counts = _sample_counts(rng, policies)
        start_counts = _sample_counts(rng, starts)
        multiplicities = {
            key: policy_counts[key[0]] * start_counts[key[1]] for key in indexed
        }
        estimate = _acquisition_estimate(indexed, groups, multiplicities)
        estimates.append(float(estimate["acquisition_delta"]))

        policy_counts = _sample_counts(policy_rng, policies)
        policy_multiplicities = {
            key: policy_counts[key[0]] for key in indexed
        }
        policy_only_estimates.append(
            float(
                _acquisition_estimate(
                    indexed,
                    groups,
                    policy_multiplicities,
                )["acquisition_delta"]
            )
        )

        start_counts = _sample_counts(start_rng, starts)
        start_multiplicities = {key: start_counts[key[1]] for key in indexed}
        start_only_estimates.append(
            float(
                _acquisition_estimate(
                    indexed,
                    groups,
                    start_multiplicities,
                )["acquisition_delta"]
            )
        )

    leave_policy = []
    for omitted in policies:
        subset = {key: value for key, value in indexed.items() if key[0] != omitted}
        estimate = _acquisition_estimate(
            subset,
            _score_groups(subset),
            {key: 1 for key in subset},
        )
        leave_policy.append({"omitted_train_seed": omitted, **estimate})

    leave_start = []
    for omitted in starts:
        subset = {key: value for key, value in indexed.items() if key[1] != omitted}
        estimate = _acquisition_estimate(
            subset,
            _score_groups(subset),
            {key: 1 for key in subset},
        )
        leave_start.append({"omitted_episode_seed": omitted, **estimate})

    return {
        "method": "two_way_policy_start_pigeonhole_bootstrap",
        "policies": list(policies),
        "starts": list(starts),
        "point": point,
        "bootstrap": _interval_summary(
            estimates,
            replicates=replicates,
            seed=seed,
        ),
        "policy_only_bootstrap": _interval_summary(
            policy_only_estimates,
            replicates=replicates,
            seed=seed + 1,
        ),
        "start_only_bootstrap": _interval_summary(
            start_only_estimates,
            replicates=replicates,
            seed=seed + 2,
        ),
        "leave_one_policy_out": leave_policy,
        "leave_one_start_out": leave_start,
    }


def _index_specificity_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[int],
    starts: Sequence[int],
) -> dict[PairingKey, tuple[int, int]]:
    allowed_cells = {(policy, start) for policy in policies for start in starts}
    seen: set[CandidateKey] = set()
    counts: dict[PairingKey, list[int]] = {
        key: [0, 0] for key in allowed_cells
    }
    for row in rows:
        pairing = int(row["train_seed"]), int(row["episode_seed"])
        key = pairing[0], pairing[1], str(row["candidate_kind"])
        if pairing not in allowed_cells:
            raise ValueError(f"candidate {key!r} falls outside the frozen design")
        if key in seen:
            raise ValueError(f"duplicate specificity candidate {key!r}")
        seen.add(key)
        primary_decision = str(row["primary_only_decision"])
        full_decision = str(row["full_decision"])
        if primary_decision not in _PRIMARY_ONLY_DECISIONS:
            raise ValueError(f"unknown primary-only decision {primary_decision!r}")
        if full_decision not in _FULL_DECISIONS:
            raise ValueError(f"unknown full-suite decision {full_decision!r}")
        if primary_decision != "supported_by_primary_only":
            continue
        if full_decision not in _EVALUABLE_FULL_DECISIONS:
            continue
        counts[pairing][0] += 1
        if full_decision in _FULL_DOWNGRADES:
            counts[pairing][1] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _specificity_estimate(
    counts: Mapping[PairingKey, tuple[int, int]],
    multiplicities: Mapping[PairingKey, int],
) -> dict[str, Any]:
    positives = sum(
        int(multiplicities.get(key, 0)) * value[0]
        for key, value in counts.items()
    )
    downgrades = sum(
        int(multiplicities.get(key, 0)) * value[1]
        for key, value in counts.items()
    )
    return {
        "evaluable_primary_positives": positives,
        "downgrades": downgrades,
        "downgrade_rate": downgrades / positives if positives else None,
    }


def specificity_downgrade_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[int],
    start_ids: Sequence[int],
    replicates: int = 10_000,
    seed: int = 20_260_922,
) -> dict[str, Any]:
    """Bootstrap the full-suite downgrade rate among primary-only positives.

    Required row fields are ``train_seed``, ``episode_seed``,
    ``candidate_kind``, ``primary_only_decision`` and ``full_decision``.
    Invalid or incomplete full decisions are not treated as specificity gains.
    """

    _validate_bootstrap_arguments(replicates, seed)
    policies = tuple(sorted({int(value) for value in policy_ids}))
    starts = tuple(sorted({int(value) for value in start_ids}))
    if not policies or not starts:
        raise ValueError("specificity analysis requires policies and starts")
    if len(policies) < 2 or len(starts) < 2:
        raise ValueError(
            "clustered specificity analysis requires at least two policies "
            "and two starts"
        )
    if len(policies) != len(policy_ids) or len(starts) != len(start_ids):
        raise ValueError("policy_ids and start_ids must be unique")

    counts = _index_specificity_candidates(
        rows,
        policies=policies,
        starts=starts,
    )
    point = _specificity_estimate(counts, {key: 1 for key in counts})
    downgrade_cells = [key for key, value in counts.items() if value[1] > 0]
    point["policies_with_downgrade"] = sorted({key[0] for key in downgrade_cells})
    point["starts_with_downgrade"] = sorted({key[1] for key in downgrade_cells})

    rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    start_rng = random.Random(seed + 2)
    estimates: list[float] = []
    policy_only_estimates: list[float] = []
    start_only_estimates: list[float] = []
    for _ in range(replicates):
        policy_counts = _sample_counts(rng, policies)
        start_counts = _sample_counts(rng, starts)
        multiplicities = {
            key: policy_counts[key[0]] * start_counts[key[1]] for key in counts
        }
        estimate = _specificity_estimate(counts, multiplicities)
        value = estimate["downgrade_rate"]
        if value is not None:
            estimates.append(float(value))

        policy_counts = _sample_counts(policy_rng, policies)
        policy_value = _specificity_estimate(
            counts,
            {key: policy_counts[key[0]] for key in counts},
        )["downgrade_rate"]
        if policy_value is not None:
            policy_only_estimates.append(float(policy_value))

        start_counts = _sample_counts(start_rng, starts)
        start_value = _specificity_estimate(
            counts,
            {key: start_counts[key[1]] for key in counts},
        )["downgrade_rate"]
        if start_value is not None:
            start_only_estimates.append(float(start_value))

    leave_policy = []
    for omitted in policies:
        multiplicities = {
            key: int(key[0] != omitted) for key in counts
        }
        leave_policy.append(
            {
                "omitted_train_seed": omitted,
                **_specificity_estimate(counts, multiplicities),
            }
        )

    leave_start = []
    for omitted in starts:
        multiplicities = {
            key: int(key[1] != omitted) for key in counts
        }
        leave_start.append(
            {
                "omitted_episode_seed": omitted,
                **_specificity_estimate(counts, multiplicities),
            }
        )

    return {
        "method": "two_way_policy_start_pigeonhole_bootstrap",
        "policies": list(policies),
        "starts": list(starts),
        "point": point,
        "bootstrap": _interval_summary(
            estimates,
            replicates=replicates,
            seed=seed,
        ),
        "policy_only_bootstrap": _interval_summary(
            policy_only_estimates,
            replicates=replicates,
            seed=seed + 1,
        ),
        "start_only_bootstrap": _interval_summary(
            start_only_estimates,
            replicates=replicates,
            seed=seed + 2,
        ),
        "leave_one_policy_out": leave_policy,
        "leave_one_start_out": leave_start,
    }
