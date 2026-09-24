"""Retrospective, development-selected factual baselines for the frozen study.

This does not modify or replace the September 22 confirmatory comparison. The
confirmation outcomes were already available when this analysis was written,
so its contrasts are descriptive, not a new prospective significance test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

from aht_diagnostics.baselines import tie_aware_top_budget_weights


ROOT = Path(__file__).resolve().parents[1]
SCORE_NAMES = (
    "ego_blocked_move_rate",
    "partner_blocked_move_rate",
    "ego_interact_no_effect_rate",
    "partner_interact_no_effect_rate",
    "combined_interact_no_effect_rate",
    "longest_blocked_swap_run",
)


def score(row: dict[str, Any], name: str) -> float:
    features = row["factual_event_features"]
    if name == "combined_interact_no_effect_rate":
        return float(features["ego_interact_no_effect_rate"]) + float(
            features["partner_interact_no_effect_rate"]
        )
    if name == "longest_blocked_swap_run":
        return float(row["swap_context"]["longest_blocked_swap_run"])
    if name not in SCORE_NAMES:
        raise ValueError(f"unknown factual score {name!r}")
    return float(features[name])


def expected_yield(rows: list[dict[str, Any]], name: str, budget: int) -> float:
    """Use the original shared zero-return priority and fractional boundary ties."""

    if len(rows) < budget:
        raise ValueError("query budget exceeds starts per policy")
    priority = [
        (int(float(row["factual_return"]) == 0.0), score(row, name))
        for row in rows
    ]
    ordered = sorted(set(priority), reverse=True)
    remaining = budget
    expected_positives = 0.0
    for key in ordered:
        group = [row for row, row_key in zip(rows, priority) if row_key == key]
        if not group:
            continue
        weight = min(1.0, remaining / len(group)) if remaining > 0 else 0.0
        expected_positives += weight * sum(
            float(row["factual_return"]) == 0.0
            and float(row["reference_return"]) >= 20.0
            for row in group
        )
        remaining -= min(remaining, len(group))
    if remaining:
        raise AssertionError("query budget not exhausted")
    return expected_positives / budget


def policy_yields(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    by_policy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[int(row["train_seed"])].append(row)
    start_sets = [{int(row["episode_seed"]) for row in group} for group in by_policy.values()]
    if len({frozenset(starts) for starts in start_sets}) != 1:
        raise ValueError("policies must share the same evaluation starts")
    return {
        str(policy): {
            name: expected_yield(group, name, budget) for name in SCORE_NAMES
        }
        for policy, group in sorted(by_policy.items())
    }


def reference_screen_yield(
    rows: list[dict[str, Any]], budget: int, screens: int, *, seed: int
) -> dict[str, Any]:
    """Simulate a random limited reference screen, then review best gaps.

    This comparator has privileged access to reference outcomes for only the
    screened cells. Its simulated screening cost is separate from review count.
    """

    rng = random.Random(seed)
    by_policy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[int(row["train_seed"])].append(row)
    estimates: dict[str, float] = {}
    for policy, group in sorted(by_policy.items()):
        zero = [row for row in group if float(row["factual_return"]) == 0.0]
        nonzero = [row for row in group if float(row["factual_return"]) != 0.0]
        replicate_yields = []
        for _ in range(2000):
            screened = rng.sample(zero, min(screens, len(zero)))
            if len(screened) < budget:
                screened += rng.sample(nonzero, budget - len(screened))
            gaps = [
                float(row["reference_return"]) - float(row["factual_return"])
                for row in screened
            ]
            weights = tie_aware_top_budget_weights(gaps, budget=budget)
            replicate_yields.append(
                sum(
                    weight
                    * (
                        float(row["factual_return"]) == 0.0
                        and float(row["reference_return"]) >= 20.0
                    )
                    for row, weight in zip(screened, weights)
                )
                / budget
            )
        estimates[str(policy)] = sum(replicate_yields) / len(replicate_yields)
    return {
        "screens_per_policy": screens,
        "reviewed_per_policy": budget,
        "Monte_Carlo_replicates_per_policy": 2000,
        "policy_yields": estimates,
        "mean_yield": sum(estimates.values()) / len(estimates),
        "note": "Retrospective simulation using stored reference outcomes; new use would require only the screened reference rollouts.",
    }


def branch_steps(path: Path, names: set[str] | None = None) -> tuple[int, int]:
    cases = json.loads(path.read_text())["cases"]
    branches = [
        (name, branch)
        for case in cases
        for name, branch in case["branches"].items()
        if names is None or name in names
    ]
    return len(branches), sum(
        int(branch["end_step"]) - int(branch["start_step"])
        for _, branch in branches
    )


def summarize(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    per_policy = policy_yields(rows, budget)
    return {
        "policy_yields": per_policy,
        "mean_yields": {
            name: sum(policy[name] for policy in per_policy.values()) / len(per_policy)
            for name in SCORE_NAMES
        },
    }


def paired_policy_interval(
    comparison: dict[str, Any], secondary: dict[str, Any], name: str
) -> dict[str, Any]:
    """Describe a paired policy-cluster contrast without confirmatory status."""

    point = comparison["confirmatory_analysis"]["acquisition"]["per_policy"]
    differences = [
        float(row["selectors"]["diagnosis"]["yield"])
        - float(secondary["policy_yields"][str(row["train_seed"])][name])
        for row in point
    ]
    rng = random.Random(20260923)
    boot = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(10000)
    )
    return {
        "difference_in_yield": sum(differences) / len(differences),
        "policy_cluster_bootstrap_ci95": [boot[249], boot[9749]],
        "policy_count": len(differences),
        "status": "retrospective_descriptive_not_a_frozen_primary_test",
    }


def run() -> dict[str, Any]:
    development = json.loads(
        (ROOT / "results/development/diagnostic_baseline_rows.json").read_text()
    )["rows"]
    confirmation = json.loads(
        (ROOT / "results/confirmation/diagnostic_baseline_confirmation_rows.json").read_text()
    )["rows"]
    dev = summarize(development, 8)
    chosen = max(SCORE_NAMES, key=lambda name: (dev["mean_yields"][name], -SCORE_NAMES.index(name)))
    confirm = summarize(confirmation, 12)
    comparison = json.loads(
        (ROOT / "results/confirmation/diagnostic_baseline_confirmation_comparison.json").read_text()
    )
    swap = ROOT / "results/confirmation/diagnostic_baseline_confirmation_swap_lock.json"
    initial = ROOT / "results/confirmation/diagnostic_baseline_confirmation_initial_soup.json"
    swap_all = branch_steps(swap)
    initial_all = branch_steps(initial)
    swap_primary = branch_steps(swap, {"no_change", "detour_down_left_2"})
    initial_primary = branch_steps(
        initial,
        {"no_change", "reference_exact_actions", "reference_actions_no_interact"},
    )
    if swap_all[0] + initial_all[0] != 2786:
        raise ValueError("full-branch count differs from the frozen report")
    if swap_primary[0] + initial_primary[0] != 875:
        raise ValueError("primary-only branch count differs from the frozen report")
    return {
        "status": "retrospective_secondary_analysis_not_new_confirmation",
        "selection_rule": "chosen by highest mean development yield among six fixed factual scores, with name order breaking exact ties",
        "development_starts": 100,
        "development_policies": len(dev["policy_yields"]),
        "development_review_budget_per_policy": 8,
        "development": dev,
        "chosen_factual_score": chosen,
        "confirmation_starts": 150,
        "confirmation_policies": len(confirm["policy_yields"]),
        "confirmation_review_budget_per_policy": 12,
        "confirmation": confirm,
        "diagnosis_minus_development_chosen_score": paired_policy_interval(
            comparison, confirm, chosen
        ),
        "diagnosis_minus_swap_persistence_score": paired_policy_interval(
            comparison, confirm, "longest_blocked_swap_run"
        ),
        "reference_screen_2x": reference_screen_yield(confirmation, 12, 24, seed=20260922),
        "cost": {
            "shared_factual_rollouts": 1800,
            "shared_factual_environment_steps": sum(
                int(row["factual_event_features"]["steps"]) for row in confirmation
            ),
            "full_reference_rollouts_if_raw_gap_used": 1800,
            "limited_reference_rollouts_if_2x_screen_used": 24 * 12,
            "full_diagnostic_branches": swap_all[0] + initial_all[0],
            "full_diagnostic_branch_environment_steps": swap_all[1] + initial_all[1],
            "primary_only_branches": swap_primary[0] + initial_primary[0],
            "primary_only_branch_environment_steps": swap_primary[1] + initial_primary[1],
            "wall_time": None,
            "wall_time_note": "Historical wall times were not stored for these separate runs; do not infer elapsed time from branch counts.",
            "ROTATE_generated_teammate_training_steps_per_policy": 5120000,
            "ROTATE_generated_teammate_policy_count": 12,
            "separate_reference_training_cost": "not independently recorded",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/confirmation/retrospective_secondary_baselines.json",
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "chosen": result["chosen_factual_score"],
        "development_yield": result["development"]["mean_yields"][result["chosen_factual_score"]],
        "confirmation_yield": result["confirmation"]["mean_yields"][result["chosen_factual_score"]],
        "reference_screen_2x_yield": result["reference_screen_2x"]["mean_yield"],
    }, indent=2))
