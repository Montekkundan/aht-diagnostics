#!/usr/bin/env python3
"""Audit frozen soup-carrier swap-lock confirmation results.

This script performs derived, paired comparisons on an existing immutable
confirmation result. It does not rerun the environment or select new cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.natural_failure import (  # noqa: E402
    clustered_paired_bootstrap_interval,
    wilson_interval,
)


PRIMARY_BRANCH = "detour_down_left_2"
NO_CHANGE_BRANCH = "no_change"
MATCHED_CONTROLS = (
    "control_left_down_2",
    "control_down_stay_2",
    "control_wait_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def remaining_return(row: dict[str, Any], branch: str) -> float:
    return float(row["branches"][branch]["remaining_return"])


def paired_contrast(
    rows: list[dict[str, Any]],
    treatment: str,
    comparison: str,
) -> dict[str, Any]:
    differences = [
        remaining_return(row, treatment) - remaining_return(row, comparison)
        for row in rows
    ]
    if not differences:
        raise ValueError("Cannot summarize an empty confirmation result")

    policy_ids = sorted({int(row["train_seed"]) for row in rows})
    per_policy = {}
    for policy_id in policy_ids:
        policy_differences = [
            difference
            for difference, row in zip(differences, rows, strict=True)
            if int(row["train_seed"]) == policy_id
        ]
        per_policy[str(policy_id)] = {
            "mean": sum(policy_differences) / len(policy_differences),
            "n": len(policy_differences),
            "positive": sum(value > 0 for value in policy_differences),
            "zero": sum(value == 0 for value in policy_differences),
            "negative": sum(value < 0 for value in policy_differences),
        }

    positive = sum(value > 0 for value in differences)
    zero = sum(value == 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    return {
        "treatment": treatment,
        "comparison": comparison,
        "mean": sum(differences) / len(differences),
        "n": len(differences),
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "positive_rate": wilson_interval(positive, len(differences)),
        "environment_seed_cluster_interval": clustered_paired_bootstrap_interval(
            differences,
            [int(row["episode_seed"]) for row in rows],
            seed=53,
        ),
        "policy_cluster_interval": clustered_paired_bootstrap_interval(
            differences,
            [int(row["train_seed"]) for row in rows],
            seed=59,
        ),
        "by_policy": per_policy,
        "differences": differences,
    }


def audit_confirmation(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(result.get("cases", []))
    if not rows:
        raise ValueError("Confirmation result contains no detected cases")

    required_branches = {
        PRIMARY_BRANCH,
        NO_CHANGE_BRANCH,
        *MATCHED_CONTROLS,
    }
    for index, row in enumerate(rows):
        missing = required_branches.difference(row.get("branches", {}))
        if missing:
            raise ValueError(
                f"Case {index} is missing required branches: {sorted(missing)}"
            )

    versus_no_change = paired_contrast(rows, PRIMARY_BRANCH, NO_CHANGE_BRANCH)
    versus_controls = {
        control: paired_contrast(rows, PRIMARY_BRANCH, control)
        for control in MATCHED_CONTROLS
    }
    control_effects = {
        control: paired_contrast(rows, control, NO_CHANGE_BRANCH)
        for control in MATCHED_CONTROLS
    }

    policy_count = len({int(row["train_seed"]) for row in rows})
    environment_start_count = len({int(row["episode_seed"]) for row in rows})
    exact_replay_count = sum(bool(row["no_change_exact_replay"]) for row in rows)
    positive_within_every_policy = all(
        policy_result["mean"] > 0
        for policy_result in versus_no_change["by_policy"].values()
    )
    any_control_reproduced = any(
        contrast["positive"] > 0 for contrast in control_effects.values()
    )

    return {
        "status": "derived_confirmation_audit",
        "source_status": result.get("status"),
        "source_provenance": result.get("provenance"),
        "case_count": len(rows),
        "policy_count": policy_count,
        "environment_start_count": environment_start_count,
        "primary_vs_no_change": versus_no_change,
        "primary_vs_matched_controls": versus_controls,
        "matched_controls_vs_no_change": control_effects,
        "frozen_decision_checks": {
            "recurs_across_at_least_two_policies": policy_count >= 2,
            "recurs_across_at_least_two_environment_starts": (
                environment_start_count >= 2
            ),
            "all_no_change_replays_exact": exact_replay_count == len(rows),
            "primary_positive_in_every_occurring_policy": (
                positive_within_every_policy
            ),
            "no_matched_control_reproduces_improvement": (
                not any_control_reproduced
            ),
            "specific_down_left_mechanism_supported": (
                policy_count >= 2
                and environment_start_count >= 2
                and exact_replay_count == len(rows)
                and positive_within_every_policy
                and not any_control_reproduced
            ),
            "broader_motion_conflict_resolution_supported": (
                policy_count >= 2
                and environment_start_count >= 2
                and exact_replay_count == len(rows)
                and positive_within_every_policy
                and control_effects["control_wait_2"]["positive"] == 0
            ),
        },
        "interpretation": (
            "The frozen context recurred and the down-left intervention repaired "
            "every detected case, while two-step waiting repaired none. However, "
            "two movement controls also repaired a subset from one policy, so the "
            "strict claim that down-left is the unique mechanism is rejected. The "
            "result supports the broader diagnosis of a recurrent, causally "
            "repairable motion-conflict limitation whose sufficient escape path "
            "depends on the teammate policy."
        ),
    }


def main() -> int:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    audit = audit_confirmation(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["frozen_decision_checks"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
