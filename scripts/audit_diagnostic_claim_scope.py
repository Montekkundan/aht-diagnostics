"""Audit what the frozen evaluator decisions can and cannot establish.

This is a retrospective description of the unchanged confirmation outputs,
not a new confirmatory experiment or a source of independent semantic labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "results/confirmation"


def audit() -> dict:
    rows = json.loads(
        (CONFIRMATION / "diagnostic_baseline_confirmation_rows.json").read_text()
    )["rows"]
    suite = json.loads(
        (CONFIRMATION / "diagnostic_baseline_confirmation_suite.json").read_text()
    )["diagnoses"]
    comparison = json.loads(
        (CONFIRMATION / "diagnostic_baseline_confirmation_comparison.json").read_text()
    )["primary_only_unit_tests"]
    by_key = {
        (int(row["train_seed"]), int(row["episode_seed"])): row for row in rows
    }
    if len(by_key) != 1800:
        raise ValueError("Expected 1,800 distinct frozen policy-start cells")
    swap = [
        decision for decision in suite
        if decision["candidate_kind"] == "soup_carrier_swap_lock"
    ]
    swap_returns = Counter(
        int(by_key[int(item["train_seed"]), int(item["episode_seed"])]["factual_return"])
        for item in swap
    )
    primary = {
        (int(item["train_seed"]), int(item["episode_seed"]), item["candidate_kind"]): item
        for item in comparison
    }
    changed = []
    for item in suite:
        key = int(item["train_seed"]), int(item["episode_seed"]), item["candidate_kind"]
        if primary[key]["decision"] != "supported_by_primary_only" or item["decision"] == "supported":
            continue
        evidence = item["intervention_evidence"]
        changed.append({
            "train_seed": key[0],
            "episode_seed": key[1],
            "candidate_kind": key[2],
            "full_decision": item["decision"],
            "primary_effect": evidence["primary_effect"],
            "duration_matched_wait_effect": evidence["wait_effect"],
            "specificity": evidence["specificity"],
        })
    if len(swap) != 403 or len(changed) != 4:
        raise ValueError("Frozen candidate counts changed unexpectedly")
    return {
        "status": "retrospective_claim_scope_audit_not_independent_ground_truth",
        "population_policy_start_cells": len(rows),
        "swap_trigger_cells": len(swap),
        "swap_trigger_by_factual_return": dict(sorted(swap_returns.items())),
        "swap_trigger_cells_with_positive_factual_return": sum(
            count for value, count in swap_returns.items() if value > 0
        ),
        "full_control_downgrades": changed,
        "interpretation": (
            "A supported intervention can improve an already successful episode. "
            "It measures an opportunity loss in those cells, not the cause of a zero-return failure. "
            "Four positive primary interventions were not mechanism-specific because "
            "a duration-matched pause improved return too. These decisions are evaluator "
            "outputs, not independently verified semantic truth."
        ),
        "opportunity_matched_specificity": None,
        "opportunity_matched_specificity_note": (
            "The capable-reference zero-trigger comparison does not condition on "
            "reaching the same reciprocal-motion opportunity. A conditional false-positive "
            "rate requires independent labels for episodes with comparable opportunities."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIRMATION / "diagnostic_claim_scope_audit.json",
    )
    output = parser.parse_args().output
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "swap_trigger_by_factual_return": result["swap_trigger_by_factual_return"],
        "full_control_downgrades": len(result["full_control_downgrades"]),
    }, indent=2))
