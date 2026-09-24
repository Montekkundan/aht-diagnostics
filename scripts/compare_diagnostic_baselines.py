#!/usr/bin/env python3
"""Compare frozen diagnostic baselines on a shared policy-start split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.baseline_evaluation import (  # noqa: E402
    acquisition_summary,
    candidate_key,
    compare_candidate_decisions,
    index_unique_rows,
    pairing_key,
    require_matching_pairing_keys,
    require_unique_candidate_keys,
)
from aht_diagnostics.baselines import (  # noqa: E402
    event_rule_decision,
    primary_only_unit_test_decision,
    tie_aware_top_budget_weights,
    weighted_event_burden,
)
from aht_diagnostics.confirmatory_statistics import (  # noqa: E402
    external_binary_acquisition_analysis,
    specificity_downgrade_analysis,
)
from aht_diagnostics.freeze import validate_frozen_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--swap-result", required=True, type=Path)
    parser.add_argument("--initial-soup-result", required=True, type=Path)
    parser.add_argument("--suite-result", required=True, type=Path)
    parser.add_argument("--capable-control-result", type=Path)
    parser.add_argument("--jaxaht", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoints" / "hf",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoints.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256")
    return parser.parse_args()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remaining_return(case: dict[str, Any], branch: str) -> float:
    return float(case["branches"][branch]["remaining_return"])


def _event_and_primary_rows(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    swap_result: dict[str, Any],
    initial_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []

    for case in swap_result.get("cases", []):
        key = pairing_key(case)
        canonical = canonical_rows[key]
        interval = canonical["swap_context"]["candidate_interval"]
        if interval is None:
            raise ValueError(f"swap case {key!r} is absent from canonical factual rows")
        event_rows.append(
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                **event_rule_decision(
                    candidate_kind="soup_carrier_swap_lock",
                    triggered=True,
                    interval=(int(interval[0]), int(interval[1])),
                    score=float(canonical["swap_context"]["longest_blocked_swap_run"]),
                ),
            }
        )
        primary_rows.append(
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                **primary_only_unit_test_decision(
                    candidate_kind="soup_carrier_swap_lock",
                    exact_replay=bool(case["no_change_exact_replay"]),
                    no_change_return=_remaining_return(case, "no_change"),
                    primary_return=_remaining_return(case, "detour_down_left_2"),
                ),
            }
        )

    for case in initial_result.get("cases", []):
        key = pairing_key(case)
        canonical = canonical_rows[key]
        interval = canonical["initial_soup_context"]["candidate_interval"]
        if interval is None:
            raise ValueError(
                f"initial-soup case {key!r} is absent from canonical factual rows"
            )
        event_rows.append(
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                **event_rule_decision(
                    candidate_kind="initial_completed_soup_drop",
                    triggered=True,
                    interval=(int(interval[0]), int(interval[1])),
                    score=1.0,
                ),
            }
        )
        primary_rows.append(
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                **primary_only_unit_test_decision(
                    candidate_kind="initial_completed_soup_drop",
                    exact_replay=bool(case["no_change_exact_replay"]),
                    no_change_return=_remaining_return(case, "no_change"),
                    primary_return=_remaining_return(
                        case,
                        "reference_actions_no_interact",
                    ),
                ),
            }
        )
    return event_rows, primary_rows


def _validate_provenance(
    canonical: dict[str, Any],
    swap: dict[str, Any],
    initial: dict[str, Any],
    suite: dict[str, Any],
    *,
    swap_path: Path,
    initial_path: Path,
    capable_control_path: Path | None,
    capable_control: dict[str, Any] | None = None,
    frozen_hashes: dict[str, Any] | None = None,
) -> None:
    canonical_provenance = canonical["provenance"]
    canonical_protocol = canonical["protocol"]
    for label, result in (("swap", swap), ("initial-soup", initial)):
        provenance = result["provenance"]
        for field in (
            "checkpoint_dataset_revision",
            "fixed_checkpoint_sha256",
            "saved_policy_sha256",
            "source_implementation_reference",
        ):
            if canonical_provenance[field] != provenance[field]:
                raise ValueError(f"canonical and {label} inputs disagree on {field}")
        for field in ("environment_condition", "max_steps"):
            if canonical_protocol[field] != result["protocol"][field]:
                raise ValueError(
                    f"canonical and {label} inputs disagree on protocol {field}"
                )
    for field in (
        "checkpoint_dataset_revision",
        "fixed_checkpoint_sha256",
        "saved_policy_sha256",
    ):
        if canonical_provenance[field] != suite["provenance"][field]:
            raise ValueError(f"canonical and suite inputs disagree on {field}")

    suite_inputs = suite.get("inputs")
    if not isinstance(suite_inputs, dict):
        raise ValueError("suite result is missing its input digest manifest")
    expected_inputs = {
        "swap_result_sha256": file_digest(swap_path),
        "initial_soup_result_sha256": file_digest(initial_path),
    }
    for field, actual_digest in expected_inputs.items():
        if suite_inputs.get(field) != actual_digest:
            raise ValueError(f"suite result was not derived from the supplied {field}")

    expected_capable_digest = suite_inputs.get("capable_control_result_sha256")
    if expected_capable_digest is not None:
        if capable_control_path is None:
            raise ValueError(
                "suite result uses a capable-control input; pass --capable-control-result"
            )
        if file_digest(capable_control_path) != expected_capable_digest:
            raise ValueError(
                "suite result was not derived from the supplied capable-control result"
            )
    elif capable_control_path is not None:
        raise ValueError(
            "a capable-control result was supplied but the suite did not record one"
        )

    if frozen_hashes is not None:
        generated_inputs = (
            (
                "canonical rows",
                canonical["provenance"],
                "scripts/export_diagnostic_baseline_rows.py",
                "configs/confirmation/diagnostic_baseline_confirmation_rows.json",
            ),
            (
                "swap result",
                swap["provenance"],
                "scripts/analyze_soup_carrier_swap_lock.py",
                "configs/confirmation/diagnostic_baseline_confirmation_swap_lock.json",
            ),
            (
                "initial-soup result",
                initial["provenance"],
                "scripts/analyze_initial_completed_soup_delivery.py",
                "configs/confirmation/diagnostic_baseline_confirmation_initial_soup.json",
            ),
        )
        for label, provenance, script_name, config_name in generated_inputs:
            if provenance.get("analysis_script_sha256") != frozen_hashes.get(
                script_name
            ):
                raise ValueError(f"{label} was generated by an unfrozen script")
            if provenance.get("config_sha256") != frozen_hashes.get(config_name):
                raise ValueError(f"{label} was generated from an unfrozen config")

        if capable_control_path is not None:
            if capable_control is None:
                raise ValueError("capable-control provenance is unavailable")
            capable_provenance = capable_control.get("provenance", {})
            if capable_provenance.get("analysis_script_sha256") != frozen_hashes.get(
                "scripts/screen_soup_carrier_swap_lock_capable_controls.py"
            ):
                raise ValueError(
                    "capable-control result was generated by an unfrozen script"
                )
            if capable_provenance.get("config_sha256") != frozen_hashes.get(
                "configs/confirmation/diagnostic_baseline_confirmation_capable_control.json"
            ):
                raise ValueError(
                    "capable-control result was generated from an unfrozen config"
                )

        if suite.get("analysis_script_sha256") != frozen_hashes.get(
            "scripts/summarize_diagnostic_suite.py"
        ):
            raise ValueError("suite result was generated by an unfrozen script")


def _validate_case_alignment(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    swap: dict[str, Any],
    initial: dict[str, Any],
) -> None:
    """Reject stale candidate artifacts that do not match the canonical reroll."""

    for case in swap.get("cases", []):
        key = pairing_key(case)
        canonical = canonical_rows.get(key)
        if canonical is None:
            raise ValueError(f"swap case {key!r} is absent from canonical rows")
        context = canonical["swap_context"]
        interval = context["candidate_interval"]
        trigger = case["trigger"]
        expected_interval = [
            int(trigger["start_step"]),
            int(trigger["confirmed_at_step"]),
        ]
        if float(case["factual_return"]) != float(canonical["factual_return"]):
            raise ValueError(f"swap case {key!r} has a stale factual return")
        if interval != expected_interval:
            raise ValueError(f"swap case {key!r} has a stale candidate interval")
        if int(trigger["intervention_step"]) != expected_interval[1] + 1:
            raise ValueError(f"swap case {key!r} has inconsistent intervention timing")

    for case in initial.get("cases", []):
        key = pairing_key(case)
        canonical = canonical_rows.get(key)
        if canonical is None:
            raise ValueError(f"initial-soup case {key!r} is absent from canonical rows")
        context = canonical["initial_soup_context"]
        expected_interval = [0, int(case["factual_disposition_step"])]
        if float(case["factual_return"]) != float(canonical["factual_return"]):
            raise ValueError(f"initial-soup case {key!r} has a stale factual return")
        if context["candidate_interval"] != expected_interval:
            raise ValueError(
                f"initial-soup case {key!r} has a stale candidate interval"
            )
        if str(context["disposition"]) != str(case["factual_disposition"]):
            raise ValueError(f"initial-soup case {key!r} has a stale disposition")
        if int(case["intervention_step"]) != expected_interval[1]:
            raise ValueError(
                f"initial-soup case {key!r} has inconsistent intervention timing"
            )


def _normalized_cost_accounting(
    *,
    canonical_row_count: int,
    swap_case_count: int,
    initial_case_count: int,
    full_suite_branch_count: int,
) -> dict[str, Any]:
    """Return shared-pipeline costs, not the historical cost of separate reruns."""

    return {
        "accounting_basis": (
            "Normalized algorithmic counts for one shared factual-rollout pipeline; "
            "the historical artifacts were generated by separate runs."
        ),
        "canonical_factual_rollouts": canonical_row_count,
        "raw_gap_additional_reference_rollouts": canonical_row_count,
        "event_frequency_additional_policy_rollouts": 0,
        "event_rule_counterfactual_branches": 0,
        "primary_only_counterfactual_branches": (
            2 * swap_case_count + 3 * initial_case_count
        ),
        "primary_only_cost_note": (
            "Each swap test uses no-change plus one fixed primary branch. Each "
            "initial-soup test first generates the case-adaptive reference action "
            "sequence, then evaluates no-change and the masked primary branch."
        ),
        "full_suite_counterfactual_branches": full_suite_branch_count,
    }


def _selection_rows(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    raw_gap_scores: dict[tuple[int, int], float],
    event_scores: dict[tuple[int, int], float],
    budget: int,
) -> list[dict[str, Any]]:
    keys = sorted(canonical_rows)
    raw_weights = tie_aware_top_budget_weights(
        [raw_gap_scores[key] for key in keys],
        budget=budget,
    )
    event_weights = tie_aware_top_budget_weights(
        [event_scores[key] for key in keys],
        budget=budget,
    )
    random_weights = tie_aware_top_budget_weights(
        [0.0 for _ in keys],
        budget=budget,
    )
    return [
        {
            "train_seed": key[0],
            "episode_seed": key[1],
            "raw_gap_score": raw_gap_scores[key],
            "raw_gap_expected_selection_weight": raw_weight,
            "event_burden_score": event_scores[key],
            "event_burden_expected_selection_weight": event_weight,
            "random_expected_selection_weight": random_weight,
        }
        for key, raw_weight, event_weight, random_weight in zip(
            keys,
            raw_weights,
            event_weights,
            random_weights,
            strict=True,
        )
    ]


def _confirmation_settings(specification: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize the single nested confirmation protocol for this analysis."""

    endpoint = specification.get("primary_acquisition_endpoint")
    if endpoint is None:
        return None
    training = specification["training"]
    evaluation = specification["evaluation"]
    bootstrap = specification["bootstrap"]
    return {
        "expected_train_seeds": [int(value) for value in training["train_seeds"]],
        "expected_episode_seed_range": evaluation["episode_seed_range"],
        "event_burden_weights": specification["factual_event_burden_weights"],
        "queries_per_policy": int(endpoint["query_budget_per_policy"]),
        "competence_return_threshold": float(
            evaluation["competence_return_threshold"]
        ),
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "bootstrap_rng_seed": int(bootstrap["rng_seed"]),
    }


def _validate_expected_confirmation_design(
    specification: dict[str, Any],
    canonical_rows: dict[tuple[int, int], Any],
) -> tuple[list[int], list[int]]:
    settings = _confirmation_settings(specification) or specification
    expected_policies = [int(value) for value in settings["expected_train_seeds"]]
    start_range = settings["expected_episode_seed_range"]
    expected_starts = list(
        range(int(start_range["start"]), int(start_range["end_exclusive"]))
    )
    expected_keys = {
        (policy, start)
        for policy in expected_policies
        for start in expected_starts
    }
    if set(canonical_rows) != expected_keys:
        missing = sorted(expected_keys - set(canonical_rows))
        extra = sorted(set(canonical_rows) - expected_keys)
        raise ValueError(
            "confirmation rows do not match the frozen policy-by-start rectangle; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    return expected_policies, expected_starts


def _diagnosis_score(row: dict[str, Any]) -> float:
    return float(
        row["swap_context"]["candidate_interval"] is not None
        or row["initial_soup_context"]["candidate_interval"] is not None
    )


def _actionable_rescue_keys(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    swap: dict[str, Any],
    initial: dict[str, Any],
    competence_return_threshold: float,
    minimum_primary_gain: float = 20.0,
) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    """Build the shared environment-level outcome independently of suite labels."""

    rescue_keys: set[tuple[int, int]] = set()
    evidence: list[dict[str, Any]] = []
    for candidate_kind, result, primary_branch in (
        ("soup_carrier_swap_lock", swap, "detour_down_left_2"),
        (
            "initial_completed_soup_drop",
            initial,
            "reference_actions_no_interact",
        ),
    ):
        for case in result.get("cases", []):
            key = pairing_key(case)
            canonical = canonical_rows[key]
            no_change = _remaining_return(case, "no_change")
            primary = _remaining_return(case, primary_branch)
            checks = {
                "factual_zero": float(canonical["factual_return"]) == 0.0,
                "exact_no_change_replay": bool(case["no_change_exact_replay"]),
                "competent_same_start_reference": (
                    float(canonical["reference_return"])
                    >= competence_return_threshold
                ),
                "primary_gain_at_least_threshold": (
                    primary - no_change >= minimum_primary_gain
                ),
            }
            actionable = all(checks.values())
            if actionable:
                rescue_keys.add(key)
            evidence.append(
                {
                    "train_seed": key[0],
                    "episode_seed": key[1],
                    "candidate_kind": candidate_kind,
                    "actionable_rescue": actionable,
                    "no_change_remaining_return": no_change,
                    "primary_remaining_return": primary,
                    "primary_gain": primary - no_change,
                    "checks": checks,
                }
            )
    return rescue_keys, evidence


def _competence_witnessed_failure_keys(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    competence_return_threshold: float,
) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    """Build a trigger-independent failure target for every policy-start cell."""

    failure_keys: set[tuple[int, int]] = set()
    evidence: list[dict[str, Any]] = []
    for key, row in sorted(canonical_rows.items()):
        checks = {
            "factual_zero": float(row["factual_return"]) == 0.0,
            "competent_same_start_reference": (
                float(row["reference_return"]) >= competence_return_threshold
            ),
        }
        target_positive = all(checks.values())
        if target_positive:
            failure_keys.add(key)
        evidence.append(
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                "competence_witnessed_failure": target_positive,
                "factual_return": float(row["factual_return"]),
                "reference_return": float(row["reference_return"]),
                "checks": checks,
            }
        )
    return failure_keys, evidence


def _external_acquisition_rows(
    *,
    canonical_rows: dict[tuple[int, int], Any],
    event_scores: dict[tuple[int, int], float],
    raw_gap_scores: dict[tuple[int, int], float],
    target_positive_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "train_seed": key[0],
            "episode_seed": key[1],
            "factual_zero": float(canonical_rows[key]["factual_return"]) == 0.0,
            "diagnosis_score": _diagnosis_score(canonical_rows[key]),
            "event_burden_score": event_scores[key],
            "raw_gap_score": raw_gap_scores[key],
            "target_positive": key in target_positive_keys,
        }
        for key in sorted(canonical_rows)
    ]


def _acquisition_decision(analysis: dict[str, Any]) -> str:
    if int(analysis.get("queries_per_policy", 0)) == 0:
        return "not_estimable"
    low, high = analysis["policy_cluster_bootstrap"]["ci95"]
    if float(low) > 0.0:
        return "supported"
    if float(high) < 0.0:
        return "contradicted"
    return "inconclusive"


def main() -> int:
    args = parse_args()
    raw_paths = {
        "config": args.config,
        "rows": args.rows,
        "swap_result": args.swap_result,
        "initial_soup_result": args.initial_soup_result,
        "suite_result": args.suite_result,
        "output": args.output,
    }
    if args.capable_control_result is not None:
        raw_paths["capable_control_result"] = args.capable_control_result
    paths = {
        name: path.resolve()
        for name, path in raw_paths.items()
    }
    specification = json.loads(paths["config"].read_text(encoding="utf-8"))
    canonical = json.loads(paths["rows"].read_text(encoding="utf-8"))
    confirmation = _confirmation_settings(specification)
    frozen_protocol_validation = None
    if confirmation is not None:
        if args.jaxaht is None:
            raise ValueError("--jaxaht is required for confirmation analysis")
        if not args.expected_protocol_sha256:
            raise ValueError(
                "--expected-protocol-sha256 is required for confirmation analysis"
            )
        checkpoint_index_path = args.checkpoint_index.resolve()
        checkpoint_index = json.loads(
            checkpoint_index_path.read_text(encoding="utf-8")
        )
        fixed_checkpoint = (
            args.checkpoint_root.resolve()
            / checkpoint_index["fixed_ego"]["checkpoint"]
        )
        frozen_protocol_validation = validate_frozen_protocol(
            protocol_path=paths["config"],
            expected_protocol_sha256=args.expected_protocol_sha256,
            repo_root=REPO_ROOT,
            jaxaht_path=args.jaxaht.resolve(),
            fixed_checkpoint_path=fixed_checkpoint,
        )
    swap = json.loads(paths["swap_result"].read_text(encoding="utf-8"))
    initial = json.loads(
        paths["initial_soup_result"].read_text(encoding="utf-8")
    )
    suite = json.loads(paths["suite_result"].read_text(encoding="utf-8"))
    capable_control = (
        json.loads(paths["capable_control_result"].read_text(encoding="utf-8"))
        if "capable_control_result" in paths
        else None
    )
    if confirmation is not None:
        canonical_validation = canonical.get("provenance", {}).get(
            "validated_frozen_protocol"
        )
        if not isinstance(canonical_validation, dict):
            raise ValueError(
                "confirmation rows are missing frozen-protocol provenance"
            )
        if canonical_validation.get("protocol_sha256") != (
            frozen_protocol_validation["protocol_sha256"]
        ):
            raise ValueError(
                "confirmation rows were exported under a different protocol digest"
            )
    _validate_provenance(
        canonical,
        swap,
        initial,
        suite,
        swap_path=paths["swap_result"],
        initial_path=paths["initial_soup_result"],
        capable_control_path=paths.get("capable_control_result"),
        capable_control=capable_control,
        frozen_hashes=(
            specification["provenance_freeze"]["sha256"]
            if confirmation is not None
            else None
        ),
    )

    canonical_rows = index_unique_rows(canonical["rows"], label="canonical rows")
    _validate_case_alignment(
        canonical_rows=canonical_rows,
        swap=swap,
        initial=initial,
    )
    exposure_rows = index_unique_rows(
        initial["pairing_exposures"],
        label="initial-soup exposures",
    )
    require_matching_pairing_keys(
        canonical_rows,
        exposure_rows,
        reference_label="canonical rows",
        comparison_label="initial-soup exposures",
    )

    event_rows, primary_rows = _event_and_primary_rows(
        canonical_rows=canonical_rows,
        swap_result=swap,
        initial_result=initial,
    )
    event_by_candidate = require_unique_candidate_keys(
        event_rows,
        label="event-rule nominations",
    )
    primary_by_candidate = require_unique_candidate_keys(
        primary_rows,
        label="primary-only unit tests",
    )
    full_by_candidate = require_unique_candidate_keys(
        suite["diagnoses"],
        label="full adjudication",
    )
    if set(event_by_candidate) != set(full_by_candidate):
        raise ValueError("event-rule and full-adjudication candidate keys differ")
    if set(primary_by_candidate) != set(full_by_candidate):
        raise ValueError("primary-only and full-adjudication candidate keys differ")

    operational_candidate_keys = {
        (key[0], key[1]) for key in full_by_candidate
    }
    supported_keys = {
        (key[0], key[1])
        for key, row in full_by_candidate.items()
        if row["decision"] == "supported"
    }
    budget = len(operational_candidate_keys)
    weights = (
        confirmation["event_burden_weights"]
        if confirmation is not None
        else specification["event_burden_weights"]
    )
    raw_gap_scores = {
        key: float(row["approximate_reference_gap"])
        for key, row in canonical_rows.items()
    }
    event_scores = {
        key: weighted_event_burden(row["factual_event_features"], weights)
        for key, row in canonical_rows.items()
    }
    random_scores = {key: 0.0 for key in canonical_rows}

    event_decisions = {
        key: str(row["decision"]) for key, row in event_by_candidate.items()
    }
    primary_decisions = {
        key: str(row["decision"]) for key, row in primary_by_candidate.items()
    }
    full_decisions = {
        key: str(row["decision"]) for key, row in full_by_candidate.items()
    }

    result = {
        "status": specification["status"],
        "claim_scope": specification["claim_scope"],
        "measurement_note": (
            "Intervention-supported acquisition is agreement with the frozen "
            "operational adjudicator, not independent ground truth."
        ),
        "protocol": {
            "pairing_key": ["train_seed", "episode_seed"],
            "query_budget": budget,
            "event_burden_weights": weights,
            "score_ties": "uniform expected inclusion at the budget boundary",
        },
        "acquisition": {
            "raw_approximate_reference_gap": acquisition_summary(
                rows=canonical_rows,
                scores=raw_gap_scores,
                budget=budget,
                operational_candidate_keys=operational_candidate_keys,
                intervention_supported_keys=supported_keys,
            ),
            "equal_weight_factual_event_burden": acquisition_summary(
                rows=canonical_rows,
                scores=event_scores,
                budget=budget,
                operational_candidate_keys=operational_candidate_keys,
                intervention_supported_keys=supported_keys,
            ),
            "random": acquisition_summary(
                rows=canonical_rows,
                scores=random_scores,
                budget=budget,
                operational_candidate_keys=operational_candidate_keys,
                intervention_supported_keys=supported_keys,
            ),
        },
        "candidate_evidence": {
            "event_rule_vs_full_adjudication": compare_candidate_decisions(
                event_decisions,
                full_decisions,
            ),
            "primary_only_unit_test_vs_full_adjudication": (
                compare_candidate_decisions(primary_decisions, full_decisions)
            ),
            "full_adjudication_decision_counts": suite["summary"][
                "decision_counts"
            ],
        },
        "cost_accounting": _normalized_cost_accounting(
            canonical_row_count=len(canonical_rows),
            swap_case_count=len(swap.get("cases", [])),
            initial_case_count=len(initial.get("cases", [])),
            full_suite_branch_count=sum(
                len(case["branches"])
                for case in [*swap.get("cases", []), *initial.get("cases", [])]
            ),
        ),
        "event_rule_nominations": event_rows,
        "primary_only_unit_tests": primary_rows,
        "selection_rows": _selection_rows(
            canonical_rows=canonical_rows,
            raw_gap_scores=raw_gap_scores,
            event_scores=event_scores,
            budget=budget,
        ),
        "inputs": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_digest(path),
            }
            for name, path in paths.items()
            if name != "output"
        },
        "analysis_script_sha256": file_digest(Path(__file__).resolve()),
        "validated_frozen_protocol": frozen_protocol_validation,
    }
    if confirmation is not None:
        expected_policies, expected_starts = _validate_expected_confirmation_design(
            specification,
            canonical_rows,
        )
        failure_keys, failure_evidence = _competence_witnessed_failure_keys(
            canonical_rows=canonical_rows,
            competence_return_threshold=confirmation[
                "competence_return_threshold"
            ],
        )
        rescue_keys, rescue_evidence = _actionable_rescue_keys(
            canonical_rows=canonical_rows,
            swap=swap,
            initial=initial,
            competence_return_threshold=confirmation[
                "competence_return_threshold"
            ],
        )
        acquisition_rows = _external_acquisition_rows(
            canonical_rows=canonical_rows,
            event_scores=event_scores,
            raw_gap_scores=raw_gap_scores,
            target_positive_keys=failure_keys,
        )
        specificity_rows = [
            {
                "train_seed": key[0],
                "episode_seed": key[1],
                "candidate_kind": key[2],
                "primary_only_decision": primary_decisions[key],
                "full_decision": full_decisions[key],
            }
            for key in sorted(full_decisions)
        ]
        acquisition_analysis = external_binary_acquisition_analysis(
            acquisition_rows,
            queries_per_policy=confirmation["queries_per_policy"],
            replicates=confirmation["bootstrap_replicates"],
            seed=confirmation["bootstrap_rng_seed"],
        )
        specificity_analysis = specificity_downgrade_analysis(
            specificity_rows,
            policy_ids=expected_policies,
            start_ids=expected_starts,
            replicates=confirmation["bootstrap_replicates"],
            seed=confirmation["bootstrap_rng_seed"],
        )
        acquisition_decision = _acquisition_decision(acquisition_analysis)
        result.pop("measurement_note", None)
        result.pop("acquisition", None)
        result.pop("selection_rows", None)
        result["superseded_analysis"] = (
            "The suite-agreement acquisition calculation was removed from the "
            "confirmation output because the suite defines its own candidate "
            "universe. It remains only in exposed development artifacts."
        )
        result["confirmatory_analysis"] = {
            "primary_endpoint": (
                "equal-budget diagnosis-aware versus factual-event selection, "
                "with shared zero-return priority, on same-start reference "
                "competence among observed failures"
            ),
            "acquisition": acquisition_analysis,
            "acquisition_decision": acquisition_decision,
            "competence_witnessed_failure_definition": {
                "factual_return": 0.0,
                "minimum_same_start_reference_return": confirmation[
                    "competence_return_threshold"
                ],
                "candidate_trigger_determines_label": False,
            },
            "competence_witnessed_failure_count": len(failure_keys),
            "competence_witnessed_failure_evidence": failure_evidence,
            "secondary_actionable_causal_rescue": {
                "interpretation": (
                    "Secondary descriptive evidence among cells where a frozen "
                    "primary intervention is defined; it is not the acquisition "
                    "target because that would incorporate the diagnosis trigger."
                ),
                "count": len(rescue_keys),
                "evidence": rescue_evidence,
            },
            "candidate_control_analysis": {
                "interpretation": (
                    "Descriptive only: without external semantic labels, changes "
                    "between primary-only and full-suite decisions are not an "
                    "accuracy or superiority test."
                ),
                "analysis": specificity_analysis,
            },
            "information_access": {
                "shared": (
                    "all selectors know whether factual return is zero and rank "
                    "zero-return cells first"
                ),
                "diagnosis_aware": (
                    "within each factual-return stratum, factual semantic candidate "
                    "triggers only"
                ),
                "event_burden": (
                    "within each factual-return stratum, factual trajectory event "
                    "frequencies only"
                ),
                "raw_gap": (
                    "within each factual-return stratum, factual return plus a "
                    "reference rollout for every cell"
                ),
                "random": (
                    "uniform expected selection within each factual-return stratum"
                ),
            },
            "claim_boundary": (
                "A positive result supports finding reference-competent cases "
                "among observed zero-return episodes for this frozen ego, layout "
                "and two predefined diagnosis families. It does not establish "
                "that the semantic diagnosis is correct or that a selected "
                "failure is intervention-rescuable."
            ),
        }
    elif "development_external_outcome_queries_per_policy" in specification:
        failure_keys, failure_evidence = _competence_witnessed_failure_keys(
            canonical_rows=canonical_rows,
            competence_return_threshold=float(
                specification.get("competence_return_threshold", 20.0)
            ),
        )
        rescue_keys, rescue_evidence = _actionable_rescue_keys(
            canonical_rows=canonical_rows,
            swap=swap,
            initial=initial,
            competence_return_threshold=float(
                specification.get("competence_return_threshold", 20.0)
            ),
        )
        acquisition_analysis = external_binary_acquisition_analysis(
            _external_acquisition_rows(
                canonical_rows=canonical_rows,
                event_scores=event_scores,
                raw_gap_scores=raw_gap_scores,
                target_positive_keys=failure_keys,
            ),
            queries_per_policy=int(
                specification["development_external_outcome_queries_per_policy"]
            ),
            replicates=int(specification.get("bootstrap_replicates", 10_000)),
            seed=int(specification.get("bootstrap_rng_seed", 20_260_922)),
        )
        result["development_external_outcome_analysis"] = {
            "warning": (
                "Development-only result on an exposed split. It validates the "
                "non-circular endpoint implementation but is not confirmatory."
            ),
            "competence_witnessed_failure_count": len(failure_keys),
            "competence_witnessed_failure_evidence": failure_evidence,
            "secondary_actionable_causal_rescue_count": len(rescue_keys),
            "secondary_actionable_causal_rescue_evidence": rescue_evidence,
            "acquisition": acquisition_analysis,
        }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    printed = {
        "candidate_evidence": result["candidate_evidence"],
        "cost_accounting": result["cost_accounting"],
    }
    if "confirmatory_analysis" in result:
        printed["confirmatory_analysis"] = result["confirmatory_analysis"]
    else:
        printed["development_acquisition"] = result["acquisition"]
        if "development_external_outcome_analysis" in result:
            printed["development_external_outcome_analysis"] = result[
                "development_external_outcome_analysis"
            ]
    print(json.dumps(printed, indent=2, sort_keys=True))
    print(f"wrote {paths['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
