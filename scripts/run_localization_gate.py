#!/usr/bin/env python3
"""Run the controlled Overcooked causal-localization gate."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.defects import (  # noqa: E402
    PLANTED_DEFECTS,
    InteractionSuppressionPolicy,
    classify_repair_event,
)
from aht_diagnostics.jaxaht_bridge import (  # noqa: E402
    load_frozen_s5_ego,
    make_overcooked_env,
    make_overcooked_partner,
)
from aht_diagnostics.rollout import (  # noqa: E402
    Episode,
    counterfactual_splice_profile,
    reference_action_diagnostics,
    rollout_episode,
)


@dataclass(frozen=True)
class PartnerSpec:
    name: str
    split: str
    role: str
    reference_checkpoint: str
    p_onion_on_counter: float = 0.0
    p_plate_on_counter: float = 0.0


PARTNERS = (
    PartnerSpec(
        "independent_agent_0",
        "discovery",
        "independent",
        "cramped_room_independent_agent_0_serious/2026-03-16_12-18-02/ego_train_run",
    ),
    PartnerSpec(
        "onion_agent_0.1",
        "heldout",
        "onion",
        "cramped_room_onion_agent_0_1_serious/2026-03-16_12-20-27/ego_train_run",
        p_onion_on_counter=0.1,
    ),
    PartnerSpec(
        "independent_agent_0.4",
        "heldout",
        "independent",
        "cramped_room_independent_agent_0_4_serious/2026-03-16_11-55-04/ego_train_run",
        p_onion_on_counter=0.4,
        p_plate_on_counter=0.4,
    ),
    PartnerSpec(
        "plate_agent_0.1",
        "discovery",
        "plate",
        "cramped_room_plate_agent_0_1_serious/2026-03-16_12-43-21/ego_train_run",
        p_plate_on_counter=0.1,
    ),
)

CHECKPOINT_PREFIX = "overcooked_v1_cramped_room"
FROZEN_EGO = (
    "cramped_room_independent_agent_0_serious/"
    "2026-03-16_12-18-02/ego_train_run"
)
DELIVERY_REWARD = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "localization_gate",
    )
    parser.add_argument("--mode", choices=("smoke", "pilot", "validation"), default="smoke")
    parser.add_argument(
        "--reference-mode",
        choices=("matched", "oracle"),
        default="matched",
        help="matched is the practical evaluator; oracle uses the clean pre-defect ego",
    )
    parser.add_argument(
        "--ego-mode",
        choices=("learned", "heuristic"),
        default="learned",
        help="heuristic is a controlled calibration; learned uses the downloaded S5 BR",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("limited", "exhaustive"),
        default="limited",
        help="limited tests four selected steps; exhaustive tests every timestep",
    )
    return parser.parse_args()


def checkpoint_path(root: Path, relative: str) -> Path:
    return root / CHECKPOINT_PREFIX / relative


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def event_rule_scores(episode: Episode) -> np.ndarray:
    scores = []
    for record in episode.steps:
        score = 0.0
        if record.ego_event == "wait" and record.ego_inventory != "empty":
            score = 3.0
        elif record.ego_event == "interact_no_effect":
            score = 2.0
        elif record.ego_event == "blocked_move":
            score = 1.0
        if record.ego_event == record.partner_event and record.ego_event.startswith("fetch_"):
            score = max(score, 2.0)
        scores.append(score)
    return np.asarray(scores, dtype=float)


def tie_aware_auc(scores: np.ndarray, positives: set[int]) -> float:
    if not positives or len(positives) == len(scores):
        return float("nan")
    positive_scores = scores[sorted(positives)]
    negative_scores = np.delete(scores, sorted(positives))
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def fractional_top_hit(scores: np.ndarray, positives: set[int]) -> float:
    if not positives or not len(scores):
        return float("nan")
    top = np.flatnonzero(scores == np.max(scores))
    return float(sum(int(index in positives) for index in top) / len(top))


def candidate_steps(
    episode: Episode,
    reference_actions: list[int],
    kl_scores: np.ndarray,
    event_scores: np.ndarray,
    limit: int,
) -> set[int]:
    """Select a fixed repair budget without reading planted ground truth."""
    n_steps = len(episode.steps)
    mismatches = [
        index
        for index, (record, reference) in enumerate(
            zip(episode.steps, reference_actions, strict=True)
        )
        if record.ego_action != reference
    ]
    ordered: list[int] = []
    for indices in (
        sorted(mismatches, key=lambda index: (-kl_scores[index], index)),
        sorted(range(n_steps), key=lambda index: (-event_scores[index], index)),
        np.linspace(0, max(0, n_steps - 1), min(limit, n_steps), dtype=int).tolist(),
    ):
        for index in indices:
            index = int(index)
            if index not in ordered:
                ordered.append(index)
            if len(ordered) >= limit:
                return set(ordered)
    return set(ordered)


def diagnose(
    env: Any,
    episode: Episode,
    ego_policy: Any,
    ego_params: Any,
    partner: Any,
    reference: Any,
    reference_params: Any,
    max_steps: int,
    repair_budget: int | None,
) -> dict[str, Any]:
    reference_actions, kl_scores = reference_action_diagnostics(
        env,
        episode,
        reference,
        reference_params,
    )
    rule_scores = event_rule_scores(episode)
    if all(
        record.ego_action == reference_action
        for record, reference_action in zip(
            episode.steps,
            reference_actions,
            strict=True,
        )
    ):
        return {
            "reference_actions": reference_actions,
            "kl_scores": kl_scores,
            "rule_scores": rule_scores,
            "gain_scores": np.zeros(len(episode.steps), dtype=float),
            "candidates": set(),
            "top_repair": None,
        }
    candidates = (
        set(range(len(episode.steps)))
        if repair_budget is None
        else candidate_steps(
            episode,
            reference_actions,
            kl_scores,
            rule_scores,
            repair_budget,
        )
    )
    repairs = counterfactual_splice_profile(
        env,
        episode,
        ego_policy,
        ego_params,
        partner,
        None,
        reference,
        reference_params,
        max_steps=max_steps,
        windows=(1,),
        candidate_steps=candidates,
    )
    gain_scores = np.zeros(len(episode.steps), dtype=float)
    for repair in repairs:
        gain_scores[repair.step] = max(gain_scores[repair.step], repair.gain)
    top_repair = max(repairs, key=lambda row: (row.gain, -row.step), default=None)
    return {
        "reference_actions": reference_actions,
        "kl_scores": kl_scores,
        "rule_scores": rule_scores,
        "gain_scores": gain_scores,
        "candidates": candidates,
        "top_repair": top_repair,
    }


def macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = [defect.name for defect in PLANTED_DEFECTS]
    scores = []
    for label in labels:
        tp = sum(row["true_type"] == label and row["predicted_type"] == label for row in rows)
        fp = sum(row["true_type"] != label and row["predicted_type"] == label for row in rows)
        fn = sum(row["true_type"] == label and row["predicted_type"] != label for row in rows)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def bootstrap_margin(rows: list[dict[str, Any]], baseline_key: str) -> tuple[float, float]:
    if len(rows) < 2:
        return float("nan"), float("nan")
    differences = np.asarray(
        [row["repair_hit"] - row[baseline_key] for row in rows],
        dtype=float,
    )
    rng = np.random.default_rng(20260810)
    draws = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return float(differences.mean()), float(np.quantile(draws, 0.025))


def aggregate(
    case_rows: list[dict[str, Any]],
    healthy_rows: list[dict[str, Any]],
    min_eligible: int,
) -> dict[str, Any]:
    heldout = [row for row in case_rows if row["split"] == "heldout" and row["eligible"]]
    heldout_healthy = [row for row in healthy_rows if row["split"] == "heldout"]
    counts = Counter(row["true_type"] for row in heldout)

    def mean(key: str) -> float:
        values = [row[key] for row in heldout if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    type_f1 = macro_f1(heldout) if heldout else float("nan")
    healthy_fpr = (
        float(np.mean([row["false_positive"] for row in heldout_healthy]))
        if heldout_healthy
        else float("nan")
    )
    baseline_hits = {
        "action_kl_hit": mean("action_kl_hit"),
        "event_rule_hit": mean("event_rule_hit"),
        "random_hit": mean("random_hit"),
    }
    best_baseline = max(baseline_hits, key=lambda key: baseline_hits[key]) if heldout else "random_hit"
    margin, margin_lower = bootstrap_margin(heldout, best_baseline)

    enough = all(counts[defect.name] >= min_eligible for defect in PLANTED_DEFECTS)
    gates = {
        "eligible_cases": {
            "value": dict(counts),
            "threshold": min_eligible,
            "pass": enough,
        },
        "weakness_type_macro_f1": {
            "value": type_f1,
            "threshold": 0.75,
            "pass": bool(type_f1 >= 0.75),
        },
        "semantic_decision_hit_rate": {
            "value": mean("repair_hit"),
            "threshold": 0.70,
            "pass": bool(mean("repair_hit") >= 0.70),
        },
        "healthy_false_positive_rate": {
            "value": healthy_fpr,
            "threshold": 0.10,
            "pass": bool(healthy_fpr <= 0.10),
        },
        "outcome_improving_repair_rate": {
            "value": mean("repair_success"),
            "threshold": 0.70,
            "pass": bool(mean("repair_success") >= 0.70),
        },
        "heldout_localization_auc": {
            "value": mean("repair_auc"),
            "threshold": 0.80,
            "pass": bool(mean("repair_auc") >= 0.80),
        },
        "better_than_localization_baselines": {
            "value": {
                "repair_hit": mean("repair_hit"),
                **baseline_hits,
                "best_baseline": best_baseline,
                "margin": margin,
                "bootstrap_95_lower": margin_lower,
            },
            "threshold": "margin >= 0.15 and 95% lower bound > 0",
            "pass": bool(margin >= 0.15 and margin_lower > 0),
        },
    }
    if not enough:
        verdict = "INCONCLUSIVE"
        reason = "Too few eligible held-out failures to test every planted weakness."
    elif all(gate["pass"] for gate in gates.values()):
        verdict = "CONTROLLED_GATE_PASS"
        reason = "All predefined controlled localization gates passed."
    else:
        verdict = "UNSUCCESSFUL"
        reason = "At least one predefined evaluator gate failed with sufficient cases."
    return {
        "verdict": verdict,
        "reason": reason,
        "gates": gates,
        "heldout_eligible_total": len(heldout),
        "heldout_healthy_total": len(heldout_healthy),
        "warning": "A controlled pass is calibration evidence, not a paper-ready result.",
    }


def plot_summary(
    case_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    heldout = [row for row in case_rows if row["split"] == "heldout" and row["eligible"]]
    labels = [defect.name.replace("_", "\n") for defect in PLANTED_DEFECTS]
    counts = Counter(row["true_type"] for row in heldout)
    methods = ["repair_hit", "action_kl_hit", "event_rule_hit", "random_hit"]
    method_labels = ["repair", "action KL", "event rules", "random"]
    hits = [float(np.mean([row[key] for row in heldout])) if heldout else 0.0 for key in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(labels, [counts[defect.name] for defect in PLANTED_DEFECTS], color="#4c78a8")
    axes[0].axhline(
        summary["gates"]["eligible_cases"]["threshold"],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_title("Held-out eligible failures")
    axes[0].set_ylabel("episodes")

    colors = ["#f2a541", "#7f8c8d", "#7f8c8d", "#7f8c8d"]
    axes[1].bar(method_labels, hits, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fractional defect-interval hit")
    axes[1].set_title("Held-out localization")

    gate_names = [name for name in summary["gates"] if name != "eligible_cases"]
    gate_values = [bool(summary["gates"][name]["pass"]) for name in gate_names]
    axes[2].barh(
        [name.replace("_", " ") for name in gate_names],
        np.ones(len(gate_names)),
        color=["#59a14f" if value else "#e15759" for value in gate_values],
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_xticks([])
    for index, passed in enumerate(gate_values):
        axes[2].text(
            0.5,
            index,
            "PASS" if passed else "FAIL",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    axes[2].set_title(summary["verdict"].replace("_", " "))

    fig.suptitle("Frozen-ego semantic localization gate", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "localization_gate.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(logging.CRITICAL)

    if args.mode == "smoke":
        (
            max_steps,
            discovery_seeds,
            heldout_seeds,
            repair_budget,
            min_eligible,
            diagnosis_quota,
            healthy_quota,
        ) = (
            100,
            range(2),
            range(1000, 1002),
            4,
            1,
            2,
            2,
        )
    elif args.mode == "pilot":
        (
            max_steps,
            discovery_seeds,
            heldout_seeds,
            repair_budget,
            min_eligible,
            diagnosis_quota,
            healthy_quota,
        ) = (
            100,
            range(6),
            range(1000, 1012),
            4,
            5,
            5,
            5,
        )
    else:
        (
            max_steps,
            discovery_seeds,
            heldout_seeds,
            repair_budget,
            min_eligible,
            diagnosis_quota,
            healthy_quota,
        ) = (
            100,
            range(20),
            range(1000, 1060),
            4,
            20,
            20,
            20,
        )

    env = make_overcooked_env(args.jaxaht, max_steps, "cramped_room", random_reset=True)
    effective_repair_budget = None if args.candidate_mode == "exhaustive" else repair_budget
    if args.ego_mode == "heuristic":
        healthy_policy = make_overcooked_partner(
            args.jaxaht,
            "independent",
            "cramped_room",
        )
        healthy_params = None
    else:
        healthy_policy, healthy_params = load_frozen_s5_ego(
            args.jaxaht,
            checkpoint_path(args.checkpoint_root, FROZEN_EGO),
            env,
        )

    case_rows: list[dict[str, Any]] = []
    healthy_rows: list[dict[str, Any]] = []
    factual_rows: list[dict[str, Any]] = []
    diagnosed_counts: Counter[tuple[str, str]] = Counter()
    healthy_counts: Counter[tuple[str, str]] = Counter()

    for partner_index, spec in enumerate(PARTNERS):
        seeds = discovery_seeds if spec.split == "discovery" else heldout_seeds
        partner = make_overcooked_partner(
            args.jaxaht,
            spec.role,
            "cramped_room",
            p_onion_on_counter=spec.p_onion_on_counter,
            p_plate_on_counter=spec.p_plate_on_counter,
        )
        if args.reference_mode == "oracle":
            reference, reference_params = healthy_policy, healthy_params
        else:
            reference, reference_params = load_frozen_s5_ego(
                args.jaxaht,
                checkpoint_path(args.checkpoint_root, spec.reference_checkpoint),
                env,
                seed=10_000 + partner_index,
            )

        print(f"partner={spec.name} split={spec.split}", flush=True)
        for seed in seeds:
            healthy = rollout_episode(
                env,
                healthy_policy,
                healthy_params,
                partner,
                None,
                seed,
                max_steps,
            )
            witness = rollout_episode(
                env,
                reference,
                reference_params,
                partner,
                None,
                seed,
                max_steps,
            )
            healthy_key = (spec.split, spec.name)
            if healthy_counts[healthy_key] < healthy_quota:
                healthy_diagnosis = diagnose(
                    env,
                    healthy,
                    healthy_policy,
                    healthy_params,
                    partner,
                    reference,
                    reference_params,
                    max_steps,
                    effective_repair_budget,
                )
                healthy_top = healthy_diagnosis["top_repair"]
                healthy_rows.append(
                    {
                        "partner": spec.name,
                        "split": spec.split,
                        "seed": seed,
                        "healthy_return": healthy.total_return,
                        "witness_return": witness.total_return,
                        "top_gain": 0.0 if healthy_top is None else healthy_top.gain,
                        "predicted_type": (
                            "unknown"
                            if healthy_top is None
                            else classify_repair_event(healthy_top.repaired_event)
                        ),
                        "false_positive": bool(
                            healthy_top is not None
                            and healthy_top.gain >= DELIVERY_REWARD
                            and classify_repair_event(healthy_top.repaired_event) != "unknown"
                        ),
                    }
                )
                healthy_counts[healthy_key] += 1

            for defect in PLANTED_DEFECTS:
                faulty_policy = InteractionSuppressionPolicy(healthy_policy, defect, env)
                faulty = rollout_episode(
                    env,
                    faulty_policy,
                    healthy_params,
                    partner,
                    None,
                    seed,
                    max_steps,
                )
                trigger_steps = {
                    record.step for record in faulty.steps if record.ego_defect_triggered
                }
                true_steps = {
                    record.step for record in faulty.steps if record.ego_defect_suppressed
                }
                planted_loss = healthy.total_return - faulty.total_return
                eligible = bool(
                    healthy.total_return >= DELIVERY_REWARD
                    and witness.total_return >= DELIVERY_REWARD
                    and planted_loss >= DELIVERY_REWARD
                    and true_steps
                )
                factual_rows.append(
                    {
                        "partner": spec.name,
                        "split": spec.split,
                        "seed": seed,
                        "defect": defect.name,
                        "healthy_return": healthy.total_return,
                        "faulty_return": faulty.total_return,
                        "witness_return": witness.total_return,
                        "planted_loss": planted_loss,
                        "first_trigger_count": len(trigger_steps),
                        "suppressed_step_count": len(true_steps),
                        "eligible": eligible,
                    }
                )
                if not eligible:
                    continue

                diagnosis_key = (spec.split, defect.name)
                if diagnosed_counts[diagnosis_key] >= diagnosis_quota:
                    continue

                diagnosis = diagnose(
                    env,
                    faulty,
                    faulty_policy,
                    healthy_params,
                    partner,
                    reference,
                    reference_params,
                    max_steps,
                    effective_repair_budget,
                )
                top = diagnosis["top_repair"]
                top_gain = 0.0 if top is None else top.gain
                predicted_type = (
                    "unknown" if top is None else classify_repair_event(top.repaired_event)
                )
                predicted_step = -1 if top is None else top.step
                recovered_fraction = (
                    0.0 if planted_loss <= 0 else min(1.0, top_gain / planted_loss)
                )
                repair_success = bool(
                    top_gain >= DELIVERY_REWARD or recovered_fraction >= 0.5
                )
                case_rows.append(
                    {
                        "partner": spec.name,
                        "split": spec.split,
                        "seed": seed,
                        "true_type": defect.name,
                        "predicted_type": predicted_type,
                        "true_steps": json.dumps(sorted(true_steps)),
                        "predicted_step": predicted_step,
                        "candidate_coverage": bool(true_steps & diagnosis["candidates"]),
                        "healthy_return": healthy.total_return,
                        "faulty_return": faulty.total_return,
                        "witness_return": witness.total_return,
                        "planted_loss": planted_loss,
                        "repair_gain": top_gain,
                        "recovered_fraction": recovered_fraction,
                        "repair_success": repair_success,
                        "repair_hit": float(predicted_step in true_steps and top_gain >= DELIVERY_REWARD),
                        "repair_auc": tie_aware_auc(diagnosis["gain_scores"], true_steps),
                        "action_kl_hit": fractional_top_hit(diagnosis["kl_scores"], true_steps),
                        "action_kl_auc": tie_aware_auc(diagnosis["kl_scores"], true_steps),
                        "event_rule_hit": fractional_top_hit(diagnosis["rule_scores"], true_steps),
                        "event_rule_auc": tie_aware_auc(diagnosis["rule_scores"], true_steps),
                        "random_hit": float(len(true_steps) / len(faulty.steps)),
                        "random_auc": 0.5,
                        "eligible": True,
                    }
                )
                diagnosed_counts[diagnosis_key] += 1
            print(f"  seed={seed} healthy={healthy.total_return:.0f} witness={witness.total_return:.0f}", flush=True)

    summary = aggregate(case_rows, healthy_rows, min_eligible)
    summary["config"] = {
        "mode": args.mode,
        "ego_mode": args.ego_mode,
        "reference_mode": args.reference_mode,
        "candidate_mode": args.candidate_mode,
        "ground_truth_mode": "all_suppressed_steps",
        "max_steps": max_steps,
        "repair_budget": "all_timesteps" if effective_repair_budget is None else repair_budget,
        "diagnosis_quota_per_split_and_type": diagnosis_quota,
        "healthy_quota_per_split_and_policy": healthy_quota,
        "discovery_policies": [spec.name for spec in PARTNERS if spec.split == "discovery"],
        "heldout_policies": [spec.name for spec in PARTNERS if spec.split == "heldout"],
        "eligibility": "healthy >= 20, matched witness >= 20, planted loss >= 20, defect triggered",
        "repair_threshold": DELIVERY_REWARD,
    }
    write_csv(args.output / "factual_cases.csv", factual_rows)
    write_csv(args.output / "eligible_diagnoses.csv", case_rows)
    write_csv(args.output / "healthy_controls.csv", healthy_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_summary(case_rows, summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["verdict"] == "CONTROLLED_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
