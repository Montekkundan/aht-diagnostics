#!/usr/bin/env python3
"""Compare the matched fixed-ego teammate-generation conditions."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.teamgen import mean_and_bootstrap_ci  # noqa: E402


PRODUCTIVE_EVENTS = {
    "fetch_onion",
    "place_onion_in_pot",
    "fetch_plate",
    "pick_up_soup",
    "deliver_soup",
}
DEGENERATE_EVENTS = {"blocked_move", "interact_no_effect", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-state", required=True, type=Path)
    parser.add_argument("--per-trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {"productive": 0.0, "degenerate": 0.0, "other": 0.0}
    productive = sum(count for event, count in counts.items() if event in PRODUCTIVE_EVENTS)
    degenerate = sum(count for event, count in counts.items() if event in DEGENERATE_EVENTS)
    return {
        "productive": productive / total,
        "degenerate": degenerate / total,
        "other": (total - productive - degenerate) / total,
    }


def condition_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in summary["seed_results"]:
        fixed_return = seed["return_with_fixed_ego"]["return"]["mean"]
        br_return = seed["return_with_best_response"]["return"]["mean"]
        rows.append(
            {
                "seed": seed["train_seed"],
                "fixed_return": fixed_return,
                "br_return": br_return,
                "regret": br_return - fixed_return,
                "feasible": br_return > 0,
                "exposes_ego": br_return > fixed_return and br_return > 0,
                "behavior_with_br": event_fractions(
                    seed["return_with_best_response"]["generated_teammate_events"]
                ),
            }
        )
    return rows


def paired_bootstrap_difference(
    state_values: list[float], trajectory_values: list[float], seed: int = 17
) -> dict[str, float | int]:
    differences = np.asarray(state_values) - np.asarray(trajectory_values)
    return mean_and_bootstrap_ci(differences.tolist(), seed=seed)


def summarize_condition(summary: dict[str, Any]) -> dict[str, Any]:
    rows = condition_rows(summary)
    behavior = {
        category: mean_and_bootstrap_ci(
            [row["behavior_with_br"][category] for row in rows],
            seed=30 + index,
        )
        for index, category in enumerate(("productive", "degenerate", "other"))
    }
    return {
        "objective": summary["objective"],
        "training_timesteps_per_seed": summary["effective_training_config"][
            "TIMESTEPS_PER_ITER_PARTNER"
        ],
        "released_training_timesteps_per_seed": 6_000_000,
        "seeds": len(rows),
        "evaluation_episodes_per_seed": summary["evaluation_episodes_per_seed"],
        "feasible_best_response_pairings": sum(row["feasible"] for row in rows),
        "ego_weakness_exposed": sum(row["exposes_ego"] for row in rows),
        "return_with_fixed_ego": summary["return_with_fixed_ego"],
        "return_with_best_response": summary["return_with_best_response"],
        "cooperative_regret": summary["cooperative_regret"],
        "generated_teammate_behavior_with_best_response": behavior,
        "seed_results": rows,
    }


def finding_text(per_state: dict[str, Any], per_trajectory: dict[str, Any]) -> str:
    conditions = [per_state, per_trajectory]
    budget = per_state["training_timesteps_per_seed"]
    feasible = [condition["feasible_best_response_pairings"] for condition in conditions]
    exposed = [condition["ego_weakness_exposed"] for condition in conditions]
    if all(count == 0 for count in feasible):
        return (
            f"At {budget:,} training steps per seed, neither objective produced a generated "
            "teammate that completed the task with its learned best response. The experiment "
            "therefore does not test the fixed-ego weakness hypothesis."
        )
    if all(count == 5 for count in feasible) and all(count == 5 for count in exposed):
        return (
            f"At {budget:,} training steps per seed, both objectives produced feasible generated "
            "teammates in all five seeds and exposed a large performance gap between the learned "
            "best response and the fixed ego. This supports the fixed-ego weakness hypothesis in "
            "this controlled Cramped Room setting."
        )
    return (
        f"At {budget:,} training steps per seed, feasibility or fixed-ego exposure was inconsistent "
        "across objectives or seeds. The result is mixed and does not support a general conclusion."
    )


def plot_analysis(analysis: dict[str, Any], output: Path) -> None:
    labels = ["Per-state regret", "Trajectory regret"]
    conditions = [analysis["per_state"], analysis["per_trajectory"]]
    colors = {"fixed": "#4C78A8", "br": "#F58518"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    budget = conditions[0]["training_timesteps_per_seed"]
    released = conditions[0]["released_training_timesteps_per_seed"]
    budget_label = "full-budget" if budget == released else "reduced-budget"
    fig.suptitle(
        f"Fixed-ego teammate generation: {budget_label} five-seed comparison",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    for condition_index, rows in enumerate(condition["seed_results"] for condition in conditions):
        fixed_x = condition_index * 3
        br_x = fixed_x + 1
        for row in rows:
            ax.plot(
                [fixed_x, br_x],
                [row["fixed_return"], row["br_return"]],
                color="#B8B8B8",
                linewidth=1,
                zorder=1,
            )
            ax.scatter(fixed_x, row["fixed_return"], color=colors["fixed"], zorder=2)
            ax.scatter(br_x, row["br_return"], color=colors["br"], zorder=2)
    ax.set_xticks([0, 1, 3, 4], ["fixed ego", "learned BR", "fixed ego", "learned BR"])
    ax.text(0.5, -0.22, labels[0], ha="center", transform=ax.get_xaxis_transform())
    ax.text(3.5, -0.22, labels[1], ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylabel("Mean episode return")
    ax.set_title("A. Pairing returns by training seed")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    for index, (label, condition) in enumerate(zip(labels, conditions)):
        values = [row["regret"] for row in condition["seed_results"]]
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=48, zorder=2)
        ax.hlines(np.mean(values), index - 0.2, index + 0.2, color="black", linewidth=2)
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("BR return minus fixed-ego return")
    ax.set_title("B. Cooperative regret")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    categories = ["productive", "other", "degenerate"]
    category_labels = ["task events", "movement/other", "blocked/idle/no effect"]
    category_colors = ["#54A24B", "#9D9DA1", "#E45756"]
    bottoms = np.zeros(2)
    for category, category_label, color in zip(
        categories, category_labels, category_colors
    ):
        values = [
            condition["generated_teammate_behavior_with_best_response"][category]["mean"]
            for condition in conditions
        ]
        ax.bar(labels, values, bottom=bottoms, label=category_label, color=color)
        bottoms += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of generated-teammate steps")
    ax.set_title("C. Behavior when paired with learned BR")
    ax.legend(frameon=False, loc="upper center")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Finding",
        "",
        f"Training: {budget:,} steps/seed ({100 * budget / released:.1f}% of released budget)",
        "Evaluation: 5 seeds × 20 episodes per pairing",
        "",
        f"Feasible BR pairings: per-state {conditions[0]['feasible_best_response_pairings']}/5",
        f"Feasible BR pairings: trajectory {conditions[1]['feasible_best_response_pairings']}/5",
        f"Ego weakness exposed: per-state {conditions[0]['ego_weakness_exposed']}/5",
        f"Ego weakness exposed: trajectory {conditions[1]['ego_weakness_exposed']}/5",
        "",
        *textwrap.wrap(analysis["finding"], width=48),
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=10.5, linespacing=1.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    per_state = summarize_condition(load_summary(args.per_state))
    per_trajectory = summarize_condition(load_summary(args.per_trajectory))
    analysis = {
        "finding": finding_text(per_state, per_trajectory),
        "per_state": per_state,
        "per_trajectory": per_trajectory,
        "paired_regret_difference": paired_bootstrap_difference(
            [row["regret"] for row in per_state["seed_results"]],
            [row["regret"] for row in per_trajectory["seed_results"]],
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_analysis(analysis, args.output / "comparison.png")
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
