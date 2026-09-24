#!/usr/bin/env python3
"""Render the frozen diagnostic-baseline confirmation figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


METHOD_LABELS = {
    "diagnosis": "Diagnosis-aware\n(factual triggers)",
    "event": "Event burden\n(factual only)",
    "random": "Random\n(factual only)",
    "raw_gap": "Raw return gap\n(+ reference rollout)",
}

METHOD_COLORS = {
    "diagnosis": "#0F766E",
    "event": "#64748B",
    "random": "#CBD5E1",
    "raw_gap": "#C0841A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/confirmation/diagnostic_baseline_confirmation_comparison.json"),
    )
    parser.add_argument(
        "--capable-control",
        type=Path,
        default=Path("results/confirmation/diagnostic_baseline_confirmation_capable_control.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/confirmation/diagnostic_baseline_confirmation"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")


def save_split_figure(fig: plt.Figure, output_prefix: Path) -> None:
    """Keep the original figure and export crops legible at report width."""
    width, height = fig.get_size_inches()
    for suffix, bottom, top in (
        ("selection", height * 0.50, height),
        ("controls", 0.0, height * 0.50),
    ):
        crop = Bbox.from_extents(0, bottom, width, top)
        path = output_prefix.with_name(f"{output_prefix.name}_{suffix}")
        fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches=crop)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=crop)
    for suffix, left, bottom in (
        ("panel_a", 0.0, height * 0.50),
        ("panel_b", width * 0.50, height * 0.50),
        ("panel_c", 0.0, 0.0),
        ("panel_d", width * 0.50, 0.0),
    ):
        crop = Bbox.from_extents(left, bottom, left + width * 0.50, bottom + height * 0.50)
        path = output_prefix.with_name(f"{output_prefix.name}_{suffix}")
        fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches=crop)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=crop)


def render_result_figure(comparison: dict, capable: dict, output_prefix: Path) -> None:
    analysis = comparison["confirmatory_analysis"]["acquisition"]
    point = analysis["point"]
    per_policy = analysis["per_policy"]
    control = comparison["confirmatory_analysis"]["candidate_control_analysis"]["analysis"]
    costs = comparison["cost_accounting"]
    capable_summary = capable["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.4))

    # Panel A: equal-budget acquisition.
    ax = axes[0, 0]
    methods = ["diagnosis", "event", "random", "raw_gap"]
    means = [point["selectors"][name]["mean_expected_positives"] for name in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=[METHOD_COLORS[name] for name in methods], width=0.68)
    offsets = np.linspace(-0.18, 0.18, len(per_policy))
    for index, method in enumerate(methods):
        values = [row["selectors"][method]["expected_positives"] for row in per_policy]
        ax.scatter(
            np.full(len(values), index) + offsets,
            values,
            color="#0F172A",
            s=13,
            alpha=0.58,
            linewidths=0,
            zorder=3,
        )
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [METHOD_LABELS[name] for name in methods], fontsize=8)
    ax.set_ylabel("Expected useful cases per 12 queries")
    ax.set_ylim(0, 13.4)
    ax.set_title("A  Equal-budget acquisition", loc="left", fontweight="semibold")
    ax.grid(axis="y", alpha=0.2)

    # Panel B: paired policy-level diagnosis minus event yield.
    ax = axes[0, 1]
    seeds = [row["train_seed"] for row in per_policy]
    differences = [row["diagnosis_minus_event_yield"] for row in per_policy]
    positions = np.arange(len(seeds))
    colors = ["#0F766E" if value >= 0 else "#B91C1C" for value in differences]
    ax.axvline(0, color="#334155", linewidth=1)
    ax.scatter(differences, positions, c=colors, s=40, zorder=3)
    ax.set_yticks(positions, [str(seed) for seed in seeds], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Diagnosis yield minus event-burden yield")
    ax.set_title("B  Policy-level paired differences", loc="left", fontweight="semibold")
    ax.grid(axis="x", alpha=0.2)
    ci_low, ci_high = analysis["policy_cluster_bootstrap"]["ci95"]
    mean = point["diagnosis_minus_event_yield"]
    ax.errorbar(
        mean,
        len(seeds) + 0.15,
        xerr=[[mean - ci_low], [ci_high - mean]],
        fmt="D",
        color="#111827",
        capsize=4,
        markersize=5,
    )
    ax.set_ylim(len(seeds) + 1.25, -0.75)

    # Panel C: counterfactual-control cost and decision impact.
    ax = axes[1, 0]
    primary_cost = costs["primary_only_counterfactual_branches"]
    full_cost = costs["full_suite_counterfactual_branches"]
    bars = ax.bar(
        [0, 1],
        [primary_cost, full_cost],
        color=["#0F766E", "#64748B"],
        width=0.6,
    )
    for bar, value in zip(bars, [primary_cost, full_cost]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 55, f"{value:,}", ha="center", fontsize=9)
    downgrade = control["point"]
    ax.set_xticks([0, 1], ["Primary intervention", "Full control battery"])
    ax.set_ylabel("Counterfactual branches")
    ax.set_ylim(0, full_cost * 1.22)
    ax.set_title("C  Controls: cost versus changed decisions", loc="left", fontweight="semibold")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.5,
        0.88,
        f"Full controls downgraded {downgrade['downgrades']} of "
        f"{downgrade['evaluable_primary_positives']} primary-positive cases\n"
        f"({100 * downgrade['downgrade_rate']:.1f}%; two-way 95% CI "
        f"[{100 * control['bootstrap']['ci95'][0]:.1f}%, "
        f"{100 * control['bootstrap']['ci95'][1]:.1f}%])",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )

    # Panel D: capable-policy negative control.
    ax = axes[1, 1]
    prevalence_rows = [
        ("Fixed ego", capable_summary["fixed_trigger_prevalence"]),
        ("Reference policy", capable_summary["reference_trigger_prevalence"]),
        (
            "Competent reference\npairings",
            capable_summary["reference_trigger_prevalence_among_competent_pairings"],
        ),
    ]
    labels = [row[0] for row in prevalence_rows]
    values = [row[1]["proportion"] for row in prevalence_rows]
    lows = [row[1]["proportion"] - row[1]["ci_low"] for row in prevalence_rows]
    highs = [row[1]["ci_high"] - row[1]["proportion"] for row in prevalence_rows]
    x = np.arange(len(labels))
    ax.bar(x, values, color=["#0F766E", "#94A3B8", "#CBD5E1"], width=0.62)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", color="#111827", capsize=4)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Reciprocal-motion trigger prevalence")
    ax.set_ylim(0, 0.29)
    ax.set_title("D  Capable-policy negative control", loc="left", fontweight="semibold")
    ax.grid(axis="y", alpha=0.2)
    counts = [
        (capable_summary["fixed_trigger_prevalence"]["successes"], capable_summary["fixed_trigger_prevalence"]["n"]),
        (capable_summary["reference_trigger_prevalence"]["successes"], capable_summary["reference_trigger_prevalence"]["n"]),
        (
            capable_summary["reference_trigger_prevalence_among_competent_pairings"]["successes"],
            capable_summary["reference_trigger_prevalence_among_competent_pairings"]["n"],
        ),
    ]
    for index, (successes, total) in enumerate(counts):
        ax.text(index, values[index] + highs[index] + 0.012, f"{successes}/{total}", ha="center", fontsize=9)

    full_figure_note = fig.text(
        0.5,
        0.012,
        "Primary endpoint: acquisition of same-start reference-competent observed failures. "
        "Raw return gap is a privileged-information comparator because it evaluates a learned reference on every cell.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#334155",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.97), h_pad=7.0, w_pad=2.0)
    save_figure(fig, output_prefix)
    full_figure_note.remove()
    save_split_figure(fig, output_prefix)
    plt.close(fig)


def render_workflow_figure(output_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 3.1))
    ax.set_axis_off()
    boxes = [
        (0.02, "1  Observe", "Run the fixed ego with\none learned teammate"),
        (0.215, "2  Detect context", "Flag one of two\npredefined contexts"),
        (0.41, "3  Intervene", "Replay the saved state\nwith a target sequence"),
        (0.605, "4  Compare controls", "Compare no-change, wait,\nand alternative movements"),
        (0.80, "5  Report or abstain", "Supported, ambiguous,\nunsupported, or abstain"),
    ]
    for x, title, body in boxes:
        rect = plt.Rectangle(
            (x, 0.24),
            0.175,
            0.56,
            transform=ax.transAxes,
            facecolor="#F8FAFC",
            edgecolor="#0F766E",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.0875, 0.67, title, transform=ax.transAxes, ha="center", fontweight="semibold")
        ax.text(x + 0.0875, 0.44, body, transform=ax.transAxes, ha="center", va="center", fontsize=8.4)
    for x in [0.195, 0.39, 0.585, 0.78]:
        ax.annotate(
            "",
            xy=(x + 0.014, 0.52),
            xytext=(x - 0.014, 0.52),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": "#475569", "linewidth": 1.4},
        )
    ax.text(
        0.5,
        0.93,
        "Frozen diagnosis-aware evaluation workflow",
        transform=ax.transAxes,
        ha="center",
        fontsize=15,
        fontweight="semibold",
    )
    ax.text(
        0.5,
        0.08,
        "The evaluator does not retrain the ego and does not assign a label outside its predefined context families.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#334155",
    )
    save_figure(fig, output_prefix)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison = load_json(args.comparison)
    capable = load_json(args.capable_control)
    render_result_figure(comparison, capable, args.output_prefix)
    render_workflow_figure(args.output_prefix.with_name(args.output_prefix.name + "_workflow"))


if __name__ == "__main__":
    main()
