#!/usr/bin/env python3
"""Test competing explanations for a soup-carrier swap lock."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.evaluation_conditions import condition_kwargs  # noqa: E402
from aht_diagnostics.interventions import (  # noqa: E402
    ActionSequencePolicy,
    mask_interactions,
)
from aht_diagnostics.jaxaht_bridge import activate_jaxaht, load_frozen_s5_ego  # noqa: E402
from aht_diagnostics.natural_failure import (  # noqa: E402
    branch_summary,
    clustered_paired_bootstrap_interval,
    first_soup_carrier_swap_lock_opportunity,
    soup_carrier_swap_context_summary,
    wilson_interval,
)
from aht_diagnostics.rollout import (  # noqa: E402
    ContinuationResult,
    Episode,
    policy_states_on_factual_trajectory,
    rollout_continuation,
    rollout_episode,
    rollout_policy_option,
)


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "stay",
    5: "interact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ego-checkpoint-name",
        help=(
            "Checkpoint key from available_best_responses. Required when the "
            "configuration declares multiple ego_checkpoint_names."
        ),
    )
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
    return parser.parse_args()


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_branch(
    env: Any,
    snapshot: Any,
    ego_policy: Any,
    ego_params: Any,
    teammate_policy: Any,
    teammate_params: Any,
    actions: tuple[int, ...],
    max_steps: int,
) -> ContinuationResult:
    controller = ActionSequencePolicy(actions)
    return rollout_policy_option(
        env,
        snapshot,
        ego_policy,
        ego_params,
        teammate_policy,
        teammate_params,
        controller,
        None,
        controller.init_hstate(1, aux_info={"agent_id": 0}),
        len(actions),
        max_steps,
    )


def reference_branch(
    env: Any,
    snapshot: Any,
    ego_policy: Any,
    ego_params: Any,
    teammate_policy: Any,
    teammate_params: Any,
    reference_policy: Any,
    reference_params: Any,
    reference_hstate: Any,
    option_steps: int,
    max_steps: int,
) -> ContinuationResult:
    return rollout_policy_option(
        env,
        snapshot,
        ego_policy,
        ego_params,
        teammate_policy,
        teammate_params,
        reference_policy,
        reference_params,
        reference_hstate,
        option_steps,
        max_steps,
    )


def exact_replay_matches(
    episode: Episode,
    snapshot_index: int,
    replay: ContinuationResult,
) -> bool:
    factual_suffix = episode.steps[snapshot_index:]
    return (
        replay.total_return == episode.total_return
        and len(replay.continuation_records) == len(factual_suffix)
        and all(
            (
                actual.ego_action,
                actual.partner_action,
                actual.reward,
                actual.ego_event,
                actual.partner_event,
            )
            == (
                expected.ego_action,
                expected.partner_action,
                expected.reward,
                expected.ego_event,
                expected.partner_event,
            )
            for actual, expected in zip(
                replay.continuation_records,
                factual_suffix,
                strict=True,
            )
        )
    )


def effect_summary(
    rows: list[dict[str, Any]],
    branch_name: str,
    comparison_name: str = "no_change",
) -> dict[str, Any] | None:
    differences = [
        row["branches"][branch_name]["remaining_return"]
        - row["branches"][comparison_name]["remaining_return"]
        for row in rows
    ]
    if not differences:
        return None
    improved = sum(value > 0 for value in differences)
    start_clustered = clustered_paired_bootstrap_interval(
        differences,
        [row["episode_seed"] for row in rows],
        seed=43,
    )
    policy_clustered = clustered_paired_bootstrap_interval(
        differences,
        [row["train_seed"] for row in rows],
        seed=47,
    )
    per_policy = {
        str(train_seed): float(
            sum(
                value
                for value, row in zip(differences, rows, strict=True)
                if row["train_seed"] == train_seed
            )
            / sum(row["train_seed"] == train_seed for row in rows)
        )
        for train_seed in sorted({row["train_seed"] for row in rows})
    }
    return {
        **start_clustered,
        "environment_seed_cluster_interval": start_clustered,
        "policy_cluster_interval": policy_clustered,
        "mean_difference_by_policy": per_policy,
        "distinct_policies": len(per_policy),
        "improvement_rate": wilson_interval(improved, len(differences)),
        "differences": differences,
    }


def main() -> int:
    args = parse_args()
    args.jaxaht = args.jaxaht.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.checkpoint_index = args.checkpoint_index.resolve()

    specification = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint_index = json.loads(args.checkpoint_index.read_text(encoding="utf-8"))
    allowed_ego_names = tuple(specification.get("ego_checkpoint_names", ()))
    ego_checkpoint_name = args.ego_checkpoint_name or specification.get(
        "ego_checkpoint_name"
    )
    if args.ego_checkpoint_name and allowed_ego_names:
        if args.ego_checkpoint_name not in allowed_ego_names:
            raise ValueError(
                f"ego checkpoint {args.ego_checkpoint_name!r} is not part of "
                f"the frozen transfer set {allowed_ego_names!r}"
            )
    if ego_checkpoint_name is None and len(allowed_ego_names) == 1:
        ego_checkpoint_name = allowed_ego_names[0]
    if ego_checkpoint_name is None and len(allowed_ego_names) > 1:
        raise ValueError(
            "--ego-checkpoint-name is required for a configuration with "
            "multiple ego_checkpoint_names"
        )
    minimum_lock_steps = int(specification.get("minimum_lock_steps", 8))
    confirmation_only = bool(specification.get("confirmation_only", False))
    reference_windows = tuple(
        int(window) for window in specification.get("reference_windows", [1, 2, 4, 8, 16])
    )
    reference_control_window = int(specification.get("reference_control_window", 4))
    if minimum_lock_steps < 1:
        raise ValueError("minimum_lock_steps must be positive")
    if not reference_windows or any(window < 1 for window in reference_windows):
        raise ValueError("reference_windows must contain positive integers")
    if not confirmation_only and reference_control_window not in reference_windows:
        raise ValueError("reference_control_window must be included in reference_windows")

    activate_jaxaht(args.jaxaht)
    from agents.mlp_actor_critic_agent import (  # noqa: PLC0415
        ActorWithDoubleCriticPolicy,
        MLPActorCriticPolicy,
    )
    from common.save_load_utils import load_train_run  # noqa: PLC0415
    from envs import make_env  # noqa: PLC0415
    from envs.log_wrapper import LogWrapper  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    task = OmegaConf.load(
        args.jaxaht
        / "evaluation"
        / "configs"
        / "task"
        / "overcooked-v1"
        / "cramped_room.yaml"
    )
    max_steps = int(specification["max_steps"])
    base_kwargs = OmegaConf.to_container(task.ENV_KWARGS, resolve=True)
    env_kwargs = condition_kwargs(
        base_kwargs,
        specification["environment_condition"],
        max_steps,
    )
    env = LogWrapper(make_env(task.ENV_NAME, env_kwargs))

    if ego_checkpoint_name is None:
        fixed_checkpoint = (
            args.checkpoint_root / checkpoint_index["fixed_ego"]["checkpoint"]
        )
        ego_checkpoint_name = "fixed_ego"
    else:
        available = checkpoint_index.get("available_best_responses", {})
        if ego_checkpoint_name not in available:
            choices = ", ".join(sorted(available))
            raise ValueError(
                f"Unknown ego_checkpoint_name {ego_checkpoint_name!r}; "
                f"available names: {choices}"
            )
        fixed_checkpoint = args.checkpoint_root / available[ego_checkpoint_name]
    ego_policy, ego_params = load_frozen_s5_ego(
        args.jaxaht,
        fixed_checkpoint,
        env,
        seed=0,
    )
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    teammate_policy = ActorWithDoubleCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    reference_policy = MLPActorCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)

    source_results = REPO_ROOT / specification["source_results"]
    source_summary = json.loads(
        (source_results / "summary.json").read_text(encoding="utf-8")
    )
    if "episode_seeds" in specification:
        episode_seeds = [int(seed) for seed in specification["episode_seeds"]]
    else:
        seed_range = specification["episode_seed_range"]
        episode_seeds = list(
            range(int(seed_range["start"]), int(seed_range["end_exclusive"]))
        )
    rows: list[dict[str, Any]] = []
    pairing_exposures: list[dict[str, Any]] = []
    policy_digests: dict[str, str] = {}
    evaluated_pairings = 0

    for seed_result in source_summary["seed_results"]:
        train_seed = int(seed_result["train_seed"])
        saved_path = source_results / f"seed_{train_seed}" / "trained_policies"
        saved = load_train_run(saved_path)
        teammate_params = saved["generated_teammate_params"]
        reference_params = saved["best_response_params"]
        policy_digests[str(train_seed)] = directory_digest(saved_path)

        for episode_seed in episode_seeds:
            evaluated_pairings += 1
            factual = rollout_episode(
                env,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                seed=episode_seed,
                max_steps=max_steps,
            )
            exposure = soup_carrier_swap_context_summary(factual)
            pairing_exposures.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "factual_return": factual.total_return,
                    **exposure,
                }
            )
            opportunity = first_soup_carrier_swap_lock_opportunity(
                factual,
                minimum_consecutive_steps=minimum_lock_steps,
            )
            if opportunity is None:
                continue
            intervention_step = opportunity.confirmed_at_step + 1
            if intervention_step >= len(factual.snapshots):
                continue
            snapshot = factual.snapshots[intervention_step]

            no_change = rollout_continuation(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                max_steps,
            )
            reference_states = policy_states_on_factual_trajectory(
                env,
                factual,
                reference_policy,
                reference_params,
            )
            full_reference = reference_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                reference_policy,
                reference_params,
                reference_states[intervention_step],
                max_steps - intervention_step,
                max_steps,
            )
            short_reference = {
                f"reference_{window}_steps": reference_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    reference_policy,
                    reference_params,
                    reference_states[intervention_step],
                    window,
                    max_steps,
                )
                for window in reference_windows
            } if not confirmation_only else {}
            reference_actions = (
                tuple(
                    record.ego_action
                    for record in short_reference[
                        f"reference_{reference_control_window}_steps"
                    ].intervention_records
                )
                if not confirmation_only
                else ()
            )
            primitive_branches = {
                f"one_step_{name}": sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    (action,),
                    max_steps,
                )
                for action, name in ACTION_NAMES.items()
            } if not confirmation_only else {}
            two_step_branches = {
                "detour_down_left_2": sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    (1, 3),
                    max_steps,
                ),
                "control_left_down_2": sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    (3, 1),
                    max_steps,
                ),
                "control_down_stay_2": sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    (1, 4),
                    max_steps,
                ),
                "control_wait_2": sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    (4, 4),
                    max_steps,
                ),
            }
            branches = {
                "no_change": no_change,
                "full_reference": full_reference,
                **short_reference,
                **primitive_branches,
                **two_step_branches,
            }
            if not confirmation_only:
                branches.update(
                    {
                        "yield_down_1": primitive_branches["one_step_down"],
                        "wait_1": primitive_branches["one_step_stay"],
                        f"reference_{reference_control_window}_actions_no_interact": sequence_branch(
                            env,
                            snapshot,
                            ego_policy,
                            ego_params,
                            teammate_policy,
                            teammate_params,
                            mask_interactions(reference_actions),
                            max_steps,
                        ),
                    }
                )
            rows.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "factual_return": factual.total_return,
                    "trigger": {
                        "start_step": opportunity.step,
                        "confirmed_at_step": opportunity.confirmed_at_step,
                        "intervention_step": intervention_step,
                        "minimum_consecutive_steps": opportunity.consecutive_steps,
                        "ego_position": list(opportunity.ego_position),
                        "partner_position": list(opportunity.partner_position),
                    },
                    "factual_action_at_intervention": factual.steps[
                        intervention_step
                    ].ego_action,
                    "reference_actions": list(reference_actions),
                    "no_change_exact_replay": exact_replay_matches(
                        factual,
                        intervention_step,
                        no_change,
                    ),
                    "branches": {
                        name: branch_summary(branch) for name, branch in branches.items()
                    },
                }
            )

    comparison_names = [
        "full_reference",
        "detour_down_left_2",
        "control_left_down_2",
        "control_down_stay_2",
        "control_wait_2",
    ]
    if not confirmation_only:
        comparison_names.extend(
            [
                *(f"reference_{window}_steps" for window in reference_windows),
                *(f"one_step_{name}" for name in ACTION_NAMES.values()),
                "yield_down_1",
                "wait_1",
                f"reference_{reference_control_window}_actions_no_interact",
            ]
        )
    effects_vs_no_change = {
        name: effect_summary(rows, name) for name in comparison_names
    }
    primary_name = str(specification.get("primary_comparison", "yield_down_1"))
    negative_name = str(specification.get("negative_control", "wait_1"))
    if primary_name not in effects_vs_no_change or negative_name not in effects_vs_no_change:
        raise ValueError("Configured primary or negative-control branch is unavailable")

    result = {
        "status": specification.get("status", "development_result"),
        "claim_scope": specification.get("claim_scope"),
        "protocol": {
            "trigger": (
                "At least eight consecutive transitions where the teammate holds "
                "soup, the agents are adjacent, each tries to enter the other's "
                "cell, and both movements are blocked."
            ),
            "minimum_lock_steps": minimum_lock_steps,
            "intervention_timing": (
                "The intervention begins on the snapshot after the eighth "
                "qualifying transition, so it uses no future trigger information."
            ),
            "primary_comparison": primary_name,
            "negative_control": negative_name,
            "reference_windows": list(reference_windows),
            "reference_control_window": reference_control_window,
            "confirmation_only": confirmation_only,
            "environment_condition": specification["environment_condition"],
            "max_steps": max_steps,
        },
        "provenance": {
            "source_results": str(source_results),
            "source_implementation_reference": source_summary.get(
                "implementation_reference"
            ),
            "fixed_checkpoint_sha256": directory_digest(fixed_checkpoint),
            "ego_checkpoint_name": ego_checkpoint_name,
            "ego_checkpoint_path": str(fixed_checkpoint),
            "saved_policy_sha256": policy_digests,
            "checkpoint_dataset_revision": checkpoint_index.get("revision"),
            "analysis_script_sha256": file_digest(Path(__file__).resolve()),
            "config_sha256": file_digest(args.config),
        },
        "pairing_exposures": pairing_exposures,
        "cases": rows,
        "summary": {
            "evaluated_pairings": evaluated_pairings,
            "triggered_cases": len(rows),
            "train_seeds": sorted({row["train_seed"] for row in rows}),
            "episode_seeds": sorted({row["episode_seed"] for row in rows}),
            "triggers_by_train_seed": dict(
                sorted(Counter(row["train_seed"] for row in rows).items())
            ),
            "triggers_by_episode_seed": dict(
                sorted(Counter(row["episode_seed"] for row in rows).items())
            ),
            "trigger_prevalence": wilson_interval(len(rows), evaluated_pairings),
            "exact_replay_cases": sum(
                bool(row["no_change_exact_replay"]) for row in rows
            ),
            "exposure_funnel": {
                "pairings_with_teammate_soup": sum(
                    bool(row["has_teammate_soup"]) for row in pairing_exposures
                ),
                "pairings_with_adjacent_soup_carrier": sum(
                    bool(row["has_adjacent_soup_carrier"])
                    for row in pairing_exposures
                ),
                "pairings_with_reciprocal_swap_attempt": sum(
                    bool(row["has_reciprocal_swap_attempt"])
                    for row in pairing_exposures
                ),
                "pairings_with_any_blocked_swap": sum(
                    int(row["blocked_swap_steps"]) > 0 for row in pairing_exposures
                ),
            },
            "effects_vs_no_change": effects_vs_no_change,
            "primary_effect_vs_no_change": effects_vs_no_change[primary_name],
            "negative_control_effect_vs_no_change": effects_vs_no_change[
                negative_name
            ],
            "primary_effect_vs_negative_control": effect_summary(
                rows,
                primary_name,
                negative_name,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
