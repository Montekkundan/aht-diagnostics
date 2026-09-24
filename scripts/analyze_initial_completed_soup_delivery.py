#!/usr/bin/env python3
"""Test delivery-completion and navigation explanations from initial soup states."""

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
from aht_diagnostics.jaxaht_bridge import (  # noqa: E402
    activate_jaxaht,
    load_frozen_s5_ego,
    make_overcooked_partner,
)
from aht_diagnostics.natural_failure import (  # noqa: E402
    branch_summary,
    clustered_paired_bootstrap_interval,
    first_initial_soup_disposition,
    initial_completed_soup_opportunity,
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
from aht_diagnostics.semantics import (  # noqa: E402
    INTERACT_ACTION,
    STAY_ACTION,
    state_view,
)


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    STAY_ACTION: "stay",
    INTERACT_ACTION: "interact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    start_interval = clustered_paired_bootstrap_interval(
        differences,
        [row["episode_seed"] for row in rows],
        seed=59,
    )
    policy_interval = clustered_paired_bootstrap_interval(
        differences,
        [row["train_seed"] for row in rows],
        seed=61,
    )
    return {
        **start_interval,
        "environment_seed_cluster_interval": start_interval,
        "policy_cluster_interval": policy_interval,
        "improvement_rate": wilson_interval(
            sum(difference > 0 for difference in differences),
            len(differences),
        ),
        "differences": differences,
    }


def direct_difference_summary(
    rows: list[dict[str, Any]],
    differences: list[float],
) -> dict[str, Any] | None:
    """Summarize paired differences that are not represented by one branch."""

    if not differences:
        return None
    if len(rows) != len(differences):
        raise ValueError("rows and differences must have the same length")
    start_interval = clustered_paired_bootstrap_interval(
        differences,
        [row["episode_seed"] for row in rows],
        seed=67,
    )
    policy_interval = clustered_paired_bootstrap_interval(
        differences,
        [row["train_seed"] for row in rows],
        seed=71,
    )
    return {
        **start_interval,
        "environment_seed_cluster_interval": start_interval,
        "policy_cluster_interval": policy_interval,
        "improvement_rate": wilson_interval(
            sum(difference > 0 for difference in differences),
            len(differences),
        ),
        "differences": differences,
    }


def intervention_delivered(result: ContinuationResult) -> bool:
    return any(record.ego_event == "deliver_soup" for record in result.intervention_records)


def branch_delivered(result: ContinuationResult) -> bool:
    """Return whether either agent delivered after the branch started."""

    return any(
        record.ego_event == "deliver_soup"
        or record.partner_event == "deliver_soup"
        for record in (*result.intervention_records, *result.continuation_records)
    )


def factual_window(
    episode: Episode,
    center_index: int,
    *,
    preceding: int = 6,
    following: int = 16,
) -> list[dict[str, Any]]:
    start = max(0, center_index - preceding)
    stop = min(len(episode.steps), center_index + following)
    return [
        {
            "step": record.step,
            "ego_action": record.ego_action,
            "partner_action": record.partner_action,
            "ego_event": record.ego_event,
            "partner_event": record.partner_event,
            "ego_inventory": record.ego_inventory,
            "partner_inventory": record.partner_inventory,
            "ego_position": list(record.ego_position),
            "partner_position": list(record.partner_position),
            "reward": record.reward,
        }
        for record in episode.steps[start:stop]
    ]


def summarize_rows(
    rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
) -> dict[str, Any]:
    branch_names = (
        "full_reference",
        "reference_to_delivery",
        "reference_exact_actions",
        "reference_actions_no_interact",
        "reference_actions_wait",
        "semantic_delivery_option",
        "semantic_delivery_exact_actions",
        "semantic_delivery_actions_no_interact",
        "semantic_delivery_actions_wait",
        *(f"one_step_{name}" for name in ACTION_NAMES.values()),
    )
    primary = "reference_actions_no_interact"
    one_step_names = tuple(f"one_step_{name}" for name in ACTION_NAMES.values())
    initial_soup_exposures = [
        exposure for exposure in exposures if exposure["initial_completed_soup"]
    ]
    disposition_counts = Counter(
        exposure["initial_soup_disposition"] or "none"
        for exposure in initial_soup_exposures
    )
    eligibility_counts = Counter(
        exposure["causal_eligibility"] for exposure in exposures
    )
    primary_minus_best_one_step = [
        row["branches"][primary]["remaining_return"]
        - max(
            row["branches"][name]["remaining_return"]
            for name in one_step_names
        )
        for row in rows
    ]
    return {
        "initial_completed_soup_exposures": len(initial_soup_exposures),
        "initial_soup_disposition_counts": dict(sorted(disposition_counts.items())),
        "causal_eligibility_counts": dict(sorted(eligibility_counts.items())),
        "drop_candidates": len(rows),
        "reference_option_deliveries": sum(
            row["reference_option_delivered"] for row in rows
        ),
        "semantic_option_deliveries": sum(
            row["semantic_option_delivered"] for row in rows
        ),
        "exact_no_change_replays": sum(row["no_change_exact_replay"] for row in rows),
        "train_seeds": sorted({row["train_seed"] for row in rows}),
        "episode_seeds": sorted({row["episode_seed"] for row in rows}),
        "drop_candidates_by_train_seed": dict(
            sorted(Counter(row["train_seed"] for row in rows).items())
        ),
        "drop_candidates_by_episode_seed": dict(
            sorted(Counter(row["episode_seed"] for row in rows).items())
        ),
        "effects_drop_candidates": {
            name: effect_summary(rows, name) for name in branch_names
        },
        "primary_movement_only_comparisons": {
            "versus_no_change": effect_summary(rows, primary),
            "versus_matched_wait": effect_summary(
                rows,
                primary,
                "reference_actions_wait",
            ),
            "versus_best_one_step_control": direct_difference_summary(
                rows,
                primary_minus_best_one_step,
            ),
        },
        "completion_specific_contrasts": {
            "reference_minus_same_actions_without_interact": effect_summary(
                rows,
                "reference_exact_actions",
                "reference_actions_no_interact",
            ),
            "semantic_minus_same_actions_without_interact": effect_summary(
                rows,
                "semantic_delivery_exact_actions",
                "semantic_delivery_actions_no_interact",
            ),
        },
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

    fixed_checkpoint = args.checkpoint_root / checkpoint_index["fixed_ego"]["checkpoint"]
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
    delivery_policy = make_overcooked_partner(
        args.jaxaht,
        "plate",
        using_log_wrapper=True,
    )

    source_results = REPO_ROOT / specification["source_results"]
    source_summary = json.loads(
        (source_results / "summary.json").read_text(encoding="utf-8")
    )
    seed_range = specification["episode_seed_range"]
    episode_seeds = range(
        int(seed_range["start"]),
        int(seed_range["end_exclusive"]),
    )
    reference_budget = int(specification["reference_option_budget"])
    delivery_budget = int(specification["delivery_option_budget"])
    minimum_remaining_steps = int(specification["minimum_remaining_steps"])
    require_zero_factual_return = bool(
        specification["require_zero_factual_return"]
    )

    rows: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
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
            initial_view = state_view(factual.snapshots[0].env_state)
            opportunity = initial_completed_soup_opportunity(factual)
            disposition = first_initial_soup_disposition(factual)
            remaining_steps = (
                max_steps - disposition.step if disposition is not None else None
            )
            if opportunity is None:
                eligibility = "no_initial_completed_soup"
            elif disposition is None:
                eligibility = "initial_soup_not_disposed"
            elif disposition.event != "drop_soup":
                eligibility = "initial_soup_delivered"
            elif require_zero_factual_return and factual.total_return > 0.0:
                eligibility = "positive_factual_task_return"
            elif remaining_steps is None or remaining_steps < minimum_remaining_steps:
                eligibility = "insufficient_remaining_horizon"
            else:
                eligibility = "eligible_drop_failure"
            exposures.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "factual_return": factual.total_return,
                    "initial_ego_inventory": initial_view.inventories[0],
                    "initial_partner_inventory": initial_view.inventories[1],
                    "initial_completed_soup": opportunity is not None,
                    "initial_soup_disposition": (
                        disposition.event if disposition is not None else None
                    ),
                    "initial_soup_disposition_step": (
                        disposition.step if disposition is not None else None
                    ),
                    "remaining_steps_at_disposition": remaining_steps,
                    "causal_eligibility": eligibility,
                }
            )
            if eligibility != "eligible_drop_failure":
                continue

            snapshot_index = next(
                index
                for index, candidate in enumerate(factual.snapshots)
                if candidate.step == disposition.step
            )
            snapshot = factual.snapshots[snapshot_index]
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
            delivery_states = policy_states_on_factual_trajectory(
                env,
                factual,
                delivery_policy,
                None,
            )
            full_reference = rollout_policy_option(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                reference_policy,
                reference_params,
                reference_states[snapshot_index],
                max_steps - snapshot.step,
                max_steps,
            )
            reference_to_delivery = rollout_policy_option(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                reference_policy,
                reference_params,
                reference_states[snapshot_index],
                reference_budget,
                max_steps,
                terminate_on_events=frozenset({"deliver_soup"}),
            )
            reference_actions = tuple(
                record.ego_action for record in reference_to_delivery.intervention_records
            )
            if not reference_actions:
                raise RuntimeError(
                    "Reference option produced no actions for "
                    f"train_seed={train_seed}, episode_seed={episode_seed}"
                )
            reference_exact = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                reference_actions,
                max_steps,
            )
            reference_no_interact = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                mask_interactions(reference_actions),
                max_steps,
            )
            reference_wait = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                tuple(4 for _ in reference_actions),
                max_steps,
            )

            semantic_delivery = rollout_policy_option(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                delivery_policy,
                None,
                delivery_states[snapshot_index],
                delivery_budget,
                max_steps,
                terminate_on_events=frozenset({"deliver_soup"}),
            )
            semantic_actions = tuple(
                record.ego_action for record in semantic_delivery.intervention_records
            )
            if not semantic_actions:
                raise RuntimeError(
                    "Semantic delivery option produced no actions for "
                    f"train_seed={train_seed}, episode_seed={episode_seed}"
                )
            semantic_exact = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                semantic_actions,
                max_steps,
            )
            semantic_no_interact = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                mask_interactions(semantic_actions),
                max_steps,
            )
            semantic_wait = sequence_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                tuple(4 for _ in semantic_actions),
                max_steps,
            )

            one_step_branches = {
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
            }

            branches = {
                "no_change": no_change,
                "full_reference": full_reference,
                "reference_to_delivery": reference_to_delivery,
                "reference_exact_actions": reference_exact,
                "reference_actions_no_interact": reference_no_interact,
                "reference_actions_wait": reference_wait,
                "semantic_delivery_option": semantic_delivery,
                "semantic_delivery_exact_actions": semantic_exact,
                "semantic_delivery_actions_no_interact": semantic_no_interact,
                "semantic_delivery_actions_wait": semantic_wait,
                **one_step_branches,
            }
            rows.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "factual_return": factual.total_return,
                    "intervention_step": snapshot.step,
                    "factual_disposition": disposition.event,
                    "factual_disposition_step": disposition.step,
                    "initial_state": {
                        "ego_position": list(opportunity.ego_position),
                        "partner_position": list(opportunity.partner_position),
                        "partner_inventory": opportunity.partner_inventory,
                    },
                    "pre_drop_state": {
                        "ego_position": list(disposition.ego_position),
                        "partner_position": list(disposition.partner_position),
                        "partner_inventory": disposition.partner_inventory,
                    },
                    "factual_window": factual_window(factual, snapshot_index),
                    "no_change_exact_replay": exact_replay_matches(
                        factual,
                        snapshot_index,
                        no_change,
                    ),
                    "reference_option_delivered": intervention_delivered(
                        reference_to_delivery
                    ),
                    "semantic_option_delivered": intervention_delivered(
                        semantic_delivery
                    ),
                    "reference_actions": list(reference_actions),
                    "semantic_delivery_actions": list(semantic_actions),
                    "reference_exact_reproduced": (
                        reference_exact.total_return == reference_to_delivery.total_return
                    ),
                    "semantic_exact_reproduced": (
                        semantic_exact.total_return == semantic_delivery.total_return
                    ),
                    "branch_delivered_after_start": {
                        name: branch_delivered(branch)
                        for name, branch in branches.items()
                    },
                    "branches": {
                        name: branch_summary(branch) for name, branch in branches.items()
                    },
                }
            )

    result = {
        "status": specification["status"],
        "claim_scope": specification["claim_scope"],
        "protocol": {
            "exposure": "The frozen ego holds completed soup at initial step 0.",
            "causal_trigger": (
                "The initially held soup is factually dropped; branching begins "
                "from the exact snapshot immediately before that action."
            ),
            "exposure_uses_future_information": False,
            "causal_trigger_uses_factual_future_information": True,
            "reference_option_budget": reference_budget,
            "delivery_option_budget": delivery_budget,
            "minimum_remaining_steps": minimum_remaining_steps,
            "require_zero_factual_return": require_zero_factual_return,
            "primary_comparison": specification["primary_comparison"],
            "negative_control": specification["negative_control"],
            "environment_condition": specification["environment_condition"],
            "max_steps": max_steps,
        },
        "provenance": {
            "source_results": str(source_results),
            "source_implementation_reference": source_summary.get(
                "implementation_reference"
            ),
            "fixed_checkpoint_sha256": directory_digest(fixed_checkpoint),
            "saved_policy_sha256": policy_digests,
            "checkpoint_dataset_revision": checkpoint_index.get("revision"),
            "analysis_script_sha256": file_digest(Path(__file__).resolve()),
            "config_sha256": file_digest(args.config),
        },
        "pairing_exposures": exposures,
        "cases": rows,
        "summary": {
            "evaluated_pairings": evaluated_pairings,
            "initial_completed_soup_prevalence": wilson_interval(
                sum(exposure["initial_completed_soup"] for exposure in exposures),
                evaluated_pairings,
            ),
            **summarize_rows(rows, exposures),
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
