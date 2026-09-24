#!/usr/bin/env python3
"""Regenerate detailed traces for selected confirmed swap-lock cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.evaluation_conditions import condition_kwargs  # noqa: E402
from aht_diagnostics.interventions import ActionSequencePolicy  # noqa: E402
from aht_diagnostics.jaxaht_bridge import activate_jaxaht, load_frozen_s5_ego  # noqa: E402
from aht_diagnostics.natural_failure import (  # noqa: E402
    branch_summary,
    first_soup_carrier_swap_lock_opportunity,
)
from aht_diagnostics.rollout import (  # noqa: E402
    ContinuationResult,
    Episode,
    policy_states_on_factual_trajectory,
    rollout_continuation,
    rollout_episode,
    rollout_policy_option,
)


BRANCH_ACTIONS = {
    "detour_down_left_2": (1, 3),
    "control_left_down_2": (3, 1),
    "control_down_stay_2": (1, 4),
    "control_wait_2": (4, 4),
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


def record_dict(record: Any) -> dict[str, Any]:
    return {
        "step": int(record.step),
        "ego_action": int(record.ego_action),
        "partner_action": int(record.partner_action),
        "ego_position": list(record.ego_position),
        "partner_position": list(record.partner_position),
        "ego_inventory": record.ego_inventory,
        "partner_inventory": record.partner_inventory,
        "ego_event": record.ego_event,
        "partner_event": record.partner_event,
        "reward": float(record.reward),
        "cumulative_return": float(record.cumulative_return),
    }


def relevant_branch_records(result: ContinuationResult) -> list[dict[str, Any]]:
    records = result.intervention_records + result.continuation_records
    first_reward_index = next(
        (index for index, record in enumerate(records) if record.reward > 0),
        None,
    )
    keep_until = min(len(records), 20)
    if first_reward_index is not None:
        keep_until = max(keep_until, min(len(records), first_reward_index + 2))
    selected = list(records[:keep_until])
    if first_reward_index is not None and first_reward_index >= keep_until:
        selected.append(records[first_reward_index])
    return [record_dict(record) for record in selected]


def action_sequence_branch(
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


def full_reference_branch(
    env: Any,
    snapshot: Any,
    ego_policy: Any,
    ego_params: Any,
    teammate_policy: Any,
    teammate_params: Any,
    reference_policy: Any,
    reference_params: Any,
    reference_hstate: Any,
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
        max_steps - snapshot.step,
        max_steps,
    )


def factual_context(
    episode: Episode,
    start_step: int,
    intervention_step: int,
) -> list[dict[str, Any]]:
    context_start = max(0, start_step - 2)
    context_end = min(len(episode.steps), intervention_step + 3)
    return [
        record_dict(record)
        for record in episode.steps[context_start:context_end]
    ]


def main() -> int:
    args = parse_args()
    args.jaxaht = args.jaxaht.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.checkpoint_index = args.checkpoint_index.resolve()

    specification = json.loads(args.config.read_text(encoding="utf-8"))
    confirmation_path = REPO_ROOT / specification["source_confirmation"]
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    confirmation_cases = {
        (int(row["train_seed"]), int(row["episode_seed"])): row
        for row in confirmation["cases"]
    }
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
    env = LogWrapper(
        make_env(
            task.ENV_NAME,
            condition_kwargs(
                base_kwargs,
                specification["environment_condition"],
                max_steps,
            ),
        )
    )

    fixed_checkpoint = (
        args.checkpoint_root / checkpoint_index["fixed_ego"]["checkpoint"]
    )
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

    cached_params: dict[int, tuple[Any, Any]] = {}
    rows = []
    for selected in specification["cases"]:
        train_seed = int(selected["train_seed"])
        episode_seed = int(selected["episode_seed"])
        key = (train_seed, episode_seed)
        if key not in confirmation_cases:
            raise ValueError(f"Selected case {key} is absent from confirmation")
        if train_seed not in cached_params:
            saved = load_train_run(
                source_results / f"seed_{train_seed}" / "trained_policies"
            )
            cached_params[train_seed] = (
                saved["generated_teammate_params"],
                saved["best_response_params"],
            )
        teammate_params, reference_params = cached_params[train_seed]
        factual = rollout_episode(
            env,
            ego_policy,
            ego_params,
            teammate_policy,
            teammate_params,
            seed=episode_seed,
            max_steps=max_steps,
        )
        opportunity = first_soup_carrier_swap_lock_opportunity(
            factual,
            minimum_consecutive_steps=int(specification["minimum_lock_steps"]),
        )
        if opportunity is None:
            raise ValueError(f"Selected case {key} no longer triggers")
        intervention_step = opportunity.confirmed_at_step + 1
        snapshot = factual.snapshots[intervention_step]
        reference_states = policy_states_on_factual_trajectory(
            env,
            factual,
            reference_policy,
            reference_params,
        )
        branches = {
            "no_change": rollout_continuation(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                max_steps,
            ),
            **{
                name: action_sequence_branch(
                    env,
                    snapshot,
                    ego_policy,
                    ego_params,
                    teammate_policy,
                    teammate_params,
                    actions,
                    max_steps,
                )
                for name, actions in BRANCH_ACTIONS.items()
            },
            "full_reference": full_reference_branch(
                env,
                snapshot,
                ego_policy,
                ego_params,
                teammate_policy,
                teammate_params,
                reference_policy,
                reference_params,
                reference_states[intervention_step],
                max_steps,
            ),
        }
        expected = confirmation_cases[key]["branches"]
        branch_rows = {
            name: {
                "summary": branch_summary(result),
                "trace": relevant_branch_records(result),
                "remaining_return_matches_confirmation": (
                    float(result.remaining_return)
                    == float(expected[name]["remaining_return"])
                ),
            }
            for name, result in branches.items()
        }
        rows.append(
            {
                "train_seed": train_seed,
                "episode_seed": episode_seed,
                "selection_reason": selected["selection_reason"],
                "trigger": {
                    "start_step": opportunity.step,
                    "confirmed_at_step": opportunity.confirmed_at_step,
                    "intervention_step": intervention_step,
                },
                "factual_return": factual.total_return,
                "factual_context": factual_context(
                    factual,
                    opportunity.step,
                    intervention_step,
                ),
                "branches": branch_rows,
                "all_branch_returns_match_confirmation": all(
                    row["remaining_return_matches_confirmation"]
                    for row in branch_rows.values()
                ),
            }
        )

    result = {
        "status": specification["status"],
        "claim_scope": specification["claim_scope"],
        "source_confirmation": str(confirmation_path),
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "all_branch_returns_match_confirmation": all(
                row["all_branch_returns_match_confirmation"] for row in rows
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
