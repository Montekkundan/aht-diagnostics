#!/usr/bin/env python3
"""Compare soup-carrier swap locks for a fixed ego and learned references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.evaluation_conditions import condition_kwargs  # noqa: E402
from aht_diagnostics.jaxaht_bridge import activate_jaxaht, load_frozen_s5_ego  # noqa: E402
from aht_diagnostics.natural_failure import (  # noqa: E402
    clustered_paired_bootstrap_interval,
    first_soup_carrier_swap_lock_opportunity,
    wilson_interval,
)
from aht_diagnostics.rollout import rollout_episode  # noqa: E402


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


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(
    rows: list[dict[str, Any]],
    competence_threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No paired control rows were produced")

    fixed_only = sum(row["fixed_trigger"] and not row["reference_trigger"] for row in rows)
    reference_only = sum(row["reference_trigger"] and not row["fixed_trigger"] for row in rows)
    both = sum(row["fixed_trigger"] and row["reference_trigger"] for row in rows)
    neither = len(rows) - fixed_only - reference_only - both
    fixed_count = fixed_only + both
    reference_count = reference_only + both
    paired_trigger_differences = [
        float(row["fixed_trigger"]) - float(row["reference_trigger"])
        for row in rows
    ]
    competent_reference_rows = [
        row for row in rows if row["reference_return"] >= competence_threshold
    ]
    fixed_trigger_rows = [row for row in rows if row["fixed_trigger"]]
    confirmed_fixed_with_competent_reference = sum(
        row["reference_return"] >= competence_threshold for row in fixed_trigger_rows
    )

    return {
        "pairings": len(rows),
        "fixed_trigger_prevalence": wilson_interval(fixed_count, len(rows)),
        "reference_trigger_prevalence": wilson_interval(reference_count, len(rows)),
        "paired_trigger_table": {
            "fixed_only": fixed_only,
            "reference_only": reference_only,
            "both": both,
            "neither": neither,
        },
        "fixed_minus_reference_trigger_prevalence": {
            "mean": sum(paired_trigger_differences) / len(paired_trigger_differences),
            "environment_seed_cluster_interval": clustered_paired_bootstrap_interval(
                paired_trigger_differences,
                [int(row["episode_seed"]) for row in rows],
                seed=61,
            ),
            "policy_cluster_interval": clustered_paired_bootstrap_interval(
                paired_trigger_differences,
                [int(row["train_seed"]) for row in rows],
                seed=67,
            ),
        },
        "competence_threshold": competence_threshold,
        "competent_reference_pairings": len(competent_reference_rows),
        "reference_trigger_prevalence_among_competent_pairings": (
            wilson_interval(
                sum(row["reference_trigger"] for row in competent_reference_rows),
                len(competent_reference_rows),
            )
            if competent_reference_rows
            else None
        ),
        "fixed_trigger_pairings_with_competent_reference": (
            confirmed_fixed_with_competent_reference
        ),
        "fixed_trigger_pairings": len(fixed_trigger_rows),
        "mean_fixed_return": sum(row["fixed_return"] for row in rows) / len(rows),
        "mean_reference_return": (
            sum(row["reference_return"] for row in rows) / len(rows)
        ),
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
    source_confirmation = REPO_ROOT / specification["source_confirmation"]
    confirmation = json.loads(source_confirmation.read_text(encoding="utf-8"))
    expected_fixed_triggers = {
        (int(row["train_seed"]), int(row["episode_seed"]))
        for row in confirmation["cases"]
    }

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

    fixed_checkpoint = (
        args.checkpoint_root / checkpoint_index["fixed_ego"]["checkpoint"]
    )
    fixed_policy, fixed_params = load_frozen_s5_ego(
        args.jaxaht,
        fixed_checkpoint,
        env,
        seed=0,
    )
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    teammate_policy = ActorWithDoubleCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    reference_policy = MLPActorCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)

    seed_range = specification["episode_seed_range"]
    episode_seeds = range(
        int(seed_range["start"]),
        int(seed_range["end_exclusive"]),
    )
    minimum_lock_steps = int(specification["minimum_lock_steps"])
    source_results = REPO_ROOT / specification["source_results"]
    source_summary = json.loads(
        (source_results / "summary.json").read_text(encoding="utf-8")
    )

    rows: list[dict[str, Any]] = []
    for seed_result in source_summary["seed_results"]:
        train_seed = int(seed_result["train_seed"])
        saved_path = source_results / f"seed_{train_seed}" / "trained_policies"
        saved = load_train_run(saved_path)
        teammate_params = saved["generated_teammate_params"]
        reference_params = saved["best_response_params"]

        for episode_seed in episode_seeds:
            fixed_episode = rollout_episode(
                env,
                fixed_policy,
                fixed_params,
                teammate_policy,
                teammate_params,
                seed=episode_seed,
                max_steps=max_steps,
            )
            reference_episode = rollout_episode(
                env,
                reference_policy,
                reference_params,
                teammate_policy,
                teammate_params,
                seed=episode_seed,
                max_steps=max_steps,
            )
            fixed_opportunity = first_soup_carrier_swap_lock_opportunity(
                fixed_episode,
                minimum_consecutive_steps=minimum_lock_steps,
            )
            reference_opportunity = first_soup_carrier_swap_lock_opportunity(
                reference_episode,
                minimum_consecutive_steps=minimum_lock_steps,
            )
            rows.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "fixed_return": fixed_episode.total_return,
                    "reference_return": reference_episode.total_return,
                    "fixed_trigger": fixed_opportunity is not None,
                    "reference_trigger": reference_opportunity is not None,
                    "fixed_trigger_start": (
                        fixed_opportunity.step if fixed_opportunity else None
                    ),
                    "reference_trigger_start": (
                        reference_opportunity.step if reference_opportunity else None
                    ),
                }
            )

    actual_fixed_triggers = {
        (int(row["train_seed"]), int(row["episode_seed"]))
        for row in rows
        if row["fixed_trigger"]
    }
    result = {
        "status": specification["status"],
        "claim_scope": specification["claim_scope"],
        "protocol": {
            "environment_condition": specification["environment_condition"],
            "max_steps": max_steps,
            "minimum_lock_steps": minimum_lock_steps,
            "episode_seed_range": specification["episode_seed_range"],
            "competence_return_threshold": specification[
                "competence_return_threshold"
            ],
        },
        "provenance": {
            "config_sha256": file_digest(args.config),
            "analysis_script_sha256": file_digest(Path(__file__).resolve()),
            "source_confirmation": str(source_confirmation),
            "source_results": str(source_results),
            "source_implementation_reference": confirmation["provenance"][
                "source_implementation_reference"
            ],
        },
        "rows": rows,
        "summary": {
            **summarize_rows(
                rows,
                float(specification["competence_return_threshold"]),
            ),
            "expected_fixed_trigger_count": len(expected_fixed_triggers),
            "actual_fixed_trigger_count": len(actual_fixed_triggers),
            "fixed_trigger_set_matches_confirmation": (
                actual_fixed_triggers == expected_fixed_triggers
            ),
            "missing_fixed_triggers": sorted(
                [list(item) for item in expected_fixed_triggers - actual_fixed_triggers]
            ),
            "unexpected_fixed_triggers": sorted(
                [list(item) for item in actual_fixed_triggers - expected_fixed_triggers]
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
