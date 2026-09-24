#!/usr/bin/env python3
"""Train only the ROTATE teammate generator against one fixed Overcooked ego."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.jaxaht_bridge import (  # noqa: E402
    activate_jaxaht,
    load_frozen_s5_ego,
)
from aht_diagnostics.behavior_penalty import OvercookedBehaviorPenalty  # noqa: E402
from aht_diagnostics.rollout import rollout_episode  # noqa: E402
from aht_diagnostics.teamgen import (  # noqa: E402
    CANONICAL_ROTATE_OBJECTIVE,
    OFFICIAL_ROTATE_REPOSITORY,
    TRAJECTORY_REGRET_BASELINE,
    evidence_status,
    mean_and_bootstrap_ci,
    summarize_pairing,
    training_overrides,
)


OBJECTIVES = {
    "per_state_regret": CANONICAL_ROTATE_OBJECTIVE,
    "per_trajectory_regret": TRAJECTORY_REGRET_BASELINE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=REPO_ROOT / "data" / "fixed_ego_experiment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "fixed_ego_teamgen",
    )
    parser.add_argument(
        "--objective",
        choices=tuple(OBJECTIVES),
        default="per_state_regret",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "profile", "preliminary"),
        default="smoke",
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[20374])
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument(
        "--training-timesteps",
        type=int,
        default=None,
        help="Override the released budget with an explicitly prespecified budget.",
    )
    parser.add_argument("--videos", action="store_true")
    parser.add_argument(
        "--behavior-penalty",
        type=float,
        default=0.0,
        help=(
            "Training-time penalty applied to each generated-teammate wait, "
            "blocked move, or ineffective interaction."
        ),
    )
    return parser.parse_args()


def load_training_config(jaxaht: Path, mode: str, objective: str) -> dict[str, Any]:
    activate_jaxaht(jaxaht)
    from omegaconf import OmegaConf

    config_root = jaxaht / "open_ended_training" / "configs"
    base = OmegaConf.load(config_root / "algorithm" / "rotate" / "_base_.yaml")
    task_algorithm = OmegaConf.load(
        config_root / "algorithm" / "rotate" / "overcooked-v1" / "cramped_room.yaml"
    )
    task = OmegaConf.load(config_root / "task" / "overcooked-v1" / "cramped_room.yaml")
    merged = OmegaConf.merge(base, task_algorithm)
    config = OmegaConf.to_container(merged, resolve=True)
    config.pop("defaults", None)
    config.update(
        {
            "ENV_NAME": task.ENV_NAME,
            "ENV_KWARGS": OmegaConf.to_container(task.ENV_KWARGS, resolve=True),
            "ROLLOUT_LENGTH": int(task.ROLLOUT_LENGTH),
            "NUM_EVAL_EPISODES": 20,
            "CONF_OBJ_TYPE": OBJECTIVES[objective],
        }
    )

    config.update(training_overrides(mode, objective, int(config["ROLLOUT_LENGTH"])))
    config["TIMESTEPS_PER_ITER_PARTNER"] = int(
        config["TIMESTEPS_PER_ITER_PARTNER"]
    )

    if mode == "smoke":
        config.update(
            {
                "ROLLOUT_LENGTH": 100,
                "NUM_EVAL_EPISODES": 2,
            }
        )
        config["ENV_KWARGS"].update(
            {"max_steps": 100, "random_obj_state": False, "random_reset": False}
        )
    return config


def git_revision(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def select_population_member(params: Any, index: int = 0) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], params)


def train_one_seed(
    *,
    jaxaht: Path,
    checkpoint: Path,
    output: Path,
    objective: str,
    mode: str,
    training_timesteps: int | None,
    train_seed: int,
    evaluation_episodes: int,
    save_videos: bool,
    behavior_penalty_coefficient: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    activate_jaxaht(jaxaht)
    from agents.mlp_actor_critic_agent import (
        ActorWithDoubleCriticPolicy,
        MLPActorCriticPolicy,
    )
    from common.save_load_utils import save_train_run
    from envs import make_env
    from envs.log_wrapper import LogWrapper
    from evaluation.vis_episodes import save_video
    from open_ended_training.rotate import train_regret_maximizing_partners

    config = load_training_config(jaxaht, mode, objective)
    if training_timesteps is not None:
        update_size = int(config["ROLLOUT_LENGTH"]) * 4 * int(config["NUM_ENVS"])
        if training_timesteps < update_size or training_timesteps % update_size:
            raise ValueError(
                f"training timesteps must be a positive multiple of {update_size}"
            )
        config["TIMESTEPS_PER_ITER_PARTNER"] = training_timesteps
    raw_training_env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    if behavior_penalty_coefficient > 0:
        raw_training_env = OvercookedBehaviorPenalty(
            raw_training_env,
            coefficient=behavior_penalty_coefficient,
        )
    training_env = LogWrapper(raw_training_env)
    fixed_ego_policy, fixed_ego_params = load_frozen_s5_ego(
        jaxaht,
        checkpoint,
        training_env,
        seed=train_seed,
    )

    action_dim = training_env.action_space(training_env.agents[0]).n
    obs_dim = training_env.observation_space(training_env.agents[0]).shape[0]
    teammate_policy = ActorWithDoubleCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    best_response_policy = MLPActorCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)

    rng = jax.random.PRNGKey(train_seed)
    rng, teammate_rng, br_rng, training_rng = jax.random.split(rng, 4)
    teammate_params = jax.tree.map(
        lambda leaf: leaf[None, ...],
        teammate_policy.init_params(teammate_rng),
    )
    best_response_params = jax.tree.map(
        lambda leaf: leaf[None, ...],
        best_response_policy.init_params(br_rng),
    )

    training_output = train_regret_maximizing_partners(
        copy.deepcopy(config),
        training_env,
        fixed_ego_params,
        fixed_ego_policy,
        teammate_params,
        teammate_policy,
        best_response_params,
        best_response_policy,
        training_rng,
    )
    learned_teammate = select_population_member(training_output["final_params_conf"])
    learned_best_response = select_population_member(training_output["final_params_br"])

    seed_output = output / f"seed_{train_seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    save_train_run(
        {
            "generated_teammate_params": learned_teammate,
            "best_response_params": learned_best_response,
        },
        str(seed_output),
        savename="trained_policies",
    )

    max_steps = 100 if mode == "smoke" else 400
    evaluation_env = make_env(
        config["ENV_NAME"],
        {**config["ENV_KWARGS"], "max_steps": max_steps},
    )
    episode_seeds = [train_seed * 1000 + index for index in range(evaluation_episodes)]
    with_fixed_ego = [
        rollout_episode(
            evaluation_env,
            fixed_ego_policy,
            fixed_ego_params,
            teammate_policy,
            learned_teammate,
            seed,
            max_steps,
        )
        for seed in episode_seeds
    ]
    with_best_response = [
        rollout_episode(
            evaluation_env,
            best_response_policy,
            learned_best_response,
            teammate_policy,
            learned_teammate,
            seed,
            max_steps,
        )
        for seed in episode_seeds
    ]

    video_paths: list[str] = []
    video_error = None
    if save_videos:
        try:
            video_env = make_env(
                "overcooked-v1",
                {
                    "layout": "cramped_room",
                    "max_steps": max_steps,
                    "random_reset": False,
                    "random_obj_state": False,
                    "do_reward_shaping": False,
                },
            )
            video_paths.append(
                save_video(
                    video_env,
                    "overcooked-v1",
                    fixed_ego_params,
                    fixed_ego_policy,
                    learned_teammate,
                    teammate_policy,
                    max_steps,
                    1,
                    True,
                    str(seed_output / "videos"),
                    "generated_teammate_with_fixed_ego",
                )
            )
            video_paths.append(
                save_video(
                    video_env,
                    "overcooked-v1",
                    learned_best_response,
                    best_response_policy,
                    learned_teammate,
                    teammate_policy,
                    max_steps,
                    1,
                    True,
                    str(seed_output / "videos"),
                    "generated_teammate_with_best_response",
                )
            )
        except Exception as exc:  # video generation should not erase training results
            video_error = f"{type(exc).__name__}: {exc}"

    fixed_summary = summarize_pairing(
        with_fixed_ego,
        seed=train_seed,
        generated_teammate_slot="agent_1",
    )
    br_summary = summarize_pairing(
        with_best_response,
        seed=train_seed + 1,
        generated_teammate_slot="agent_1",
    )
    regret = (
        br_summary["return"]["mean"] - fixed_summary["return"]["mean"]
    )
    return {
        "train_seed": train_seed,
        "objective": objective,
        "return_with_fixed_ego": fixed_summary,
        "return_with_best_response": br_summary,
        "cooperative_regret": float(regret),
        "videos": video_paths,
        "video_error": video_error,
        "training_seconds": float(time.perf_counter() - started_at),
        "behavior_penalty_per_event": behavior_penalty_coefficient,
        "effective_training_config": {
            key: config[key]
            for key in (
                "CONF_OBJ_TYPE",
                "TIMESTEPS_PER_ITER_PARTNER",
                "NUM_ENVS",
                "ROLLOUT_LENGTH",
                "NUM_CHECKPOINTS",
                "LR",
                "UPDATE_EPOCHS",
                "NUM_MINIBATCHES",
                "CLIP_EPS",
                "ENT_COEF",
                "LAMBDA_1",
                "LAMBDA_2",
                "ANNEAL_LR",
            )
        },
    }


def main() -> int:
    args = parse_args()
    if args.behavior_penalty < 0:
        raise ValueError("behavior penalty must be nonnegative")
    args.output = args.output.resolve()
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint_root / experiment["fixed_ego"]["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing fixed ego checkpoint: {checkpoint}")

    evaluation_episodes = args.evaluation_episodes
    if evaluation_episodes is None:
        evaluation_episodes = 2 if args.mode == "smoke" else 20
    args.output.mkdir(parents=True, exist_ok=True)

    results = [
        train_one_seed(
            jaxaht=args.jaxaht.resolve(),
            checkpoint=checkpoint.resolve(),
            output=args.output,
            objective=args.objective,
            mode=args.mode,
            training_timesteps=args.training_timesteps,
            train_seed=train_seed,
            evaluation_episodes=evaluation_episodes,
            save_videos=args.videos,
            behavior_penalty_coefficient=args.behavior_penalty,
        )
        for train_seed in args.train_seeds
    ]
    fixed_returns = [row["return_with_fixed_ego"]["return"]["mean"] for row in results]
    br_returns = [row["return_with_best_response"]["return"]["mean"] for row in results]
    regrets = [row["cooperative_regret"] for row in results]
    summary = {
        "research_question": experiment["research_question"],
        "hypothesis": experiment["hypothesis"],
        "objective": args.objective,
        "objective_config_key": results[0]["effective_training_config"]["CONF_OBJ_TYPE"],
        "mode": args.mode,
        "evidence_status": evidence_status(
            args.mode,
            training_timesteps_overridden=args.training_timesteps is not None,
        ),
        "fixed_ego": experiment["fixed_ego"],
        "implementation_reference": {
            "repository": OFFICIAL_ROTATE_REPOSITORY,
            "local_jaxaht": str(args.jaxaht.resolve()),
            "local_head": git_revision(args.jaxaht.resolve(), "rev-parse", "HEAD"),
            "rotate_source_commit": git_revision(
                args.jaxaht.resolve(),
                "log",
                "-1",
                "--format=%H",
                "--",
                "open_ended_training/rotate.py",
            ),
            "working_tree_clean": not bool(
                git_revision(args.jaxaht.resolve(), "status", "--porcelain")
            ),
        },
        "effective_training_config": results[0]["effective_training_config"],
        "training_seeds": args.train_seeds,
        "evaluation_episodes_per_seed": evaluation_episodes,
        "behavior_penalty": {
            "coefficient_per_event": args.behavior_penalty,
            "events": ["wait", "blocked_move", "interact_no_effect"],
            "applied_during": "generated_teammate_training_only",
        },
        "return_with_fixed_ego": mean_and_bootstrap_ci(fixed_returns, seed=11),
        "return_with_best_response": mean_and_bootstrap_ci(br_returns, seed=12),
        "cooperative_regret": mean_and_bootstrap_ci(regrets, seed=13),
        "seed_results": results,
        "qualitative_analysis": {
            "question": (
                "Does the generated teammate follow a coherent cooperative convention that "
                "the fixed ego cannot accommodate, rather than blocking, idling, or otherwise sabotaging?"
            ),
            "status": (
                "This short run only checks that training and evaluation execute. It does not test the hypothesis."
                if args.mode == "smoke"
                else "Interpret the returns together with the videos and semantic event counts."
            ),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
