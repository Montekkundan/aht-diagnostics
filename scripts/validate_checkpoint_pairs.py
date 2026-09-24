#!/usr/bin/env python3
"""Validate saved Overcooked best-response checkpoints against their training partners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.jaxaht_bridge import (  # noqa: E402
    activate_jaxaht,
    load_frozen_s5_ego,
    make_overcooked_partner,
)
from aht_diagnostics.teamgen import mean_and_bootstrap_ci  # noqa: E402


PARTNER_SPECS = {
    "independent_agent_0": {
        "role": "independent",
        "p_onion_on_counter": 0.0,
        "p_plate_on_counter": 0.0,
    },
    "independent_agent_0.4": {
        "role": "independent",
        "p_onion_on_counter": 0.4,
        "p_plate_on_counter": 0.4,
    },
    "onion_agent_0.1": {
        "role": "onion",
        "p_onion_on_counter": 0.1,
        "p_plate_on_counter": 0.0,
    },
    "plate_agent_0.1": {
        "role": "plate",
        "p_onion_on_counter": 0.0,
        "p_plate_on_counter": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "checkpoint_pair_validation.json",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    return parser.parse_args()


def make_environment(jaxaht: Path, max_steps: int, *, training_conditions: bool) -> Any:
    activate_jaxaht(jaxaht)
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    return LogWrapper(
        make_env(
            "overcooked-v1",
            {
                "layout": "cramped_room",
                "max_steps": max_steps,
                "random_reset": False,
                "random_obj_state": training_conditions,
                "do_reward_shaping": training_conditions,
            },
        )
    )


def evaluate_returns(
    env: Any,
    agent_0_policy: Any,
    agent_0_params: Any,
    agent_1_policy: Any,
    agent_1_params: Any,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> list[float]:
    from common.run_episodes import run_episodes

    metrics = run_episodes(
        jax.random.PRNGKey(seed),
        env,
        agent_0_params,
        agent_0_policy,
        agent_1_params,
        agent_1_policy,
        max_steps,
        episodes,
        agent_0_test_mode=True,
        agent_1_test_mode=True,
    )
    returns = np.asarray(metrics["returned_episode_returns"])
    return returns[:, 0].astype(float).tolist()


def main() -> int:
    args = parse_args()
    args.jaxaht = args.jaxaht.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.checkpoint_index = args.checkpoint_index.resolve()
    args.output = args.output.resolve()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    index = json.loads(args.checkpoint_index.read_text(encoding="utf-8"))
    checkpoints = index["available_best_responses"]
    missing = set(checkpoints) - set(PARTNER_SPECS)
    if missing:
        raise ValueError(f"Missing partner specifications: {sorted(missing)}")

    results: dict[str, object] = {}
    for condition_name, training_conditions in (
        ("diagnostic_sparse_reward", False),
        ("training_shaped_reward", True),
    ):
        env = make_environment(args.jaxaht, args.max_steps, training_conditions=training_conditions)
        condition_results: dict[str, object] = {}
        for checkpoint_index, (name, relative_path) in enumerate(checkpoints.items()):
            spec = PARTNER_SPECS[name]
            partner = make_overcooked_partner(
                args.jaxaht,
                spec["role"],
                p_onion_on_counter=spec["p_onion_on_counter"],
                p_plate_on_counter=spec["p_plate_on_counter"],
                using_log_wrapper=True,
            )
            learned, params = load_frozen_s5_ego(
                args.jaxaht,
                args.checkpoint_root / relative_path,
                env,
                seed=checkpoint_index,
            )
            base_seed = 10_000 * checkpoint_index
            learned_as_agent_0 = evaluate_returns(
                env,
                learned,
                params,
                partner,
                None,
                seed=base_seed,
                episodes=args.episodes,
                max_steps=args.max_steps,
            )
            learned_as_agent_1 = evaluate_returns(
                env,
                partner,
                None,
                learned,
                params,
                seed=base_seed,
                episodes=args.episodes,
                max_steps=args.max_steps,
            )
            condition_results[name] = {
                "partner": spec,
                "checkpoint": relative_path,
                "learned_as_agent_0": mean_and_bootstrap_ci(
                    learned_as_agent_0,
                    seed=base_seed + 1,
                ),
                "learned_as_agent_1": mean_and_bootstrap_ci(
                    learned_as_agent_1,
                    seed=base_seed + 2,
                ),
            }
        results[condition_name] = condition_results

    output = {
        "purpose": (
            "Check that each saved learned best response succeeds with the heuristic "
            "partner used to train it, and whether agent-slot order changes the result."
        ),
        "episodes_per_pair": args.episodes,
        "max_steps": args.max_steps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
