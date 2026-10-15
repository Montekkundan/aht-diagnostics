"""Summary statistics for fixed-ego teammate-generation experiments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .rollout import Episode


CANONICAL_ROTATE_OBJECTIVE = "gae_ps_regret_ts_weighted"
TRAJECTORY_REGRET_BASELINE = "traj_regret"
OFFICIAL_ROTATE_REPOSITORY = "https://github.com/LARG/jax-aht"

# Current JaxAHT expts-branch Cramped Room settings. The runner reads the
# selected checkout directly; this table is used for profile-sized runs.
# With 64 environments this budget gives each teammate and best response only
# 50 PPO updates, too few for competent Cramped Room teammates.
OFFICIAL_CRAMPED_ROOM_CONFIG = {
    "TIMESTEPS_PER_ITER_PARTNER": 5_220_000,
    "NUM_ENVS": 64,
    "NUM_CHECKPOINTS": 2,
    "LR": 1e-4,
    "UPDATE_EPOCHS": 20,
    "NUM_MINIBATCHES": 8,
    "CLIP_EPS": 0.3,
    "ENT_COEF": 0.03,
    "LAMBDA_1": 10.0,
    "LAMBDA_2": 1.0,
    "ANNEAL_LR": True,
}


# Teammate-generation settings reported for Cramped Room in the ROTATE
# manuscript (Table 14): 16 environments and 6M steps give 234 PPO updates.
# NUM_MINIBATCHES is not reported there and follows the JaxAHT base config.
ROTATE_PAPER_CRAMPED_ROOM_CONFIG = {
    "TIMESTEPS_PER_ITER_PARTNER": 6_000_000,
    "NUM_ENVS": 16,
    "LR": 1e-4,
    "UPDATE_EPOCHS": 20,
    "NUM_MINIBATCHES": 4,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "LAMBDA_1": 4.0,
    "LAMBDA_2": 4.0,
    "ANNEAL_LR": True,
}

TRAINING_PRESETS = {
    "jaxaht-expts": {},
    "rotate-paper": ROTATE_PAPER_CRAMPED_ROOM_CONFIG,
}


def partner_num_updates(config: dict[str, object]) -> int:
    """PPO updates JaxAHT's ROTATE gives one teammate and its best response."""

    per_update = (
        int(config["ROLLOUT_LENGTH"])
        * 4
        * int(config["NUM_ENVS"])
        * int(config.get("PARTNER_POP_SIZE", 1))
    )
    return int(config["TIMESTEPS_PER_ITER_PARTNER"]) // per_update


def evidence_status(mode: str, *, training_timesteps_overridden: bool) -> str:
    """Label evidence without treating one historical step count as universal."""

    if mode in {"smoke", "profile"}:
        return "technical_check_only"
    if mode != "preliminary":
        raise ValueError(f"Unknown mode: {mode}")
    return (
        "prespecified_budget_preliminary_experiment"
        if training_timesteps_overridden
        else "released_budget_preliminary_experiment"
    )


def training_overrides(mode: str, objective: str, rollout_length: int) -> dict[str, object]:
    objective_names = {
        "per_state_regret": CANONICAL_ROTATE_OBJECTIVE,
        "per_trajectory_regret": TRAJECTORY_REGRET_BASELINE,
    }
    if objective not in objective_names:
        raise ValueError(f"Unknown objective: {objective}")
    if mode not in {"smoke", "profile", "preliminary"}:
        raise ValueError(f"Unknown mode: {mode}")

    overrides: dict[str, object] = {"CONF_OBJ_TYPE": objective_names[objective]}
    if mode == "smoke":
        overrides.update(
            {
                "NUM_ENVS": 2,
                "NUM_CHECKPOINTS": 1,
                "NUM_MINIBATCHES": 1,
                "UPDATE_EPOCHS": 1,
                "TIMESTEPS_PER_ITER_PARTNER": 800,
            }
        )
    elif mode == "profile":
        overrides["TIMESTEPS_PER_ITER_PARTNER"] = (
            rollout_length * 4 * int(OFFICIAL_CRAMPED_ROOM_CONFIG["NUM_ENVS"])
        )
    return overrides


def mean_and_bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("At least one value is required")
    if data.size == 1:
        value = float(data[0])
        return {"mean": value, "ci_low": value, "ci_high": value, "n": 1}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(samples, data.size))
    bootstrap_means = data[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "mean": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(data.size),
    }


def semantic_event_counts(
    episodes: Iterable[Episode],
    *,
    agent: str,
) -> dict[str, int]:
    if agent not in {"generated_teammate", "partner"}:
        raise ValueError("agent must be generated_teammate or partner")
    attribute = "ego_event" if agent == "generated_teammate" else "partner_event"
    counts: Counter[str] = Counter()
    for episode in episodes:
        counts.update(getattr(step, attribute) for step in episode.steps)
    return dict(sorted(counts.items()))


def summarize_pairing(
    episodes: Sequence[Episode],
    *,
    seed: int,
    generated_teammate_slot: str = "agent_0",
) -> dict[str, object]:
    if generated_teammate_slot not in {"agent_0", "agent_1"}:
        raise ValueError("generated_teammate_slot must be agent_0 or agent_1")
    partner_slot = "agent_1" if generated_teammate_slot == "agent_0" else "agent_0"
    returns = [episode.total_return for episode in episodes]
    return {
        "return": mean_and_bootstrap_ci(returns, seed=seed),
        "generated_teammate_events": semantic_event_counts(
            episodes,
            agent=(
                "generated_teammate"
                if generated_teammate_slot == "agent_0"
                else "partner"
            ),
        ),
        "partner_events": semantic_event_counts(
            episodes,
            agent="generated_teammate" if partner_slot == "agent_0" else "partner",
        ),
    }
