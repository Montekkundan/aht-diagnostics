"""Environment conditions and return summaries for teammate evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .teamgen import mean_and_bootstrap_ci


CONDITIONS = {
    "standard_sparse": {
        "random_obj_state": False,
        "do_reward_shaping": False,
    },
    "randomized_sparse": {
        "random_obj_state": True,
        "do_reward_shaping": False,
    },
    "training_shaped": {
        "random_obj_state": True,
        "do_reward_shaping": True,
    },
}


def condition_kwargs(base: dict[str, Any], condition: str, max_steps: int) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    kwargs = dict(base)
    kwargs.update(CONDITIONS[condition])
    kwargs["max_steps"] = max_steps
    kwargs["random_reset"] = False
    if not kwargs["do_reward_shaping"]:
        kwargs.pop("reward_shaping_params", None)
    return kwargs


def summarize_returns(values: list[float], seed: int, *, sparse: bool) -> dict[str, Any]:
    result = dict(mean_and_bootstrap_ci(values, seed=seed))
    if sparse:
        completed = np.asarray(values, dtype=float) > 0.0
        result["completion_rate"] = float(completed.mean())
        result["completed_episodes"] = int(completed.sum())
    return result
