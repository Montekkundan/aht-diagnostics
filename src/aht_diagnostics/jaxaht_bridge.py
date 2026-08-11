"""Thin adapters around a pinned JaxAHT checkout."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax


S5_CONFIG = {
    "actor_type": "s5",
    "ckpt_key": "final_params",
    "idx_list": [0],
    "test_mode": True,
    "S5_D_MODEL": 128,
    "S5_SSM_SIZE": 128,
    "S5_ACTOR_CRITIC_HIDDEN_DIM": 1024,
    "FC_N_LAYERS": 3,
}


def activate_jaxaht(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / "common" / "agent_loader_from_config.py").exists():
        raise FileNotFoundError(f"Not a JaxAHT checkout: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def make_overcooked_env(
    jaxaht_path: str | Path,
    max_steps: int,
    layout: str = "cramped_room",
    random_reset: bool = True,
) -> Any:
    activate_jaxaht(jaxaht_path)
    from envs import make_env

    return make_env(
        "overcooked-v1",
        {
            "layout": layout,
            "max_steps": max_steps,
            "random_reset": random_reset,
            "random_obj_state": False,
            "do_reward_shaping": False,
        },
    )


def make_overcooked_partner(
    jaxaht_path: str | Path,
    role: str,
    layout: str = "cramped_room",
    *,
    p_onion_on_counter: float = 0.0,
    p_plate_on_counter: float = 0.0,
) -> Any:
    activate_jaxaht(jaxaht_path)
    from agents.overcooked.agent_policy_wrappers import (
        OvercookedIndependentPolicyWrapper,
        OvercookedOnionPolicyWrapper,
        OvercookedPlatePolicyWrapper,
        OvercookedRandomPolicyWrapper,
        OvercookedStaticPolicyWrapper,
    )
    from envs.overcooked.augmented_layouts import augmented_layouts

    layout_config = augmented_layouts[layout]
    if role == "independent":
        return OvercookedIndependentPolicyWrapper(
            layout_config,
            using_log_wrapper=False,
            p_onion_on_counter=p_onion_on_counter,
            p_plate_on_counter=p_plate_on_counter,
        )
    if role == "onion":
        return OvercookedOnionPolicyWrapper(
            layout_config,
            p_onion_on_counter=p_onion_on_counter,
            using_log_wrapper=False,
        )
    if role == "plate":
        return OvercookedPlatePolicyWrapper(
            layout_config,
            p_plate_on_counter=p_plate_on_counter,
            using_log_wrapper=False,
        )
    if role == "static":
        return OvercookedStaticPolicyWrapper(layout_config, using_log_wrapper=False)
    if role == "random":
        return OvercookedRandomPolicyWrapper(layout_config, using_log_wrapper=False)
    raise ValueError(f"Unknown Overcooked teammate role: {role}")


def load_frozen_s5_ego(
    jaxaht_path: str | Path,
    checkpoint_path: str | Path,
    env: Any,
    seed: int = 0,
) -> tuple[Any, Any]:
    activate_jaxaht(jaxaht_path)
    from common.agent_loader_from_config import initialize_rl_agent_from_config

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing frozen ego checkpoint: {checkpoint}")
    config = dict(S5_CONFIG)
    config["path"] = str(checkpoint)
    policy, selected_params, init_params, labels = initialize_rl_agent_from_config(
        config,
        "frozen_ego",
        env,
        jax.random.PRNGKey(seed),
    )
    selected_shape = jax.tree.leaves(selected_params)[0].shape
    model_shape = jax.tree.leaves(init_params)[0].shape
    if selected_shape == model_shape:
        model = selected_params
    else:
        flattened = jax.tree.map(
            lambda selected, initial: selected.reshape((-1,) + initial.shape),
            selected_params,
            init_params,
        )
        count = jax.tree.leaves(flattened)[0].shape[0]
        if count != 1:
            raise RuntimeError(f"Expected one frozen ego, loaded {count}")
        model = jax.tree.map(lambda leaf: leaf[0], flattened)
    return policy, model
