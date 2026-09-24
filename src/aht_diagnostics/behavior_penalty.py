"""Training-time penalties for inactive or obstructive Overcooked actions."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxmarl.wrappers.baselines import JaxMARLWrapper


MOVE_ACTIONS = 4
STAY_ACTION = 4
INTERACT_ACTION = 5


def behavior_penalty(
    *,
    action,
    position_before,
    position_after,
    direction_before,
    direction_after,
    inventory_before,
    inventory_after,
    base_reward,
    shaped_reward,
    coefficient: float,
):
    is_wait = action == STAY_ACTION
    is_blocked_move = (
        (action < MOVE_ACTIONS)
        & jnp.all(position_before == position_after)
        & (direction_before == direction_after)
    )
    is_ineffective_interaction = (
        (action == INTERACT_ACTION)
        & (inventory_before == inventory_after)
        & (base_reward == 0)
        & (shaped_reward == 0)
    )
    triggered = is_wait | is_blocked_move | is_ineffective_interaction
    return jnp.asarray(coefficient, dtype=jnp.float32) * triggered.astype(jnp.float32)


class OvercookedBehaviorPenalty(JaxMARLWrapper):
    def __init__(
        self,
        env,
        *,
        coefficient: float,
        penalized_agent_index: int = 1,
    ):
        super().__init__(env)
        self.coefficient = float(coefficient)
        self.penalized_agent_index = int(penalized_agent_index)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        before = state.env_state
        obs, next_state, rewards, dones, info = self._env.step(key, state, actions)
        after = next_state.env_state
        agent = f"agent_{self.penalized_agent_index}"
        index = self.penalized_agent_index
        penalty = behavior_penalty(
            action=actions[agent],
            position_before=before.agent_pos[index],
            position_after=after.agent_pos[index],
            direction_before=before.agent_dir_idx[index],
            direction_after=after.agent_dir_idx[index],
            inventory_before=before.agent_inv[index],
            inventory_after=after.agent_inv[index],
            base_reward=info["base_reward"][index],
            shaped_reward=info["shaped_reward"][index],
            coefficient=self.coefficient,
        )
        rewards = {**rewards, agent: rewards[agent] - penalty}
        info = {
            **info,
            "behavior_penalty": jnp.zeros(self.num_agents, dtype=jnp.float32)
            .at[index]
            .set(penalty),
        }
        return obs, next_state, rewards, dones, info
