"""Small policy adapters used to test competing behavioural explanations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from .semantics import INTERACT_ACTION, STAY_ACTION


class InteractionMaskedPolicy:
    """Preserve a controller's navigation while disabling interactions.

    This creates a movement-matched control for an option controller. It is not
    a generally competent policy: whenever the wrapped controller would use the
    Overcooked ``INTERACT`` action, this adapter executes ``STAY`` instead.
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def init_hstate(self, batch_size: int, aux_info: Any = None) -> Any:
        return self.policy.init_hstate(batch_size, aux_info=aux_info)

    def get_action(
        self,
        params: Any,
        obs: Any,
        done: Any,
        avail_actions: Any,
        hstate: Any,
        rng: Any,
        env_state: Any,
        aux_obs: Any = None,
        test_mode: bool = False,
    ) -> tuple[Any, Any]:
        action, new_hstate = self.policy.get_action(
            params=params,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=hstate,
            rng=rng,
            env_state=env_state,
            aux_obs=aux_obs,
            test_mode=test_mode,
        )
        masked = jnp.where(action == INTERACT_ACTION, STAY_ACTION, action)
        return masked, new_hstate


class FixedActionPolicy:
    """Execute one specified primitive action for a controlled intervention."""

    def __init__(self, action: int):
        self.action = int(action)

    def init_hstate(self, batch_size: int, aux_info: Any = None) -> Any:
        del batch_size, aux_info
        return jnp.asarray(0)

    def get_action(
        self,
        params: Any,
        obs: Any,
        done: Any,
        avail_actions: Any,
        hstate: Any,
        rng: Any,
        env_state: Any,
        aux_obs: Any = None,
        test_mode: bool = False,
    ) -> tuple[Any, Any]:
        del params, obs, done, rng, env_state, aux_obs, test_mode
        if not bool(avail_actions[self.action]):
            raise ValueError(f"Fixed action {self.action} is unavailable")
        return jnp.asarray([[[self.action]]]), hstate


class ActionSequencePolicy:
    """Replay a pre-recorded primitive-action sequence exactly once.

    This controller is intended for paired causal controls. It preserves the
    action count and every primitive action chosen by a successful branch, so a
    comparison can replace only the hypothesized action type while leaving the
    rest of the sequence fixed.
    """

    def __init__(self, actions: Sequence[int]):
        if not actions:
            raise ValueError("ActionSequencePolicy requires at least one action")
        self.actions = tuple(int(action) for action in actions)

    def init_hstate(self, batch_size: int, aux_info: Any = None) -> Any:
        del batch_size, aux_info
        return jnp.asarray(0, dtype=jnp.int32)

    def get_action(
        self,
        params: Any,
        obs: Any,
        done: Any,
        avail_actions: Any,
        hstate: Any,
        rng: Any,
        env_state: Any,
        aux_obs: Any = None,
        test_mode: bool = False,
    ) -> tuple[Any, Any]:
        del params, obs, done, rng, env_state, aux_obs, test_mode
        index = int(jnp.asarray(hstate).reshape(-1)[0])
        if index >= len(self.actions):
            raise IndexError("Action sequence is exhausted")
        action = self.actions[index]
        if not bool(avail_actions[action]):
            raise ValueError(
                f"Recorded action {action} is unavailable at sequence index {index}"
            )
        return (
            jnp.asarray([[[action]]]),
            jnp.asarray(index + 1, dtype=jnp.int32),
        )


def mask_interactions(actions: Sequence[int]) -> tuple[int, ...]:
    """Replace only Overcooked interaction actions with no-op actions."""

    return tuple(
        STAY_ACTION if int(action) == INTERACT_ACTION else int(action)
        for action in actions
    )
