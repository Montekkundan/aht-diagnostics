"""Controlled semantic defects for the Overcooked localization gate.

The wrapper never updates or retrains the underlying policy. It only suppresses
one class of interaction so that the evaluator has known decision-level ground
truth to recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .semantics import INTERACT_ACTION, STAY_ACTION, semantic_event, state_view


@dataclass(frozen=True)
class PlantedDefect:
    name: str
    held_inventory: str
    expected_repair_event: str


@dataclass(frozen=True)
class DefectHState:
    base_hstate: Any
    active: bool = False
    resolved: bool = False


PLANTED_DEFECTS = (
    PlantedDefect(
        name="onion_placement_avoidance",
        held_inventory="onion",
        expected_repair_event="place_onion_in_pot",
    ),
    PlantedDefect(
        name="soup_pickup_avoidance",
        held_inventory="plate",
        expected_repair_event="pick_up_soup",
    ),
    PlantedDefect(
        name="delivery_avoidance",
        held_inventory="dish",
        expected_repair_event="deliver_soup",
    ),
)


class InteractionSuppressionPolicy:
    """Freeze a policy while replacing one semantic interaction with `stay`."""

    def __init__(self, base_policy: Any, defect: PlantedDefect, env: Any):
        self.base_policy = base_policy
        self.defect = defect
        self.env = env

    def init_hstate(self, batch_size: int, aux_info: Any = None) -> Any:
        return DefectHState(
            self.base_policy.init_hstate(batch_size, aux_info=aux_info)
        )

    def get_action_diagnostic(
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
    ) -> tuple[Any, Any, dict[str, Any]]:
        base_action, new_base_hstate = self.base_policy.get_action(
            params=params,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=hstate.base_hstate,
            rng=rng,
            env_state=env_state,
            aux_obs=aux_obs,
            test_mode=test_mode,
        )
        base_action_int = int(np.asarray(base_action).squeeze())
        before = state_view(env_state)
        resolved = hstate.resolved or (
            hstate.active and before.inventories[0] != self.defect.held_inventory
        )
        candidate = (
            before.inventories[0] == self.defect.held_inventory
            and base_action_int == INTERACT_ACTION
        )
        event = "none"
        if candidate:
            simulated_actions = {
                "agent_0": jnp.asarray(base_action_int, dtype=jnp.int32),
                "agent_1": jnp.asarray(STAY_ACTION, dtype=jnp.int32),
            }
            _, next_state, reward, _, _ = self.env.step(
                rng,
                env_state,
                simulated_actions,
            )
            event = semantic_event(
                before,
                state_view(next_state),
                0,
                base_action_int,
                float(np.asarray(reward["agent_0"]).reshape(-1)[0]),
            )
        starts_now = (
            not hstate.active
            and not resolved
            and event == self.defect.expected_repair_event
        )
        active = (hstate.active or starts_now) and not resolved
        suppressed = active and event == self.defect.expected_repair_event
        action = jnp.full_like(base_action, STAY_ACTION) if suppressed else base_action
        if bool(np.asarray(done).squeeze()):
            new_hstate = DefectHState(new_base_hstate)
        else:
            new_hstate = DefectHState(new_base_hstate, active=active, resolved=resolved)
        return action, new_hstate, {
            "base_action": base_action_int,
            "defect_type": self.defect.name,
            "defect_triggered": starts_now,
            "expected_repair_event": self.defect.expected_repair_event,
            "suppressed_event": event,
        }

    def get_action(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        action, hstate, _ = self.get_action_diagnostic(*args, **kwargs)
        return action, hstate


REPAIR_EVENT_TO_DEFECT = {
    defect.expected_repair_event: defect.name for defect in PLANTED_DEFECTS
}


def classify_repair_event(event: str) -> str:
    return REPAIR_EVENT_TO_DEFECT.get(event, "unknown")
