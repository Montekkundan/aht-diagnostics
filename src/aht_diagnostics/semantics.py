"""Translate Overcooked state transitions into human-readable events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OBJECT_NAMES = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "onion",
    4: "onion_pile",
    5: "plate",
    6: "plate_pile",
    7: "goal",
    8: "pot",
    9: "dish",
    10: "agent",
}
POT_INDEX = 8
STAY_ACTION = 4
INTERACT_ACTION = 5


@dataclass(frozen=True)
class OvercookedStateView:
    positions: tuple[tuple[int, int], ...]
    inventories: tuple[str, ...]
    pot_statuses: tuple[int, ...]
    directions: tuple[int, ...] = ()


def _unwrap_state(env_state: Any) -> Any:
    state = env_state
    for _ in range(5):
        if hasattr(state, "agent_pos") and hasattr(state, "agent_inv"):
            return state
        if not hasattr(state, "env_state"):
            break
        state = state.env_state
    raise TypeError("Could not find an Overcooked state inside the environment wrapper")


def state_view(env_state: Any) -> OvercookedStateView:
    raw = _unwrap_state(env_state)
    positions = tuple(
        (int(position[0]), int(position[1]))
        for position in np.asarray(raw.agent_pos).reshape(-1, 2)
    )
    inventories = tuple(
        OBJECT_NAMES.get(int(item), f"object_{int(item)}")
        for item in np.asarray(raw.agent_inv).reshape(-1)
    )
    maze = np.asarray(raw.maze_map)
    pot_cells = np.argwhere(maze[:, :, 0] == POT_INDEX)
    pot_statuses = tuple(sorted(int(maze[row, column, 2]) for row, column in pot_cells))
    directions = tuple(int(value) for value in np.asarray(raw.agent_dir_idx).reshape(-1))
    return OvercookedStateView(
        positions=positions,
        inventories=inventories,
        pot_statuses=pot_statuses,
        directions=directions,
    )


def semantic_event(
    before: OvercookedStateView,
    after: OvercookedStateView,
    agent_index: int,
    action: int,
    reward: float,
) -> str:
    """Name the observable event caused by one agent action."""
    before_item = before.inventories[agent_index]
    after_item = after.inventories[agent_index]

    if before_item == "empty" and after_item == "onion":
        return "fetch_onion"
    if before_item == "empty" and after_item == "plate":
        return "fetch_plate"
    if before_item == "empty" and after_item == "dish":
        return "pick_up_soup"
    if before_item == "plate" and after_item == "dish":
        return "pick_up_soup"
    if before_item == "dish" and after_item == "empty":
        return "deliver_soup" if reward > 0 else "drop_soup"
    if before_item == "onion" and after_item == "empty":
        placed = any(
            old > 20 and new == old - 1
            for old, new in zip(before.pot_statuses, after.pot_statuses, strict=True)
        )
        return "place_onion_in_pot" if placed else "drop_onion"
    if before_item == "plate" and after_item == "empty":
        return "drop_plate"

    if action == INTERACT_ACTION:
        return "interact_no_effect"
    if action == STAY_ACTION:
        return "wait"
    if before.positions[agent_index] != after.positions[agent_index]:
        return "move"
    if (
        action < STAY_ACTION
        and len(before.directions) > agent_index
        and len(after.directions) > agent_index
        and before.directions[agent_index] != after.directions[agent_index]
    ):
        return "orientation_change"
    return "blocked_move"


COOPERATIVE_EVENTS = frozenset(
    {
        "fetch_onion",
        "fetch_plate",
        "place_onion_in_pot",
        "pick_up_soup",
        "deliver_soup",
    }
)
