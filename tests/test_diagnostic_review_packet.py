from types import SimpleNamespace

import numpy as np

from scripts.build_diagnostic_review_packet import (
    kitchen_layout,
    observation_viewer,
    pot_stage,
    records,
)


def _episode():
    maze = np.full((12, 13, 3), 2, dtype=np.int32)
    maze[3:9, 3:10, 0] = 1
    maze[4, 6, 0] = 8
    maze[4, 6, 2] = 0
    maze[6, 6, 0] = 10
    raw = SimpleNamespace(
        maze_map=maze,
        agent_pos=np.array([[2, 2], [3, 2]]),
        agent_inv=np.array([1, 5]),
        agent_dir_idx=np.array([0, 1]),
    )
    snapshot = SimpleNamespace(env_state=SimpleNamespace(env_state=raw))
    step = SimpleNamespace(
        step=0, ego_action=4, partner_action=5,
        ego_position=(2, 2), partner_position=(3, 2),
        ego_inventory="empty", partner_inventory="plate",
        ego_event="wait", partner_event="interact_no_effect",
        ego_direction=0, partner_direction=1, reward=0.0,
    )
    return SimpleNamespace(snapshots=(snapshot,), steps=(step,))


def test_pot_stages_use_overcooked_v1_countdown():
    assert [pot_stage(status) for status in (23, 22, 21, 20, 19, 0)] == [
        "empty", "1 onion(s)", "2 onion(s)", "3 onions, cooking",
        "cooking, 19 steps remaining", "ready",
    ]


def test_review_records_include_state_needed_for_soup_cases():
    episode = _episode()
    layout = kitchen_layout(episode, agent_view_size=5)
    assert layout[1][3] == "pot"
    assert layout[3][3] == "empty"
    row = records(episode)[0]
    assert row["pot_statuses"] == ["ready"]
    assert row["ego_direction"] == 0
    assert row["partner_direction"] == 1


def test_observation_viewer_does_not_embed_evaluator_answer():
    case = {
        "case_id": "D01", "condition_A_return": 0,
        "condition_B_return": 20, "kitchen_layout": [["empty"]],
        "condition_A_steps": [], "condition_B_steps": [],
        "evaluator_decision": "supported",
        "sampling_stratum": "supported_swap_competent_reference",
    }
    viewer = observation_viewer([case])
    assert "D01" in viewer
    assert "evaluator_decision" not in viewer
    assert "sampling_stratum" not in viewer
    assert 'id="step"' in viewer
