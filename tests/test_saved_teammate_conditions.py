from aht_diagnostics.evaluation_conditions import condition_kwargs, summarize_returns


def test_sparse_condition_removes_shaping_parameters() -> None:
    kwargs = condition_kwargs(
        {"reward_shaping_params": {"SOUP_PICKUP_REWARD": 1.0}},
        "standard_sparse",
        400,
    )

    assert kwargs == {
        "random_obj_state": False,
        "do_reward_shaping": False,
        "max_steps": 400,
        "random_reset": False,
    }


def test_sparse_summary_reports_task_completion() -> None:
    summary = summarize_returns([0.0, 20.0, 40.0, 0.0], 3, sparse=True)

    assert summary["completion_rate"] == 0.5
    assert summary["completed_episodes"] == 2
