import unittest
from types import SimpleNamespace

import numpy as np

from aht_diagnostics.natural_failure import (
    branch_summary,
    clustered_paired_bootstrap_interval,
    first_mutual_plate_block_opportunity,
    first_initial_soup_disposition,
    initial_completed_soup_opportunity,
    first_soup_carrier_swap_lock_opportunity,
    mutual_plate_block_summary,
    paired_bootstrap_interval,
    soup_carrier_swap_context_summary,
    wilson_interval,
)


def raw_state(*, inventories=(5, 5), positions=((3, 2), (3, 1)), pot_status=0):
    maze = np.zeros((5, 5, 3), dtype=int)
    maze[1, 1, 0] = 8
    maze[1, 1, 2] = pot_status
    return SimpleNamespace(
        agent_pos=np.asarray(positions),
        agent_inv=np.asarray(inventories),
        agent_dir_idx=np.asarray((0, 0)),
        maze_map=maze,
    )


def swap_step(step, *, ego_action=3, partner_action=2):
    return SimpleNamespace(
        step=step,
        ego_action=ego_action,
        partner_action=partner_action,
        ego_event="blocked_move",
        partner_event="blocked_move",
    )


class NaturalFailureStatisticsTests(unittest.TestCase):
    def test_initial_completed_soup_opportunity_uses_only_initial_state(self):
        snapshots = (
            SimpleNamespace(step=0, env_state=raw_state(inventories=(9, 3))),
            SimpleNamespace(step=1, env_state=raw_state(inventories=(1, 3))),
        )

        opportunity = initial_completed_soup_opportunity(
            SimpleNamespace(snapshots=snapshots)
        )

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.step, 0)
        self.assertEqual(opportunity.ego_position, (3, 2))
        self.assertEqual(opportunity.partner_inventory, "onion")

    def test_initial_completed_soup_opportunity_rejects_later_pickup(self):
        snapshots = (
            SimpleNamespace(step=0, env_state=raw_state(inventories=(1, 3))),
            SimpleNamespace(step=1, env_state=raw_state(inventories=(9, 3))),
        )

        self.assertIsNone(
            initial_completed_soup_opportunity(SimpleNamespace(snapshots=snapshots))
        )

    def test_first_initial_soup_disposition_ignores_later_soup_cycle(self):
        snapshots = tuple(
            SimpleNamespace(step=step, env_state=raw_state(inventories=(9, 3)))
            for step in range(3)
        )
        steps = (
            SimpleNamespace(step=0, ego_event="move"),
            SimpleNamespace(step=1, ego_event="deliver_soup"),
            SimpleNamespace(step=2, ego_event="drop_soup"),
        )

        disposition = first_initial_soup_disposition(
            SimpleNamespace(snapshots=snapshots, steps=steps)
        )

        self.assertIsNotNone(disposition)
        self.assertEqual(disposition.step, 1)
        self.assertEqual(disposition.event, "deliver_soup")

    def test_first_initial_soup_disposition_requires_initial_soup(self):
        snapshots = (
            SimpleNamespace(step=0, env_state=raw_state(inventories=(1, 3))),
        )
        steps = (SimpleNamespace(step=0, ego_event="drop_soup"),)

        self.assertIsNone(
            first_initial_soup_disposition(
                SimpleNamespace(snapshots=snapshots, steps=steps)
            )
        )

    def test_branch_summary_counts_partner_delivery_as_team_delivery(self):
        result = SimpleNamespace(
            start_step=10,
            end_step=12,
            starting_return=0.0,
            total_return=20.0,
            remaining_return=20.0,
            intervention_steps=1,
            intervention_records=(
                SimpleNamespace(
                    step=10,
                    reward=0.0,
                    ego_event="move",
                    partner_event="move",
                ),
            ),
            continuation_records=(
                SimpleNamespace(
                    step=11,
                    reward=20.0,
                    ego_event="wait",
                    partner_event="deliver_soup",
                ),
            ),
        )

        summary = branch_summary(result)

        self.assertEqual(summary["delivered_dishes"], 1)
        self.assertEqual(summary["time_to_first_delivery"], 1)

    def test_paired_interval_uses_episode_differences(self):
        result = paired_bootstrap_interval([20.0, 0.0, 20.0, 20.0], seed=7)

        self.assertEqual(result["mean"], 15.0)
        self.assertEqual(result["n"], 4)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])

    def test_paired_interval_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_interval([])

    def test_clustered_interval_resamples_whole_episode_groups(self):
        result = clustered_paired_bootstrap_interval(
            [20.0, 20.0, 0.0],
            [40006, 40006, 40001],
            samples=1_000,
            seed=7,
        )

        self.assertEqual(result["mean"], 40.0 / 3.0)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["clusters"], 2)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])

    def test_clustered_interval_requires_matching_labels(self):
        with self.assertRaises(ValueError):
            clustered_paired_bootstrap_interval([20.0, 0.0], [40001])

    def test_clustered_interval_does_not_invent_uncertainty_from_one_cluster(self):
        result = clustered_paired_bootstrap_interval(
            [20.0, 20.0],
            [140027, 140027],
        )

        self.assertEqual(result["mean"], 20.0)
        self.assertEqual(result["clusters"], 1)
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])

    def test_wilson_interval_contains_observed_proportion(self):
        result = wilson_interval(4, 4)

        self.assertEqual(result["proportion"], 1.0)
        self.assertLess(result["ci_low"], 1.0)
        self.assertEqual(result["ci_high"], 1.0)

    def test_mutual_plate_block_summary_counts_contiguous_trigger(self):
        snapshots = tuple(
            SimpleNamespace(env_state=raw_state()) for _ in range(4)
        )
        steps = (
            SimpleNamespace(step=0, ego_event="blocked_move", partner_event="blocked_move"),
            SimpleNamespace(step=1, ego_event="blocked_move", partner_event="blocked_move"),
            SimpleNamespace(step=2, ego_event="move", partner_event="blocked_move"),
            SimpleNamespace(step=3, ego_event="blocked_move", partner_event="blocked_move"),
        )
        episode = SimpleNamespace(snapshots=snapshots, steps=steps)

        self.assertEqual(
            mutual_plate_block_summary(episode),
            {
                "triggered": True,
                "total_steps": 3,
                "longest_run": 2,
                "first_step": 0,
                "run_count": 2,
            },
        )

    def test_first_mutual_plate_block_opportunity_returns_run_start(self):
        snapshots = tuple(
            SimpleNamespace(env_state=raw_state()) for _ in range(4)
        )
        steps = tuple(
            SimpleNamespace(
                step=step,
                ego_event="blocked_move",
                partner_event="blocked_move",
            )
            for step in range(4)
        )
        opportunity = first_mutual_plate_block_opportunity(
            SimpleNamespace(snapshots=snapshots, steps=steps),
            minimum_consecutive_steps=3,
        )

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.step, 0)
        self.assertEqual(opportunity.confirmed_at_step, 2)

    def test_soup_carrier_swap_lock_requires_repeated_reciprocal_swap(self):
        snapshots = tuple(
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 9),
                    positions=((3, 1), (2, 1)),
                )
            )
            for _ in range(4)
        )
        steps = tuple(swap_step(step) for step in range(4))

        opportunity = first_soup_carrier_swap_lock_opportunity(
            SimpleNamespace(snapshots=snapshots, steps=steps),
            minimum_consecutive_steps=3,
        )

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity.step, 0)
        self.assertEqual(opportunity.confirmed_at_step, 2)
        self.assertEqual(opportunity.ego_position, (3, 1))
        self.assertEqual(opportunity.partner_position, (2, 1))

    def test_soup_carrier_swap_lock_rejects_parallel_blocking(self):
        snapshots = tuple(
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 9),
                    positions=((3, 1), (2, 1)),
                )
            )
            for _ in range(4)
        )
        steps = tuple(
            swap_step(step, ego_action=3, partner_action=3) for step in range(4)
        )

        opportunity = first_soup_carrier_swap_lock_opportunity(
            SimpleNamespace(snapshots=snapshots, steps=steps),
            minimum_consecutive_steps=3,
        )

        self.assertIsNone(opportunity)

    def test_soup_carrier_swap_lock_rejects_non_soup_partner(self):
        snapshots = tuple(
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 5),
                    positions=((3, 1), (2, 1)),
                )
            )
            for _ in range(4)
        )
        steps = tuple(swap_step(step) for step in range(4))

        opportunity = first_soup_carrier_swap_lock_opportunity(
            SimpleNamespace(snapshots=snapshots, steps=steps),
            minimum_consecutive_steps=3,
        )

        self.assertIsNone(opportunity)

    def test_soup_carrier_swap_context_summary_reports_exposure_funnel(self):
        snapshots = (
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 9),
                    positions=((3, 1), (2, 1)),
                )
            ),
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 9),
                    positions=((3, 1), (2, 1)),
                )
            ),
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 9),
                    positions=((3, 1), (2, 1)),
                )
            ),
            SimpleNamespace(
                env_state=raw_state(
                    inventories=(3, 5),
                    positions=((3, 1), (2, 1)),
                )
            ),
        )
        steps = (
            swap_step(0),
            swap_step(1),
            swap_step(2, ego_action=3, partner_action=3),
            swap_step(3),
        )

        summary = soup_carrier_swap_context_summary(
            SimpleNamespace(snapshots=snapshots, steps=steps)
        )

        self.assertEqual(
            summary,
            {
                "has_teammate_soup": True,
                "has_adjacent_soup_carrier": True,
                "has_reciprocal_swap_attempt": True,
                "teammate_soup_steps": 3,
                "adjacent_soup_steps": 3,
                "reciprocal_attempt_steps": 2,
                "blocked_swap_steps": 2,
                "longest_blocked_swap_run": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
