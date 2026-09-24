import unittest

import jax.numpy as jnp

from aht_diagnostics.behavior_penalty import behavior_penalty


def penalty(
    action: int,
    *,
    before=(1, 1),
    after=(1, 1),
    direction_before=0,
    direction_after=0,
    inventory_before=1,
    inventory_after=1,
    base_reward=0.0,
    shaped_reward=0.0,
) -> float:
    return float(
        behavior_penalty(
            action=jnp.asarray(action),
            position_before=jnp.asarray(before),
            position_after=jnp.asarray(after),
            direction_before=jnp.asarray(direction_before),
            direction_after=jnp.asarray(direction_after),
            inventory_before=jnp.asarray(inventory_before),
            inventory_after=jnp.asarray(inventory_after),
            base_reward=jnp.asarray(base_reward),
            shaped_reward=jnp.asarray(shaped_reward),
            coefficient=0.01,
        )
    )


class BehaviorPenaltyTests(unittest.TestCase):
    def test_penalizes_wait(self):
        self.assertAlmostEqual(penalty(4), 0.01)

    def test_penalizes_blocked_move_but_not_successful_move(self):
        self.assertAlmostEqual(penalty(0), 0.01)
        self.assertEqual(penalty(0, after=(1, 0)), 0.0)
        self.assertEqual(penalty(0, direction_before=1, direction_after=0), 0.0)

    def test_penalizes_only_ineffective_interactions(self):
        self.assertAlmostEqual(penalty(5), 0.01)
        self.assertEqual(penalty(5, inventory_after=3), 0.0)
        self.assertEqual(penalty(5, shaped_reward=0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
