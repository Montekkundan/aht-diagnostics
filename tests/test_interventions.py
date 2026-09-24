import unittest

import jax.numpy as jnp

from aht_diagnostics.interventions import (
    ActionSequencePolicy,
    FixedActionPolicy,
    InteractionMaskedPolicy,
    mask_interactions,
)


class AlwaysInteractPolicy:
    def init_hstate(self, batch_size, aux_info=None):
        return jnp.asarray(0)

    def get_action(self, **kwargs):
        return jnp.asarray([[[5]]]), kwargs["hstate"] + 1


class InteractionMaskedPolicyTests(unittest.TestCase):
    def test_replaces_interact_with_stay_and_preserves_state_update(self):
        policy = InteractionMaskedPolicy(AlwaysInteractPolicy())
        action, state = policy.get_action(
            params=None,
            obs=jnp.zeros((1, 1, 1)),
            done=jnp.asarray([[False]]),
            avail_actions=jnp.ones(6),
            hstate=jnp.asarray(4),
            rng=jnp.asarray([0, 1], dtype=jnp.uint32),
            env_state=None,
        )

        self.assertEqual(int(action.squeeze()), 4)
        self.assertEqual(int(state), 5)

    def test_fixed_action_policy_returns_requested_available_action(self):
        policy = FixedActionPolicy(1)
        action, state = policy.get_action(
            params=None,
            obs=jnp.zeros((1, 1, 1)),
            done=jnp.asarray([[False]]),
            avail_actions=jnp.ones(6, dtype=bool),
            hstate=jnp.asarray(4),
            rng=jnp.asarray([0, 1], dtype=jnp.uint32),
            env_state=None,
        )

        self.assertEqual(int(action.squeeze()), 1)
        self.assertEqual(int(state), 4)

    def test_fixed_action_policy_rejects_unavailable_action(self):
        policy = FixedActionPolicy(1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            policy.get_action(
                params=None,
                obs=jnp.zeros((1, 1, 1)),
                done=jnp.asarray([[False]]),
                avail_actions=jnp.asarray([1, 0, 1, 1, 1, 1], dtype=bool),
                hstate=jnp.asarray(4),
                rng=jnp.asarray([0, 1], dtype=jnp.uint32),
                env_state=None,
            )

    def test_action_sequence_policy_replays_actions_in_order(self):
        policy = ActionSequencePolicy((1, 5, 3))
        state = policy.init_hstate(1)
        observed = []
        for _ in range(3):
            action, state = policy.get_action(
                params=None,
                obs=jnp.zeros((1, 1, 1)),
                done=jnp.asarray([[False]]),
                avail_actions=jnp.ones(6, dtype=bool),
                hstate=state,
                rng=jnp.asarray([0, 1], dtype=jnp.uint32),
                env_state=None,
            )
            observed.append(int(action.squeeze()))

        self.assertEqual(observed, [1, 5, 3])
        self.assertEqual(int(state), 3)
        with self.assertRaisesRegex(IndexError, "exhausted"):
            policy.get_action(
                params=None,
                obs=jnp.zeros((1, 1, 1)),
                done=jnp.asarray([[False]]),
                avail_actions=jnp.ones(6, dtype=bool),
                hstate=state,
                rng=jnp.asarray([0, 1], dtype=jnp.uint32),
                env_state=None,
            )

    def test_mask_interactions_preserves_every_other_action(self):
        self.assertEqual(mask_interactions((1, 5, 3, 5, 4)), (1, 4, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
