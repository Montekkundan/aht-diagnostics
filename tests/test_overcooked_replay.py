import os
from pathlib import Path
import unittest

from aht_diagnostics.jaxaht_bridge import (
    make_overcooked_env,
    make_overcooked_partner,
)
from aht_diagnostics.rollout import reference_action_trace, rollout_episode


JAXAHT = os.environ.get("JAXAHT_PATH")


class OvercookedReplayTests(unittest.TestCase):
    @unittest.skipUnless(JAXAHT, "JAXAHT_PATH is not set")
    def test_exact_seed_replay_is_deterministic_and_semantic(self):
        root = Path(JAXAHT)
        env = make_overcooked_env(root, 80, random_reset=False)
        ego = make_overcooked_partner(root, "independent")
        partner = make_overcooked_partner(root, "onion", p_onion_on_counter=0.1)

        first = rollout_episode(env, ego, None, partner, None, seed=17, max_steps=80)
        second = rollout_episode(env, ego, None, partner, None, seed=17, max_steps=80)

        self.assertEqual(first.total_return, second.total_return)
        self.assertEqual(
            [step.ego_action for step in first.steps],
            [step.ego_action for step in second.steps],
        )
        self.assertEqual(
            [step.ego_event for step in first.steps],
            [step.ego_event for step in second.steps],
        )
        self.assertTrue(all(step.ego_event for step in first.steps))

        _, action_kl = reference_action_trace(env, first, ego, None)
        self.assertAlmostEqual(action_kl, 0.0)


if __name__ == "__main__":
    unittest.main()
