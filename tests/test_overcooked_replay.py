import os
from pathlib import Path
import unittest

from aht_diagnostics.jaxaht_bridge import (
    make_overcooked_env,
    make_overcooked_partner,
)
from aht_diagnostics.rollout import (
    reference_action_trace,
    rollout_continuation,
    rollout_episode,
)


JAXAHT = os.environ.get("JAXAHT_PATH")


class OvercookedReplayTests(unittest.TestCase):
    @unittest.skipUnless(JAXAHT, "JAXAHT_PATH is not set")
    def test_terminal_autoreset_is_not_labeled_as_an_action(self):
        root = Path(JAXAHT)
        env = make_overcooked_env(root, 20, random_reset=False)
        static = make_overcooked_partner(root, "static")

        episode = rollout_episode(env, static, None, static, None, seed=17, max_steps=20)

        self.assertEqual(len(episode.steps), 20)
        self.assertTrue(all(step.ego_event == "wait" for step in episode.steps[:-1]))
        self.assertTrue(all(step.partner_event == "wait" for step in episode.steps[:-1]))
        self.assertEqual(episode.steps[-1].ego_event, "terminal_state_unobserved")
        self.assertEqual(episode.steps[-1].partner_event, "terminal_state_unobserved")

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

    @unittest.skipUnless(JAXAHT, "JAXAHT_PATH is not set")
    def test_no_change_continuation_reproduces_episode_suffix(self):
        root = Path(JAXAHT)
        env = make_overcooked_env(root, 100, random_reset=False)
        ego = make_overcooked_partner(root, "independent")
        partner = make_overcooked_partner(root, "onion", p_onion_on_counter=0.1)
        episode = rollout_episode(env, ego, None, partner, None, seed=29, max_steps=100)
        snapshot_index = 30

        replay = rollout_continuation(
            env,
            episode.snapshots[snapshot_index],
            ego,
            None,
            partner,
            None,
            max_steps=100,
        )

        self.assertEqual(replay.total_return, episode.total_return)
        self.assertEqual(
            [record.ego_action for record in replay.continuation_records],
            [record.ego_action for record in episode.steps[snapshot_index:]],
        )
        self.assertEqual(
            [record.partner_action for record in replay.continuation_records],
            [record.partner_action for record in episode.steps[snapshot_index:]],
        )
        self.assertEqual(
            [record.ego_event for record in replay.continuation_records],
            [record.ego_event for record in episode.steps[snapshot_index:]],
        )


if __name__ == "__main__":
    unittest.main()
