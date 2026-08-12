import os
from pathlib import Path
import unittest

from aht_diagnostics.defects import InteractionSuppressionPolicy, PLANTED_DEFECTS
from aht_diagnostics.jaxaht_bridge import make_overcooked_env, make_overcooked_partner
from aht_diagnostics.rollout import counterfactual_splice_profile, rollout_episode


JAXAHT = os.environ.get("JAXAHT_PATH")


class PlantedDefectTests(unittest.TestCase):
    @unittest.skipUnless(JAXAHT, "JAXAHT_PATH is not set")
    def test_delivery_defect_has_one_ground_truth_and_is_repairable(self):
        root = Path(JAXAHT)
        env = make_overcooked_env(root, 100, random_reset=True)
        healthy = make_overcooked_partner(root, "independent")
        teammate = make_overcooked_partner(
            root,
            "onion",
            p_onion_on_counter=0.1,
        )
        defect = next(
            defect for defect in PLANTED_DEFECTS if defect.name == "delivery_avoidance"
        )
        faulty = InteractionSuppressionPolicy(healthy, defect, env)

        episode = rollout_episode(
            env,
            faulty,
            None,
            teammate,
            None,
            seed=1000,
            max_steps=100,
        )
        trigger_steps = [
            record.step for record in episode.steps if record.ego_defect_triggered
        ]
        suppressed_steps = [
            record.step for record in episode.steps if record.ego_defect_suppressed
        ]
        self.assertEqual(len(trigger_steps), 1)
        self.assertIn(trigger_steps[0], suppressed_steps)

        repairs = counterfactual_splice_profile(
            env,
            episode,
            faulty,
            None,
            teammate,
            None,
            healthy,
            None,
            max_steps=100,
            windows=(1,),
            candidate_steps={trigger_steps[0]},
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0].step, trigger_steps[0])
        self.assertEqual(repairs[0].repaired_event, "deliver_soup")
        self.assertGreaterEqual(repairs[0].gain, 20.0)


if __name__ == "__main__":
    unittest.main()
