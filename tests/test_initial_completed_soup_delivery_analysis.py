import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_initial_completed_soup_delivery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_initial_completed_soup_delivery",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class InitialCompletedSoupDeliveryAnalysisTests(unittest.TestCase):
    def test_effect_summary_uses_clustered_paired_return_differences(self):
        rows = [
            {
                "episode_seed": 1,
                "train_seed": 10,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "delivery": {"remaining_return": 20.0},
                },
            },
            {
                "episode_seed": 1,
                "train_seed": 11,
                "branches": {
                    "no_change": {"remaining_return": 20.0},
                    "delivery": {"remaining_return": 20.0},
                },
            },
        ]

        result = MODULE.effect_summary(rows, "delivery")

        self.assertEqual(result["mean"], 10.0)
        self.assertEqual(result["differences"], [20.0, 0.0])
        self.assertEqual(result["environment_seed_cluster_interval"]["clusters"], 1)
        self.assertIsNone(result["environment_seed_cluster_interval"]["ci_low"])


if __name__ == "__main__":
    unittest.main()
