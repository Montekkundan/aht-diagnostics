import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_soup_carrier_swap_lock.py"
SPEC = importlib.util.spec_from_file_location("analyze_soup_carrier_swap_lock", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SoupCarrierSwapAnalysisTests(unittest.TestCase):
    def test_action_names_match_overcooked_v1_order(self):
        self.assertEqual(
            MODULE.ACTION_NAMES,
            {
                0: "up",
                1: "down",
                2: "right",
                3: "left",
                4: "stay",
                5: "interact",
            },
        )

    def test_effect_summary_uses_paired_remaining_return(self):
        rows = [
            {
                "episode_seed": 1,
                "train_seed": 10,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "yield_down_1": {"remaining_return": 20.0},
                },
            },
            {
                "episode_seed": 2,
                "train_seed": 11,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "yield_down_1": {"remaining_return": 0.0},
                },
            },
        ]

        result = MODULE.effect_summary(rows, "yield_down_1")

        self.assertEqual(result["mean"], 10.0)
        self.assertEqual(result["differences"], [20.0, 0.0])
        self.assertEqual(result["improvement_rate"]["successes"], 1)


if __name__ == "__main__":
    unittest.main()
