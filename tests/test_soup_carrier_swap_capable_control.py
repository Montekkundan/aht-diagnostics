import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "screen_soup_carrier_swap_lock_capable_controls.py"
)
SPEC = importlib.util.spec_from_file_location(
    "screen_soup_carrier_swap_lock_capable_controls",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SoupCarrierCapableControlTests(unittest.TestCase):
    def test_summary_keeps_paired_trigger_categories(self):
        rows = [
            {
                "train_seed": 1,
                "episode_seed": 10,
                "fixed_return": 0.0,
                "reference_return": 20.0,
                "fixed_trigger": True,
                "reference_trigger": False,
            },
            {
                "train_seed": 2,
                "episode_seed": 11,
                "fixed_return": 20.0,
                "reference_return": 20.0,
                "fixed_trigger": True,
                "reference_trigger": True,
            },
            {
                "train_seed": 2,
                "episode_seed": 12,
                "fixed_return": 0.0,
                "reference_return": 0.0,
                "fixed_trigger": False,
                "reference_trigger": False,
            },
        ]

        summary = MODULE.summarize_rows(rows, competence_threshold=20.0)

        self.assertEqual(
            summary["paired_trigger_table"],
            {"fixed_only": 1, "reference_only": 0, "both": 1, "neither": 1},
        )
        self.assertEqual(summary["competent_reference_pairings"], 2)
        self.assertEqual(summary["fixed_trigger_pairings_with_competent_reference"], 2)
        self.assertEqual(
            summary["reference_trigger_prevalence_among_competent_pairings"][
                "successes"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
