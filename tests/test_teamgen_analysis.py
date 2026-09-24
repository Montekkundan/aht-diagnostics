import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_fixed_ego_teamgen.py"
SPEC = importlib.util.spec_from_file_location("analyze_fixed_ego_teamgen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TeamGenerationAnalysisTests(unittest.TestCase):
    def test_event_fractions_separate_task_progress_from_degenerate_actions(self):
        fractions = MODULE.event_fractions(
            {
                "fetch_onion": 2,
                "move": 3,
                "blocked_move": 4,
                "interact_no_effect": 1,
            }
        )
        self.assertAlmostEqual(fractions["productive"], 0.2)
        self.assertAlmostEqual(fractions["other"], 0.3)
        self.assertAlmostEqual(fractions["degenerate"], 0.5)

    def test_empty_event_counts_are_safe(self):
        self.assertEqual(
            MODULE.event_fractions({}),
            {"productive": 0.0, "degenerate": 0.0, "other": 0.0},
        )

    def test_finding_reports_full_feasibility_without_reduced_budget_text(self):
        condition = {
            "training_timesteps_per_seed": 6_000_000,
            "feasible_best_response_pairings": 5,
            "ego_weakness_exposed": 5,
        }
        finding = MODULE.finding_text(condition, condition)
        self.assertIn("supports the fixed-ego weakness hypothesis", finding)
        self.assertNotIn("1,024,000", finding)


if __name__ == "__main__":
    unittest.main()
