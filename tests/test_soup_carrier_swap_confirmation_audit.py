import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_soup_carrier_swap_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_soup_carrier_swap_confirmation",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def case(policy, episode, primary, left_down, down_stay, wait):
    return {
        "train_seed": policy,
        "episode_seed": episode,
        "no_change_exact_replay": True,
        "branches": {
            "no_change": {"remaining_return": 0.0},
            "detour_down_left_2": {"remaining_return": primary},
            "control_left_down_2": {"remaining_return": left_down},
            "control_down_stay_2": {"remaining_return": down_stay},
            "control_wait_2": {"remaining_return": wait},
        },
    }


class SoupCarrierConfirmationAuditTests(unittest.TestCase):
    def test_rejects_specific_path_when_a_control_also_repairs_cases(self):
        result = {
            "status": "prospective_confirmation_result",
            "cases": [
                case(1, 10, 20.0, 0.0, 0.0, 0.0),
                case(2, 11, 20.0, 20.0, 20.0, 0.0),
            ],
        }

        audit = MODULE.audit_confirmation(result)

        checks = audit["frozen_decision_checks"]
        self.assertFalse(checks["no_matched_control_reproduces_improvement"])
        self.assertFalse(checks["specific_down_left_mechanism_supported"])
        self.assertTrue(checks["broader_motion_conflict_resolution_supported"])
        self.assertEqual(
            audit["primary_vs_matched_controls"]["control_left_down_2"][
                "positive"
            ],
            1,
        )

    def test_supports_specific_path_when_all_controls_fail(self):
        result = {
            "status": "prospective_confirmation_result",
            "cases": [
                case(1, 10, 20.0, 0.0, 0.0, 0.0),
                case(2, 11, 20.0, 0.0, 0.0, 0.0),
            ],
        }

        audit = MODULE.audit_confirmation(result)

        self.assertTrue(
            audit["frozen_decision_checks"][
                "specific_down_left_mechanism_supported"
            ]
        )


if __name__ == "__main__":
    unittest.main()
