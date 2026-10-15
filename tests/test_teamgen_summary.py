import unittest

from aht_diagnostics.rollout import Episode, StepRecord
from aht_diagnostics.teamgen import (
    CANONICAL_ROTATE_OBJECTIVE,
    OFFICIAL_CRAMPED_ROOM_CONFIG,
    ROTATE_PAPER_CRAMPED_ROOM_CONFIG,
    TRAINING_PRESETS,
    TRAJECTORY_REGRET_BASELINE,
    evidence_status,
    mean_and_bootstrap_ci,
    partner_num_updates,
    semantic_event_counts,
    summarize_pairing,
    training_overrides,
)


def step(event: str, partner_event: str) -> StepRecord:
    return StepRecord(
        step=0,
        ego_action=0,
        partner_action=0,
        reward=0.0,
        cumulative_return=0.0,
        ego_probs=[],
        ego_position=(0, 0),
        partner_position=(0, 1),
        ego_inventory="empty",
        partner_inventory="empty",
        ego_event=event,
        partner_event=partner_event,
        ego_base_action=0,
        ego_defect_type="healthy",
        ego_defect_triggered=False,
        ego_defect_suppressed=False,
        expected_repair_event="none",
    )


class TeamGenerationSummaryTests(unittest.TestCase):
    def test_preliminary_uses_released_rotate_configuration(self):
        overrides = training_overrides("preliminary", "per_state_regret", 400)
        self.assertEqual(overrides["CONF_OBJ_TYPE"], CANONICAL_ROTATE_OBJECTIVE)
        self.assertEqual(set(overrides), {"CONF_OBJ_TYPE"})

    def test_profile_is_one_official_shape_update(self):
        overrides = training_overrides("profile", "per_trajectory_regret", 400)
        self.assertEqual(overrides["CONF_OBJ_TYPE"], TRAJECTORY_REGRET_BASELINE)
        self.assertEqual(overrides["TIMESTEPS_PER_ITER_PARTNER"], 102_400)

    def test_update_count_matches_jaxaht_partner_schedule(self):
        expts = {**OFFICIAL_CRAMPED_ROOM_CONFIG, "ROLLOUT_LENGTH": 400,
                 "TIMESTEPS_PER_ITER_PARTNER": 5_120_000}
        self.assertEqual(partner_num_updates(expts), 50)
        paper = {**ROTATE_PAPER_CRAMPED_ROOM_CONFIG, "ROLLOUT_LENGTH": 400}
        self.assertEqual(partner_num_updates(paper), 234)

    def test_presets_keep_the_released_default_and_offer_the_paper_budget(self):
        self.assertEqual(TRAINING_PRESETS["jaxaht-expts"], {})
        self.assertEqual(TRAINING_PRESETS["rotate-paper"]["NUM_ENVS"], 16)
        self.assertEqual(TRAINING_PRESETS["rotate-paper"]["LAMBDA_1"], 4.0)

    def test_prespecified_budget_is_not_labeled_timeboxed(self):
        self.assertEqual(
            evidence_status(
                "preliminary",
                training_timesteps_overridden=True,
            ),
            "prespecified_budget_preliminary_experiment",
        )

    def test_bootstrap_summary_is_deterministic(self):
        first = mean_and_bootstrap_ci([0.0, 20.0, 40.0], samples=1000, seed=3)
        second = mean_and_bootstrap_ci([0.0, 20.0, 40.0], samples=1000, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], 20.0)
        self.assertEqual(first["n"], 3)

    def test_semantic_counts_distinguish_the_two_agents(self):
        episode = Episode(
            seed=1,
            total_return=20.0,
            snapshots=(),
            steps=(step("fetch_onion", "fetch_plate"), step("fetch_onion", "deliver_soup")),
        )
        self.assertEqual(
            semantic_event_counts([episode], agent="generated_teammate"),
            {"fetch_onion": 2},
        )
        self.assertEqual(
            semantic_event_counts([episode], agent="partner"),
            {"deliver_soup": 1, "fetch_plate": 1},
        )

    def test_pair_summary_tracks_generated_teammate_in_agent_1(self):
        episode = Episode(
            seed=1,
            total_return=20.0,
            snapshots=(),
            steps=(step("fetch_onion", "fetch_plate"), step("fetch_onion", "deliver_soup")),
        )
        summary = summarize_pairing(
            [episode],
            seed=1,
            generated_teammate_slot="agent_1",
        )
        self.assertEqual(
            summary["generated_teammate_events"],
            {"deliver_soup": 1, "fetch_plate": 1},
        )
        self.assertEqual(summary["partner_events"], {"fetch_onion": 2})


if __name__ == "__main__":
    unittest.main()
