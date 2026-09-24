from aht_diagnostics.diagnostic_suite import summarize_diagnostic_suite


def branch(value):
    return {"remaining_return": float(value)}


def provenance():
    return {
        "checkpoint_dataset_revision": "dataset-revision",
        "fixed_checkpoint_sha256": "fixed-checkpoint",
        "saved_policy_sha256": {"1": "partner-checkpoint"},
        "source_results": "saved-policies",
        "source_implementation_reference": {"local_head": "implementation"},
    }


def swap_case():
    return {
        "train_seed": 1,
        "episode_seed": 10,
        "no_change_exact_replay": True,
        "branches": {
            "no_change": branch(0),
            "detour_down_left_2": branch(20),
            "control_wait_2": branch(0),
            "control_down_stay_2": branch(20),
            "control_left_down_2": branch(0),
        },
    }


def initial_case():
    branches = {
        "no_change": branch(0),
        "reference_actions_no_interact": branch(20),
        "reference_actions_wait": branch(20),
    }
    branches.update(
        {
            f"one_step_{action}": branch(0)
            for action in ("up", "down", "left", "right", "stay", "interact")
        }
    )
    return {
        "train_seed": 1,
        "episode_seed": 11,
        "no_change_exact_replay": True,
        "branches": branches,
    }


def test_suite_distinguishes_supported_ambiguous_and_abstained_failures():
    common_provenance = provenance()
    swap_result = {
        "provenance": common_provenance,
        "cases": [swap_case()],
        "summary": {"evaluated_pairings": 3},
    }
    initial_result = {
        "provenance": common_provenance,
        "cases": [initial_case()],
        "pairing_exposures": [
            {"train_seed": 1, "episode_seed": 10, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 11, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 12, "factual_return": 0},
        ],
        "summary": {"evaluated_pairings": 3},
    }

    result = summarize_diagnostic_suite(
        swap_result=swap_result,
        initial_soup_result=initial_result,
    )

    assert result["summary"]["supported_pairings"] == 1
    assert result["summary"]["ambiguous_pairings"] == 1
    assert result["summary"]["abstained_zero_return_pairings"] == 1
    assert result["summary"]["supported_coverage_over_zero_return"] == 1 / 3
    assert result["summary"]["candidate_coverage_over_zero_return"] == 2 / 3
    diagnoses = {row["episode_seed"]: row for row in result["diagnoses"]}
    assert diagnoses[10]["diagnosis"] == "reciprocal_motion_conflict"
    assert diagnoses[10]["specificity"] == "multiple_movement_repairs"
    assert diagnoses[11]["decision"] == "ambiguous"


def test_suite_rejects_mismatched_policy_provenance():
    swap_result = {
        "provenance": provenance(),
        "cases": [],
        "summary": {"evaluated_pairings": 1},
    }
    initial_provenance = provenance()
    initial_provenance["saved_policy_sha256"] = {"2": "different"}
    initial_result = {
        "provenance": initial_provenance,
        "cases": [],
        "pairing_exposures": [
            {"train_seed": 1, "episode_seed": 10, "factual_return": 0}
        ],
        "summary": {"evaluated_pairings": 1},
    }

    try:
        summarize_diagnostic_suite(
            swap_result=swap_result,
            initial_soup_result=initial_result,
        )
    except ValueError as error:
        assert "saved_policy_sha256" in str(error)
    else:
        raise AssertionError("mismatched provenance should be rejected")


def capable_control():
    return {
        "provenance": {
            "source_results": "saved-policies",
            "source_implementation_reference": {"local_head": "implementation"},
        },
        "rows": [
            {
                "train_seed": 1,
                "episode_seed": 10,
                "fixed_return": 0,
                "fixed_trigger": True,
                "reference_return": 20,
                "reference_trigger": False,
            },
            {
                "train_seed": 1,
                "episode_seed": 11,
                "fixed_return": 0,
                "fixed_trigger": False,
                "reference_return": 0,
                "reference_trigger": False,
            },
            {
                "train_seed": 1,
                "episode_seed": 12,
                "fixed_return": 0,
                "fixed_trigger": False,
                "reference_return": 20,
                "reference_trigger": False,
            },
        ],
        "summary": {"competence_threshold": 20},
    }


def test_suite_joins_start_specific_competence_evidence():
    common_provenance = provenance()
    swap_result = {
        "provenance": common_provenance,
        "cases": [swap_case()],
        "summary": {"evaluated_pairings": 3},
    }
    initial_result = {
        "provenance": common_provenance,
        "cases": [initial_case()],
        "pairing_exposures": [
            {"train_seed": 1, "episode_seed": 10, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 11, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 12, "factual_return": 0},
        ],
        "summary": {"evaluated_pairings": 3},
    }

    result = summarize_diagnostic_suite(
        swap_result=swap_result,
        initial_soup_result=initial_result,
        capable_control_result=capable_control(),
    )

    diagnoses = {row["episode_seed"]: row for row in result["diagnoses"]}
    assert diagnoses[10]["initial_state_competence_witness"] is True
    assert diagnoses[11]["initial_state_competence_witness"] is False
    assert result["summary"][
        "supported_diagnoses_with_initial_state_competence_witness"
    ] == 1
    assert result["summary"]["supported_diagnoses_reference_inconclusive"] == 0
    assert result["summary"]["swap_lock_capable_control"] == {
        "competence_threshold": 20.0,
        "competent_reference_pairings": 2,
        "reference_swap_lock_triggers": 0,
        "reference_swap_lock_triggers_among_competent_pairings": 0,
    }


def test_suite_rejects_mismatched_capable_control_pairings():
    common_provenance = provenance()
    swap_result = {
        "provenance": common_provenance,
        "cases": [swap_case()],
        "summary": {"evaluated_pairings": 3},
    }
    initial_result = {
        "provenance": common_provenance,
        "cases": [initial_case()],
        "pairing_exposures": [
            {"train_seed": 1, "episode_seed": 10, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 11, "factual_return": 0},
            {"train_seed": 1, "episode_seed": 12, "factual_return": 0},
        ],
        "summary": {"evaluated_pairings": 3},
    }
    control = capable_control()
    control["rows"][-1]["episode_seed"] = 99

    try:
        summarize_diagnostic_suite(
            swap_result=swap_result,
            initial_soup_result=initial_result,
            capable_control_result=control,
        )
    except ValueError as error:
        assert "different pairings" in str(error)
    else:
        raise AssertionError("mismatched capable-control pairings should be rejected")
