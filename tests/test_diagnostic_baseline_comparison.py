import importlib.util
import hashlib
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_diagnostic_baselines.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_diagnostic_baselines",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical_row(*, factual_return=0.0, swap_interval=None, soup_interval=None):
    return {
        "factual_return": factual_return,
        "reference_return": 20.0,
        "swap_context": {"candidate_interval": swap_interval},
        "initial_soup_context": {
            "candidate_interval": soup_interval,
            "disposition": "drop_soup",
        },
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_fixture(*, swap_digest: str, initial_digest: str):
    shared = {
        "checkpoint_dataset_revision": "dataset-revision",
        "fixed_checkpoint_sha256": "fixed-checkpoint",
        "saved_policy_sha256": {"1": "policy"},
    }
    canonical = {
        "provenance": {
            **shared,
            "source_implementation_reference": {"head": "abc"},
        },
        "protocol": {"environment_condition": "random", "max_steps": 400},
    }
    component = {
        "provenance": {
            **shared,
            "source_implementation_reference": {"head": "abc"},
        },
        "protocol": {"environment_condition": "random", "max_steps": 400},
    }
    suite = {
        "provenance": shared,
        "inputs": {
            "swap_result_sha256": swap_digest,
            "initial_soup_result_sha256": initial_digest,
        },
    }
    return canonical, component, suite


def test_normalized_cost_counts_case_adaptive_action_generation():
    result = MODULE._normalized_cost_accounting(
        canonical_row_count=500,
        swap_case_count=29,
        initial_case_count=11,
        full_suite_branch_count=350,
    )

    assert result["primary_only_counterfactual_branches"] == 91
    assert result["full_suite_counterfactual_branches"] == 350
    assert "Normalized algorithmic counts" in result["accounting_basis"]


def test_case_alignment_accepts_matching_returns_and_intervals():
    canonical = {
        (1, 2): canonical_row(factual_return=0.0, swap_interval=[19, 26]),
        (1, 3): canonical_row(factual_return=0.0, soup_interval=[0, 3]),
    }
    swap = {
        "cases": [
            {
                "train_seed": 1,
                "episode_seed": 2,
                "factual_return": 0.0,
                "trigger": {
                    "start_step": 19,
                    "confirmed_at_step": 26,
                    "intervention_step": 27,
                },
            }
        ]
    }
    initial = {
        "cases": [
            {
                "train_seed": 1,
                "episode_seed": 3,
                "factual_return": 0.0,
                "factual_disposition": "drop_soup",
                "factual_disposition_step": 3,
                "intervention_step": 3,
            }
        ]
    }

    MODULE._validate_case_alignment(
        canonical_rows=canonical,
        swap=swap,
        initial=initial,
    )


def test_case_alignment_rejects_stale_candidate_interval():
    canonical = {
        (1, 2): canonical_row(factual_return=0.0, swap_interval=[20, 27]),
    }
    swap = {
        "cases": [
            {
                "train_seed": 1,
                "episode_seed": 2,
                "factual_return": 0.0,
                "trigger": {
                    "start_step": 19,
                    "confirmed_at_step": 26,
                    "intervention_step": 27,
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="stale candidate interval"):
        MODULE._validate_case_alignment(
            canonical_rows=canonical,
            swap=swap,
            initial={"cases": []},
        )


def test_provenance_requires_suite_to_match_supplied_component_files(tmp_path):
    swap_path = tmp_path / "swap.json"
    initial_path = tmp_path / "initial.json"
    swap_path.write_text("swap", encoding="utf-8")
    initial_path.write_text("initial", encoding="utf-8")
    canonical, component, suite = provenance_fixture(
        swap_digest=digest(swap_path),
        initial_digest=digest(initial_path),
    )

    MODULE._validate_provenance(
        canonical,
        component,
        component,
        suite,
        swap_path=swap_path,
        initial_path=initial_path,
        capable_control_path=None,
    )

    suite["inputs"]["swap_result_sha256"] = "stale"
    with pytest.raises(ValueError, match="supplied swap_result_sha256"):
        MODULE._validate_provenance(
            canonical,
            component,
            component,
            suite,
            swap_path=swap_path,
            initial_path=initial_path,
            capable_control_path=None,
        )


def test_provenance_requires_and_validates_capable_control(tmp_path):
    swap_path = tmp_path / "swap.json"
    initial_path = tmp_path / "initial.json"
    capable_path = tmp_path / "capable.json"
    swap_path.write_text("swap", encoding="utf-8")
    initial_path.write_text("initial", encoding="utf-8")
    capable_path.write_text("capable", encoding="utf-8")
    canonical, component, suite = provenance_fixture(
        swap_digest=digest(swap_path),
        initial_digest=digest(initial_path),
    )
    suite["inputs"]["capable_control_result_sha256"] = digest(capable_path)

    with pytest.raises(ValueError, match="pass --capable-control-result"):
        MODULE._validate_provenance(
            canonical,
            component,
            component,
            suite,
            swap_path=swap_path,
            initial_path=initial_path,
            capable_control_path=None,
        )

    MODULE._validate_provenance(
        canonical,
        component,
        component,
        suite,
        swap_path=swap_path,
        initial_path=initial_path,
        capable_control_path=capable_path,
    )


def test_confirmation_provenance_requires_frozen_generators(tmp_path):
    swap_path = tmp_path / "swap.json"
    initial_path = tmp_path / "initial.json"
    capable_path = tmp_path / "capable.json"
    swap_path.write_text("swap", encoding="utf-8")
    initial_path.write_text("initial", encoding="utf-8")
    capable_path.write_text("capable", encoding="utf-8")
    canonical, component, suite = provenance_fixture(
        swap_digest=digest(swap_path),
        initial_digest=digest(initial_path),
    )
    canonical["provenance"].update(
        analysis_script_sha256="export-script",
        config_sha256="rows-config",
    )
    component["provenance"].update(
        analysis_script_sha256="component-script",
        config_sha256="component-config",
    )
    suite["analysis_script_sha256"] = "suite-script"
    suite["inputs"]["capable_control_result_sha256"] = digest(capable_path)
    capable = {
        "provenance": {
            "analysis_script_sha256": "capable-script",
            "config_sha256": "capable-config",
        }
    }
    frozen = {
        "scripts/export_diagnostic_baseline_rows.py": "export-script",
        "configs/confirmation/diagnostic_baseline_confirmation_rows.json": "rows-config",
        "scripts/analyze_soup_carrier_swap_lock.py": "component-script",
        "configs/confirmation/diagnostic_baseline_confirmation_swap_lock.json": "component-config",
        "scripts/analyze_initial_completed_soup_delivery.py": "component-script",
        "configs/confirmation/diagnostic_baseline_confirmation_initial_soup.json": "component-config",
        "scripts/screen_soup_carrier_swap_lock_capable_controls.py": "capable-script",
        "configs/confirmation/diagnostic_baseline_confirmation_capable_control.json": "capable-config",
        "scripts/summarize_diagnostic_suite.py": "suite-script",
    }

    MODULE._validate_provenance(
        canonical,
        component,
        component,
        suite,
        swap_path=swap_path,
        initial_path=initial_path,
        capable_control_path=capable_path,
        capable_control=capable,
        frozen_hashes=frozen,
    )

    canonical["provenance"]["analysis_script_sha256"] = "stale"
    with pytest.raises(ValueError, match="canonical rows.*unfrozen script"):
        MODULE._validate_provenance(
            canonical,
            component,
            component,
            suite,
            swap_path=swap_path,
            initial_path=initial_path,
            capable_control_path=capable_path,
            capable_control=capable,
            frozen_hashes=frozen,
        )


def test_confirmation_design_requires_the_complete_frozen_rectangle():
    specification = {
        "expected_train_seeds": [1, 2],
        "expected_episode_seed_range": {"start": 10, "end_exclusive": 12},
    }
    rows = {(policy, start): {} for policy in [1, 2] for start in [10, 11]}

    assert MODULE._validate_expected_confirmation_design(specification, rows) == (
        [1, 2],
        [10, 11],
    )
    rows.pop((2, 11))
    with pytest.raises(ValueError, match="frozen policy-by-start rectangle"):
        MODULE._validate_expected_confirmation_design(specification, rows)


def test_confirmation_settings_normalize_the_single_nested_protocol():
    specification = {
        "training": {"train_seeds": [1, 2]},
        "evaluation": {
            "episode_seed_range": {"start": 10, "end_exclusive": 12},
            "competence_return_threshold": 20.0,
        },
        "factual_event_burden_weights": {"ego_blocked_move_rate": 1.0},
        "primary_acquisition_endpoint": {"query_budget_per_policy": 1},
        "bootstrap": {"replicates": 100, "rng_seed": 9},
    }

    assert MODULE._confirmation_settings(specification) == {
        "expected_train_seeds": [1, 2],
        "expected_episode_seed_range": {"start": 10, "end_exclusive": 12},
        "event_burden_weights": {"ego_blocked_move_rate": 1.0},
        "queries_per_policy": 1,
        "competence_return_threshold": 20.0,
        "bootstrap_replicates": 100,
        "bootstrap_rng_seed": 9,
    }


def test_actionable_rescue_is_independent_of_full_suite_decisions():
    canonical = {
        (1, 10): canonical_row(factual_return=0.0, swap_interval=[1, 8]),
        (1, 11): canonical_row(factual_return=0.0, soup_interval=[0, 3]),
        (1, 12): {
            **canonical_row(factual_return=0.0, swap_interval=[1, 8]),
            "reference_return": 0.0,
        },
    }
    swap = {
        "cases": [
            {
                "train_seed": 1,
                "episode_seed": 10,
                "no_change_exact_replay": True,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "detour_down_left_2": {"remaining_return": 20.0},
                },
            },
            {
                "train_seed": 1,
                "episode_seed": 12,
                "no_change_exact_replay": True,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "detour_down_left_2": {"remaining_return": 20.0},
                },
            },
        ]
    }
    initial = {
        "cases": [
            {
                "train_seed": 1,
                "episode_seed": 11,
                "no_change_exact_replay": True,
                "branches": {
                    "no_change": {"remaining_return": 0.0},
                    "reference_actions_no_interact": {
                        "remaining_return": 10.0
                    },
                },
            }
        ]
    }

    keys, evidence = MODULE._actionable_rescue_keys(
        canonical_rows=canonical,
        swap=swap,
        initial=initial,
        competence_return_threshold=20.0,
    )

    assert keys == {(1, 10)}
    by_key = {
        (row["train_seed"], row["episode_seed"]): row for row in evidence
    }
    assert by_key[(1, 11)]["checks"]["primary_gain_at_least_threshold"] is False
    assert by_key[(1, 12)]["checks"]["competent_same_start_reference"] is False


def test_competence_witnessed_failure_is_trigger_independent():
    canonical = {
        (1, 10): canonical_row(
            factual_return=0.0,
            swap_interval=None,
            soup_interval=None,
        ),
        (1, 11): canonical_row(
            factual_return=20.0,
            swap_interval=[1, 8],
        ),
        (1, 12): {
            **canonical_row(factual_return=0.0, swap_interval=[1, 8]),
            "reference_return": 0.0,
        },
    }

    keys, evidence = MODULE._competence_witnessed_failure_keys(
        canonical_rows=canonical,
        competence_return_threshold=20.0,
    )

    assert keys == {(1, 10)}
    by_key = {
        (row["train_seed"], row["episode_seed"]): row for row in evidence
    }
    assert by_key[(1, 10)]["competence_witnessed_failure"] is True
    assert by_key[(1, 11)]["checks"]["factual_zero"] is False
    assert by_key[(1, 12)]["checks"]["competent_same_start_reference"] is False


def test_external_acquisition_rows_expose_shared_factual_failure_priority():
    canonical = {
        (1, 10): canonical_row(factual_return=0.0, swap_interval=[1, 8]),
        (1, 11): canonical_row(factual_return=20.0, soup_interval=[0, 3]),
    }

    rows = MODULE._external_acquisition_rows(
        canonical_rows=canonical,
        event_scores={(1, 10): 1.0, (1, 11): 2.0},
        raw_gap_scores={(1, 10): 20.0, (1, 11): 0.0},
        target_positive_keys={(1, 10)},
    )

    by_key = {(row["train_seed"], row["episode_seed"]): row for row in rows}
    assert by_key[(1, 10)]["factual_zero"] is True
    assert by_key[(1, 11)]["factual_zero"] is False


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        ([0.01, 0.2], "supported"),
        ([-0.2, -0.01], "contradicted"),
        ([-0.1, 0.1], "inconclusive"),
    ],
)
def test_acquisition_decision_is_symmetric(ci, expected):
    analysis = {
        "queries_per_policy": 4,
        "policy_cluster_bootstrap": {"ci95": ci},
    }

    assert MODULE._acquisition_decision(analysis) == expected
