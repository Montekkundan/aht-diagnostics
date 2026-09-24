import pytest

from aht_diagnostics.confirmatory_statistics import (
    acquisition_delta_analysis,
    external_binary_acquisition_analysis,
    specificity_downgrade_analysis,
)


def acquisition_rows():
    return [
        {
            "train_seed": 1,
            "episode_seed": 10,
            "event_burden_score": 10.0,
            "suite_nominated": True,
            "suite_supported": False,
        },
        {
            "train_seed": 1,
            "episode_seed": 11,
            "event_burden_score": 8.0,
            "suite_nominated": True,
            "suite_supported": True,
        },
        {
            "train_seed": 2,
            "episode_seed": 10,
            "event_burden_score": 8.0,
            "suite_nominated": False,
            "suite_supported": False,
        },
        {
            "train_seed": 2,
            "episode_seed": 11,
            "event_burden_score": 1.0,
            "suite_nominated": False,
            "suite_supported": False,
        },
    ]


def test_acquisition_delta_uses_equal_expected_budget_and_fractional_ties():
    result = acquisition_delta_analysis(acquisition_rows(), replicates=25, seed=7)

    assert result["point"]["query_budget"] == 2
    assert result["point"]["suite_supported"] == 1
    assert result["point"]["event_expected_supported"] == 0.5
    assert result["point"]["acquisition_delta"] == pytest.approx(0.125)
    assert result["point"]["score_boundary"] == 8.0
    assert result["point"]["boundary_ties"] == 2


def test_acquisition_bootstrap_is_deterministic_and_reports_sensitivities():
    first = acquisition_delta_analysis(acquisition_rows(), replicates=40, seed=11)
    second = acquisition_delta_analysis(acquisition_rows(), replicates=40, seed=11)

    assert first == second
    assert first["bootstrap"]["valid_replicates"] == 40
    assert len(first["bootstrap"]["estimates"]) == 40
    assert len(first["policy_only_bootstrap"]["estimates"]) == 40
    assert len(first["start_only_bootstrap"]["estimates"]) == 40
    assert len(first["leave_one_policy_out"]) == 2
    assert len(first["leave_one_start_out"]) == 2


def test_acquisition_boundary_is_reported_when_no_tie_is_fractional():
    rows = acquisition_rows()
    rows[1] = {**rows[1], "event_burden_score": 9.0}

    result = acquisition_delta_analysis(rows, replicates=5, seed=3)

    assert result["point"]["score_boundary"] == 9.0
    assert result["point"]["boundary_ties"] == 1


def test_acquisition_analysis_requires_complete_crossed_design():
    with pytest.raises(ValueError, match="complete policy-by-start rectangle"):
        acquisition_delta_analysis(acquisition_rows()[:-1], replicates=5)


def test_acquisition_analysis_rejects_supported_but_unnominated_cell():
    rows = acquisition_rows()
    rows[-1] = {**rows[-1], "suite_supported": True}

    with pytest.raises(ValueError, match="also be nominated"):
        acquisition_delta_analysis(rows, replicates=5)


def external_binary_rows():
    rescues = {
        1: [True, True, False, False],
        2: [True, False, True, False],
    }
    diagnosis_scores = {
        1: [4.0, 3.0, 2.0, 1.0],
        2: [4.0, 2.0, 3.0, 1.0],
    }
    event_scores = {
        1: [1.0, 2.0, 4.0, 3.0],
        2: [1.0, 4.0, 2.0, 3.0],
    }
    raw_gap_scores = {
        1: [4.0, 1.0, 3.0, 2.0],
        2: [4.0, 3.0, 2.0, 1.0],
    }
    return [
        {
            "train_seed": policy,
            "episode_seed": start,
            "diagnosis_score": diagnosis_scores[policy][index],
            "event_burden_score": event_scores[policy][index],
            "raw_gap_score": raw_gap_scores[policy][index],
            "factual_zero": True,
            "target_positive": rescues[policy][index],
        }
        for policy in [1, 2]
        for index, start in enumerate([10, 11, 12, 13])
    ]


def test_external_binary_analysis_uses_equal_within_policy_budgets():
    result = external_binary_acquisition_analysis(
        external_binary_rows(),
        queries_per_policy=2,
        replicates=25,
        seed=7,
    )

    assert result["point"]["selectors"]["diagnosis"] == {
        "mean_expected_positives": 2.0,
        "mean_yield": 1.0,
    }
    assert result["point"]["selectors"]["event"] == {
        "mean_expected_positives": 0.0,
        "mean_yield": 0.0,
    }
    assert result["point"]["selectors"]["raw_gap"] == {
        "mean_expected_positives": 1.0,
        "mean_yield": 0.5,
    }
    assert result["point"]["selectors"]["random"] == {
        "mean_expected_positives": 1.0,
        "mean_yield": 0.5,
    }
    assert result["point"]["diagnosis_minus_event_expected_positives"] == 2.0
    assert result["point"]["diagnosis_minus_event_yield"] == 1.0
    assert all(
        row["selectors"]["diagnosis"]["queries"] == 2
        for row in result["per_policy"]
    )
    assert result["exact_paired_sign_flip"]["p_value"] == 0.5


def test_external_binary_analysis_fractionally_splits_score_ties():
    rows = external_binary_rows()
    for row in rows:
        if row["train_seed"] == 1:
            row["diagnosis_score"] = {
                10: 4.0,
                11: 3.0,
                12: 3.0,
                13: 1.0,
            }[row["episode_seed"]]

    result = external_binary_acquisition_analysis(
        rows,
        queries_per_policy=2,
        replicates=5,
        seed=3,
    )
    policy_one = result["per_policy"][0]["selectors"]["diagnosis"]

    assert policy_one["expected_positives"] == 1.5
    assert policy_one["yield"] == 0.75
    assert policy_one["score_boundary"] == 3.0
    assert policy_one["factual_zero_boundary"] is True
    assert policy_one["boundary_ties"] == 2


def test_external_binary_analysis_gives_every_selector_shared_failure_priority():
    rows = external_binary_rows()
    for row in rows:
        row["target_positive"] = False
        row["factual_zero"] = row["episode_seed"] in {10, 11}
        if row["episode_seed"] == 10:
            row["target_positive"] = True
        # A positive-return cell has the largest selector scores, but must not
        # outrank an observed zero-return failure for any selector.
        if row["episode_seed"] == 12:
            row["diagnosis_score"] = 100.0
            row["event_burden_score"] = 100.0
            row["raw_gap_score"] = 100.0

    result = external_binary_acquisition_analysis(
        rows,
        queries_per_policy=1,
        replicates=5,
        seed=3,
    )

    expected_by_selector = {
        "diagnosis": 1.0,
        "event": 0.0,
        "raw_gap": 1.0,
        "random": 0.5,
    }
    for policy in result["per_policy"]:
        for selector, expected in expected_by_selector.items():
            selected = policy["selectors"][selector]
            assert selected["factual_zero_boundary"] is True
            assert selected["expected_positives"] == pytest.approx(expected)


def test_external_binary_bootstraps_are_deterministic_and_recompute_two_way():
    first = external_binary_acquisition_analysis(
        external_binary_rows(),
        queries_per_policy=2,
        replicates=40,
        seed=11,
    )
    second = external_binary_acquisition_analysis(
        external_binary_rows(),
        queries_per_policy=2,
        replicates=40,
        seed=11,
    )

    assert first == second
    assert first["policy_cluster_bootstrap"]["valid_replicates"] == 40
    assert set(first["policy_cluster_bootstrap"]["estimates"]) == {1.0}
    assert (
        first["two_way_policy_start_pigeonhole_bootstrap"]["valid_replicates"]
        == 40
    )
    assert (
        len(set(first["two_way_policy_start_pigeonhole_bootstrap"]["estimates"]))
        > 1
    )
    assert len(first["leave_one_policy_out"]) == 2
    assert len(first["leave_one_start_out"]) == 4
    assert {
        row["omitted_episode_seed"] for row in first["leave_one_start_out"]
    } == {10, 11, 12, 13}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda rows: rows.pop(),
            "complete policy-by-start rectangle",
        ),
        (
            lambda rows: rows.append(dict(rows[0])),
            "duplicate external-outcome cell",
        ),
        (
            lambda rows: rows[0].update(diagnosis_score=float("inf")),
            "diagnosis scores must be finite",
        ),
        (
            lambda rows: rows[0].update(target_positive=1),
            "target_positive must be a boolean",
        ),
        (
            lambda rows: rows[0].update(factual_zero=1),
            "factual_zero must be a boolean",
        ),
        (
            lambda rows: rows[0].update(factual_zero=False),
            "target-positive cell must have factual_zero=True",
        ),
    ],
)
def test_external_binary_analysis_rejects_invalid_rows(mutator, message):
    rows = external_binary_rows()
    mutator(rows)

    with pytest.raises(ValueError, match=message):
        external_binary_acquisition_analysis(
            rows,
            queries_per_policy=2,
            replicates=5,
        )


@pytest.mark.parametrize("queries_per_policy", [0, 5, 1.5, True])
def test_external_binary_analysis_rejects_invalid_fixed_budget(queries_per_policy):
    with pytest.raises(ValueError, match="queries_per_policy"):
        external_binary_acquisition_analysis(
            external_binary_rows(),
            queries_per_policy=queries_per_policy,
            replicates=5,
        )


def specificity_rows():
    return [
        {
            "train_seed": 1,
            "episode_seed": 10,
            "candidate_kind": "swap",
            "primary_only_decision": "supported_by_primary_only",
            "full_decision": "ambiguous",
        },
        {
            "train_seed": 1,
            "episode_seed": 10,
            "candidate_kind": "initial",
            "primary_only_decision": "supported_by_primary_only",
            "full_decision": "supported",
        },
        {
            "train_seed": 2,
            "episode_seed": 10,
            "candidate_kind": "swap",
            "primary_only_decision": "supported_by_primary_only",
            "full_decision": "invalid",
        },
        {
            "train_seed": 2,
            "episode_seed": 11,
            "candidate_kind": "swap",
            "primary_only_decision": "no_primary_effect",
            "full_decision": "no_supported_limitation",
        },
    ]


def test_specificity_rate_excludes_invalid_and_primary_negative_candidates():
    result = specificity_downgrade_analysis(
        specificity_rows(),
        policy_ids=[1, 2],
        start_ids=[10, 11],
        replicates=40,
        seed=5,
    )

    assert result["point"]["evaluable_primary_positives"] == 2
    assert result["point"]["downgrades"] == 1
    assert result["point"]["downgrade_rate"] == 0.5
    assert result["point"]["policies_with_downgrade"] == [1]
    assert result["point"]["starts_with_downgrade"] == [10]


def test_specificity_bootstrap_is_deterministic_and_keeps_empty_replicates():
    first = specificity_downgrade_analysis(
        specificity_rows(),
        policy_ids=[1, 2],
        start_ids=[10, 11],
        replicates=100,
        seed=13,
    )
    second = specificity_downgrade_analysis(
        specificity_rows(),
        policy_ids=[1, 2],
        start_ids=[10, 11],
        replicates=100,
        seed=13,
    )

    assert first == second
    assert first["bootstrap"]["invalid_replicates"] > 0
    assert (
        first["bootstrap"]["valid_replicates"]
        + first["bootstrap"]["invalid_replicates"]
        == 100
    )
    assert (
        first["policy_only_bootstrap"]["valid_replicates"]
        + first["policy_only_bootstrap"]["invalid_replicates"]
        == 100
    )
    assert (
        first["start_only_bootstrap"]["valid_replicates"]
        + first["start_only_bootstrap"]["invalid_replicates"]
        == 100
    )
    assert len(first["leave_one_policy_out"]) == 2
    assert len(first["leave_one_start_out"]) == 2


def test_specificity_analysis_rejects_duplicate_candidate_keys():
    rows = specificity_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate specificity candidate"):
        specificity_downgrade_analysis(
            rows,
            policy_ids=[1, 2],
            start_ids=[10, 11],
            replicates=5,
        )


def test_specificity_analysis_rejects_unknown_decisions():
    rows = specificity_rows()
    rows[0] = {**rows[0], "full_decision": "unsupported"}

    with pytest.raises(ValueError, match="unknown full-suite decision"):
        specificity_downgrade_analysis(
            rows,
            policy_ids=[1, 2],
            start_ids=[10, 11],
            replicates=5,
        )
