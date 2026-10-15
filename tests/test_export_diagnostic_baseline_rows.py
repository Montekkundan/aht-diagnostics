from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_diagnostic_baseline_rows as exporter


PROJECT_HEAD = "1" * 40
JAXAHT_HEAD = "2" * 40
ROTATE_HEAD = "3" * 40


def test_layout_resolution_preserves_legacy_checkpoint_and_requires_explicit_transfer(
    tmp_path: Path,
) -> None:
    checkpoint_index = {"fixed_ego": {"checkpoint": "legacy/ego"}}
    layout, checkpoint = exporter._resolve_environment_and_checkpoint(
        specification={},
        source_summary={},
        checkpoint_index=checkpoint_index,
        checkpoint_root=tmp_path,
    )
    assert layout == "cramped_room"
    assert checkpoint == (tmp_path / "legacy/ego").resolve()

    with pytest.raises(ValueError, match="source summary environment"):
        exporter._resolve_environment_and_checkpoint(
            specification={"environment": "overcooked-v1/counter_circuit"},
            source_summary={},
            checkpoint_index=checkpoint_index,
            checkpoint_root=tmp_path,
        )
    with pytest.raises(ValueError, match="fixed_ego_checkpoint"):
        exporter._resolve_environment_and_checkpoint(
            specification={"environment": "overcooked-v1/counter_circuit"},
            source_summary={"environment": "overcooked-v1/counter_circuit"},
            checkpoint_index=checkpoint_index,
            checkpoint_root=tmp_path,
        )

    layout, checkpoint = exporter._resolve_environment_and_checkpoint(
        specification={
            "environment": "overcooked-v1/counter_circuit",
            "fixed_ego_checkpoint": "counter/ego",
        },
        source_summary={"environment": "overcooked-v1/counter_circuit"},
        checkpoint_index=checkpoint_index,
        checkpoint_root=tmp_path,
    )
    assert layout == "counter_circuit"
    assert checkpoint == (tmp_path / "counter/ego").resolve()


def test_layout_resolution_rejects_source_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        exporter._resolve_environment_and_checkpoint(
            specification={
                "environment": "overcooked-v1/counter_circuit",
                "fixed_ego_checkpoint": "counter/ego",
            },
            source_summary={"environment": "overcooked-v1/cramped_room"},
            checkpoint_index={},
            checkpoint_root=tmp_path,
        )


def _confirmation_specification() -> dict[str, object]:
    return {
        "status": "prospective_diagnostic_baseline_confirmation_rows",
        "source_results": "results/confirmation",
        "episode_seed_range": {"start": 20, "end_exclusive": 23},
        "expected_episode_seed_range": {"start": 20, "end_exclusive": 23},
        "expected_episode_count": 3,
        "expected_train_seeds": [1, 2],
        "project_head": PROJECT_HEAD,
        "expected_jaxaht_provenance": {
            "local_head": JAXAHT_HEAD,
            "rotate_source_commit": ROTATE_HEAD,
            "working_tree_clean": True,
        },
    }


def _source_summary(jaxaht_path: Path) -> dict[str, object]:
    return {
        "training_seeds": [1, 2],
        "seed_results": [{"train_seed": 1}, {"train_seed": 2}],
        "implementation_reference": {
            "local_jaxaht": str(jaxaht_path.resolve()),
            "local_head": JAXAHT_HEAD,
            "rotate_source_commit": ROTATE_HEAD,
            "working_tree_clean": True,
        },
    }


def _make_policy_directories(source_results: Path, seeds: list[int]) -> None:
    for seed in seeds:
        policy_dir = source_results / f"seed_{seed}" / "trained_policies"
        policy_dir.mkdir(parents=True)
        (policy_dir / "weights.bin").write_bytes(b"weights")


def _mock_git_revision(monkeypatch: pytest.MonkeyPatch, jaxaht_path: Path) -> None:
    def fake_git_revision(repository: Path, *arguments: str) -> str:
        if repository == exporter.REPO_ROOT:
            assert arguments == ("rev-parse", "HEAD")
            return PROJECT_HEAD
        assert repository == jaxaht_path.resolve()
        if arguments == ("rev-parse", "HEAD"):
            return JAXAHT_HEAD
        if arguments == (
            "log",
            "-1",
            "--format=%H",
            "--",
            "open_ended_training/rotate.py",
        ):
            return ROTATE_HEAD
        if arguments == ("status", "--porcelain"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(exporter, "_git_revision", fake_git_revision)


def test_confirmation_validation_accepts_exact_frozen_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_results = tmp_path / "results"
    jaxaht_path = tmp_path / "jaxaht"
    jaxaht_path.mkdir()
    _make_policy_directories(source_results, [1, 2])
    _mock_git_revision(monkeypatch, jaxaht_path)

    validated = exporter._validate_confirmation_inputs(
        specification=_confirmation_specification(),
        config_path=tmp_path / "confirmation_rows.json",
        source_results=source_results,
        source_summary=_source_summary(jaxaht_path),
        jaxaht_path=jaxaht_path.resolve(),
    )

    assert validated == {
        "project_head": PROJECT_HEAD,
        "jaxaht": {
            "local_head": JAXAHT_HEAD,
            "rotate_source_commit": ROTATE_HEAD,
            "working_tree_clean": True,
        },
        "expected_train_seeds": [1, 2],
    }


@pytest.mark.parametrize(
    ("summary_seeds", "policy_seeds", "message"),
    [
        ([1], [1, 2], "source summary training_seeds"),
        ([1, 2, 3], [1, 2], "source summary training_seeds"),
        ([1, 2], [1], "saved policy directories"),
        ([1, 2], [1, 2, 3], "saved policy directories"),
    ],
)
def test_confirmation_validation_rejects_missing_or_extra_policies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_seeds: list[int],
    policy_seeds: list[int],
    message: str,
) -> None:
    source_results = tmp_path / "results"
    jaxaht_path = tmp_path / "jaxaht"
    jaxaht_path.mkdir()
    _make_policy_directories(source_results, policy_seeds)
    _mock_git_revision(monkeypatch, jaxaht_path)
    source_summary = _source_summary(jaxaht_path)
    source_summary["training_seeds"] = summary_seeds

    with pytest.raises(ValueError, match=message):
        exporter._validate_confirmation_inputs(
            specification=_confirmation_specification(),
            config_path=tmp_path / "confirmation_rows.json",
            source_results=source_results,
            source_summary=source_summary,
            jaxaht_path=jaxaht_path.resolve(),
        )


def test_confirmation_validation_requires_project_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_results = tmp_path / "results"
    jaxaht_path = tmp_path / "jaxaht"
    jaxaht_path.mkdir()
    _make_policy_directories(source_results, [1, 2])
    _mock_git_revision(monkeypatch, jaxaht_path)
    specification = _confirmation_specification()
    specification["project_head"] = None

    with pytest.raises(ValueError, match="project_head must be a non-empty string"):
        exporter._validate_confirmation_inputs(
            specification=specification,
            config_path=tmp_path / "confirmation_rows.json",
            source_results=source_results,
            source_summary=_source_summary(jaxaht_path),
            jaxaht_path=jaxaht_path.resolve(),
        )


def test_confirmation_validation_rejects_dirty_jaxaht_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_results = tmp_path / "results"
    jaxaht_path = tmp_path / "jaxaht"
    jaxaht_path.mkdir()
    _make_policy_directories(source_results, [1, 2])
    _mock_git_revision(monkeypatch, jaxaht_path)
    source_summary = _source_summary(jaxaht_path)
    source_summary["implementation_reference"]["working_tree_clean"] = False

    with pytest.raises(ValueError, match="working_tree_clean"):
        exporter._validate_confirmation_inputs(
            specification=_confirmation_specification(),
            config_path=tmp_path / "confirmation_rows.json",
            source_results=source_results,
            source_summary=source_summary,
            jaxaht_path=jaxaht_path.resolve(),
        )


def test_confirmation_validation_enforces_optional_episode_count() -> None:
    specification = _confirmation_specification()
    specification["expected_episode_count"] = 4

    with pytest.raises(ValueError, match="expected_episode_count"):
        exporter._validate_expected_episode_design(
            specification,
            specification["episode_seed_range"],
        )


def test_main_refuses_incomplete_confirmation_before_jaxaht_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_results = tmp_path / "source_results"
    source_results.mkdir()
    (source_results / "summary.json").write_text(
        json.dumps({"training_seeds": [1], "seed_results": [{"train_seed": 1}]}),
        encoding="utf-8",
    )
    config = _confirmation_specification()
    config["source_results"] = str(source_results)
    config_path = tmp_path / "confirmation_rows.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint_index_path = tmp_path / "checkpoints.json"
    checkpoint_index_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        exporter,
        "parse_args",
        lambda: SimpleNamespace(
            jaxaht=tmp_path / "jaxaht",
            config=config_path,
            output=tmp_path / "output.json",
            checkpoint_root=tmp_path / "checkpoints",
            checkpoint_index=checkpoint_index_path,
        ),
    )
    monkeypatch.setattr(
        exporter,
        "activate_jaxaht",
        lambda *_: pytest.fail("JaxAHT activated before confirmation validation"),
    )

    with pytest.raises(ValueError, match="source summary training_seeds"):
        exporter.main()
