from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_confirmation_training.py"
SPEC = importlib.util.spec_from_file_location("run_confirmation_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _frozen_fixture(tmp_path: Path, timesteps: int = 160) -> dict[str, Path]:
    repo = tmp_path / "repo"
    jaxaht = tmp_path / "jaxaht"
    checkpoint_root = repo / "checkpoints"
    checkpoint = checkpoint_root / "fixed/ego"
    _write(checkpoint / "weights", "fixed")
    _write(
        repo / "data/fixed_ego_experiment.json",
        json.dumps({"fixed_ego": {"checkpoint": "fixed/ego"}}),
    )
    for name in driver.TRAINING_HASH_INPUTS:
        path = driver._resolve_frozen_input(repo, jaxaht, name)
        if path.exists():
            continue
        content = f"frozen {name}\n"
        if name.endswith("_base_.yaml"):
            content += "NUM_ENVS: 2\n"
        if "algorithm/rotate/overcooked-v1" in name:
            content += "LAMBDA_1: 10\nLAMBDA_2: 1\n"
        if name.endswith("configs/task/overcooked-v1/cramped_room.yaml"):
            content += "ROLLOUT_LENGTH: 10\n"
        _write(path, content)

    frozen_hashes = {
        name: _digest(driver._resolve_frozen_input(repo, jaxaht, name))
        for name in driver.TRAINING_HASH_INPUTS
    }
    protocol = {
        "status": "frozen_before_training_and_outcome_access",
        "training": {
            "train_seeds": list(driver.EXPECTED_CONFIRMATION_SEEDS),
            "objective": driver.EXPECTED_OBJECTIVE,
            "lambda_1": 10.0,
            "lambda_2": 1.0,
            "timesteps_per_policy": timesteps,
            "retain_all_completed_policies": True,
            "replace_failed_seed_after_outcome_access": False,
        },
        "evaluation": {"expected_policy_count": 12},
        "provenance_freeze": {
            "aht_diagnostics_git_head": "repo-head",
            "jaxaht_git_head": "jax-head",
            "jaxaht_rotate_source_commit": "rotate-head",
            "jaxaht_working_tree_clean": True,
            "fixed_ego_checkpoint_tree_sha256": _tree_digest(checkpoint),
            "sha256": frozen_hashes,
        },
    }
    protocol_path = repo / "protocol.json"
    _write(protocol_path, json.dumps(protocol))
    return {
        "repo": repo,
        "jaxaht": jaxaht,
        "checkpoint_root": checkpoint_root,
        "protocol": protocol_path,
    }


def _fake_git(path: Path, *args: str) -> str:
    if args == ("rev-parse", "HEAD"):
        return "jax-head" if path.name == "jaxaht" else "repo-head"
    if args == ("status", "--porcelain"):
        return ""
    if args == (
        "log",
        "-1",
        "--format=%H",
        "--",
        "open_ended_training/rotate.py",
    ):
        return "rotate-head"
    raise AssertionError((path, args))


def _successful_runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
    output = Path(command[command.index("--output") + 1])
    seed = int(command[command.index("--train-seeds") + 1])
    timesteps = int(command[command.index("--training-timesteps") + 1])
    _write(output / f"seed_{seed}/trained_policies/weights", f"seed={seed}\n")
    summary = {
        "training_seeds": [seed],
        "implementation_reference": {
            "local_head": "jax-head",
            "rotate_source_commit": "rotate-head",
            "working_tree_clean": True,
        },
        "seed_results": [
            {
                "train_seed": seed,
                "effective_training_config": {
                    "CONF_OBJ_TYPE": driver.EXPECTED_OBJECTIVE,
                    "TIMESTEPS_PER_ITER_PARTNER": timesteps,
                    "LAMBDA_1": 10.0,
                    "LAMBDA_2": 1.0,
                },
            }
        ],
    }
    _write(output / "summary.json", json.dumps(summary))
    kwargs["stdout"].write(f"trained {seed}\n".encode())
    kwargs["stderr"].write(b"")
    return SimpleNamespace(returncode=0)


def test_validation_rejects_nondivisible_explicit_budget(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path, timesteps=161)
    with pytest.raises(driver.ConfirmationDriverError, match="not divisible"):
        driver.validate_protocol(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            git_reader=_fake_git,
        )


def test_validation_rejects_training_input_drift(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    _write(paths["repo"] / "src/aht_diagnostics/teamgen.py", "changed\n")
    with pytest.raises(driver.ConfirmationDriverError, match="input drifted"):
        driver.validate_protocol(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            git_reader=_fake_git,
        )


def test_validation_rejects_protocol_digest_drift(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    expected = _digest(paths["protocol"])
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    protocol["training"]["timesteps_per_policy"] = 320
    _write(paths["protocol"], json.dumps(protocol))

    with pytest.raises(driver.ConfirmationDriverError, match="protocol digest drifted"):
        driver.validate_protocol(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=expected,
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            git_reader=_fake_git,
        )


def test_runs_each_seed_once_and_resumes_only_verified_outputs(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    output = tmp_path / "confirmation"
    manifest = driver.run_confirmation(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_digest(paths["protocol"]),
        repo_root=paths["repo"],
        jaxaht=paths["jaxaht"],
        checkpoint_root=paths["checkpoint_root"],
        output=output,
        python=Path("/fake/python"),
        evaluation_episodes=20,
        git_reader=_fake_git,
        process_runner=_successful_runner,
    )
    assert manifest["status"] == "completed"
    assert set(manifest["completed_seeds"]) == {
        str(seed) for seed in driver.EXPECTED_CONFIRMATION_SEEDS
    }
    assert len(manifest["attempts"]) == len(driver.EXPECTED_CONFIRMATION_SEEDS)
    aggregate = output / "per_state"
    aggregate_summary = json.loads((aggregate / "summary.json").read_text())
    assert aggregate_summary["training_seeds"] == list(
        driver.EXPECTED_CONFIRMATION_SEEDS
    )
    assert [row["train_seed"] for row in aggregate_summary["seed_results"]] == list(
        driver.EXPECTED_CONFIRMATION_SEEDS
    )
    assert aggregate_summary["implementation_reference"] == {
        "local_jaxaht": str(paths["jaxaht"].resolve()),
        "local_head": "jax-head",
        "rotate_source_commit": "rotate-head",
        "working_tree_clean": True,
    }
    for seed in driver.EXPECTED_CONFIRMATION_SEEDS:
        completion = output / "completed" / f"seed_{seed}"
        assert (completion / "checkpoint/trained_policies/weights").is_file()
        assert (completion / "seed_result.json").is_file()
        assert (completion / "completion.json").is_file()
        assert (output / f"attempts/seed_{seed}/attempt_1/stdout.log").is_file()
        assert (output / f"attempts/seed_{seed}/attempt_1/stderr.log").is_file()
        assert (output / f"attempts/seed_{seed}/attempt_1/attempt.json").is_file()
        aggregate_seed = aggregate / f"seed_{seed}"
        assert aggregate_seed.is_symlink()
        assert aggregate_seed.resolve() == (completion / "checkpoint").resolve()
        assert (aggregate_seed / "trained_policies").is_dir()
        assert driver.directory_digest(aggregate_seed) == json.loads(
            (completion / "completion.json").read_text()
        )["checkpoint_tree_sha256"]

    def should_not_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("verified completed seeds must be skipped on resume")

    resumed = driver.run_confirmation(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_digest(paths["protocol"]),
        repo_root=paths["repo"],
        jaxaht=paths["jaxaht"],
        checkpoint_root=paths["checkpoint_root"],
        output=output,
        python=Path("/fake/python"),
        evaluation_episodes=20,
        git_reader=_fake_git,
        process_runner=should_not_run,
    )
    assert resumed == manifest


def test_resume_refuses_tampered_completed_checkpoint(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    output = tmp_path / "confirmation"
    driver.run_confirmation(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_digest(paths["protocol"]),
        repo_root=paths["repo"],
        jaxaht=paths["jaxaht"],
        checkpoint_root=paths["checkpoint_root"],
        output=output,
        python=Path("/fake/python"),
        evaluation_episodes=20,
        git_reader=_fake_git,
        process_runner=_successful_runner,
    )
    seed = driver.EXPECTED_CONFIRMATION_SEEDS[0]
    _write(
        output / f"completed/seed_{seed}/checkpoint/trained_policies/weights",
        "tampered\n",
    )
    with pytest.raises(driver.ConfirmationDriverError, match="Checkpoint hash mismatch"):
        driver.run_confirmation(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            output=output,
            python=Path("/fake/python"),
            evaluation_episodes=20,
            git_reader=_fake_git,
            process_runner=_successful_runner,
        )


def test_resume_refuses_tampered_exporter_aggregate(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    output = tmp_path / "confirmation"
    driver.run_confirmation(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_digest(paths["protocol"]),
        repo_root=paths["repo"],
        jaxaht=paths["jaxaht"],
        checkpoint_root=paths["checkpoint_root"],
        output=output,
        python=Path("/fake/python"),
        evaluation_episodes=20,
        git_reader=_fake_git,
        process_runner=_successful_runner,
    )
    summary_path = output / "per_state/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["training_seeds"] = summary["training_seeds"][:-1]
    _write(summary_path, json.dumps(summary))
    with pytest.raises(driver.ConfirmationDriverError, match="wrong training seeds"):
        driver.run_confirmation(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            output=output,
            python=Path("/fake/python"),
            evaluation_episodes=20,
            git_reader=_fake_git,
            process_runner=_successful_runner,
        )


def test_refuses_nonempty_output_without_manifest(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    output = tmp_path / "confirmation"
    _write(output / "unrelated.txt", "do not overwrite\n")
    with pytest.raises(driver.ConfirmationDriverError, match="non-empty output"):
        driver.run_confirmation(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            output=output,
            python=Path("/fake/python"),
            evaluation_episodes=20,
            git_reader=_fake_git,
            process_runner=_successful_runner,
        )


def test_failed_attempt_is_preserved_and_retry_uses_new_attempt(tmp_path: Path) -> None:
    paths = _frozen_fixture(tmp_path)
    output = tmp_path / "confirmation"

    def failed_runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
        kwargs["stdout"].write(b"partial output\n")
        kwargs["stderr"].write(b"infrastructure failure\n")
        return SimpleNamespace(returncode=9)

    with pytest.raises(driver.ConfirmationDriverError, match="exit code 9"):
        driver.run_confirmation(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_digest(paths["protocol"]),
            repo_root=paths["repo"],
            jaxaht=paths["jaxaht"],
            checkpoint_root=paths["checkpoint_root"],
            output=output,
            python=Path("/fake/python"),
            evaluation_episodes=20,
            git_reader=_fake_git,
            process_runner=failed_runner,
        )
    first_seed = driver.EXPECTED_CONFIRMATION_SEEDS[0]
    first_attempt = output / f"attempts/seed_{first_seed}/attempt_1"
    assert json.loads((first_attempt / "attempt.json").read_text())["status"] == "failed"
    assert (first_attempt / "stdout.log").read_text() == "partial output\n"
    assert (first_attempt / "stderr.log").read_text() == "infrastructure failure\n"

    manifest = driver.run_confirmation(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_digest(paths["protocol"]),
        repo_root=paths["repo"],
        jaxaht=paths["jaxaht"],
        checkpoint_root=paths["checkpoint_root"],
        output=output,
        python=Path("/fake/python"),
        evaluation_episodes=20,
        git_reader=_fake_git,
        process_runner=_successful_runner,
    )
    assert manifest["status"] == "completed"
    assert (output / f"attempts/seed_{first_seed}/attempt_2/attempt.json").is_file()
    assert len(manifest["attempts"]) == len(driver.EXPECTED_CONFIRMATION_SEEDS) + 1
