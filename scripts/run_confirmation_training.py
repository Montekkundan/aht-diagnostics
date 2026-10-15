#!/usr/bin/env python3
"""Run the frozen confirmation training rectangle one verified seed at a time.

This driver is intentionally conservative.  It validates the machine-readable
confirmation protocol before starting any expensive work, writes every attempt
to a new directory, and only records a seed as complete after its checkpoint
tree and canonical seed result have been hashed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OBJECTIVE = "gae_ps_regret_ts_weighted"
RUNNER_OBJECTIVE = "per_state_regret"
ALLOWED_PREOUTCOME_STATUSES = {"frozen_before_training_and_outcome_access"}
TRAINING_HASH_INPUTS = (
    "data/fixed_ego_experiment.json",
    "scripts/run_confirmation_training.py",
    "scripts/run_fixed_ego_teamgen.py",
    "src/aht_diagnostics/__init__.py",
    "src/aht_diagnostics/behavior_penalty.py",
    "src/aht_diagnostics/defects.py",
    "src/aht_diagnostics/jaxaht_bridge.py",
    "src/aht_diagnostics/rollout.py",
    "src/aht_diagnostics/semantics.py",
    "src/aht_diagnostics/teamgen.py",
    "jaxaht/open_ended_training/rotate.py",
    "jaxaht/open_ended_training/configs/algorithm/rotate/_base_.yaml",
    "jaxaht/open_ended_training/configs/algorithm/rotate/overcooked-v1/cramped_room.yaml",
    "jaxaht/open_ended_training/configs/task/overcooked-v1/cramped_room.yaml",
)


class ConfirmationDriverError(RuntimeError):
    """Raised when a frozen confirmation invariant is not satisfied."""


@dataclass(frozen=True)
class ValidatedRun:
    protocol_sha256: str
    seeds: tuple[int, ...]
    timesteps: int
    update_size: int
    checkpoint: Path
    checkpoint_sha256: str
    repo_head: str
    jaxaht_head: str
    rotate_source_commit: str
    lambda_1: float
    lambda_2: float


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path) -> str:
    if not path.is_dir():
        raise ConfirmationDriverError(f"Expected directory: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ConfirmationDriverError(f"Refusing to hash empty directory: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_output(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationDriverError(f"Cannot read JSON {path}: {exc}") from exc


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfirmationDriverError(f"{label} must be a JSON object")
    return value


def _yaml_numeric_scalar(path: Path, key: str) -> float:
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:#.*)?$"
    )
    matches = [
        float(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line))
    ]
    if len(matches) != 1:
        raise ConfirmationDriverError(
            f"Expected exactly one numeric {key} in {path}, found {len(matches)}"
        )
    return matches[0]


def _resolve_frozen_input(repo_root: Path, jaxaht: Path, name: str) -> Path:
    prefix = "jaxaht/"
    return jaxaht / name[len(prefix) :] if name.startswith(prefix) else repo_root / name


def validate_protocol(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    repo_root: Path,
    jaxaht: Path,
    checkpoint_root: Path,
    git_reader: Callable[..., str] = git_output,
) -> tuple[dict[str, Any], ValidatedRun]:
    protocol_path = protocol_path.resolve()
    repo_root = repo_root.resolve()
    jaxaht = jaxaht.resolve()
    checkpoint_root = checkpoint_root.resolve()
    actual_protocol_sha256 = file_digest(protocol_path)
    if actual_protocol_sha256 != expected_protocol_sha256:
        raise ConfirmationDriverError(
            "Confirmation protocol digest drifted: "
            f"{actual_protocol_sha256} != {expected_protocol_sha256}"
        )
    protocol = _require_mapping(_load_json(protocol_path), "protocol")
    if protocol.get("status") not in ALLOWED_PREOUTCOME_STATUSES:
        raise ConfirmationDriverError("Protocol is not marked frozen before outcome access")
    training = _require_mapping(protocol.get("training"), "protocol.training")
    evaluation = _require_mapping(protocol.get("evaluation"), "protocol.evaluation")
    provenance = _require_mapping(
        protocol.get("provenance_freeze"), "protocol.provenance_freeze"
    )
    frozen_hashes = _require_mapping(provenance.get("sha256"), "provenance sha256")

    seeds_value = training.get("train_seeds")
    if not isinstance(seeds_value, list) or not seeds_value or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_value
    ):
        raise ConfirmationDriverError("training.train_seeds must be a non-empty integer list")
    seeds = tuple(seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ConfirmationDriverError("training.train_seeds must be unique")
    if evaluation.get("expected_policy_count") != len(seeds):
        raise ConfirmationDriverError("expected_policy_count does not match exact seeds")
    if training.get("objective") != EXPECTED_OBJECTIVE:
        raise ConfirmationDriverError(
            f"Expected objective {EXPECTED_OBJECTIVE!r}, got {training.get('objective')!r}"
        )
    if training.get("retain_all_completed_policies") is not True:
        raise ConfirmationDriverError("Protocol must retain every completed policy")
    if training.get("replace_failed_seed_after_outcome_access") is not False:
        raise ConfirmationDriverError("Protocol must forbid outcome-based seed replacement")

    timesteps = training.get("timesteps_per_policy")
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
        raise ConfirmationDriverError(
            "training.timesteps_per_policy must be an explicit positive integer"
        )
    lambda_1 = float(training.get("lambda_1"))
    lambda_2 = float(training.get("lambda_2"))

    repo_head = git_reader(repo_root, "rev-parse", "HEAD")
    if repo_head != provenance.get("aht_diagnostics_git_head"):
        raise ConfirmationDriverError(
            f"aht-diagnostics HEAD drifted: {repo_head} != {provenance.get('aht_diagnostics_git_head')}"
        )
    jaxaht_head = git_reader(jaxaht, "rev-parse", "HEAD")
    if jaxaht_head != provenance.get("jaxaht_git_head"):
        raise ConfirmationDriverError(
            f"JaxAHT HEAD drifted: {jaxaht_head} != {provenance.get('jaxaht_git_head')}"
        )
    if git_reader(jaxaht, "status", "--porcelain"):
        raise ConfirmationDriverError("JaxAHT working tree must be clean")
    if provenance.get("jaxaht_working_tree_clean") is not True:
        raise ConfirmationDriverError("Protocol does not require a clean JaxAHT tree")
    rotate_source_commit = git_reader(
        jaxaht,
        "log",
        "-1",
        "--format=%H",
        "--",
        "open_ended_training/rotate.py",
    )
    if rotate_source_commit != provenance.get("jaxaht_rotate_source_commit"):
        raise ConfirmationDriverError(
            "JaxAHT ROTATE source commit does not match the frozen protocol"
        )

    for name in TRAINING_HASH_INPUTS:
        expected = frozen_hashes.get(name)
        if not isinstance(expected, str):
            raise ConfirmationDriverError(f"Protocol is missing training input hash: {name}")
        path = _resolve_frozen_input(repo_root, jaxaht, name)
        if not path.is_file():
            raise ConfirmationDriverError(f"Frozen training input is missing: {path}")
        actual = file_digest(path)
        if actual != expected:
            raise ConfirmationDriverError(
                f"Frozen training input drifted: {name}: {actual} != {expected}"
            )

    fixed_experiment = _require_mapping(
        _load_json(repo_root / "data" / "fixed_ego_experiment.json"),
        "fixed ego experiment",
    )
    fixed_ego = _require_mapping(fixed_experiment.get("fixed_ego"), "fixed_ego")
    checkpoint_name = fixed_ego.get("checkpoint")
    if not isinstance(checkpoint_name, str):
        raise ConfirmationDriverError("fixed_ego.checkpoint must be a string")
    checkpoint = checkpoint_root / checkpoint_name
    checkpoint_sha256 = directory_digest(checkpoint)
    if checkpoint_sha256 != provenance.get("fixed_ego_checkpoint_tree_sha256"):
        raise ConfirmationDriverError("Fixed ego checkpoint tree does not match protocol")

    algorithm_base = jaxaht / "open_ended_training/configs/algorithm/rotate/_base_.yaml"
    task_algorithm = (
        jaxaht
        / "open_ended_training/configs/algorithm/rotate/overcooked-v1/cramped_room.yaml"
    )
    task = jaxaht / "open_ended_training/configs/task/overcooked-v1/cramped_room.yaml"
    num_envs = int(_yaml_numeric_scalar(algorithm_base, "NUM_ENVS"))
    rollout_length = int(_yaml_numeric_scalar(task, "ROLLOUT_LENGTH"))
    configured_lambda_1 = _yaml_numeric_scalar(task_algorithm, "LAMBDA_1")
    configured_lambda_2 = _yaml_numeric_scalar(task_algorithm, "LAMBDA_2")
    if configured_lambda_1 != lambda_1 or configured_lambda_2 != lambda_2:
        raise ConfirmationDriverError(
            "Protocol lambda weights do not match the frozen JaxAHT layout config"
        )
    update_size = rollout_length * 4 * num_envs
    if timesteps < update_size or timesteps % update_size:
        raise ConfirmationDriverError(
            "Explicit training budget is not divisible by the runner update size: "
            f"{timesteps} % {update_size} = {timesteps % update_size}"
        )

    return protocol, ValidatedRun(
        protocol_sha256=actual_protocol_sha256,
        seeds=seeds,
        timesteps=timesteps,
        update_size=update_size,
        checkpoint=checkpoint.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        repo_head=repo_head,
        jaxaht_head=jaxaht_head,
        rotate_source_commit=rotate_source_commit,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
    )


def _manifest_identity(validated: ValidatedRun) -> dict[str, Any]:
    return {
        "protocol_sha256": validated.protocol_sha256,
        "seeds": list(validated.seeds),
        "timesteps_per_policy": validated.timesteps,
        "update_size": validated.update_size,
        "fixed_ego_checkpoint_tree_sha256": validated.checkpoint_sha256,
        "aht_diagnostics_git_head": validated.repo_head,
        "jaxaht_git_head": validated.jaxaht_head,
        "jaxaht_rotate_source_commit": validated.rotate_source_commit,
        "lambda_1": validated.lambda_1,
        "lambda_2": validated.lambda_2,
    }


def _new_manifest(validated: ValidatedRun) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "identity": _manifest_identity(validated),
        "completed_seeds": {},
        "attempts": [],
    }


def _load_or_create_manifest(output: Path, validated: ValidatedRun) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "confirmation_training_manifest.json"
    if manifest_path.exists():
        manifest = _require_mapping(_load_json(manifest_path), "training manifest")
        if manifest.get("identity") != _manifest_identity(validated):
            raise ConfirmationDriverError(
                "Existing manifest belongs to a different frozen confirmation run"
            )
        if not isinstance(manifest.get("completed_seeds"), dict):
            raise ConfirmationDriverError("Manifest completed_seeds is invalid")
        if not isinstance(manifest.get("attempts"), list):
            raise ConfirmationDriverError("Manifest attempts is invalid")
        return manifest

    allowed = {".confirmation_training.lock"}
    unexpected = [path for path in output.iterdir() if path.name not in allowed]
    if unexpected:
        raise ConfirmationDriverError(
            "Refusing to use a non-empty output without a manifest: "
            + ", ".join(sorted(path.name for path in unexpected))
        )
    manifest = _new_manifest(validated)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _seed_completion_dir(output: Path, seed: int) -> Path:
    return output / "completed" / f"seed_{seed}"


def _validate_seed_result(result: Any, seed: int, validated: ValidatedRun) -> dict[str, Any]:
    result = _require_mapping(result, f"seed {seed} result")
    if result.get("train_seed") != seed:
        raise ConfirmationDriverError(f"Seed result mismatch for seed {seed}")
    config = _require_mapping(result.get("effective_training_config"), "effective config")
    expected = {
        "CONF_OBJ_TYPE": EXPECTED_OBJECTIVE,
        "TIMESTEPS_PER_ITER_PARTNER": validated.timesteps,
        "LAMBDA_1": validated.lambda_1,
        "LAMBDA_2": validated.lambda_2,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ConfirmationDriverError(
                f"Seed {seed} effective {key}={config.get(key)!r}, expected {expected_value!r}"
            )
    return result


def _read_completion(output: Path, seed: int, validated: ValidatedRun) -> dict[str, Any] | None:
    completion_dir = _seed_completion_dir(output, seed)
    if not completion_dir.exists():
        return None
    checkpoint = completion_dir / "checkpoint"
    result_path = completion_dir / "seed_result.json"
    record_path = completion_dir / "completion.json"
    if not (checkpoint.is_dir() and result_path.is_file() and record_path.is_file()):
        raise ConfirmationDriverError(f"Incomplete promoted output for seed {seed}")
    record = _require_mapping(_load_json(record_path), f"seed {seed} completion")
    _validate_seed_result(_load_json(result_path), seed, validated)
    checkpoint_sha256 = directory_digest(checkpoint)
    result_sha256 = file_digest(result_path)
    if record.get("train_seed") != seed:
        raise ConfirmationDriverError(f"Completion seed mismatch for {seed}")
    if record.get("checkpoint_tree_sha256") != checkpoint_sha256:
        raise ConfirmationDriverError(f"Checkpoint hash mismatch for completed seed {seed}")
    if record.get("seed_result_sha256") != result_sha256:
        raise ConfirmationDriverError(f"Result hash mismatch for completed seed {seed}")
    return record


def _reconcile_completed(
    output: Path, manifest: dict[str, Any], validated: ValidatedRun
) -> tuple[dict[str, Any], bool]:
    completed = manifest["completed_seeds"]
    changed = False
    for seed in validated.seeds:
        record = _read_completion(output, seed, validated)
        stored = completed.get(str(seed))
        if record is None and stored is not None:
            raise ConfirmationDriverError(
                f"Manifest claims seed {seed} is complete but its verified output is missing"
            )
        if record is not None:
            if stored is not None and stored != record:
                raise ConfirmationDriverError(
                    f"Manifest completion record drifted for seed {seed}"
                )
            if stored is None:
                completed[str(seed)] = record
                changed = True
            attempt_path = output / str(record["attempt"]) / "attempt.json"
            if not attempt_path.is_file():
                raise ConfirmationDriverError(
                    f"Completed seed {seed} is missing its attempt metadata"
                )
            attempt = _require_mapping(
                _load_json(attempt_path), f"seed {seed} attempt metadata"
            )
            if attempt.get("train_seed") != seed:
                raise ConfirmationDriverError(
                    f"Attempt metadata seed mismatch for completed seed {seed}"
                )
            if attempt.get("status") != "completed":
                attempt.update(
                    {
                        "status": "completed",
                        "recovered_after_interruption_at": utc_now(),
                        "checkpoint_tree_sha256": record["checkpoint_tree_sha256"],
                        "seed_result_sha256": record["seed_result_sha256"],
                    }
                )
                atomic_write_json(attempt_path, attempt)
                changed = True
            if not any(
                item.get("attempt") == attempt.get("attempt")
                for item in manifest["attempts"]
                if isinstance(item, dict)
            ):
                manifest["attempts"].append(attempt)
                changed = True
    return manifest, changed


def _attempt_number(output: Path, seed: int) -> int:
    parent = output / "attempts" / f"seed_{seed}"
    if not parent.exists():
        return 1
    values: list[int] = []
    for path in parent.glob("attempt_*"):
        try:
            values.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    return max(values, default=0) + 1


def _promote_seed(
    *,
    output: Path,
    attempt_output: Path,
    seed: int,
    result: dict[str, Any],
    attempt_relative: str,
) -> dict[str, Any]:
    source_checkpoint = attempt_output / f"seed_{seed}"
    if not source_checkpoint.is_dir():
        raise ConfirmationDriverError(
            f"Runner did not create the checkpoint tree for seed {seed}"
        )
    target = _seed_completion_dir(output, seed)
    if target.exists():
        raise ConfirmationDriverError(f"Refusing to overwrite completed seed {seed}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = output / f".seed_{seed}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        shutil.move(str(source_checkpoint), str(staging / "checkpoint"))
        result_path = staging / "seed_result.json"
        atomic_write_json(result_path, result)
        record = {
            "train_seed": seed,
            "completed_at": utc_now(),
            "attempt": attempt_relative,
            "checkpoint_tree_sha256": directory_digest(staging / "checkpoint"),
            "seed_result_sha256": file_digest(result_path),
        }
        atomic_write_json(staging / "completion.json", record)
        os.replace(staging, target)
        return record
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _runner_command(
    *,
    python: Path,
    repo_root: Path,
    jaxaht: Path,
    checkpoint_root: Path,
    experiment: Path,
    attempt_output: Path,
    seed: int,
    validated: ValidatedRun,
    evaluation_episodes: int,
) -> list[str]:
    return [
        str(python),
        str(repo_root / "scripts" / "run_fixed_ego_teamgen.py"),
        "--jaxaht",
        str(jaxaht),
        "--checkpoint-root",
        str(checkpoint_root),
        "--experiment",
        str(experiment),
        "--output",
        str(attempt_output),
        "--objective",
        RUNNER_OBJECTIVE,
        "--mode",
        "preliminary",
        "--train-seeds",
        str(seed),
        "--evaluation-episodes",
        str(evaluation_episodes),
        "--training-timesteps",
        str(validated.timesteps),
    ]


def _run_seed(
    *,
    output: Path,
    repo_root: Path,
    jaxaht: Path,
    checkpoint_root: Path,
    experiment: Path,
    python: Path,
    seed: int,
    validated: ValidatedRun,
    evaluation_episodes: int,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> tuple[dict[str, Any], dict[str, Any]]:
    number = _attempt_number(output, seed)
    attempt_dir = output / "attempts" / f"seed_{seed}" / f"attempt_{number}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt_output = attempt_dir / "runner_output"
    command = _runner_command(
        python=python,
        repo_root=repo_root,
        jaxaht=jaxaht,
        checkpoint_root=checkpoint_root,
        experiment=experiment,
        attempt_output=attempt_output,
        seed=seed,
        validated=validated,
        evaluation_episodes=evaluation_episodes,
    )
    attempt_relative = str(attempt_dir.relative_to(output))
    metadata: dict[str, Any] = {
        "train_seed": seed,
        "attempt_number": number,
        "attempt": attempt_relative,
        "status": "running",
        "started_at": utc_now(),
        "command": command,
        "stdout": str((attempt_dir / "stdout.log").relative_to(output)),
        "stderr": str((attempt_dir / "stderr.log").relative_to(output)),
    }
    atomic_write_json(attempt_dir / "attempt.json", metadata)
    try:
        with (attempt_dir / "stdout.log").open("xb") as stdout, (
            attempt_dir / "stderr.log"
        ).open("xb") as stderr:
            completed = process_runner(
                command,
                cwd=repo_root,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        metadata["returncode"] = int(completed.returncode)
        metadata["finished_at"] = utc_now()
        if completed.returncode:
            metadata["status"] = "failed"
            atomic_write_json(attempt_dir / "attempt.json", metadata)
            raise ConfirmationDriverError(
                f"Training seed {seed} failed with exit code {completed.returncode}; "
                f"see {attempt_dir}"
            )

        summary_path = attempt_output / "summary.json"
        summary = _require_mapping(_load_json(summary_path), f"seed {seed} summary")
        if summary.get("training_seeds") != [seed]:
            raise ConfirmationDriverError(f"Runner summary has wrong seed for {seed}")
        results = summary.get("seed_results")
        if not isinstance(results, list) or len(results) != 1:
            raise ConfirmationDriverError("Runner summary must contain one seed result")
        result = _validate_seed_result(results[0], seed, validated)
        reference = _require_mapping(
            summary.get("implementation_reference"), "implementation reference"
        )
        if reference.get("local_head") != validated.jaxaht_head:
            raise ConfirmationDriverError("Runner used a different JaxAHT HEAD")
        if reference.get("rotate_source_commit") != validated.rotate_source_commit:
            raise ConfirmationDriverError("Runner used a different ROTATE source commit")
        if reference.get("working_tree_clean") is not True:
            raise ConfirmationDriverError("Runner observed a dirty JaxAHT tree")

        record = _promote_seed(
            output=output,
            attempt_output=attempt_output,
            seed=seed,
            result=result,
            attempt_relative=attempt_relative,
        )
        metadata.update(
            {
                "status": "completed",
                "checkpoint_tree_sha256": record["checkpoint_tree_sha256"],
                "seed_result_sha256": record["seed_result_sha256"],
            }
        )
        atomic_write_json(attempt_dir / "attempt.json", metadata)
        return record, metadata
    except BaseException as exc:
        if metadata.get("status") == "running":
            metadata.update(
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "finished_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            atomic_write_json(attempt_dir / "attempt.json", metadata)
        raise


def _aggregate_reference(validated: ValidatedRun, jaxaht: Path) -> dict[str, Any]:
    return {
        "local_jaxaht": str(jaxaht.resolve()),
        "local_head": validated.jaxaht_head,
        "rotate_source_commit": validated.rotate_source_commit,
        "working_tree_clean": True,
    }


def _verify_aggregate(
    *,
    aggregate: Path,
    output: Path,
    validated: ValidatedRun,
    jaxaht: Path,
) -> dict[str, Any]:
    summary_path = aggregate / "summary.json"
    if not aggregate.is_dir() or not summary_path.is_file():
        raise ConfirmationDriverError("Exporter aggregate is incomplete")
    summary = _require_mapping(_load_json(summary_path), "aggregate summary")
    if summary.get("training_seeds") != list(validated.seeds):
        raise ConfirmationDriverError("Aggregate summary has the wrong training seeds")
    if summary.get("implementation_reference") != _aggregate_reference(validated, jaxaht):
        raise ConfirmationDriverError("Aggregate summary has wrong JaxAHT provenance")
    seed_results = summary.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != len(validated.seeds):
        raise ConfirmationDriverError("Aggregate summary has the wrong seed result count")

    seed_records: dict[str, Any] = {}
    for index, seed in enumerate(validated.seeds):
        completion = _read_completion(output, seed, validated)
        if completion is None:
            raise ConfirmationDriverError(
                f"Cannot verify aggregate without completed seed {seed}"
            )
        expected_target = Path("..") / "completed" / f"seed_{seed}" / "checkpoint"
        link = aggregate / f"seed_{seed}"
        if not link.is_symlink() or Path(os.readlink(link)) != expected_target:
            raise ConfirmationDriverError(
                f"Aggregate seed {seed} must be the expected relative symlink"
            )
        if link.resolve() != (_seed_completion_dir(output, seed) / "checkpoint").resolve():
            raise ConfirmationDriverError(f"Aggregate seed {seed} symlink escapes its target")
        aggregate_checkpoint_sha256 = directory_digest(link)
        if aggregate_checkpoint_sha256 != completion["checkpoint_tree_sha256"]:
            raise ConfirmationDriverError(
                f"Aggregate checkpoint hash mismatch for seed {seed}"
            )
        stable_result_path = _seed_completion_dir(output, seed) / "seed_result.json"
        stable_result = _validate_seed_result(
            _load_json(stable_result_path), seed, validated
        )
        if seed_results[index] != stable_result:
            raise ConfirmationDriverError(
                f"Aggregate summary result mismatch for seed {seed}"
            )
        seed_records[str(seed)] = {
            "checkpoint_tree_sha256": aggregate_checkpoint_sha256,
            "seed_result_sha256": completion["seed_result_sha256"],
            "relative_symlink": str(expected_target),
        }
    return {
        "path": str(aggregate.relative_to(output)),
        "summary_sha256": file_digest(summary_path),
        "seed_records": seed_records,
    }


def _materialize_aggregate(
    *,
    output: Path,
    validated: ValidatedRun,
    jaxaht: Path,
) -> dict[str, Any]:
    aggregate = output / "per_state"
    if aggregate.exists() or aggregate.is_symlink():
        return _verify_aggregate(
            aggregate=aggregate,
            output=output,
            validated=validated,
            jaxaht=jaxaht,
        )

    staging = output / f".per_state.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        seed_results: list[dict[str, Any]] = []
        for seed in validated.seeds:
            completion = _read_completion(output, seed, validated)
            if completion is None:
                raise ConfirmationDriverError(
                    f"Cannot materialize aggregate without completed seed {seed}"
                )
            stable_result_path = _seed_completion_dir(output, seed) / "seed_result.json"
            seed_results.append(
                _validate_seed_result(_load_json(stable_result_path), seed, validated)
            )
            relative_target = Path("..") / "completed" / f"seed_{seed}" / "checkpoint"
            os.symlink(relative_target, staging / f"seed_{seed}", target_is_directory=True)

        summary = {
            "training_seeds": list(validated.seeds),
            "seed_results": seed_results,
            "implementation_reference": _aggregate_reference(validated, jaxaht),
        }
        atomic_write_json(staging / "summary.json", summary)
        os.replace(staging, aggregate)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _verify_aggregate(
        aggregate=aggregate,
        output=output,
        validated=validated,
        jaxaht=jaxaht,
    )


def run_confirmation(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    repo_root: Path,
    jaxaht: Path,
    checkpoint_root: Path,
    output: Path,
    python: Path,
    evaluation_episodes: int,
    validate_only: bool = False,
    git_reader: Callable[..., str] = git_output,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    if evaluation_episodes <= 0:
        raise ConfirmationDriverError("evaluation_episodes must be positive")
    _, validated = validate_protocol(
        protocol_path=protocol_path,
        expected_protocol_sha256=expected_protocol_sha256,
        repo_root=repo_root,
        jaxaht=jaxaht,
        checkpoint_root=checkpoint_root,
        git_reader=git_reader,
    )
    validation = {
        "status": "validated",
        "protocol": str(protocol_path.resolve()),
        "identity": _manifest_identity(validated),
    }
    if validate_only:
        return validation

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".confirmation_training.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfirmationDriverError(
                f"Another confirmation driver holds {lock_path}"
            ) from exc
        manifest = _load_or_create_manifest(output, validated)
        manifest, changed = _reconcile_completed(output, manifest, validated)
        if changed:
            manifest["updated_at"] = utc_now()
            atomic_write_json(output / "confirmation_training_manifest.json", manifest)

        for seed in validated.seeds:
            if str(seed) in manifest["completed_seeds"]:
                continue
            _, current_validation = validate_protocol(
                protocol_path=protocol_path,
                expected_protocol_sha256=expected_protocol_sha256,
                repo_root=repo_root,
                jaxaht=jaxaht,
                checkpoint_root=checkpoint_root,
                git_reader=git_reader,
            )
            if _manifest_identity(current_validation) != _manifest_identity(validated):
                raise ConfirmationDriverError(
                    f"Frozen inputs changed before seed {seed}; refusing to continue"
                )
            try:
                record, attempt = _run_seed(
                    output=output,
                    repo_root=repo_root.resolve(),
                    jaxaht=jaxaht.resolve(),
                    checkpoint_root=checkpoint_root.resolve(),
                    experiment=(repo_root / "data/fixed_ego_experiment.json").resolve(),
                    python=python.resolve(),
                    seed=seed,
                    validated=validated,
                    evaluation_episodes=evaluation_episodes,
                    process_runner=process_runner,
                )
            except BaseException:
                attempt_path = sorted(
                    (output / "attempts" / f"seed_{seed}").glob("attempt_*/attempt.json")
                )[-1]
                manifest["attempts"].append(_load_json(attempt_path))
                manifest["status"] = "incomplete"
                manifest["updated_at"] = utc_now()
                atomic_write_json(output / "confirmation_training_manifest.json", manifest)
                raise
            manifest["completed_seeds"][str(seed)] = record
            manifest["attempts"].append(attempt)
            manifest["status"] = (
                "training_complete"
                if len(manifest["completed_seeds"]) == len(validated.seeds)
                else "running"
            )
            manifest["updated_at"] = utc_now()
            atomic_write_json(output / "confirmation_training_manifest.json", manifest)

        aggregate = _materialize_aggregate(
            output=output,
            validated=validated,
            jaxaht=jaxaht,
        )
        if manifest.get("aggregate") not in (None, aggregate):
            raise ConfirmationDriverError("Manifest aggregate provenance drifted")
        if manifest.get("aggregate") == aggregate and manifest.get("status") == "completed":
            return manifest
        manifest["aggregate"] = aggregate
        manifest["status"] = "completed"
        manifest["updated_at"] = utc_now()
        atomic_write_json(output / "confirmation_training_manifest.json", manifest)
        return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "configs/confirmation/diagnostic_baseline_confirmation_protocol.json",
    )
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "data/checkpoints/hf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/fixed_ego_teamgen_diagnostic_baseline_confirmation",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_confirmation(
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
        repo_root=REPO_ROOT,
        jaxaht=args.jaxaht,
        checkpoint_root=args.checkpoint_root,
        output=args.output,
        python=args.python,
        evaluation_episodes=args.evaluation_episodes,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfirmationDriverError as exc:
        print(f"confirmation training refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
