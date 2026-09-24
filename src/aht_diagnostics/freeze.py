"""Validate an immutable confirmation protocol before outcome analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable


FROZEN_STATUS = "frozen_before_training_and_outcome_access"


class FreezeValidationError(RuntimeError):
    """Raised when a confirmation input differs from its frozen digest."""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path) -> str:
    if not path.is_dir():
        raise FreezeValidationError(f"Expected directory: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise FreezeValidationError(f"Refusing to hash empty directory: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_output(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreezeValidationError(f"{label} must be a JSON object")
    return value


def validate_frozen_protocol(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    repo_root: Path,
    jaxaht_path: Path,
    fixed_checkpoint_path: Path,
    git_reader: Callable[..., str] = git_output,
) -> dict[str, Any]:
    """Fail closed if any frozen local, upstream or checkpoint input drifted."""

    protocol_path = protocol_path.resolve()
    repo_root = repo_root.resolve()
    jaxaht_path = jaxaht_path.resolve()
    fixed_checkpoint_path = fixed_checkpoint_path.resolve()
    actual_protocol_sha256 = file_digest(protocol_path)
    if actual_protocol_sha256 != expected_protocol_sha256:
        raise FreezeValidationError(
            "Confirmation protocol digest drifted: "
            f"{actual_protocol_sha256} != {expected_protocol_sha256}"
        )
    protocol = _mapping(
        json.loads(protocol_path.read_text(encoding="utf-8")),
        "protocol",
    )
    if protocol.get("status") != FROZEN_STATUS:
        raise FreezeValidationError(
            "Confirmation protocol must be frozen before outcome analysis"
        )
    provenance = _mapping(
        protocol.get("provenance_freeze"),
        "protocol.provenance_freeze",
    )
    frozen_hashes = _mapping(provenance.get("sha256"), "provenance sha256")

    repo_head = git_reader(repo_root, "rev-parse", "HEAD")
    if repo_head != provenance.get("aht_diagnostics_git_head"):
        raise FreezeValidationError("aht-diagnostics HEAD drifted from the protocol")
    jaxaht_head = git_reader(jaxaht_path, "rev-parse", "HEAD")
    if jaxaht_head != provenance.get("jaxaht_git_head"):
        raise FreezeValidationError("JaxAHT HEAD drifted from the protocol")
    if provenance.get("jaxaht_working_tree_clean") is not True:
        raise FreezeValidationError("Protocol must require a clean JaxAHT tree")
    if git_reader(jaxaht_path, "status", "--porcelain"):
        raise FreezeValidationError("JaxAHT working tree is not clean")
    rotate_source_commit = git_reader(
        jaxaht_path,
        "log",
        "-1",
        "--format=%H",
        "--",
        "open_ended_training/rotate.py",
    )
    if rotate_source_commit != provenance.get("jaxaht_rotate_source_commit"):
        raise FreezeValidationError("JaxAHT ROTATE source commit drifted")

    for name, expected in sorted(frozen_hashes.items()):
        if not isinstance(expected, str):
            raise FreezeValidationError(f"Frozen hash must be a string: {name}")
        if name.startswith("jaxaht/"):
            path = jaxaht_path / name.removeprefix("jaxaht/")
        else:
            path = repo_root / name
        if not path.is_file():
            raise FreezeValidationError(f"Frozen input is missing: {path}")
        actual = file_digest(path)
        if actual != expected:
            raise FreezeValidationError(
                f"Frozen input drifted: {name}: {actual} != {expected}"
            )

    checkpoint_digest = directory_digest(fixed_checkpoint_path)
    if checkpoint_digest != provenance.get("fixed_ego_checkpoint_tree_sha256"):
        raise FreezeValidationError("Fixed ego checkpoint tree drifted")

    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": actual_protocol_sha256,
        "aht_diagnostics_git_head": repo_head,
        "jaxaht_git_head": jaxaht_head,
        "jaxaht_rotate_source_commit": rotate_source_commit,
        "fixed_ego_checkpoint_tree_sha256": checkpoint_digest,
        "validated_frozen_file_count": len(frozen_hashes),
    }
