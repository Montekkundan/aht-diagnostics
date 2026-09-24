from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aht_diagnostics.freeze import FreezeValidationError, validate_frozen_protocol


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def test_validate_frozen_protocol_rejects_analysis_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    jaxaht = tmp_path / "jaxaht"
    checkpoint = tmp_path / "checkpoint"
    repo.mkdir()
    jaxaht.mkdir()
    checkpoint.mkdir()
    local_input = repo / "analysis.py"
    upstream_input = jaxaht / "rotate.py"
    local_input.write_text("locked", encoding="utf-8")
    upstream_input.write_text("upstream", encoding="utf-8")
    (checkpoint / "weights").write_text("weights", encoding="utf-8")
    protocol = {
        "status": "frozen_before_training_and_outcome_access",
        "provenance_freeze": {
            "aht_diagnostics_git_head": "repo-head",
            "jaxaht_git_head": "jax-head",
            "jaxaht_rotate_source_commit": "rotate-head",
            "jaxaht_working_tree_clean": True,
            "fixed_ego_checkpoint_tree_sha256": _directory_digest(checkpoint),
            "sha256": {
                "analysis.py": _file_digest(local_input),
                "jaxaht/rotate.py": _file_digest(upstream_input),
            },
        },
    }
    protocol_path = repo / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha256 = _file_digest(protocol_path)

    def git_reader(path: Path, *args: str) -> str:
        if path == repo:
            return "repo-head"
        if args == ("rev-parse", "HEAD"):
            return "jax-head"
        if args == ("status", "--porcelain"):
            return ""
        return "rotate-head"

    validated = validate_frozen_protocol(
        protocol_path=protocol_path,
        expected_protocol_sha256=protocol_sha256,
        repo_root=repo,
        jaxaht_path=jaxaht,
        fixed_checkpoint_path=checkpoint,
        git_reader=git_reader,
    )
    assert validated["validated_frozen_file_count"] == 2

    local_input.write_text("drifted", encoding="utf-8")
    with pytest.raises(FreezeValidationError, match="Frozen input drifted"):
        validate_frozen_protocol(
            protocol_path=protocol_path,
            expected_protocol_sha256=protocol_sha256,
            repo_root=repo,
            jaxaht_path=jaxaht,
            fixed_checkpoint_path=checkpoint,
            git_reader=git_reader,
        )


def test_validate_frozen_protocol_rejects_protocol_digest_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    jaxaht = tmp_path / "jaxaht"
    checkpoint = tmp_path / "checkpoint"
    repo.mkdir()
    jaxaht.mkdir()
    checkpoint.mkdir()
    local_input = repo / "analysis.py"
    upstream_input = jaxaht / "rotate.py"
    local_input.write_text("locked", encoding="utf-8")
    upstream_input.write_text("upstream", encoding="utf-8")
    (checkpoint / "weights").write_text("weights", encoding="utf-8")
    protocol = {
        "status": "frozen_before_training_and_outcome_access",
        "provenance_freeze": {
            "aht_diagnostics_git_head": "repo-head",
            "jaxaht_git_head": "jax-head",
            "jaxaht_rotate_source_commit": "rotate-head",
            "jaxaht_working_tree_clean": True,
            "fixed_ego_checkpoint_tree_sha256": _directory_digest(checkpoint),
            "sha256": {
                "analysis.py": _file_digest(local_input),
                "jaxaht/rotate.py": _file_digest(upstream_input),
            },
        },
    }
    protocol_path = repo / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    def git_reader(path: Path, *args: str) -> str:
        if path == repo:
            return "repo-head"
        if args == ("rev-parse", "HEAD"):
            return "jax-head"
        if args == ("status", "--porcelain"):
            return ""
        return "rotate-head"

    with pytest.raises(FreezeValidationError, match="protocol digest drifted"):
        validate_frozen_protocol(
            protocol_path=protocol_path,
            expected_protocol_sha256="0" * 64,
            repo_root=repo,
            jaxaht_path=jaxaht,
            fixed_checkpoint_path=checkpoint,
            git_reader=git_reader,
        )
