#!/usr/bin/env python3
"""Export canonical factual/reference rows for diagnostic baseline studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.baselines import (  # noqa: E402
    approximate_reference_gap,
    factual_event_features,
)
from aht_diagnostics.evaluation_conditions import condition_kwargs  # noqa: E402
from aht_diagnostics.freeze import validate_frozen_protocol  # noqa: E402
from aht_diagnostics.jaxaht_bridge import activate_jaxaht, load_frozen_s5_ego  # noqa: E402
from aht_diagnostics.natural_failure import (  # noqa: E402
    first_initial_soup_disposition,
    first_soup_carrier_swap_lock_opportunity,
    initial_completed_soup_opportunity,
    soup_carrier_swap_context_summary,
)
from aht_diagnostics.rollout import rollout_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaxaht", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoints" / "hf",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoints.json",
    )
    return parser.parse_args()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _is_confirmation_export(
    specification: dict[str, Any],
    config_path: Path,
) -> bool:
    identifying_text = " ".join(
        (
            str(specification.get("status", "")),
            str(specification.get("claim_scope", "")),
            str(config_path),
        )
    ).lower()
    return "confirmation" in identifying_text


def _integer_seed_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list for confirmation")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise ValueError(f"{field} must contain only integer seeds")
    seeds = [int(seed) for seed in value]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{field} must not contain duplicate seeds")
    return seeds


def _assert_exact_seeds(
    *,
    observed: list[int],
    expected: list[int],
    source: str,
) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing or extra or len(observed) != len(expected):
        raise ValueError(
            f"{source} does not match expected_train_seeds: "
            f"missing={missing}, extra={extra}, observed={observed}"
        )


def _validate_expected_episode_design(
    specification: dict[str, Any],
    episode_seed_range: dict[str, Any],
) -> None:
    start = int(episode_seed_range["start"])
    end_exclusive = int(episode_seed_range["end_exclusive"])
    if end_exclusive <= start:
        raise ValueError("episode_seed_range must contain at least one seed")

    expected_range = specification.get("expected_episode_seed_range")
    if expected_range is not None:
        if not isinstance(expected_range, dict):
            raise ValueError("expected_episode_seed_range must be an object")
        expected_start = int(expected_range["start"])
        expected_end = int(expected_range["end_exclusive"])
        if (start, end_exclusive) != (expected_start, expected_end):
            raise ValueError(
                "episode_seed_range does not match expected_episode_seed_range: "
                f"observed=[{start}, {end_exclusive}), "
                f"expected=[{expected_start}, {expected_end})"
            )

    expected_count = specification.get("expected_episode_count")
    if expected_count is not None and end_exclusive - start != int(expected_count):
        raise ValueError(
            "episode_seed_range count does not match expected_episode_count: "
            f"observed={end_exclusive - start}, expected={int(expected_count)}"
        )


def _validate_confirmation_inputs(
    *,
    specification: dict[str, Any],
    config_path: Path,
    source_results: Path,
    source_summary: dict[str, Any],
    jaxaht_path: Path,
) -> dict[str, Any] | None:
    """Reject incomplete or provenance-drifted confirmation inputs up front."""
    if not _is_confirmation_export(specification, config_path):
        return None

    expected_seeds = _integer_seed_list(
        specification.get("expected_train_seeds"),
        field="expected_train_seeds",
    )
    summary_seeds = _integer_seed_list(
        source_summary.get("training_seeds"),
        field="source summary training_seeds",
    )
    seed_results = source_summary.get("seed_results")
    if not isinstance(seed_results, list):
        raise ValueError("source summary seed_results must be a list")
    result_seeds = _integer_seed_list(
        [row.get("train_seed") for row in seed_results if isinstance(row, dict)],
        field="source summary seed_results train_seed values",
    )
    policy_seeds = sorted(
        int(path.name.removeprefix("seed_"))
        for path in source_results.glob("seed_*")
        if path.name.removeprefix("seed_").isdigit()
        and (path / "trained_policies").is_dir()
    )
    _assert_exact_seeds(
        observed=summary_seeds,
        expected=expected_seeds,
        source="source summary training_seeds",
    )
    _assert_exact_seeds(
        observed=result_seeds,
        expected=expected_seeds,
        source="source summary seed_results",
    )
    _assert_exact_seeds(
        observed=policy_seeds,
        expected=expected_seeds,
        source="saved policy directories",
    )
    for train_seed in expected_seeds:
        saved_path = source_results / f"seed_{train_seed}" / "trained_policies"
        if not any(candidate.is_file() for candidate in saved_path.rglob("*")):
            raise ValueError(f"saved policy directory is empty: {saved_path}")

    _validate_expected_episode_design(
        specification,
        specification["episode_seed_range"],
    )

    expected_project_head = specification.get("project_head")
    if not isinstance(expected_project_head, str) or not expected_project_head.strip():
        raise ValueError("project_head must be a non-empty string for confirmation")
    actual_project_head = _git_revision(REPO_ROOT, "rev-parse", "HEAD")
    if actual_project_head != expected_project_head:
        raise ValueError(
            "project_head does not match the current repository HEAD: "
            f"observed={actual_project_head}, expected={expected_project_head}"
        )

    expected_jaxaht = specification.get("expected_jaxaht_provenance")
    if not isinstance(expected_jaxaht, dict):
        raise ValueError(
            "expected_jaxaht_provenance must be an object for confirmation"
        )
    required_provenance = {"local_head", "rotate_source_commit", "working_tree_clean"}
    missing_provenance = sorted(required_provenance - set(expected_jaxaht))
    if missing_provenance:
        raise ValueError(
            "expected_jaxaht_provenance is missing required fields: "
            f"{missing_provenance}"
        )
    if expected_jaxaht["working_tree_clean"] is not True:
        raise ValueError(
            "expected_jaxaht_provenance.working_tree_clean must be true "
            "for confirmation"
        )

    source_reference = source_summary.get("implementation_reference")
    if not isinstance(source_reference, dict):
        raise ValueError("source summary implementation_reference must be an object")
    for field in required_provenance:
        if source_reference.get(field) != expected_jaxaht[field]:
            raise ValueError(
                f"source summary JaxAHT {field} does not match confirmation config: "
                f"observed={source_reference.get(field)!r}, "
                f"expected={expected_jaxaht[field]!r}"
            )
    source_jaxaht = source_reference.get("local_jaxaht")
    if (
        not isinstance(source_jaxaht, str)
        or Path(source_jaxaht).resolve() != jaxaht_path
    ):
        raise ValueError(
            "source summary local_jaxaht does not match --jaxaht: "
            f"observed={source_jaxaht!r}, expected={str(jaxaht_path)!r}"
        )

    actual_jaxaht = {
        "local_head": _git_revision(jaxaht_path, "rev-parse", "HEAD"),
        "rotate_source_commit": _git_revision(
            jaxaht_path,
            "log",
            "-1",
            "--format=%H",
            "--",
            "open_ended_training/rotate.py",
        ),
        "working_tree_clean": not bool(
            _git_revision(jaxaht_path, "status", "--porcelain")
        ),
    }
    for field in required_provenance:
        if actual_jaxaht[field] != expected_jaxaht[field]:
            raise ValueError(
                f"current JaxAHT {field} does not match confirmation config: "
                f"observed={actual_jaxaht[field]!r}, "
                f"expected={expected_jaxaht[field]!r}"
            )

    return {
        "project_head": actual_project_head,
        "jaxaht": actual_jaxaht,
        "expected_train_seeds": expected_seeds,
    }


def _initial_soup_context(episode: Any) -> dict[str, Any]:
    opportunity = initial_completed_soup_opportunity(episode)
    disposition = first_initial_soup_disposition(episode)
    return {
        "has_initial_completed_soup": opportunity is not None,
        "initial_ego_position": (
            list(opportunity.ego_position) if opportunity is not None else None
        ),
        "initial_partner_position": (
            list(opportunity.partner_position) if opportunity is not None else None
        ),
        "initial_partner_inventory": (
            opportunity.partner_inventory if opportunity is not None else None
        ),
        "disposition": disposition.event if disposition is not None else None,
        "disposition_step": disposition.step if disposition is not None else None,
        "drop_candidate": (
            disposition is not None and disposition.event == "drop_soup"
        ),
        "candidate_interval": (
            [0, disposition.step]
            if disposition is not None and disposition.event == "drop_soup"
            else None
        ),
    }


def _swap_context(episode: Any, minimum_lock_steps: int) -> dict[str, Any]:
    summary = soup_carrier_swap_context_summary(episode)
    opportunity = first_soup_carrier_swap_lock_opportunity(
        episode,
        minimum_consecutive_steps=minimum_lock_steps,
    )
    return {
        **summary,
        "triggered": opportunity is not None,
        "candidate_interval": (
            [opportunity.step, opportunity.confirmed_at_step]
            if opportunity is not None
            else None
        ),
    }


def _summarize(rows: list[dict[str, Any]], competence_threshold: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("No canonical baseline rows were produced")
    return {
        "evaluated_pairings": len(rows),
        "train_seeds": sorted({int(row["train_seed"]) for row in rows}),
        "episode_seeds": sorted({int(row["episode_seed"]) for row in rows}),
        "zero_return_pairings": sum(float(row["factual_return"]) == 0.0 for row in rows),
        "competence_threshold": competence_threshold,
        "competent_reference_pairings": sum(
            bool(row["initial_state_competence_witness"]) for row in rows
        ),
        "swap_lock_candidates": sum(
            bool(row["swap_context"]["triggered"]) for row in rows
        ),
        "initial_soup_drop_candidates": sum(
            bool(row["initial_soup_context"]["drop_candidate"]) for row in rows
        ),
        "mean_factual_return": sum(float(row["factual_return"]) for row in rows)
        / len(rows),
        "mean_reference_return": sum(float(row["reference_return"]) for row in rows)
        / len(rows),
        "mean_approximate_reference_gap": sum(
            float(row["approximate_reference_gap"]) for row in rows
        )
        / len(rows),
        "negative_approximate_reference_gaps": sum(
            float(row["approximate_reference_gap"]) < 0.0 for row in rows
        ),
    }


def main() -> int:
    args = parse_args()
    jaxaht_path = args.jaxaht.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_index_path = args.checkpoint_index.resolve()

    specification = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_index = json.loads(
        checkpoint_index_path.read_text(encoding="utf-8")
    )
    source_results_relative = Path(specification["source_results"])
    source_results = (REPO_ROOT / source_results_relative).resolve()
    source_summary_path = source_results / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))

    confirmation_inputs = _validate_confirmation_inputs(
        specification=specification,
        config_path=config_path,
        source_results=source_results,
        source_summary=source_summary,
        jaxaht_path=jaxaht_path,
    )
    fixed_checkpoint = checkpoint_root / checkpoint_index["fixed_ego"]["checkpoint"]
    frozen_protocol_validation = None
    if confirmation_inputs is not None:
        if not args.expected_protocol_sha256:
            raise ValueError(
                "--expected-protocol-sha256 is required for confirmation export"
            )
        protocol_reference = specification.get("machine_readable_protocol")
        if not isinstance(protocol_reference, str) or not protocol_reference:
            raise ValueError(
                "machine_readable_protocol is required for confirmation export"
            )
        frozen_protocol_validation = validate_frozen_protocol(
            protocol_path=REPO_ROOT / protocol_reference,
            expected_protocol_sha256=args.expected_protocol_sha256,
            repo_root=REPO_ROOT,
            jaxaht_path=jaxaht_path,
            fixed_checkpoint_path=fixed_checkpoint,
        )

    max_steps = int(specification["max_steps"])
    minimum_lock_steps = int(specification["minimum_lock_steps"])
    competence_threshold = float(specification["competence_return_threshold"])
    if max_steps < 1 or minimum_lock_steps < 1:
        raise ValueError("max_steps and minimum_lock_steps must be positive")

    activate_jaxaht(jaxaht_path)
    from agents.mlp_actor_critic_agent import (  # noqa: PLC0415
        ActorWithDoubleCriticPolicy,
        MLPActorCriticPolicy,
    )
    from common.save_load_utils import load_train_run  # noqa: PLC0415
    from envs import make_env  # noqa: PLC0415
    from envs.log_wrapper import LogWrapper  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    task = OmegaConf.load(
        jaxaht_path
        / "evaluation"
        / "configs"
        / "task"
        / "overcooked-v1"
        / "cramped_room.yaml"
    )
    base_kwargs = OmegaConf.to_container(task.ENV_KWARGS, resolve=True)
    env_kwargs = condition_kwargs(
        base_kwargs,
        specification["environment_condition"],
        max_steps,
    )
    env = LogWrapper(make_env(task.ENV_NAME, env_kwargs))

    fixed_policy, fixed_params = load_frozen_s5_ego(
        jaxaht_path,
        fixed_checkpoint,
        env,
        seed=0,
    )
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    teammate_policy = ActorWithDoubleCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    reference_policy = MLPActorCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)

    seed_range = specification["episode_seed_range"]
    episode_seeds = range(
        int(seed_range["start"]),
        int(seed_range["end_exclusive"]),
    )
    rows: list[dict[str, Any]] = []
    saved_policy_sha256: dict[str, str] = {}
    for seed_result in source_summary["seed_results"]:
        train_seed = int(seed_result["train_seed"])
        saved_path = source_results / f"seed_{train_seed}" / "trained_policies"
        saved_policy_sha256[str(train_seed)] = directory_digest(saved_path)
        saved = load_train_run(saved_path)
        teammate_params = saved["generated_teammate_params"]
        reference_params = saved["best_response_params"]

        for episode_seed in episode_seeds:
            factual = rollout_episode(
                env,
                fixed_policy,
                fixed_params,
                teammate_policy,
                teammate_params,
                seed=episode_seed,
                max_steps=max_steps,
            )
            reference = rollout_episode(
                env,
                reference_policy,
                reference_params,
                teammate_policy,
                teammate_params,
                seed=episode_seed,
                max_steps=max_steps,
            )
            rows.append(
                {
                    "train_seed": train_seed,
                    "episode_seed": episode_seed,
                    "factual_return": factual.total_return,
                    "reference_return": reference.total_return,
                    "approximate_reference_gap": approximate_reference_gap(
                        factual.total_return,
                        reference.total_return,
                    ),
                    "initial_state_competence_witness": (
                        reference.total_return >= competence_threshold
                    ),
                    "factual_event_features": factual_event_features(factual),
                    "swap_context": _swap_context(factual, minimum_lock_steps),
                    "initial_soup_context": _initial_soup_context(factual),
                }
            )

    expected_pairings = len(source_summary["seed_results"]) * len(episode_seeds)
    if len(rows) != expected_pairings:
        raise RuntimeError(
            f"expected {expected_pairings} canonical rows, produced {len(rows)}"
        )

    result = {
        "status": specification["status"],
        "claim_scope": specification["claim_scope"],
        "protocol": {
            "source_results": str(source_results_relative),
            "environment_condition": specification["environment_condition"],
            "max_steps": max_steps,
            "minimum_lock_steps": minimum_lock_steps,
            "episode_seed_range": specification["episode_seed_range"],
            "competence_return_threshold": competence_threshold,
            "player_order": {
                "agent_0": "fixed ego or learned reference",
                "agent_1": "generated teammate",
            },
            "event_features_use_counterfactual_outcomes": False,
        },
        "provenance": {
            "project_head": (
                confirmation_inputs["project_head"]
                if confirmation_inputs is not None
                else specification.get("project_head")
            ),
            "validated_confirmation_inputs": confirmation_inputs,
            "validated_frozen_protocol": frozen_protocol_validation,
            "source_implementation_reference": source_summary.get(
                "implementation_reference"
            ),
            "checkpoint_dataset_revision": checkpoint_index.get("revision"),
            "fixed_checkpoint_sha256": directory_digest(fixed_checkpoint),
            "fixed_checkpoint_path": str(fixed_checkpoint),
            "saved_policy_sha256": saved_policy_sha256,
            "source_summary_sha256": file_digest(source_summary_path),
            "config_sha256": file_digest(config_path),
            "analysis_script_sha256": file_digest(Path(__file__).resolve()),
        },
        "rows": rows,
        "summary": _summarize(rows, competence_threshold),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
