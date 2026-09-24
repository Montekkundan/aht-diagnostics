#!/usr/bin/env python3
"""Combine frozen natural-failure candidate tests into one summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aht_diagnostics.diagnostic_suite import summarize_diagnostic_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap-result", required=True, type=Path)
    parser.add_argument("--initial-soup-result", required=True, type=Path)
    parser.add_argument("--capable-control-result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_metadata(*results: dict[str, object]) -> dict[str, str]:
    statuses = " ".join(str(result.get("status", "")) for result in results).lower()
    if "confirmation" in statuses or "prospective" in statuses:
        return {
            "status": "prospective_diagnostic_suite_summary",
            "claim_scope": (
                "Derived summary of the prespecified diagnostic components on "
                "the prospective confirmation split. Semantic decisions remain "
                "operational outputs rather than external ground truth."
            ),
        }
    return {
        "status": "derived_diagnostic_suite_summary",
        "claim_scope": (
            "Derived summary of two already exposed frozen candidate tests. "
            "It is an implementation check, not independent confirmation."
        ),
    }


def main() -> int:
    args = parse_args()
    swap_path = args.swap_result.resolve()
    initial_path = args.initial_soup_result.resolve()
    output_path = args.output.resolve()
    swap_result = json.loads(swap_path.read_text(encoding="utf-8"))
    initial_result = json.loads(initial_path.read_text(encoding="utf-8"))
    capable_path = (
        args.capable_control_result.resolve()
        if args.capable_control_result is not None
        else None
    )
    capable_result = (
        json.loads(capable_path.read_text(encoding="utf-8"))
        if capable_path is not None
        else None
    )
    result = summarize_diagnostic_suite(
        swap_result=swap_result,
        initial_soup_result=initial_result,
        capable_control_result=capable_result,
    )
    metadata_inputs = [swap_result, initial_result]
    if capable_result is not None:
        metadata_inputs.append(capable_result)
    result.update(
        {
            **summary_metadata(*metadata_inputs),
            "inputs": {
                "swap_result": str(swap_path),
                "swap_result_sha256": file_digest(swap_path),
                "initial_soup_result": str(initial_path),
                "initial_soup_result_sha256": file_digest(initial_path),
            },
            "analysis_script_sha256": file_digest(Path(__file__).resolve()),
        }
    )
    if capable_path is not None:
        result["inputs"].update(
            {
                "capable_control_result": str(capable_path),
                "capable_control_result_sha256": file_digest(capable_path),
            }
        )
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
