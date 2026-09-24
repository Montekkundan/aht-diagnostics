#!/usr/bin/env python3
"""Make a fixed, label-blinded review sample from the completed confirmation.

The two-stage packet separates observation from intervention evidence. The
answer key records sampling strata and evaluator outputs, not human truth.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aht_diagnostics.evaluation_conditions import condition_kwargs  # noqa: E402
from aht_diagnostics.jaxaht_bridge import activate_jaxaht, load_frozen_s5_ego  # noqa: E402
from aht_diagnostics.natural_failure import soup_carrier_swap_context_summary  # noqa: E402
from aht_diagnostics.rollout import rollout_episode  # noqa: E402
from aht_diagnostics.semantics import OBJECT_NAMES, state_view  # noqa: E402


SAMPLE_COUNTS = {
    "supported_swap_competent_reference": 2,
    "supported_swap_reference_inconclusive": 2,
    "supported_initial_soup": 3,
    "ambiguous": 3,
    "unsupported": 3,
    "unflagged_zero_competent_reference": 3,
    "unflagged_zero_reference_inconclusive": 2,
    "successful_control": 2,
}


def select_cases(
    rows: list[dict[str, Any]], diagnoses: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratify before viewing traces; diversify training seeds within strata."""

    indexed = {(int(row["train_seed"]), int(row["episode_seed"])): row for row in rows}
    by_key = {
        (int(row["train_seed"]), int(row["episode_seed"])): row
        for row in diagnoses
    }
    if len(indexed) != len(rows) or len(by_key) != len(diagnoses):
        raise ValueError("duplicate policy-start pair")
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for key, row in indexed.items():
        diagnosis = by_key.get(key)
        if diagnosis is not None:
            decision = diagnosis["decision"]
            if decision == "supported" and diagnosis["candidate_kind"] == "soup_carrier_swap_lock":
                group = (
                    "supported_swap_competent_reference"
                    if float(row["reference_return"]) >= 20
                    else "supported_swap_reference_inconclusive"
                )
            elif decision == "supported":
                group = "supported_initial_soup"
            elif decision == "ambiguous":
                group = "ambiguous"
            else:
                group = "unsupported"
        elif float(row["factual_return"]) == 0:
            group = (
                "unflagged_zero_competent_reference"
                if float(row["reference_return"]) >= 20
                else "unflagged_zero_reference_inconclusive"
            )
        elif float(row["factual_return"]) >= 20 and float(row["reference_return"]) >= 20:
            group = "successful_control"
        else:
            continue
        strata[group].append(key)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group, count in SAMPLE_COUNTS.items():
        candidates = sorted(strata[group])
        if len(candidates) < count:
            raise ValueError(f"stratum {group} has {len(candidates)} < {count} cases")
        rng.shuffle(candidates)
        picked: list[tuple[int, int]] = []
        used_seeds: set[int] = set()
        for key in candidates:
            if key[0] not in used_seeds:
                picked.append(key)
                used_seeds.add(key[0])
            if len(picked) == count:
                break
        for key in candidates:
            if len(picked) == count:
                break
            if key not in picked:
                picked.append(key)
        selected.extend(
            {"train_seed": key[0], "episode_seed": key[1], "sampling_stratum": group}
            for key in picked
        )
    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        item["case_id"] = f"D{index:02d}"
    return selected, {group: len(strata[group]) for group in SAMPLE_COUNTS}


def pot_stage(status: int) -> str:
    """Interpret the Overcooked-v1 pot countdown without inferring an agent's intent."""

    if status == 23:
        return "empty"
    if status in (22, 21):
        return f"{23 - status} onion(s)"
    if status == 20:
        return "3 onions, cooking"
    if 0 < status < 20:
        return f"cooking, {status} steps remaining"
    if status == 0:
        return "ready"
    return f"unknown status {status}"


def _raw_state(env_state: Any) -> Any:
    state = env_state
    for _ in range(5):
        if hasattr(state, "maze_map") and hasattr(state, "agent_pos"):
            return state
        state = state.env_state
    raise TypeError("No Overcooked state in snapshot")


def kitchen_layout(episode: Any, agent_view_size: int) -> list[list[str]]:
    """Return the visible kitchen grid, with the initial agent cells cleared."""

    padding = agent_view_size - 2
    maze = np.asarray(_raw_state(episode.snapshots[0].env_state).maze_map)
    tiles = maze[padding:-padding, padding:-padding, 0]
    return [
        [OBJECT_NAMES.get(int(tile), f"object_{int(tile)}") if tile != 10 else "empty"
         for tile in row]
        for row in tiles
    ]


def records(episode: Any) -> list[dict[str, Any]]:
    return [
        {
            "step": int(step.step),
            "ego_action": int(step.ego_action),
            "partner_action": int(step.partner_action),
            "ego_position": list(step.ego_position),
            "partner_position": list(step.partner_position),
            "ego_inventory": step.ego_inventory,
            "partner_inventory": step.partner_inventory,
            "ego_event": step.ego_event,
            "partner_event": step.partner_event,
            "ego_direction": step.ego_direction,
            "partner_direction": step.partner_direction,
            "pot_statuses": [pot_stage(status) for status in state_view(snapshot.env_state).pot_statuses],
            "reward": float(step.reward),
        }
        for step, snapshot in zip(episode.steps, episode.snapshots, strict=True)
    ]


def observation_viewer(cases: list[dict[str, Any]]) -> str:
    """Make a self-contained, label-blinded viewer for the observation stage."""

    allowed = (
        "case_id", "condition_A_return", "condition_B_return", "kitchen_layout",
        "condition_A_steps", "condition_B_steps",
    )
    blinded = [{key: case[key] for key in allowed} for case in cases]
    payload = json.dumps(blinded, separators=(",", ":")).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overcooked observation review</title>
<style>
body { font: 16px/1.45 system-ui, sans-serif; max-width: 1050px; margin: 2rem auto; padding: 0 1rem; color: #17212b; }
h1 { font-size: 1.5rem; margin-bottom: .4rem; }
.controls { display: flex; flex-wrap: wrap; gap: .8rem; align-items: center; margin: 1.5rem 0; }
#step { width: min(30rem, 90vw); }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
.panel { border: 1px solid #bdc7d0; border-radius: 6px; padding: 1rem; }
.grid { display: grid; width: max-content; border: 1px solid #8a969e; margin: 1rem 0; }
.tile { width: 40px; height: 40px; display: grid; place-items: center; border: 1px solid #d5dcdf; box-sizing: border-box; font-weight: 600; }
.wall { background: #697b84; color: white; }
.empty { background: #f5f6f4; }
.pot { background: #dce9f4; }
.onion_pile { background: #f9e4a6; }
.plate_pile { background: #e9e7f4; }
.goal { background: #d7eddb; }
.agent0 { outline: 3px solid #b34433; outline-offset: -3px; }
.agent1 { outline: 3px solid #285d9b; outline-offset: -3px; }
.details { white-space: pre-line; }
.muted { color: #53626d; }
button, select, input { font: inherit; }
</style>
<h1>Observation review</h1>
<p>These are two policies paired with the same teammate from the same start. Inspect behavior before opening the separate intervention packet. Returns show outcomes, not causes. The reviewer-facing file contains no evaluator diagnosis or sampling stratum.</p>
<div class="controls">
  <label>Case <select id="case"></select></label>
  <button id="previous" type="button">Previous step</button>
  <input id="step" type="range" min="0" value="0" aria-label="Episode step">
  <button id="next" type="button">Next step</button>
  <strong id="counter"></strong>
</div>
<p class="muted">Kitchen tiles: # wall, . floor, O onion supply, P plate supply, T pot, G delivery. 0 is the evaluated ego in condition A; condition B uses a different ego policy. 1 is the same teammate. Pot state is shown before the selected action. Use the left and right arrow keys to step through the episode.</p>
<div id="pair" class="pair"></div>
<script id="cases" type="application/json">__CASES__</script>
<script>
const cases = JSON.parse(document.getElementById('cases').textContent);
const picker = document.getElementById('case');
const slider = document.getElementById('step');
const counter = document.getElementById('counter');
const pair = document.getElementById('pair');
const tileSymbol = {wall:'#', empty:'.', onion_pile:'O', plate_pile:'P', pot:'T', goal:'G'};
cases.forEach((item, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = item.case_id;
  picker.appendChild(option);
});
function panel(label, steps, totalReturn, layout, stepIndex) {
  const box = document.createElement('section');
  box.className = 'panel';
  const heading = document.createElement('h2');
  heading.textContent = `${label} · return ${totalReturn}`;
  box.appendChild(heading);
  const record = steps[stepIndex];
  if (!record) {
    const end = document.createElement('p');
    end.textContent = 'Episode ended before this step.';
    box.appendChild(end);
    return box;
  }
  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.style.gridTemplateColumns = `repeat(${layout[0].length}, 40px)`;
  layout.forEach((row, y) => row.forEach((type, x) => {
    const cell = document.createElement('div');
    cell.className = `tile ${type}`;
    cell.textContent = tileSymbol[type] || '?';
    if (record.ego_position[0] + 1 === x && record.ego_position[1] + 1 === y) {
      cell.textContent = '0'; cell.classList.add('agent0');
    }
    if (record.partner_position[0] + 1 === x && record.partner_position[1] + 1 === y) {
      cell.textContent = '1'; cell.classList.add('agent1');
    }
    grid.appendChild(cell);
  }));
  box.appendChild(grid);
  const details = document.createElement('p');
  details.className = 'details';
  details.textContent = `Ego: ${record.ego_event}, holding ${record.ego_inventory}, action ${record.ego_action}\n` +
    `Teammate: ${record.partner_event}, holding ${record.partner_inventory}, action ${record.partner_action}\n` +
    `Pot: ${record.pot_statuses.join('; ') || 'none'}\nReward at step: ${record.reward}`;
  box.appendChild(details);
  return box;
}
function render() {
  const item = cases[Number(picker.value)];
  const maxStep = Math.max(item.condition_A_steps.length, item.condition_B_steps.length) - 1;
  slider.max = String(maxStep);
  if (Number(slider.value) > maxStep) slider.value = String(maxStep);
  const step = Number(slider.value);
  counter.textContent = `step ${step} of ${maxStep}`;
  pair.replaceChildren(
    panel('Condition A', item.condition_A_steps, item.condition_A_return, item.kitchen_layout, step),
    panel('Condition B', item.condition_B_steps, item.condition_B_return, item.kitchen_layout, step)
  );
}
picker.addEventListener('change', () => { slider.value = '0'; render(); });
slider.addEventListener('input', render);
document.getElementById('previous').addEventListener('click', () => { slider.value = String(Math.max(0, Number(slider.value) - 1)); render(); });
document.getElementById('next').addEventListener('click', () => { slider.value = String(Math.min(Number(slider.max), Number(slider.value) + 1)); render(); });
document.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    slider.value = String(Math.max(0, Math.min(Number(slider.max), Number(slider.value) + (event.key === 'ArrowRight' ? 1 : -1))));
    render();
  }
});
render();
</script>
</html>
"""
    return template.replace("__CASES__", payload)


def build_packet(jaxaht: Path, *, seed: int) -> dict[str, Any]:
    canonical = json.loads(
        (ROOT / "results/confirmation/diagnostic_baseline_confirmation_rows.json").read_text()
    )
    suite = json.loads(
        (ROOT / "results/confirmation/diagnostic_baseline_confirmation_suite.json").read_text()
    )
    sampled, denominators = select_cases(canonical["rows"], suite["diagnoses"], seed=seed)
    diagnosis_by_key = {
        (int(row["train_seed"]), int(row["episode_seed"])): row
        for row in suite["diagnoses"]
    }

    activate_jaxaht(jaxaht)
    from agents.mlp_actor_critic_agent import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy
    from common.save_load_utils import load_train_run
    from envs import make_env
    from envs.log_wrapper import LogWrapper
    from omegaconf import OmegaConf

    task = OmegaConf.load(jaxaht / "evaluation/configs/task/overcooked-v1/cramped_room.yaml")
    environment_kwargs = condition_kwargs(
        OmegaConf.to_container(task.ENV_KWARGS, resolve=True), "randomized_sparse", 400
    )
    env = LogWrapper(make_env(task.ENV_NAME, environment_kwargs))
    checkpoint_index = json.loads((ROOT / "data/checkpoints.json").read_text())
    fixed_policy, fixed_params = load_frozen_s5_ego(
        jaxaht,
        ROOT / "data/checkpoints/hf" / checkpoint_index["fixed_ego"]["checkpoint"],
        env,
        seed=0,
    )
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    teammate_policy = ActorWithDoubleCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    reference_policy = MLPActorCriticPolicy(action_dim=action_dim, obs_dim=obs_dim)
    trained_root = ROOT / "results/confirmation/diagnostic_baseline_training/per_state"
    params: dict[int, tuple[Any, Any]] = {}
    observation_cases = []
    intervention_cases = []
    answer_key = []
    for item in sampled:
        train_seed = item["train_seed"]
        episode_seed = item["episode_seed"]
        if train_seed not in params:
            saved = load_train_run(trained_root / f"seed_{train_seed}/trained_policies")
            params[train_seed] = (
                saved["generated_teammate_params"], saved["best_response_params"]
            )
        teammate_params, reference_params = params[train_seed]
        factual = rollout_episode(
            env, fixed_policy, fixed_params, teammate_policy, teammate_params,
            seed=episode_seed, max_steps=400,
        )
        reference = rollout_episode(
            env, reference_policy, reference_params, teammate_policy, teammate_params,
            seed=episode_seed, max_steps=400,
        )
        key = train_seed, episode_seed
        canonical_row = next(
            row for row in canonical["rows"]
            if (int(row["train_seed"]), int(row["episode_seed"])) == key
        )
        if (
            float(factual.total_return) != float(canonical_row["factual_return"])
            or float(reference.total_return) != float(canonical_row["reference_return"])
        ):
            raise ValueError(f"replayed return differs from canonical row {key}")
        observation_cases.append({
            "case_id": item["case_id"],
            "condition_A_return": float(factual.total_return),
            "condition_B_return": float(reference.total_return),
            "kitchen_layout": kitchen_layout(factual, env.agent_view_size),
            "condition_A_steps": records(factual),
            "condition_B_steps": records(reference),
            "instructions": "Describe observable coordination behavior before opening the intervention packet. Do not infer a unique cause from return alone.",
        })
        diagnosis = diagnosis_by_key.get(key)
        evidence = diagnosis["intervention_evidence"] if diagnosis else None
        intervention = (
            {
                "no_change_remaining_return": evidence["no_change_return"],
                "proposed_intervention_gain": evidence["primary_effect"],
                "duration_matched_pause_gain": evidence["wait_effect"],
                "alternative_intervention_gains": evidence["alternative_effects"],
            }
            if evidence else None
        )
        intervention_cases.append({
            "case_id": item["case_id"],
            "comparison": intervention,
            "instructions": "After the observation label, assess whether a replay changes return and whether the available controls isolate a unique mechanism. Null means no candidate test was run.",
        })
        answer_key.append({
            **item,
            "evaluator_candidate_kind": diagnosis["candidate_kind"] if diagnosis else None,
            "evaluator_decision": diagnosis["decision"] if diagnosis else "abstain",
            "evaluator_diagnosis": diagnosis["diagnosis"] if diagnosis else None,
            "reference_swap_context": soup_carrier_swap_context_summary(reference),
            "factual_swap_context": soup_carrier_swap_context_summary(factual),
        })
    return {
        "status": "annotation_pending_no_independent_human_labels",
        "sampling_seed": seed,
        "stratum_population_denominators": denominators,
        "observation_packet": observation_cases,
        "intervention_packet": intervention_cases,
        "evaluator_answer_key_not_ground_truth": answer_key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaxaht", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/validation/diagnostic_review_packet",
    )
    args = parser.parse_args()
    result = build_packet(args.jaxaht.resolve(), seed=20260923)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("observation_packet", "intervention_packet", "evaluator_answer_key_not_ground_truth"):
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(result[name], indent=2, sort_keys=True) + "\n")
    (args.output_dir / "observation_viewer.html").write_text(
        observation_viewer(result["observation_packet"]), encoding="utf-8"
    )
    metadata = {key: value for key, value in result.items() if not isinstance(value, list)}
    (args.output_dir / "sampling_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "blank_annotation_form.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_id", "reviewer_id", "observable_behavior", "step_start", "step_end",
            "observation_confidence", "plausible_mechanism", "intervention_improved_return",
            "mechanism_distinguished_from_alternatives", "alternative_explanation",
            "uncertain_or_insufficient_evidence_reason",
        ])
        writer.writerows([[case["case_id"]] + [""] * 10 for case in result["observation_packet"]])
    print(json.dumps({"cases": len(result["observation_packet"]), **metadata}, indent=2))


if __name__ == "__main__":
    main()
