"""Exact-state Overcooked rollouts and counterfactual action repairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .semantics import COOPERATIVE_EVENTS, semantic_event, state_view


@dataclass(frozen=True)
class Snapshot:
    step: int
    obs: dict[str, Any]
    env_state: Any
    done: dict[str, Any]
    reward: dict[str, Any]
    previous_actions: dict[str, Any]
    ego_hstate: Any
    partner_hstate: Any
    rng: Any
    cumulative_return: float


@dataclass(frozen=True)
class StepRecord:
    step: int
    ego_action: int
    partner_action: int
    reward: float
    cumulative_return: float
    ego_probs: np.ndarray
    ego_position: tuple[int, int]
    partner_position: tuple[int, int]
    ego_inventory: str
    partner_inventory: str
    ego_event: str
    partner_event: str
    ego_base_action: int
    ego_defect_type: str
    ego_defect_triggered: bool
    ego_defect_suppressed: bool
    expected_repair_event: str
    ego_direction: int | None = None
    partner_direction: int | None = None


@dataclass(frozen=True)
class Episode:
    seed: int
    total_return: float
    snapshots: tuple[Snapshot, ...]
    steps: tuple[StepRecord, ...]


@dataclass(frozen=True)
class RepairRecord:
    seed: int
    step: int
    original_action: int
    repaired_action: int
    original_return: float
    repaired_return: float
    gain: float
    label: str
    original_event: str
    repaired_event: str
    ego_position: tuple[int, int]
    partner_position: tuple[int, int]
    window: int = 1


@dataclass(frozen=True)
class ContinuationResult:
    """Outcome of replaying one branch from an exact saved snapshot."""

    start_step: int
    end_step: int
    starting_return: float
    total_return: float
    remaining_return: float
    intervention_steps: int
    intervention_records: tuple[StepRecord, ...]
    continuation_records: tuple[StepRecord, ...]


def _scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _policy_step(
    policy: Any,
    params: Any,
    obs: Any,
    done: Any,
    avail_actions: Any,
    hstate: Any,
    rng: Any,
    env_state: Any,
    test_mode: bool,
    aux_obs: Any = None,
) -> tuple[int, Any, np.ndarray, dict[str, Any]]:
    action_args = {
        "params": params,
        "obs": obs.reshape(1, 1, -1),
        "done": done.reshape(1, 1),
        "avail_actions": avail_actions.astype(jnp.float32),
        "hstate": hstate,
        "rng": rng,
        "aux_obs": aux_obs,
        "env_state": env_state,
        "test_mode": test_mode,
    }
    if hasattr(policy, "get_action_diagnostic"):
        action, new_hstate, metadata = policy.get_action_diagnostic(**action_args)
    else:
        action, new_hstate = policy.get_action(**action_args)
        metadata = {}
    action_int = int(np.asarray(action).squeeze())

    probs = np.zeros(int(np.asarray(avail_actions).size), dtype=np.float64)
    if hasattr(policy, "get_action_value_policy"):
        try:
            _, _, distribution, _ = policy.get_action_value_policy(
                params=params,
                obs=obs.reshape(1, 1, -1),
                done=done.reshape(1, 1),
                avail_actions=avail_actions.astype(jnp.float32),
                hstate=hstate,
                rng=rng,
                aux_obs=aux_obs,
                env_state=env_state,
            )
            probs = np.asarray(distribution.probs).reshape(-1).astype(np.float64)
        except (AttributeError, TypeError):
            probs[action_int] = 1.0
    else:
        probs[action_int] = 1.0
    metadata = {
        "base_action": action_int,
        "defect_type": "healthy",
        "defect_triggered": False,
        "defect_suppressed": False,
        "expected_repair_event": "none",
        **metadata,
    }
    return action_int, new_hstate, probs, metadata


def _initial_transition(env: Any, ego_policy: Any, partner_policy: Any, seed: int) -> Snapshot:
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {agent: jnp.zeros((1,), dtype=bool) for agent in env.agents + ["__all__"]}
    reward = {agent: jnp.zeros((1,), dtype=jnp.float32) for agent in env.agents}
    previous_actions = {
        agent: jnp.zeros((env.action_space(agent).n,), dtype=jnp.float32)
        for agent in env.agents
    }
    return Snapshot(
        step=0,
        obs=obs,
        env_state=env_state,
        done=done,
        reward=reward,
        previous_actions=previous_actions,
        ego_hstate=ego_policy.init_hstate(1, aux_info={"agent_id": 0}),
        partner_hstate=partner_policy.init_hstate(1, aux_info={"agent_id": 1}),
        rng=rng,
        cumulative_return=0.0,
    )


def _ego_aux(snapshot: Snapshot) -> tuple[Any, Any, Any]:
    joint_previous = jnp.concatenate(
        (
            snapshot.previous_actions["agent_0"].reshape(1, 1, -1),
            snapshot.previous_actions["agent_1"].reshape(1, 1, -1),
        ),
        axis=-1,
    )
    return (
        snapshot.previous_actions["agent_0"].reshape(1, 1, -1),
        joint_previous,
        snapshot.reward["agent_0"].reshape(1, 1, -1),
    )


def _take_step(
    env: Any,
    snapshot: Snapshot,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    ego_test_mode: bool,
    partner_test_mode: bool,
    ego_action_override: int | None = None,
) -> tuple[Snapshot, StepRecord]:
    avail = jax.lax.stop_gradient(env.get_avail_actions(snapshot.env_state))
    ego_aux = _ego_aux(snapshot)

    rng, ego_rng, partner_rng, step_rng = jax.random.split(snapshot.rng, 4)
    ego_action, ego_hstate, ego_probs, ego_metadata = _policy_step(
        ego_policy,
        ego_params,
        snapshot.obs["agent_0"],
        snapshot.done["agent_0"],
        avail["agent_0"],
        snapshot.ego_hstate,
        ego_rng,
        snapshot.env_state,
        ego_test_mode,
        ego_aux,
    )
    partner_action, partner_hstate, _, _ = _policy_step(
        partner_policy,
        partner_params,
        snapshot.obs["agent_1"],
        snapshot.done["agent_1"],
        avail["agent_1"],
        snapshot.partner_hstate,
        partner_rng,
        snapshot.env_state,
        partner_test_mode,
    )

    if ego_action_override is not None:
        ego_action = int(ego_action_override)

    actions = {
        "agent_0": jnp.asarray(ego_action, dtype=jnp.int32),
        "agent_1": jnp.asarray(partner_action, dtype=jnp.int32),
    }
    before = state_view(snapshot.env_state)
    next_obs, next_state, reward, done, _ = env.step(step_rng, snapshot.env_state, actions)
    after = state_view(next_state)
    step_reward = _scalar(reward["agent_0"])
    cumulative_return = snapshot.cumulative_return + step_reward
    next_previous = {
        agent: jax.nn.one_hot(actions[agent], env.action_space(agent).n)
        for agent in env.agents
    }

    record = StepRecord(
        step=snapshot.step,
        ego_action=ego_action,
        partner_action=partner_action,
        reward=step_reward,
        cumulative_return=cumulative_return,
        ego_probs=ego_probs,
        ego_position=before.positions[0],
        partner_position=before.positions[1],
        ego_inventory=before.inventories[0],
        partner_inventory=before.inventories[1],
        ego_event=semantic_event(before, after, 0, ego_action, step_reward),
        partner_event=semantic_event(
            before,
            after,
            1,
            partner_action,
            _scalar(reward["agent_1"]),
        ),
        ego_base_action=int(ego_metadata["base_action"]),
        ego_defect_type=str(ego_metadata["defect_type"]),
        ego_defect_triggered=bool(ego_metadata["defect_triggered"]),
        ego_defect_suppressed=bool(ego_metadata["defect_suppressed"]),
        expected_repair_event=str(ego_metadata["expected_repair_event"]),
        ego_direction=before.directions[0] if before.directions else None,
        partner_direction=before.directions[1] if len(before.directions) > 1 else None,
    )
    next_snapshot = Snapshot(
        step=snapshot.step + 1,
        obs=next_obs,
        env_state=next_state,
        done=done,
        reward=reward,
        previous_actions=next_previous,
        ego_hstate=ego_hstate,
        partner_hstate=partner_hstate,
        rng=rng,
        cumulative_return=cumulative_return,
    )
    return next_snapshot, record


def rollout_episode(
    env: Any,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    seed: int,
    max_steps: int,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
) -> Episode:
    snapshot = _initial_transition(env, ego_policy, partner_policy, seed)
    snapshots: list[Snapshot] = []
    records: list[StepRecord] = []
    for _ in range(max_steps):
        if bool(np.asarray(snapshot.done["__all__"]).squeeze()):
            break
        snapshots.append(snapshot)
        snapshot, record = _take_step(
            env,
            snapshot,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
        )
        records.append(record)
    return Episode(seed, snapshot.cumulative_return, tuple(snapshots), tuple(records))


def rollout_continuation(
    env: Any,
    snapshot: Snapshot,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    max_steps: int,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
) -> ContinuationResult:
    """Resume both factual policies from a saved snapshot without intervention."""

    current = snapshot
    records: list[StepRecord] = []
    while current.step < max_steps and not bool(
        np.asarray(current.done["__all__"]).squeeze()
    ):
        current, record = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
        )
        records.append(record)
    return ContinuationResult(
        start_step=snapshot.step,
        end_step=current.step,
        starting_return=snapshot.cumulative_return,
        total_return=current.cumulative_return,
        remaining_return=current.cumulative_return - snapshot.cumulative_return,
        intervention_steps=0,
        intervention_records=(),
        continuation_records=tuple(records),
    )


def rollout_repair(
    env: Any,
    snapshot: Snapshot,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    repaired_action: int,
    max_steps: int,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
) -> tuple[float, str]:
    current, repaired_record = _take_step(
        env,
        snapshot,
        ego_policy,
        ego_params,
        partner_policy,
        partner_params,
        ego_test_mode,
        partner_test_mode,
        ego_action_override=repaired_action,
    )
    while current.step < max_steps and not bool(np.asarray(current.done["__all__"]).squeeze()):
        current, _ = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
        )
    return current.cumulative_return, repaired_record.ego_event


def policy_states_on_factual_trajectory(
    env: Any,
    episode: Episode,
    policy: Any,
    params: Any,
    test_mode: bool = True,
) -> list[Any]:
    """Track a policy's memory on the factual states without changing the rollout."""

    hstate = policy.init_hstate(1, aux_info={"agent_id": 0})
    states: list[Any] = []
    for snapshot in episode.snapshots:
        states.append(hstate)
        avail = env.get_avail_actions(snapshot.env_state)["agent_0"]
        _, hstate, _, _ = _policy_step(
            policy,
            params,
            snapshot.obs["agent_0"],
            snapshot.done["agent_0"],
            avail,
            hstate,
            jax.random.fold_in(snapshot.rng, 9173),
            snapshot.env_state,
            test_mode,
            _ego_aux(snapshot),
        )
    return states


def rollout_policy_option(
    env: Any,
    snapshot: Snapshot,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    controller_policy: Any,
    controller_params: Any,
    controller_hstate: Any,
    max_option_steps: int,
    max_steps: int,
    *,
    terminate_on_events: frozenset[str] = frozenset(),
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
    controller_test_mode: bool = True,
) -> ContinuationResult:
    """Execute a temporary multi-step controller, then resume the frozen ego.

    The teammate remains a live policy throughout the branch. The frozen ego's
    recurrent state is updated using the observations and actual overridden
    actions experienced during the intervention.
    """

    if max_option_steps < 0:
        raise ValueError("max_option_steps must be non-negative")

    current = snapshot
    controller_rng = jax.random.fold_in(snapshot.rng, 9173)
    intervention_records: list[StepRecord] = []
    for _ in range(max_option_steps):
        if current.step >= max_steps or bool(
            np.asarray(current.done["__all__"]).squeeze()
        ):
            break
        controller_rng, action_rng = jax.random.split(controller_rng)
        avail = env.get_avail_actions(current.env_state)["agent_0"]
        controller_action, controller_hstate, _, _ = _policy_step(
            controller_policy,
            controller_params,
            current.obs["agent_0"],
            current.done["agent_0"],
            avail,
            controller_hstate,
            action_rng,
            current.env_state,
            controller_test_mode,
            _ego_aux(current),
        )
        current, record = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
            ego_action_override=controller_action,
        )
        intervention_records.append(record)
        if record.ego_event in terminate_on_events:
            break

    continuation_records: list[StepRecord] = []
    while current.step < max_steps and not bool(
        np.asarray(current.done["__all__"]).squeeze()
    ):
        current, record = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
        )
        continuation_records.append(record)

    return ContinuationResult(
        start_step=snapshot.step,
        end_step=current.step,
        starting_return=snapshot.cumulative_return,
        total_return=current.cumulative_return,
        remaining_return=current.cumulative_return - snapshot.cumulative_return,
        intervention_steps=len(intervention_records),
        intervention_records=tuple(intervention_records),
        continuation_records=tuple(continuation_records),
    )


def rollout_policy_splice(
    env: Any,
    snapshot: Snapshot,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    reference_policy: Any,
    reference_params: Any,
    reference_hstate: Any,
    repair_window: int,
    max_steps: int,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
    reference_test_mode: bool = True,
) -> tuple[float, int, str]:
    """Replace ego actions with a feasible response policy for a short window."""
    current = snapshot
    reference_rng = jax.random.fold_in(snapshot.rng, 9173)
    first_action = -1
    first_event = "none"
    for splice_step in range(repair_window):
        if current.step >= max_steps or bool(np.asarray(current.done["__all__"]).squeeze()):
            break
        reference_rng, action_rng = jax.random.split(reference_rng)
        avail = env.get_avail_actions(current.env_state)["agent_0"]
        reference_action, reference_hstate, _, _ = _policy_step(
            reference_policy,
            reference_params,
            current.obs["agent_0"],
            current.done["agent_0"],
            avail,
            reference_hstate,
            action_rng,
            current.env_state,
            reference_test_mode,
            _ego_aux(current),
        )
        if splice_step == 0:
            first_action = reference_action
        current, repaired_record = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
            ego_action_override=reference_action,
        )
        if splice_step == 0:
            first_event = repaired_record.ego_event

    while current.step < max_steps and not bool(np.asarray(current.done["__all__"]).squeeze()):
        current, _ = _take_step(
            env,
            current,
            ego_policy,
            ego_params,
            partner_policy,
            partner_params,
            ego_test_mode,
            partner_test_mode,
        )
    return current.cumulative_return, first_action, first_event


def _repair_label(record: StepRecord, repaired_event: str) -> str:
    if (
        record.ego_event == record.partner_event
        and record.ego_event in COOPERATIVE_EVENTS
        and repaired_event != record.ego_event
    ):
        return f"duplicate_{record.ego_event}"
    if repaired_event in COOPERATIVE_EVENTS and repaired_event != record.ego_event:
        return f"missed_{repaired_event}"
    if repaired_event == "wait" and record.partner_event in COOPERATIVE_EVENTS:
        return "failure_to_yield"
    if record.ego_event == "blocked_move" and repaired_event == "move":
        return "navigation_block"
    return "action_selection"


def counterfactual_repair_profile(
    env: Any,
    episode: Episode,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    max_steps: int,
    min_gain: float = 1e-8,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
) -> list[RepairRecord]:
    repairs: list[RepairRecord] = []
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        available = np.asarray(env.get_avail_actions(snapshot.env_state)["agent_0"]).astype(bool)
        best_action = record.ego_action
        best_return = episode.total_return
        best_event = record.ego_event
        for action in np.flatnonzero(available):
            action = int(action)
            if action == record.ego_action:
                continue
            repaired_return, repaired_event = rollout_repair(
                env,
                snapshot,
                ego_policy,
                ego_params,
                partner_policy,
                partner_params,
                action,
                max_steps,
                ego_test_mode,
                partner_test_mode,
            )
            if repaired_return > best_return:
                best_return = repaired_return
                best_action = action
                best_event = repaired_event
        gain = best_return - episode.total_return
        if best_action != record.ego_action and gain > min_gain:
            repairs.append(
                RepairRecord(
                    seed=episode.seed,
                    step=record.step,
                    original_action=record.ego_action,
                    repaired_action=best_action,
                    original_return=episode.total_return,
                    repaired_return=best_return,
                    gain=gain,
                    label=_repair_label(record, best_event),
                    original_event=record.ego_event,
                    repaired_event=best_event,
                    ego_position=record.ego_position,
                    partner_position=record.partner_position,
                )
            )
    return repairs


def counterfactual_splice_profile(
    env: Any,
    episode: Episode,
    ego_policy: Any,
    ego_params: Any,
    partner_policy: Any,
    partner_params: Any,
    reference_policy: Any,
    reference_params: Any,
    max_steps: int,
    windows: tuple[int, ...] = (1, 2, 4, 8, 16),
    min_gain: float = 1e-8,
    ego_test_mode: bool = True,
    partner_test_mode: bool = True,
    reference_test_mode: bool = True,
    candidate_steps: set[int] | None = None,
) -> list[RepairRecord]:
    """Find short feasible-policy interventions that repair a failed rollout."""
    reference_hstates = policy_states_on_factual_trajectory(
        env,
        episode,
        reference_policy,
        reference_params,
        reference_test_mode,
    )
    repairs: list[RepairRecord] = []
    for snapshot, record, reference_hstate in zip(
        episode.snapshots,
        episode.steps,
        reference_hstates,
        strict=True,
    ):
        if candidate_steps is not None and record.step not in candidate_steps:
            continue
        best_return = episode.total_return
        best_action = record.ego_action
        best_event = record.ego_event
        best_window = 0
        for window in windows:
            repaired_return, first_action, first_event = rollout_policy_splice(
                env,
                snapshot,
                ego_policy,
                ego_params,
                partner_policy,
                partner_params,
                reference_policy,
                reference_params,
                reference_hstate,
                window,
                max_steps,
                ego_test_mode,
                partner_test_mode,
                reference_test_mode,
            )
            if repaired_return > best_return:
                best_return = repaired_return
                best_action = first_action
                best_event = first_event
                best_window = window
        gain = best_return - episode.total_return
        if best_window > 0 and gain > min_gain:
            repairs.append(
                RepairRecord(
                    seed=episode.seed,
                    step=record.step,
                    original_action=record.ego_action,
                    repaired_action=best_action,
                    original_return=episode.total_return,
                    repaired_return=best_return,
                    gain=gain,
                    label=_repair_label(record, best_event),
                    original_event=record.ego_event,
                    repaired_event=best_event,
                    ego_position=record.ego_position,
                    partner_position=record.partner_position,
                    window=best_window,
                )
            )
    return repairs


def reference_action_trace(
    env: Any,
    episode: Episode,
    reference_policy: Any,
    reference_params: Any,
    reference_test_mode: bool = True,
) -> tuple[list[int], float]:
    """Return feasible-response actions and KL(reference policy || ego policy)."""
    hstate = reference_policy.init_hstate(1, aux_info={"agent_id": 0})
    actions: list[int] = []
    losses: list[float] = []
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        avail = env.get_avail_actions(snapshot.env_state)["agent_0"]
        action, hstate, reference_probs, _ = _policy_step(
            reference_policy,
            reference_params,
            snapshot.obs["agent_0"],
            snapshot.done["agent_0"],
            avail,
            hstate,
            jax.random.fold_in(snapshot.rng, 9173),
            snapshot.env_state,
            reference_test_mode,
            _ego_aux(snapshot),
        )
        actions.append(action)
        reference_probs = np.asarray(reference_probs, dtype=np.float64)
        ego_probs = np.asarray(record.ego_probs, dtype=np.float64)
        reference_probs = np.clip(reference_probs, 0.0, 1.0)
        ego_probs = np.clip(ego_probs, 1e-8, 1.0)
        if reference_probs.sum() <= 0:
            reference_probs = np.zeros_like(ego_probs)
            reference_probs[action] = 1.0
        else:
            reference_probs = reference_probs / reference_probs.sum()
        positive = reference_probs > 0
        losses.append(
            float(
                np.sum(
                    reference_probs[positive]
                    * (
                        np.log(reference_probs[positive])
                        - np.log(ego_probs[positive])
                    )
                )
            )
        )
    return actions, float(np.mean(losses)) if losses else 0.0


def reference_action_diagnostics(
    env: Any,
    episode: Episode,
    reference_policy: Any,
    reference_params: Any,
    reference_test_mode: bool = True,
) -> tuple[list[int], np.ndarray]:
    """Return reference actions and one KL score for every factual step."""
    hstate = reference_policy.init_hstate(1, aux_info={"agent_id": 0})
    actions: list[int] = []
    losses: list[float] = []
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        avail = env.get_avail_actions(snapshot.env_state)["agent_0"]
        action, hstate, reference_probs, _ = _policy_step(
            reference_policy,
            reference_params,
            snapshot.obs["agent_0"],
            snapshot.done["agent_0"],
            avail,
            hstate,
            jax.random.fold_in(snapshot.rng, 9173),
            snapshot.env_state,
            reference_test_mode,
            _ego_aux(snapshot),
        )
        actions.append(action)
        reference_probs = np.asarray(reference_probs, dtype=np.float64)
        ego_probs = np.asarray(record.ego_probs, dtype=np.float64)
        reference_probs = np.clip(reference_probs, 0.0, 1.0)
        ego_probs = np.clip(ego_probs, 1e-8, 1.0)
        if reference_probs.sum() <= 0:
            reference_probs = np.zeros_like(ego_probs)
            reference_probs[action] = 1.0
        else:
            reference_probs = reference_probs / reference_probs.sum()
        positive = reference_probs > 0
        losses.append(
            float(
                np.sum(
                    reference_probs[positive]
                    * (
                        np.log(reference_probs[positive])
                        - np.log(ego_probs[positive])
                    )
                )
            )
        )
    return actions, np.asarray(losses, dtype=np.float64)
