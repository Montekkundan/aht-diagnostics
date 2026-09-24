"""Utilities for paired diagnosis of one naturally occurring interaction failure."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np

from .rollout import ContinuationResult, Episode, StepRecord
from .semantics import state_view


@dataclass(frozen=True)
class DeliveryOpportunity:
    """First predeclared state at which a delivery takeover can be tested."""

    step: int
    ego_inventory: str
    partner_inventory: str
    pot_statuses: tuple[int, ...]


@dataclass(frozen=True)
class MutualPlateBlockOpportunity:
    """First confirmed run of the predefined mutual plate-block trigger."""

    step: int
    confirmed_at_step: int
    consecutive_steps: int


@dataclass(frozen=True)
class SoupCarrierSwapLockOpportunity:
    """First confirmed swap lock involving a soup-carrying teammate."""

    step: int
    confirmed_at_step: int
    consecutive_steps: int
    ego_position: tuple[int, int]
    partner_position: tuple[int, int]


@dataclass(frozen=True)
class InitialCompletedSoupOpportunity:
    """Initial state in which the evaluated ego already holds completed soup."""

    step: int
    ego_position: tuple[int, int]
    partner_position: tuple[int, int]
    partner_inventory: str


@dataclass(frozen=True)
class InitialSoupDispositionOpportunity:
    """First event that disposes of completed soup held at episode start."""

    step: int
    event: str
    ego_position: tuple[int, int]
    partner_position: tuple[int, int]
    partner_inventory: str


def initial_completed_soup_opportunity(
    episode: Episode,
) -> InitialCompletedSoupOpportunity | None:
    """Return the first snapshot when the ego starts with completed soup.

    The predicate uses only the initial observable state. It does not inspect
    the episode return, later events, or any counterfactual branch outcome.
    """

    if not episode.snapshots:
        return None
    view = state_view(episode.snapshots[0].env_state)
    if len(view.positions) < 2 or len(view.inventories) < 2:
        return None
    if view.inventories[0] != "dish":
        return None
    return InitialCompletedSoupOpportunity(
        step=episode.snapshots[0].step,
        ego_position=view.positions[0],
        partner_position=view.positions[1],
        partner_inventory=view.inventories[1],
    )


def first_initial_soup_disposition(
    episode: Episode,
) -> InitialSoupDispositionOpportunity | None:
    """Find whether initially held completed soup is delivered or dropped first.

    The scan stops at the first transition that removes the initial soup from
    the ego inventory. Later soup pickups or dispositions are intentionally
    excluded.
    """

    if initial_completed_soup_opportunity(episode) is None:
        return None
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        if record.ego_event not in {"deliver_soup", "drop_soup"}:
            continue
        view = state_view(snapshot.env_state)
        return InitialSoupDispositionOpportunity(
            step=record.step,
            event=record.ego_event,
            ego_position=view.positions[0],
            partner_position=view.positions[1],
            partner_inventory=view.inventories[1],
        )
    return None


def first_ready_soup_opportunity(
    episode: Episode,
    *,
    max_steps: int,
    minimum_remaining_steps: int,
) -> DeliveryOpportunity | None:
    """Select the first state with ready soup and enough time for a fixed test.

    In Overcooked v1, a pot status of zero denotes ready soup. Selection uses
    only the factual state and the fixed time budget, never branch outcomes.
    """

    for snapshot in episode.snapshots:
        if max_steps - snapshot.step < minimum_remaining_steps:
            continue
        view = state_view(snapshot.env_state)
        if 0 in view.pot_statuses and view.inventories[0] != "dish":
            return DeliveryOpportunity(
                step=snapshot.step,
                ego_inventory=view.inventories[0],
                partner_inventory=view.inventories[1],
                pot_statuses=view.pot_statuses,
            )
    return None


def _mutual_plate_block_active(snapshot: object, record: StepRecord) -> bool:
    view = state_view(snapshot.env_state)
    adjacent = (
        abs(view.positions[0][0] - view.positions[1][0])
        + abs(view.positions[0][1] - view.positions[1][1])
        == 1
    )
    return (
        0 in view.pot_statuses
        and view.inventories[:2] == ("plate", "plate")
        and adjacent
        and record.ego_event == "blocked_move"
        and record.partner_event == "blocked_move"
    )


def first_mutual_plate_block_opportunity(
    episode: Episode,
    *,
    minimum_consecutive_steps: int,
) -> MutualPlateBlockOpportunity | None:
    """Return the start of the first sufficiently long factual lock run."""

    if minimum_consecutive_steps < 1:
        raise ValueError("minimum_consecutive_steps must be positive")
    run_start = 0
    run_length = 0
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        if _mutual_plate_block_active(snapshot, record):
            if run_length == 0:
                run_start = record.step
            run_length += 1
            if run_length == minimum_consecutive_steps:
                return MutualPlateBlockOpportunity(
                    step=run_start,
                    confirmed_at_step=record.step,
                    consecutive_steps=run_length,
                )
        else:
            run_length = 0
    return None


_MOVEMENT_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0),
}


def _movement_target(position: tuple[int, int], action: int) -> tuple[int, int] | None:
    delta = _MOVEMENT_DELTAS.get(int(action))
    if delta is None:
        return None
    return position[0] + delta[0], position[1] + delta[1]


def _soup_carrier_swap_lock_active(snapshot: object, record: StepRecord) -> bool:
    """Check an observable face-to-face swap conflict.

    The teammate must hold a completed soup, the agents must occupy adjacent
    cells, and each must attempt to move into the other's current cell. Both
    transitions must be recorded as true blocked movement. This describes the
    interaction trajectory only; it does not infer either policy's intent.
    """

    view = state_view(snapshot.env_state)
    if len(view.positions) < 2 or len(view.inventories) < 2:
        return False
    ego_position, partner_position = view.positions[:2]
    return (
        view.inventories[1] == "dish"
        and _movement_target(ego_position, record.ego_action) == partner_position
        and _movement_target(partner_position, record.partner_action) == ego_position
        and record.ego_event == "blocked_move"
        and record.partner_event == "blocked_move"
    )


def first_soup_carrier_swap_lock_opportunity(
    episode: Episode,
    *,
    minimum_consecutive_steps: int,
) -> SoupCarrierSwapLockOpportunity | None:
    """Return the first sufficiently long soup-carrier swap-lock run."""

    if minimum_consecutive_steps < 1:
        raise ValueError("minimum_consecutive_steps must be positive")
    run_start = 0
    run_length = 0
    start_positions: tuple[tuple[int, int], tuple[int, int]] | None = None
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        if _soup_carrier_swap_lock_active(snapshot, record):
            if run_length == 0:
                run_start = record.step
                view = state_view(snapshot.env_state)
                start_positions = (view.positions[0], view.positions[1])
            run_length += 1
            if run_length == minimum_consecutive_steps:
                assert start_positions is not None
                return SoupCarrierSwapLockOpportunity(
                    step=run_start,
                    confirmed_at_step=record.step,
                    consecutive_steps=run_length,
                    ego_position=start_positions[0],
                    partner_position=start_positions[1],
                )
        else:
            run_length = 0
            start_positions = None
    return None


def soup_carrier_swap_context_summary(episode: Episode) -> dict[str, int | bool]:
    """Summarize how far an episode progresses toward the swap-lock trigger.

    The counts form an exposure funnel. They distinguish episodes in which the
    teammate never carries soup from episodes with an adjacent soup carrier,
    reciprocal swap attempts, or a sustained true blocked swap. This function
    is descriptive only and does not assign a behavioural diagnosis.
    """

    teammate_soup_steps = 0
    adjacent_soup_steps = 0
    reciprocal_attempt_steps = 0
    blocked_swap_steps = 0
    longest_blocked_swap_run = 0
    current_blocked_swap_run = 0

    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        view = state_view(snapshot.env_state)
        if len(view.positions) < 2 or len(view.inventories) < 2:
            current_blocked_swap_run = 0
            continue
        ego_position, partner_position = view.positions[:2]
        teammate_has_soup = view.inventories[1] == "dish"
        adjacent = (
            abs(ego_position[0] - partner_position[0])
            + abs(ego_position[1] - partner_position[1])
            == 1
        )
        reciprocal_attempt = (
            teammate_has_soup
            and _movement_target(ego_position, record.ego_action) == partner_position
            and _movement_target(partner_position, record.partner_action) == ego_position
        )
        blocked_swap = reciprocal_attempt and _soup_carrier_swap_lock_active(
            snapshot,
            record,
        )

        teammate_soup_steps += int(teammate_has_soup)
        adjacent_soup_steps += int(teammate_has_soup and adjacent)
        reciprocal_attempt_steps += int(reciprocal_attempt)
        blocked_swap_steps += int(blocked_swap)
        if blocked_swap:
            current_blocked_swap_run += 1
            longest_blocked_swap_run = max(
                longest_blocked_swap_run,
                current_blocked_swap_run,
            )
        else:
            current_blocked_swap_run = 0

    return {
        "has_teammate_soup": teammate_soup_steps > 0,
        "has_adjacent_soup_carrier": adjacent_soup_steps > 0,
        "has_reciprocal_swap_attempt": reciprocal_attempt_steps > 0,
        "teammate_soup_steps": teammate_soup_steps,
        "adjacent_soup_steps": adjacent_soup_steps,
        "reciprocal_attempt_steps": reciprocal_attempt_steps,
        "blocked_swap_steps": blocked_swap_steps,
        "longest_blocked_swap_run": longest_blocked_swap_run,
    }


def event_counts(records: Iterable[StepRecord]) -> dict[str, int]:
    """Count ego semantic events in a branch."""

    return dict(sorted(Counter(record.ego_event for record in records).items()))


def branch_summary(result: ContinuationResult) -> dict[str, object]:
    """Return task and behavioural outcomes for one paired continuation."""

    records = result.intervention_records + result.continuation_records
    delivery_offsets = [
        record.step - result.start_step
        for record in records
        if record.ego_event == "deliver_soup"
        or record.partner_event == "deliver_soup"
    ]
    blocked = sum(record.ego_event == "blocked_move" for record in records)
    oriented = sum(record.ego_event == "orientation_change" for record in records)
    return {
        "start_step": result.start_step,
        "end_step": result.end_step,
        "starting_return": result.starting_return,
        "total_return": result.total_return,
        "remaining_return": result.remaining_return,
        "intervention_steps": result.intervention_steps,
        "ego_event_counts": event_counts(records),
        "intervention_event_counts": event_counts(result.intervention_records),
        "delivered_dishes": len(delivery_offsets),
        "time_to_first_delivery": delivery_offsets[0] if delivery_offsets else None,
        "blocked_move_fraction": blocked / len(records) if records else 0.0,
        "orientation_change_fraction": oriented / len(records) if records else 0.0,
        "immobile_motion_fraction": (
            (blocked + oriented) / len(records) if records else 0.0
        ),
    }


def mutual_plate_block_summary(episode: Episode) -> dict[str, object]:
    """Describe repeated adjacent blocking while both agents hold plates.

    The trigger is deliberately observable and narrow: soup is ready, both
    agents hold plates, they occupy adjacent cells, and both attempted motions
    leave their positions and orientations unchanged. This is a candidate
    coordination-lock signature, not proof of an internal policy mechanism.
    """

    active_steps: list[int] = []
    runs: list[list[int]] = []
    current_run: list[int] = []
    for snapshot, record in zip(episode.snapshots, episode.steps, strict=True):
        active = _mutual_plate_block_active(snapshot, record)
        if active:
            active_steps.append(record.step)
            current_run.append(record.step)
        elif current_run:
            runs.append(current_run)
            current_run = []
    if current_run:
        runs.append(current_run)

    return {
        "triggered": bool(active_steps),
        "total_steps": len(active_steps),
        "longest_run": max((len(run) for run in runs), default=0),
        "first_step": active_steps[0] if active_steps else None,
        "run_count": len(runs),
    }


def paired_bootstrap_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Episode-level percentile bootstrap interval for paired differences."""

    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        raise ValueError("At least one paired difference is required")
    if data.size == 1:
        value = float(data[0])
        return {"mean": value, "ci_low": value, "ci_high": value, "n": 1}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(samples, data.size))
    means = data[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(data.size),
    }


def clustered_paired_bootstrap_interval(
    values: Iterable[float],
    clusters: Iterable[object],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Bootstrap paired differences while resampling whole dependence clusters."""

    data = np.asarray(list(values), dtype=float)
    labels = np.asarray(list(clusters), dtype=object)
    if data.size == 0:
        raise ValueError("At least one paired difference is required")
    if data.size != labels.size:
        raise ValueError("values and clusters must have equal length")
    unique_labels = list(dict.fromkeys(labels.tolist()))
    grouped = [data[labels == label] for label in unique_labels]
    if len(grouped) == 1:
        value = float(data.mean())
        return {
            "mean": value,
            "ci_low": None,
            "ci_high": None,
            "n": int(data.size),
            "clusters": 1,
        }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        bootstrap_means[index] = np.concatenate(
            [grouped[group_index] for group_index in selected]
        ).mean()
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "mean": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(data.size),
        "clusters": len(grouped),
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Wilson score interval for a binomial success proportion."""

    if total < 1:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return {
        "proportion": float(proportion),
        "ci_low": 0.0 if successes == 0 else float(max(0.0, center - margin)),
        "ci_high": (
            1.0 if successes == total else float(min(1.0, center + margin))
        ),
        "successes": successes,
        "n": total,
    }
