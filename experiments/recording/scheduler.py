"""
Trial pool generation, randomization strategies, balance validation, and
schedule.json persistence (Part III of the spec).

Owns: trial pool, randomization, balance validation, schedule.json.
Never: audio, UI, WAV (REQ-57).
"""

from __future__ import annotations

import dataclasses
import heapq
import json
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

from .errors import ScheduleError

SCHEDULE_SCHEMA_VERSION = "1.0"


@dataclasses.dataclass(frozen=True)
class ScheduledTrial:
    trial_id: int
    label: str
    repetition: int

    def to_dict(self) -> dict:
        return {"trial_id": self.trial_id, "label": self.label, "repetition": self.repetition}


@dataclasses.dataclass
class Schedule:
    schema_version: str
    seed: int
    strategy: str
    class_definition_version: str
    repetitions_per_class: int
    total_trials: int
    trials: list  # list[ScheduledTrial]

    def by_trial_id(self) -> dict:
        return {t.trial_id: t for t in self.trials}

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "strategy": self.strategy,
            "class_definition_version": self.class_definition_version,
            "repetitions_per_class": self.repetitions_per_class,
            "total_trials": self.total_trials,
            "trials": [t.to_dict() for t in self.trials],
        }


# ---------------------------------------------------------------------------
# Pool generation (REQ-9)
# ---------------------------------------------------------------------------


def build_pool(classes: Sequence[str], repetitions_per_class: int) -> dict:
    """Cartesian product of classes x repetitions, grouped by class (REQ-9.1)."""
    return {label: [(label, rep) for rep in range(1, repetitions_per_class + 1)] for label in classes}


# ---------------------------------------------------------------------------
# Randomization strategies (REQ-10.2). Each strategy takes the per-class
# pool (dict[label] -> list[(label, repetition)]) plus a seeded RNG and any
# strategy-specific kwargs, and returns a flat, reordered list of
# (label, repetition) tuples. New strategies register here without any
# change to the recorder, trial engine, or writer (REQ-10.4).
# ---------------------------------------------------------------------------

StrategyFn = Callable[..., list]

_STRATEGIES: dict[str, StrategyFn] = {}


def register_strategy(name: str) -> Callable[[StrategyFn], StrategyFn]:
    def deco(fn: StrategyFn) -> StrategyFn:
        _STRATEGIES[name] = fn
        return fn

    return deco


@register_strategy("random")
def _strategy_random(pool_by_label: dict, rng: random.Random, **_kwargs) -> list:
    flat = [item for items in pool_by_label.values() for item in items]
    rng.shuffle(flat)
    return flat


@register_strategy("balanced_random")
def _strategy_balanced_random(pool_by_label: dict, rng: random.Random, **_kwargs) -> list:
    """Exact per-class counts (guaranteed by pool construction), then a
    global reordering that never places two trials of the same class back
    to back (REQ-10.1), via a randomized largest-remaining-count merge.
    """
    remaining = {label: list(items) for label, items in pool_by_label.items()}
    for items in remaining.values():
        rng.shuffle(items)

    heap = []
    for label, items in remaining.items():
        if items:
            heapq.heappush(heap, (-len(items), rng.random(), label))

    result: list = []
    prev_label: Optional[str] = None
    deferred = None  # (neg_count, tiebreak, label) temporarily set aside

    while heap or deferred:
        if deferred is not None:
            heapq.heappush(heap, deferred)
            deferred = None

        neg_count, _tie, label = heapq.heappop(heap)
        if label == prev_label:
            if not heap:
                # Only one class has items left; a same-class run is
                # unavoidable (shouldn't happen for a balanced multi-class
                # pool, but never crash on it).
                heapq.heappush(heap, (neg_count, rng.random(), label))
            else:
                deferred = (neg_count, rng.random(), label)
                neg_count, _tie, label = heapq.heappop(heap)

        item = remaining[label].pop()
        result.append(item)
        count = -neg_count - 1
        if count > 0:
            heapq.heappush(heap, (-count, rng.random(), label))
        prev_label = label

    return result


@register_strategy("block_random")
def _strategy_block_random(pool_by_label: dict, rng: random.Random, block_size: int, **_kwargs) -> list:
    """Balanced within blocks of `block_size` trials: each block is filled
    by round-robin sampling across classes (so class share stays even
    within the block), then shuffled internally.
    """
    if not block_size or block_size <= 0:
        raise ScheduleError("block_random requires a positive block_size")

    remaining = {label: list(items) for label, items in pool_by_label.items()}
    for items in remaining.values():
        rng.shuffle(items)

    labels_cycle = sorted(remaining)
    result: list = []
    while any(remaining.values()):
        block: list = []
        idx = 0
        active = [l for l in labels_cycle if remaining[l]]
        while len(block) < block_size and active:
            label = active[idx % len(active)]
            if remaining[label]:
                block.append(remaining[label].pop())
                idx += 1
            active = [l for l in labels_cycle if remaining[l]]
        rng.shuffle(block)
        result.extend(block)
    return result


@register_strategy("manual")
def _strategy_manual(pool_by_label: dict, rng: random.Random, manual_order_file: str, **_kwargs) -> list:
    """Order supplied by an external JSON file: a list of
    {"label": ..., "repetition": ...} objects that must be exactly a
    permutation of the generated pool (Open Point E.5)."""
    path = Path(manual_order_file)
    if not path.is_file():
        raise ScheduleError(f"manual_order_file not found: {path}")
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScheduleError(f"manual_order_file is not valid JSON: {exc}") from exc

    flat = [(e["label"], e["repetition"]) for e in entries]
    expected = sorted(item for items in pool_by_label.values() for item in items)
    if sorted(flat) != expected:
        raise ScheduleError(
            "manual_order_file does not contain exactly one entry per "
            "(class, repetition) pair in the generated pool"
        )
    return flat


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------


def generate_schedule(
    classes: Sequence[str],
    repetitions_per_class: int,
    strategy: str,
    seed: int,
    class_definition_version: str,
    block_size: Optional[int] = None,
    manual_order_file: Optional[str] = None,
) -> Schedule:
    if strategy not in _STRATEGIES:
        raise ScheduleError(f"unknown randomization strategy: {strategy!r}")
    if repetitions_per_class <= 0:
        raise ScheduleError("repetitions_per_class must be positive")
    if not classes:
        raise ScheduleError("class list is empty")

    pool_by_label = build_pool(classes, repetitions_per_class)

    # REQ-11.4: a dedicated, explicitly seeded RNG instance. Global random
    # module state is never touched.
    rng = random.Random(seed)

    ordered = _STRATEGIES[strategy](
        pool_by_label, rng, block_size=block_size, manual_order_file=manual_order_file
    )

    trials = [
        ScheduledTrial(trial_id=i + 1, label=label, repetition=rep)
        for i, (label, rep) in enumerate(ordered)
    ]

    schedule = Schedule(
        schema_version=SCHEDULE_SCHEMA_VERSION,
        seed=seed,
        strategy=strategy,
        class_definition_version=class_definition_version,
        repetitions_per_class=repetitions_per_class,
        total_trials=len(trials),
        trials=trials,
    )
    validate_balance(schedule, classes, repetitions_per_class)
    return schedule


# ---------------------------------------------------------------------------
# Balance validation (§12)
# ---------------------------------------------------------------------------


def validate_balance(schedule: Schedule, classes: Sequence[str], repetitions_per_class: int) -> None:
    counts: dict = {}
    for t in schedule.trials:
        counts[t.label] = counts.get(t.label, 0) + 1

    for label in classes:
        found = counts.get(label, 0)
        if found != repetitions_per_class:
            raise ScheduleError(
                f"class balance check failed: expected {repetitions_per_class} "
                f"trials for {label!r}, found {found}"
            )

    extra = set(counts) - set(classes)
    if extra:
        raise ScheduleError(f"schedule contains trials for undefined class(es): {sorted(extra)}")

    expected_total = len(classes) * repetitions_per_class
    if len(schedule.trials) != expected_total:
        raise ScheduleError(
            f"schedule length mismatch: expected {expected_total}, got {len(schedule.trials)}"
        )

    # Per-class repetition indices must be exactly 1..N with no duplicates.
    reps_by_label: dict = {}
    for t in schedule.trials:
        reps_by_label.setdefault(t.label, set()).add(t.repetition)
    for label in classes:
        if reps_by_label.get(label, set()) != set(range(1, repetitions_per_class + 1)):
            raise ScheduleError(f"repetition indices for {label!r} are not exactly 1..{repetitions_per_class}")


def longest_run(schedule: Schedule) -> int:
    """Longest run of consecutive trials sharing the same label (T-6)."""
    longest = 0
    current = 0
    prev = None
    for t in schedule.trials:
        if t.label == prev:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        prev = t.label
    return longest


# ---------------------------------------------------------------------------
# Persistence (§13)
# ---------------------------------------------------------------------------


def write_schedule(schedule: Schedule, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(schedule.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_schedule(path: str | Path) -> Schedule:
    path = Path(path)
    if not path.is_file():
        raise ScheduleError(f"schedule file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    trials = [ScheduledTrial(**t) for t in raw["trials"]]
    return Schedule(
        schema_version=raw["schema_version"],
        seed=raw["seed"],
        strategy=raw["strategy"],
        class_definition_version=raw["class_definition_version"],
        repetitions_per_class=raw["repetitions_per_class"],
        total_trials=raw["total_trials"],
        trials=trials,
    )
