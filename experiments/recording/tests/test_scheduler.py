import json

import pytest

from experiments.recording.errors import ScheduleError
from experiments.recording.scheduler import (
    generate_schedule,
    load_schedule,
    longest_run,
    validate_balance,
    write_schedule,
)

CLASSES = tuple(f"C{i:02d}" for i in range(10))
REPS = 20


def test_t1_schedule_size():
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    assert len(sched.trials) == len(CLASSES) * REPS
    assert sched.total_trials == len(CLASSES) * REPS


def test_t2_class_balance_rejects_unbalanced_pool():
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    # Artificially unbalance: drop one trial.
    sched.trials.pop()
    with pytest.raises(ScheduleError, match="class balance"):
        validate_balance(sched, CLASSES, REPS)


def test_t3_seed_determinism():
    s1 = generate_schedule(CLASSES, REPS, "balanced_random", seed=777, class_definition_version="v1")
    s2 = generate_schedule(CLASSES, REPS, "balanced_random", seed=777, class_definition_version="v1")
    assert [t.to_dict() for t in s1.trials] == [t.to_dict() for t in s2.trials]


def test_t4_seed_sensitivity():
    s1 = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    s2 = generate_schedule(CLASSES, REPS, "balanced_random", seed=2, class_definition_version="v1")
    assert [t.to_dict() for t in s1.trials] != [t.to_dict() for t in s2.trials]


def test_t5_repetition_indexing():
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    seen = {}
    for t in sched.trials:
        seen.setdefault(t.label, set()).add(t.repetition)
    for label in CLASSES:
        assert seen[label] == set(range(1, REPS + 1))


def test_t6_non_consecutiveness():
    for seed in range(20):
        sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=seed, class_definition_version="v1")
        assert longest_run(sched) == 1


def test_balanced_random_is_not_round_robin():
    """A round-robin makes the last class of every n-class window
    predictable to the participant; a global permutation does not."""
    classes = tuple(f"K{i:02d}" for i in range(38))
    sched = generate_schedule(classes, 50, "balanced_random", seed=3, class_definition_version="v1")
    labels = [t.label for t in sched.trials]
    full_windows = sum(len(set(labels[i:i + 38])) == 38 for i in range(0, len(labels), 38))
    assert full_windows <= 1


def test_full_scale_schedule_is_balanced_and_repeat_free():
    classes = tuple(f"K{i:02d}" for i in range(38))
    sched = generate_schedule(classes, 500, "balanced_random", seed=20260922, class_definition_version="v1")
    validate_balance(sched, classes, 500)
    assert longest_run(sched) == 1


def test_block_random_does_not_push_classes_to_the_end():
    """REQ-10.5 with more classes than the block size."""
    classes = tuple(f"K{i:02d}" for i in range(38))
    sched = generate_schedule(classes, 20, "block_random", seed=1, class_definition_version="v1", block_size=25)
    n = len(sched.trials)
    positions = {}
    for t in sched.trials:
        positions.setdefault(t.label, []).append(t.trial_id / n)
    for label, pos in positions.items():
        assert 0.4 < sum(pos) / len(pos) < 0.6, label
        assert min(pos) * n <= 2 * 25, label


@pytest.mark.parametrize("strategy,kwargs", [
    ("random", {}),
    ("balanced_random", {}),
    ("block_random", {"block_size": 25}),
])
def test_strategies_produce_valid_balanced_schedule(strategy, kwargs):
    sched = generate_schedule(
        CLASSES, REPS, strategy, seed=5, class_definition_version="v1", **kwargs
    )
    validate_balance(sched, CLASSES, REPS)


def test_schedule_persistence_round_trip(tmp_path):
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=9, class_definition_version="v1")
    path = tmp_path / "schedule.json"
    write_schedule(sched, path)
    loaded = load_schedule(path)
    assert loaded.to_dict() == sched.to_dict()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["seed"] == 9
    assert raw["strategy"] == "balanced_random"


@pytest.mark.parametrize("strategy,kwargs,digest", [
    ("balanced_random", {}, "d4d7608082e3c719"),
    ("random", {}, "7a2088e33f01d315"),
    ("block_random", {"block_size": 25}, "0781e20bd830e422"),
])
def test_schedule_is_byte_identical_across_python_versions(strategy, kwargs, digest):
    """REQ-2.4.1. These digests were produced on Python 3.10-3.13; CI runs
    this on each version, so a change in the RNG or the strategy code that
    would silently reorder a resumed session fails here."""
    import hashlib

    classes = tuple(f"K{i:02d}" for i in range(38))
    sched = generate_schedule(classes, 50, strategy, seed=20260922, class_definition_version="v", **kwargs)
    blob = json.dumps(sched.to_dict(), sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:16] == digest
