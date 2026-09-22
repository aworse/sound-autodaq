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
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    assert longest_run(sched) <= 2  # sanity bound well below REPS


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
