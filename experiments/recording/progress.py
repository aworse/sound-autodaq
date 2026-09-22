"""
progress.json persistence and resume decisions (§47-48).

Owns: progress.json, resume decisions. Never: schedule generation
(REQ-57).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from .metadata import read_json, write_json_atomic
from .trial import utc_now_iso


@dataclasses.dataclass
class Progress:
    session_uid: str
    schedule_file: str
    random_seed: int
    total_trials: int
    completed_trials: int
    last_trial_id: Optional[int]
    next_trial_id: int
    updated_utc: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_progress(path: Path, progress: Progress) -> None:
    write_json_atomic(path, progress.to_dict())


def read_progress(path: Path) -> Optional[Progress]:
    path = Path(path)
    if not path.exists():
        return None
    d = read_json(path)
    return Progress(**d)


def new_progress(session_uid: str, schedule_file: str, random_seed: int, total_trials: int) -> Progress:
    return Progress(
        session_uid=session_uid,
        schedule_file=schedule_file,
        random_seed=random_seed,
        total_trials=total_trials,
        completed_trials=0,
        last_trial_id=None,
        next_trial_id=1,
        updated_utc=utc_now_iso(),
    )


def advance(progress: Progress, completed_trial_id: int) -> Progress:
    progress.completed_trials += 1
    progress.last_trial_id = completed_trial_id
    progress.next_trial_id = completed_trial_id + 1
    progress.updated_utc = utc_now_iso()
    return progress


def is_session_incomplete(session_dir: Path) -> bool:
    """A session dir looks like a prior, unfinished run: it has a
    schedule but no session_summary.json marking completion, or its
    progress is short of its schedule's total."""
    session_dir = Path(session_dir)
    schedule_path = session_dir / "schedule.json"
    summary_path = session_dir / "session_summary.json"
    if not schedule_path.exists():
        return False
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("attempted_trials", 0) >= summary.get("expected_trials", 0):
            return False
    return True
