"""
progress.json persistence and resume decisions (§47-48).

Owns: progress.json, resume decisions. Never: schedule generation
(REQ-57).

The manifest is the authority (REQ-47.6): what is left to record is
always recomputed from schedule.json + manifest.jsonl, never read from
progress.json, so a stale progress file cannot cause a gap or a re-record.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from .metadata import read_json, write_json_atomic
from .scheduler import ScheduledTrial
from .trial import Status, utc_now_iso

REQUEUE_IMMEDIATE = "immediate"
REQUEUE_END = "end"

STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_BREAK = "break"
STATE_STOPPED = "stopped"
STATE_COMPLETE = "complete"


@dataclasses.dataclass
class Progress:
    session_uid: str
    schedule_file: str
    random_seed: int
    total_trials: int
    completed_trials: int
    last_trial_id: Optional[int]
    next_trial_id: Optional[int]
    updated_utc: str
    state: str = STATE_RUNNING
    current_trial_id: Optional[int] = None
    current_label: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_progress(path: Path, progress: Progress) -> None:
    progress.updated_utc = utc_now_iso()
    write_json_atomic(path, progress.to_dict())


def read_progress(path: Path) -> Optional[Progress]:
    path = Path(path)
    if not path.exists():
        return None
    d = read_json(path)
    known = {f.name for f in dataclasses.fields(Progress)}
    return Progress(**{k: v for k, v in d.items() if k in known})


def reconstruct_queue(schedule, rows: list) -> tuple:
    """Return (pending: list[ScheduledTrial], next_new_trial_id: int).

    pending = retries marked for immediate re-run (oldest first), then
    scheduled trials that have no manifest row (schedule order), then
    retries marked for the end of the session (oldest first). A retry is a
    (label, repetition) pair with no valid row whose latest row names, via
    `superseded_by`, a trial id that has not been recorded yet.
    """
    done_ids = {r["trial_id"] for r in rows}
    by_pair: dict = {}
    for r in rows:
        by_pair.setdefault((r["label"], r["repetition"]), []).append(r)

    immediate, end = [], []
    for (label, rep), pair_rows in by_pair.items():
        if any(r["status"] == Status.VALID.value for r in pair_rows):
            continue
        last = max(pair_rows, key=lambda r: r["trial_id"])
        retry_id = last.get("superseded_by")
        if retry_id is None or retry_id in done_ids:
            continue
        trial = ScheduledTrial(trial_id=retry_id, label=label, repetition=rep)
        (immediate if last.get("requeue") == REQUEUE_IMMEDIATE else end).append(trial)

    unrecorded = [t for t in schedule.trials if t.trial_id not in done_ids]
    pending = sorted(immediate, key=lambda t: t.trial_id) + unrecorded + sorted(end, key=lambda t: t.trial_id)

    known_ids = [schedule.total_trials] + list(done_ids) + [
        r["superseded_by"] for r in rows if r.get("superseded_by") is not None
    ]
    return pending, max(known_ids) + 1


def session_dir_state(session_dir: Path) -> str:
    """'absent' (nothing recorded there), 'incomplete', or 'complete'."""
    session_dir = Path(session_dir)
    if not (session_dir / "schedule.json").exists() and not (session_dir / "manifest.jsonl").exists():
        return "absent"
    summary_path = session_dir / "session_summary.json"
    if summary_path.exists() and read_json(summary_path).get("completed"):
        return "complete"
    return "incomplete"
