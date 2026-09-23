"""
Read-only status snapshot for the web dashboard.

Pure read of a session directory's own files (schedule.json, session.json,
progress.json, manifest.jsonl, session_summary.json) — same files the
recorder and validator already write, so the dashboard is just another
reader, never a source of truth (REQ-57: never mutates the session dir).
It works whether or not a recording process is currently running against
that directory.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from ..progress import reconstruct_queue
from ..scheduler import load_schedule
from ..writer import read_manifest_jsonl


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tail_jsonl(path: Path, n: int) -> list:
    """Last n JSON objects from a JSONL file, without loading huge files
    entirely into memory line objects beyond what's needed for counts."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _count_by(path: Path):
    """Single pass over manifest.jsonl: (total_rows, valid_count_by_label,
    status_counts, last_row)."""
    valid_by_label: dict = {}
    status_counts: dict = {}
    total = 0
    last_row = None
    if not path.exists():
        return total, valid_by_label, status_counts, last_row
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            last_row = row
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
            if row["status"] == "valid":
                valid_by_label[row["label"]] = valid_by_label.get(row["label"], 0) + 1
    return total, valid_by_label, status_counts, last_row


def _parse_iso(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        return None


def read_status(session_dir: Path) -> dict:
    session_dir = Path(session_dir)

    schedule = _read_json(session_dir / "schedule.json")
    session_meta = _read_json(session_dir / "session.json")
    progress = _read_json(session_dir / "progress.json")
    summary = _read_json(session_dir / "session_summary.json")

    if schedule is None or session_meta is None:
        return {"found": False, "session_dir": str(session_dir)}

    total_trials = schedule["total_trials"]
    repetitions_per_class = schedule["repetitions_per_class"]

    total_rows, valid_by_label, status_counts, last_row = _count_by(session_dir / "manifest.jsonl")

    # The trial being recorded right now is the head of the queue rebuilt
    # from schedule + manifest (its row is written only when it ends), not
    # the last row — which is the trial that already finished.
    pending, _ = reconstruct_queue(
        load_schedule(session_dir / "schedule.json"), read_manifest_jsonl(session_dir / "manifest.jsonl")
    )
    current = pending[0] if pending else None
    nxt = pending[1] if len(pending) > 1 else None
    current_label = current.label if current else None
    pending_pairs = {(t.label, t.repetition) for t in pending}
    state = (progress or {}).get("state") or ("complete" if summary and summary.get("completed") else "running")
    current_status = state if current else ("complete" if summary and summary.get("completed") else state)

    started = _parse_iso(session_meta.get("started_utc"))
    ended = _parse_iso(session_meta.get("ended_utc")) if session_meta.get("ended_utc") else None
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed_s = ((ended or now) - started).total_seconds() if started else 0.0

    completed = total_trials - len(pending_pairs)
    remaining_s = None
    if total_rows > 0 and elapsed_s > 0 and pending:
        remaining_s = elapsed_s / total_rows * len(pending)

    recent = _tail_jsonl(session_dir / "manifest.jsonl", 25)
    recent_overflow = any(r["status"] == "audio_overflow" for r in recent)
    recent_silence = any(r["status"] == "suspicious_silence" for r in recent)
    recent_clipping = any((r.get("clipping_ratio") or 0) > 0 for r in recent)

    is_complete = bool(summary and summary.get("completed"))

    return {
        "found": True,
        "session_dir": str(session_dir),
        "session_uid": session_meta.get("session_uid"),
        "participant": session_meta.get("participant_id"),
        "scenario": session_meta.get("scenario_id"),
        "session": session_meta.get("session_id"),
        "randomization_strategy": session_meta.get("randomization_strategy"),
        "random_seed": session_meta.get("random_seed"),
        "sample_rate": session_meta.get("sample_rate"),
        "channels": session_meta.get("channels"),
        "overall_completed": completed,
        "overall_total": total_trials,
        "overall_valid": status_counts.get("valid", 0),
        "current_trial_id": current.trial_id if current else None,
        "current_label": current_label,
        "current_status": current_status,
        "state": state,
        "last_trial_status": last_row["status"] if last_row else None,
        "class_completed": valid_by_label.get(current_label, 0) if current_label else 0,
        "class_total": repetitions_per_class,
        "next_label": nxt.label if nxt else None,
        "elapsed_s": elapsed_s,
        "remaining_s": remaining_s,
        "status_counts": status_counts,
        "last_peak": last_row.get("peak") if last_row else None,
        "last_rms": last_row.get("rms") if last_row else None,
        "last_clipping_ratio": last_row.get("clipping_ratio") if last_row else None,
        "warnings": {
            "overflow": recent_overflow,
            "silence": recent_silence,
            "clipping": recent_clipping,
        },
        "is_complete": is_complete,
        "validation_result": summary.get("validation_result") if summary else None,
        "stop_reason": summary.get("stop_reason") if summary else None,
        "completion_rate": summary.get("completion_rate") if summary else (
            (status_counts.get("valid", 0) / total_trials) if total_trials else 0.0
        ),
    }
