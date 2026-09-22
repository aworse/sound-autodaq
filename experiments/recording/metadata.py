"""
Session metadata construction: session.json, session_summary.json, and
versioning (§36-38).
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

RECORDING_SOFTWARE_VERSION = "0.1.0"
DATASET_SCHEMA_VERSION = "1.0"


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def get_git_dirty(repo_dir: Optional[Path] = None) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def build_session_metadata(
    config,
    class_definition_version: str,
    num_classes: int,
    total_trials: int,
    device_info_dict: dict,
    keyboard_dict: dict,
    microphone_dict: dict,
    environment_dict: dict,
    started_utc: str,
    mic_test_result: Optional[dict] = None,
    repo_dir: Optional[Path] = None,
) -> dict:
    """REQ-36.1: everything that affects the data, in one file."""
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "recording_software_version": RECORDING_SOFTWARE_VERSION,
        "class_definition_version": class_definition_version,
        "experiment_config_version": config.experiment.config_version,
        "participant_id": config.participant.id,
        "scenario_id": config.scenario.id,
        "scenario_name": config.scenario.name,
        "scenario_description": config.scenario.description,
        "session_id": config.session.id,
        "session_uid": config.session_uid,
        "input_mode": config.input.mode,
        "sample_rate": config.recording.sample_rate,
        "channels": config.recording.channels,
        "sample_format": config.recording.format,
        "repetitions_per_class": config.trial.repetitions_per_class,
        "num_classes": num_classes,
        "total_trials": total_trials,
        "randomization_strategy": config.randomization.strategy,
        "random_seed": config.randomization.seed,
        "countdown_ms": config.trial.countdown_ms,
        "pre_roll_ms": config.trial.pre_roll_ms,
        "input_window_ms": config.trial.input_window_ms,
        "post_roll_ms": config.trial.post_roll_ms,
        "inter_trial_ms": config.trial.inter_trial_ms,
        "started_utc": started_utc,
        "ended_utc": None,
        "software_version": RECORDING_SOFTWARE_VERSION,
        "git_commit": get_git_commit(repo_dir),
        "git_dirty": get_git_dirty(repo_dir),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "audio": device_info_dict,
        "keyboard": keyboard_dict,
        "microphone": microphone_dict,
        "environment": environment_dict,
        "mic_test": mic_test_result,
        "resolved_config": config.resolved_dict(),
        "breaks": [],
        "resumes": [],
    }


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclasses.dataclass
class SessionSummary:
    session_uid: str
    expected_trials: int
    attempted_trials: int
    valid_trials: int
    invalid_trials: int
    skipped_trials: int
    corrupted_trials: int
    mismatch_trials: int
    overflow_trials: int
    suspicious_silence_trials: int
    completion_rate: float
    per_class_valid: dict
    duration_s: float
    total_break_s: float
    validated: bool
    validation_result: Optional[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_summary(session_uid: str, expected_trials: int, records: list, duration_s: float, total_break_s: float) -> SessionSummary:
    from .trial import Status

    counts = {s.value: 0 for s in Status}
    per_class_valid: dict = {}
    for r in records:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        if status == Status.VALID.value:
            per_class_valid[r["label"]] = per_class_valid.get(r["label"], 0) + 1

    valid = counts.get(Status.VALID.value, 0)
    return SessionSummary(
        session_uid=session_uid,
        expected_trials=expected_trials,
        attempted_trials=len(records),
        valid_trials=valid,
        invalid_trials=counts.get(Status.INVALID.value, 0) + counts.get(Status.OPERATOR_MARKED_INVALID.value, 0),
        skipped_trials=counts.get(Status.SKIPPED.value, 0),
        corrupted_trials=counts.get(Status.CORRUPTED.value, 0),
        mismatch_trials=counts.get(Status.MISMATCH.value, 0),
        overflow_trials=counts.get(Status.AUDIO_OVERFLOW.value, 0),
        suspicious_silence_trials=counts.get(Status.SUSPICIOUS_SILENCE.value, 0),
        completion_rate=(valid / expected_trials) if expected_trials else 0.0,
        per_class_valid=per_class_valid,
        duration_s=duration_s,
        total_break_s=total_break_s,
        validated=False,
        validation_result=None,
    )
