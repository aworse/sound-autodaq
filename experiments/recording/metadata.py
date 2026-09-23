"""
Session metadata construction: session.json, session_summary.json, and
versioning (§36-38).
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

RECORDING_SOFTWARE_VERSION = "0.2.0"
# 1.1: manifest rows gained `requeue` (additive, REQ-35.2 would not
# require a bump) and summaries gained `completed` / `stop_reason`.
DATASET_SCHEMA_VERSION = "1.1"

SOURCE_DIR = Path(__file__).resolve().parent

# Appendix E asks these to be decided and recorded per session.
IMPLEMENTATION_DECISIONS = {
    "keystroke_detection": (
        "none (E.1, input.key_detection = none): the input window is purely time-based; "
        "input_detected_ns and observed_label are always null and status 'mismatch' is never produced"
    ),
    "out_of_window_flagging": "not applicable without a key hook (E.2)",
    "requeue_placement": (
        "E.3: operator repeat and pause-with-discard re-run the pair immediately; "
        "invalid / silence / overflow / corrupted pairs are appended at the end "
        "of the session when repeat_on_invalid is true; skipped pairs are never requeued"
    ),
    "audio_backend": (
        "E.4: sounddevice/PortAudio InputStream, dtype int16, native rate verified "
        "with check_input_settings before opening; no software resampling"
    ),
    "manual_order_format": "E.5: JSON list of {\"label\": str, \"repetition\": int}",
    "operator_control_keys": (
        "1=repeat 2=skip 3=invalid 4=pause/resume 0=quit; digits instead of "
        "SPACE/R/S/I/Q because R/S/I/Q are the dubeolsik keys for ㄱ/ㄴ/ㅑ/ㅂ "
        "(REQ-28.2 over REQ-28.1, per the §2.7 priority order)"
    ),
    "operator_key_window": (
        "a control key applies to the trial whose countdown, recording, or "
        "inter-trial gap it was pressed in"
    ),
}


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


KEY_DETECTION_DECISIONS = {
    "keystroke_detection": (
        "terminal (E.1): every key typed into the recorder terminal is stamped with "
        "time.monotonic_ns() when read and mapped through the dubeolsik layout; the first target "
        "key inside the recorded segment gives observed_label and input_detected_ns. IME must be "
        "in English mode (Hangul composition delays and merges keys) and Caps Lock off"
    ),
    "out_of_window_flagging": (
        "E.2: status 'invalid' (existing value, no schema bump) when the keystroke is outside the "
        "recorded audio (pre-roll start to post-roll end), when none is detected, when more than one "
        "key lands in the recording, when the key is not a jamo, when the IME is in Hangul mode, or "
        "when an operator digit is pressed during the recording; 'mismatch' when the jamo differs "
        "from the target. Keystrokes in the pre-/post-roll are kept with their timestamp"
    ),
}


KEYPRESS_CAPTURE_DECISION = (
    "trial.capture = keypress (deviation from the fixed REQ-22 timeline, for human participants): "
    "after the countdown the input window stays open until the participant presses a key (at most "
    "input_window_ms, 0 = no limit). The stored segment runs from pre_roll_ms before to post_roll_ms "
    "after the keystroke, cut by absolute sample index from the continuous ring buffer (REQ-15.4 "
    "holds). The keypress time is converted to a sample index through (block-arrival time, samples "
    "captured) anchors recorded in the audio callback; boundaries are accurate to about one audio "
    "block (~10 ms on hardware). Stored length is pre_roll_ms + post_roll_ms. input_expected_ns is "
    "when PRESS appeared; input_detected_ns is the keystroke."
)


def implementation_decisions(key_detection: str, capture: str = "scheduled") -> dict:
    decisions = dict(IMPLEMENTATION_DECISIONS)
    if key_detection == "terminal":
        decisions.update(KEY_DETECTION_DECISIONS)
    decisions["capture"] = (
        KEYPRESS_CAPTURE_DECISION if capture == "keypress"
        else "trial.capture = scheduled: fixed pre-roll / input window / post-roll timeline (REQ-22)"
    )
    return decisions


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
    random_seed: int,
    mic_test_result: Optional[dict] = None,
    repo_dir: Optional[Path] = None,
) -> dict:
    """REQ-36.1: everything that affects the data, in one file.

    `random_seed` is the seed actually used — generated at startup when
    the configuration left it null (REQ-11.2/11.3) — and it also replaces
    the null inside resolved_config, so a resume rebuilt from this file
    regenerates nothing.
    """
    repo_dir = repo_dir or SOURCE_DIR
    resolved = config.resolved_dict()
    resolved["randomization"]["seed"] = random_seed
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
        "random_seed": random_seed,
        "capture": config.trial.capture,
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
        "resolved_config": resolved,
        "implementation_decisions": implementation_decisions(config.input.key_detection, config.trial.capture),
        "breaks": [],
        "resumes": [],
    }


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    interrupted_trials: int
    completion_rate: float
    per_class_valid: dict
    duration_s: float
    total_break_s: float
    validated: bool
    validation_result: Optional[str]
    # REQ-46: complete only when every pair is processed AND validation
    # passes; set by the engine, never inferred from process exit.
    completed: bool = False
    stop_reason: Optional[str] = None

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
        interrupted_trials=counts.get(Status.INTERRUPTED.value, 0),
        completion_rate=(valid / expected_trials) if expected_trials else 0.0,
        per_class_valid=per_class_valid,
        duration_s=duration_s,
        total_break_s=total_break_s,
        validated=False,
        validation_result=None,
    )
