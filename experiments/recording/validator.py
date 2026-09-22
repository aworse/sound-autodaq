"""
Read-only dataset validation (§43-44).

Owns: read-only dataset verification. Never: mutation of the session
directory (REQ-57).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

from .metadata import read_json
from .scheduler import load_schedule
from .trial import Status
from .writer import read_manifest_csv, read_manifest_jsonl, read_wav


@dataclasses.dataclass
class ValidationReport:
    session_dir: str
    expected_trials: int
    manifest_rows: int
    wav_files_found: int
    status_breakdown: dict
    per_class_valid: dict
    class_balance_ok: bool
    class_balance_warnings: dict
    orphan_audio: list
    missing_audio: list
    duplicate_trial: list
    schedule_drift: list
    format_mismatch: list
    count_mismatch: bool
    audio_format_ok: bool
    passed: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def render(self) -> str:
        lines = []
        lines.append("CLASSISM DATASET VALIDATION")
        lines.append(f"Session: {self.session_dir}")
        lines.append("")
        lines.append("Schedule")
        lines.append(f"  Expected trials : {self.expected_trials}")
        lines.append(f"  Manifest rows   : {self.manifest_rows}")
        lines.append(f"  WAV files found : {self.wav_files_found}")
        lines.append("")
        lines.append("Status breakdown")
        for status, count in sorted(self.status_breakdown.items()):
            if count:
                lines.append(f"  {status:<18} {count}")
        lines.append("")
        lines.append("Consistency")
        lines.append(f"  ORPHAN_AUDIO     : {len(self.orphan_audio)}")
        lines.append(f"  MISSING_AUDIO    : {len(self.missing_audio)}")
        lines.append(f"  DUPLICATE_TRIAL  : {len(self.duplicate_trial)}")
        lines.append(f"  SCHEDULE_DRIFT   : {len(self.schedule_drift)}")
        lines.append(f"  FORMAT_MISMATCH  : {len(self.format_mismatch)}")
        lines.append(f"  COUNT_MISMATCH   : {self.count_mismatch}")
        lines.append("")
        result = "PASS" if self.passed else "FAIL"
        if self.passed and self.class_balance_warnings:
            result += " (with warnings)"
        lines.append(f"RESULT: {result}")
        return "\n".join(lines)


def validate_session(session_dir: str | Path) -> ValidationReport:
    session_dir = Path(session_dir)
    schedule = load_schedule(session_dir / "schedule.json")
    session_json = read_json(session_dir / "session.json") if (session_dir / "session.json").exists() else {}

    jsonl_records = read_manifest_jsonl(session_dir / "manifest.jsonl")
    csv_rows = read_manifest_csv(session_dir / "manifest.csv")

    by_trial_id = {r["trial_id"]: r for r in jsonl_records}
    schedule_by_id = schedule.by_trial_id()

    # DUPLICATE_TRIAL
    seen = {}
    duplicate_trial = []
    for r in jsonl_records:
        tid = r["trial_id"]
        if tid in seen:
            duplicate_trial.append(tid)
        seen[tid] = True

    audio_dir = session_dir / "audio"
    wav_files = set(p.name for p in audio_dir.glob("*.wav")) if audio_dir.exists() else set()
    manifest_files = {r["file"] for r in jsonl_records if r.get("file")}
    manifest_files_basename = {Path(f).name for f in manifest_files}

    orphan_audio = sorted(wav_files - manifest_files_basename)
    missing_audio = []
    format_mismatch = []
    schedule_drift = []
    status_breakdown = {s.value: 0 for s in Status}
    per_class_valid: dict = {}

    expected_rate = session_json.get("sample_rate")
    expected_channels = session_json.get("channels")

    for r in jsonl_records:
        status_breakdown[r["status"]] = status_breakdown.get(r["status"], 0) + 1
        if r["status"] == Status.VALID.value:
            per_class_valid[r["label"]] = per_class_valid.get(r["label"], 0) + 1

        sched = schedule_by_id.get(r["trial_id"])
        if sched is not None and r.get("scheduled_label", r["label"]) != sched.label:
            schedule_drift.append(r["trial_id"])

        if r.get("file"):
            wav_path = session_dir / r["file"]
            if not wav_path.exists():
                missing_audio.append(r["trial_id"])
            elif expected_rate is not None:
                try:
                    _, sr, ch = read_wav(wav_path)
                    if sr != expected_rate or ch != expected_channels:
                        format_mismatch.append(r["trial_id"])
                except Exception:
                    format_mismatch.append(r["trial_id"])

    class_balance_warnings = {}
    for label in {t.label for t in schedule.trials}:
        expected = schedule.repetitions_per_class
        found = per_class_valid.get(label, 0)
        if found != expected:
            class_balance_warnings[label] = found - expected

    count_mismatch = len(jsonl_records) != len(schedule.trials)

    audio_format_ok = not format_mismatch

    passed = not (
        orphan_audio
        or missing_audio
        or duplicate_trial
        or schedule_drift
        or format_mismatch
        or count_mismatch
    )

    return ValidationReport(
        session_dir=str(session_dir),
        expected_trials=len(schedule.trials),
        manifest_rows=len(jsonl_records),
        wav_files_found=len(wav_files),
        status_breakdown=status_breakdown,
        per_class_valid=per_class_valid,
        class_balance_ok=not class_balance_warnings,
        class_balance_warnings=class_balance_warnings,
        orphan_audio=orphan_audio,
        missing_audio=missing_audio,
        duplicate_trial=duplicate_trial,
        schedule_drift=schedule_drift,
        format_mismatch=format_mismatch,
        count_mismatch=count_mismatch,
        audio_format_ok=audio_format_ok,
        passed=passed,
    )
