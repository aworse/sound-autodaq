"""
Read-only dataset validation (§43-44).

Owns: read-only dataset verification. Never: mutation of the session
directory (REQ-57).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .metadata import read_json
from .progress import read_progress, reconstruct_queue
from .scheduler import load_schedule
from .trial import Status
from .writer import MANIFEST_CSV_COLUMNS, read_manifest_csv, read_manifest_jsonl, wav_params


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
    csv_jsonl_divergence: list
    unprocessed_trials: list
    audio_format_ok: bool
    passed: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def render(self) -> str:
        lines = [
            "CLASSISM DATASET VALIDATION",
            f"Session: {self.session_dir}",
            "",
            "Schedule",
            f"  Expected trials : {self.expected_trials}",
            f"  Manifest rows   : {self.manifest_rows}",
            f"  WAV files found : {self.wav_files_found}",
            f"  Still to record : {len(self.unprocessed_trials)}",
            "",
            "Status breakdown",
        ]
        for status, count in sorted(self.status_breakdown.items()):
            if count:
                lines.append(f"  {status:<24} {count}")
        lines += ["", "Class balance (valid trials)"]
        if self.class_balance_warnings:
            for label, delta in sorted(self.class_balance_warnings.items()):
                lines.append(f"  {label}  WARN ({delta:+d})")
        else:
            lines.append("  all classes OK")
        lines += [
            "",
            "Consistency",
            f"  ORPHAN_AUDIO     : {len(self.orphan_audio)}",
            f"  MISSING_AUDIO    : {len(self.missing_audio)}",
            f"  DUPLICATE_TRIAL  : {len(self.duplicate_trial)}",
            f"  SCHEDULE_DRIFT   : {len(self.schedule_drift)}",
            f"  COUNT_MISMATCH   : {self.count_mismatch}",
            f"  FORMAT_MISMATCH  : {len(self.format_mismatch)}",
            f"  CSV_JSONL_DIVERGE: {len(self.csv_jsonl_divergence)}",
            "",
        ]
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

    schedule_by_id = schedule.by_trial_id()
    schedule_pairs = {(t.label, t.repetition) for t in schedule.trials}

    seen: set = set()
    duplicate_trial = []
    for r in jsonl_records:
        if r["trial_id"] in seen:
            duplicate_trial.append(r["trial_id"])
        seen.add(r["trial_id"])

    audio_dir = session_dir / "audio"
    wav_files = {p.name for p in audio_dir.glob("*.wav")} if audio_dir.exists() else set()
    manifest_files = {Path(r["file"]).name for r in jsonl_records if r.get("file")}
    orphan_audio = sorted(wav_files - manifest_files)

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

        # A scheduled id must carry its scheduled label; a retry id (beyond
        # the schedule) must retry a pair that exists in the schedule.
        sched = schedule_by_id.get(r["trial_id"])
        if sched is not None:
            drifted = (r["label"], r["repetition"]) != (sched.label, sched.repetition)
        else:
            drifted = (r["label"], r["repetition"]) not in schedule_pairs
        if drifted or r.get("scheduled_label", r["label"]) != r["label"]:
            schedule_drift.append(r["trial_id"])

        if r.get("file"):
            wav_path = session_dir / r["file"]
            if not wav_path.exists():
                missing_audio.append(r["trial_id"])
            elif expected_rate is not None:
                try:
                    sr, ch, width, _ = wav_params(wav_path)
                    if sr != expected_rate or ch != expected_channels or width != 2:
                        format_mismatch.append(r["trial_id"])
                except Exception:
                    format_mismatch.append(r["trial_id"])

    # REQ-34.3: CSV and JSONL must agree on trial_id, file, label, status.
    csv_jsonl_divergence = []
    if len(csv_rows) != len(jsonl_records):
        csv_jsonl_divergence.append(f"row count csv={len(csv_rows)} jsonl={len(jsonl_records)}")
    for c, j in zip(csv_rows, jsonl_records):
        for col in ("trial_id", "file", "label", "status"):
            jv = "" if j.get(col) is None else str(j.get(col))
            if c.get(col, "") != jv:
                csv_jsonl_divergence.append(f"trial {j['trial_id']}: {col} csv={c.get(col)!r} jsonl={jv!r}")
    if csv_rows and list(csv_rows[0].keys())[: len(MANIFEST_CSV_COLUMNS)] != MANIFEST_CSV_COLUMNS:
        csv_jsonl_divergence.append("manifest.csv header does not match the required column order")

    # REQ-43.1 COUNT_MISMATCH: completed trials vs progress.json. Progress
    # may lag by up to progress_flush_every rows, never lead.
    prog = read_progress(session_dir / "progress.json")
    count_mismatch = False
    if prog is not None:
        flush_every = (session_json.get("resolved_config") or {}).get("output", {}).get("progress_flush_every", 1)
        lag = len(jsonl_records) - prog.completed_trials
        count_mismatch = not (0 <= lag < max(1, flush_every))

    pending, _ = reconstruct_queue(schedule, jsonl_records)
    unprocessed_trials = [t.trial_id for t in pending]

    class_balance_warnings = {}
    for label in {t.label for t in schedule.trials}:
        found = per_class_valid.get(label, 0)
        if found != schedule.repetitions_per_class:
            class_balance_warnings[label] = found - schedule.repetitions_per_class

    passed = not (
        orphan_audio
        or missing_audio
        or duplicate_trial
        or schedule_drift
        or format_mismatch
        or count_mismatch
        or csv_jsonl_divergence
        or unprocessed_trials
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
        csv_jsonl_divergence=csv_jsonl_divergence,
        unprocessed_trials=unprocessed_trials,
        audio_format_ok=not format_mismatch,
        passed=passed,
    )
