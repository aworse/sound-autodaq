"""
Export a recorded session as classism training data (aworse/classism
spec §3.1 / §7, dataset.py).

    <out>/labels/<session_uid>.csv   one row per valid trial
    <out>/audio/<clip_id>.wav        that trial's recording

Columns: the classism label table (onset_s, jamo, keytype, shift,
scenario, participant, clip_id), then extras its loader ignores:
wav_onset_s (the keystroke's offset inside the WAV), onset_source,
trial_id, observed_key.

The log-mel tensors classism trains on (`<out>/mel/<clip_id>.npy`, 100 ms
from 5 ms before the onset) belong to the Mel component: it cuts each
WAV at wav_onset_s. Many sessions can be exported into one <out>; each
gets its own CSV, and re-exporting a session replaces its files.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from classism.labels import CLASSES

from .errors import RecorderError
from .keylog import SHIFTED_JAMO
from .metadata import read_json
from .trial import Status
from .validator import validate_session
from .writer import read_manifest_jsonl

LABEL_COLUMNS = ["onset_s", "jamo", "keytype", "shift", "scenario", "participant", "clip_id"]
EXTRA_COLUMNS = ["wav_onset_s", "onset_source", "trial_id", "observed_key"]

KEYTYPE_OF_SPECIAL = {"<sp>": "space", "<bs>": "backspace", "<shift>": "shift", "<caps>": "caps", "<other>": "other"}

# classism's frozen log-mel regime (melio.Regime): 48 kHz, onset -5 ms ... +95 ms
CLASSISM_SAMPLE_RATE = 48000
CLIP_BEFORE_S = 0.005
CLIP_AFTER_S = 0.095


class ExportError(RecorderError):
    """The session cannot be exported."""


@dataclasses.dataclass
class ExportResult:
    session_uid: str
    labels_csv: Path
    exported: int
    skipped: list  # (trial_id, reason)
    still_to_record: int
    warnings: list

    def render(self) -> str:
        lines = [f"Exported {self.exported} clips of {self.session_uid} -> {self.labels_csv}"]
        if self.still_to_record:
            lines.append(f"  session incomplete: {self.still_to_record} trials still to record")
        for trial_id, reason in self.skipped:
            lines.append(f"  skipped trial {trial_id}: {reason}")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _utc(ts: str) -> datetime.datetime:
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of the audio when on the same disk
    except OSError:
        shutil.copy2(src, dst)


def wav_onset_s(row: dict, capture: str, pre_roll_ms: int) -> tuple:
    """(offset of the keystroke inside the trial's WAV in seconds, source).

    keypress capture cuts pre_roll_ms before the keystroke, so it sits at
    exactly that offset. scheduled capture starts the file when the
    pre-roll phase starts, pre_roll_ms before PRESS appeared; the
    keystroke is wherever it was typed after that. Without key detection
    only the prompt time is known."""
    pre_s = pre_roll_ms / 1000.0
    if capture == "keypress":
        return pre_s, "keystroke"
    if row.get("input_detected_ns") is not None:
        return pre_s + (row["input_detected_ns"] - row["input_expected_ns"]) / 1e9, "keystroke"
    return pre_s, "prompt"


def export_session(session_dir, out_root) -> ExportResult:
    session_dir, out_root = Path(session_dir), Path(out_root)
    if not (session_dir / "session.json").exists():
        raise ExportError(f"no session.json in {session_dir}")

    report = validate_session(session_dir)
    problems = {
        "orphan audio": report.orphan_audio, "missing audio": report.missing_audio,
        "duplicate trial ids": report.duplicate_trial, "schedule drift": report.schedule_drift,
        "audio format mismatch": report.format_mismatch, "manifest csv/jsonl divergence": report.csv_jsonl_divergence,
        "progress count mismatch": report.count_mismatch,
    }
    found = [f"{name}: {value}" for name, value in problems.items() if value]
    if found:
        # An incomplete session is fine to export; a damaged one is not.
        raise ExportError(f"{session_dir} fails validation ({'; '.join(found)}); "
                          f"inspect it with --validate before exporting")

    session = read_json(session_dir / "session.json")
    resolved = session["resolved_config"]
    trial_cfg = resolved["trial"]
    capture = trial_cfg.get("capture", "scheduled")
    pre_roll_ms = trial_cfg["pre_roll_ms"]
    session_uid = session["session_uid"]

    warnings = []
    if session["sample_rate"] != CLASSISM_SAMPLE_RATE:
        warnings.append(f"recorded at {session['sample_rate']} Hz; classism's log-mel regime is "
                        f"{CLASSISM_SAMPLE_RATE} Hz and must not be fed resampled audio silently")
    if resolved["input"].get("key_detection", "none") == "none":
        warnings.append("no key detection: onsets are the prompt time, not the keystroke "
                        "(onset_source=prompt); the Mel component must find the onset acoustically")

    rows = read_manifest_jsonl(session_dir / "manifest.jsonl")
    valid = [r for r in rows if r["status"] == Status.VALID.value]
    unknown = sorted({r["label"] for r in valid} - set(CLASSES))
    if unknown:
        raise ExportError(f"labels {unknown} are not in the classism label space")

    labels_dir, audio_dir = out_root / "labels", out_root / "audio"
    labels_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    # The session's clock: the first trial's wall-clock start. Monotonic
    # stamps restart when a session is resumed in a new process; the wall
    # clock does not.
    t0: Optional[datetime.datetime] = _utc(rows[0]["wall_clock_utc"]) if rows else None

    skipped, out_rows = [], []
    for r in valid:
        offset, source = wav_onset_s(r, capture, pre_roll_ms)
        duration_s = r["num_samples"] / r["sample_rate"]
        if offset - CLIP_BEFORE_S < 0 or offset + CLIP_AFTER_S > duration_s:
            skipped.append((r["trial_id"], f"the 100 ms classism window around the onset at {offset:.3f} s "
                                           f"does not fit the {duration_s:.3f} s recording"))
            continue
        key_ns = r["input_detected_ns"] if r.get("input_detected_ns") is not None else r["input_expected_ns"]
        onset = (_utc(r["wall_clock_utc"]) - t0).total_seconds() + (key_ns - r["trial_start_ns"]) / 1e9

        label = r["label"]
        keytype = KEYTYPE_OF_SPECIAL.get(label, "normal")
        clip_id = f"{session_uid}_{r['trial_id']:08d}"
        _link_or_copy(session_dir / r["file"], audio_dir / f"{clip_id}.wav")
        out_rows.append({
            "onset_s": f"{onset:.3f}",
            "jamo": label,
            "keytype": keytype,
            "shift": int(label in SHIFTED_JAMO or keytype == "shift"),
            "scenario": r["scenario"],
            "participant": r["participant"],
            "clip_id": clip_id,
            "wav_onset_s": f"{offset:.4f}",
            "onset_source": source,
            "trial_id": r["trial_id"],
            "observed_key": r.get("observed_key") or "",
        })

    # A re-export replaces this session's clips: drop ones no longer valid.
    mine = re.compile(re.escape(session_uid) + r"_\d{8}\.wav")
    keep = {f"{row['clip_id']}.wav" for row in out_rows}
    for old in audio_dir.iterdir():
        if mine.fullmatch(old.name) and old.name not in keep:
            old.unlink()

    labels_csv = labels_dir / f"{session_uid}.csv"
    tmp = labels_csv.with_name(labels_csv.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp, labels_csv)

    return ExportResult(session_uid, labels_csv, len(out_rows), skipped, len(report.unprocessed_trials), warnings)
