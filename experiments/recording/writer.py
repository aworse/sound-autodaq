"""
WAV encoding, atomic writes, and manifest append/flush (§31-34, §47).

Owns: WAV encoding, atomic writes, manifest append/flush. Never: audio
capture, DSP (REQ-57). No module outside this one may create files inside
`audio/` (REQ-57.1).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import DuplicateTrialError

MANIFEST_CSV_COLUMNS = [
    "trial_id",
    "file",
    "label",
    "participant",
    "scenario",
    "session",
    "repetition",
    "status",
]


def trial_filename(trial_id: int) -> str:
    return f"trial_{trial_id:08d}.wav"


def write_wav_atomic(path: Path, samples: np.ndarray, sample_rate: int, channels: int) -> None:
    """Write a PCM16 WAV atomically: temp file in the same directory,
    fsync, then rename into place (REQ-47.3). Never overwrites an
    existing file under its final name.
    """
    path = Path(path)
    if path.exists():
        raise DuplicateTrialError(f"refusing to overwrite existing file: {path}")

    tmp_path = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # PCM_16
        wf.setframerate(sample_rate)
        wf.writeframes(np.ascontiguousarray(samples, dtype=np.int16).tobytes())

    fd = os.open(str(tmp_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    os.replace(str(tmp_path), str(path))

    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_wav(path: Path) -> tuple:
    """Returns (samples: np.ndarray[int16], sample_rate, channels)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, sample_rate, channels


def wav_params(path: Path) -> tuple:
    """(sample_rate, channels, sample_width_bytes, n_frames) from the header."""
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()


@dataclasses.dataclass
class TrialRecord:
    trial_id: int
    file: Optional[str]
    label: str
    scheduled_label: str
    observed_label: Optional[str]
    participant: str
    scenario: str
    session: str
    repetition: int
    status: str
    input_mode: str
    sample_rate: int
    channels: int
    sample_format: str
    num_samples: int
    duration_ms: float
    segment_start_sample: Optional[int]
    segment_end_sample: Optional[int]
    trial_start_ns: Optional[int]
    input_expected_ns: Optional[int]
    input_detected_ns: Optional[int]
    trial_end_ns: Optional[int]
    wall_clock_utc: Optional[str]
    peak: Optional[float]
    rms: Optional[float]
    clipping_ratio: Optional[float]
    overflow: Optional[bool]
    notes: Optional[str] = None
    superseded_by: Optional[int] = None
    # "immediate" | "end" | None: where the retry named by superseded_by
    # goes in the queue, so a resumed session re-runs it in the same place.
    requeue: Optional[str] = None
    # Key detection (input.key_detection = terminal): the raw key behind
    # observed_label, and how many target keys landed in the recording.
    observed_key: Optional[str] = None
    keystrokes: Optional[int] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_csv_row(self) -> dict:
        return {col: getattr(self, col) for col in MANIFEST_CSV_COLUMNS}


class ManifestWriter:
    """Appends and flushes one row per trial to both manifest.csv and
    manifest.jsonl (REQ-33.4, REQ-34.1, REQ-47.4)."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.csv_path = self.session_dir / "manifest.csv"
        self.jsonl_path = self.session_dir / "manifest.jsonl"
        self._seen_trial_ids: set = set()
        self.torn_tails: list = []

        for path in (self.csv_path, self.jsonl_path):
            torn = _set_aside_torn_tail(path)
            if torn is not None:
                self.torn_tails.append(torn)

        csv_is_new = not self.csv_path.exists()
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=MANIFEST_CSV_COLUMNS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        if csv_is_new:
            self._csv_writer.writeheader()
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
        else:
            self._load_existing_ids()

        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    def _load_existing_ids(self) -> None:
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._seen_trial_ids.add(json.loads(line)["trial_id"])

    def has_trial(self, trial_id: int) -> bool:
        return trial_id in self._seen_trial_ids

    def append(self, record: TrialRecord) -> None:
        if record.trial_id in self._seen_trial_ids:
            raise DuplicateTrialError(f"trial_id {record.trial_id} already present in manifest")
        self._seen_trial_ids.add(record.trial_id)

        self._csv_writer.writerow(record.to_csv_row())
        self._csv_file.flush()
        os.fsync(self._csv_file.fileno())

        self._jsonl_file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._jsonl_file.flush()
        os.fsync(self._jsonl_file.fileno())

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _set_aside_torn_tail(path: Path) -> Optional[Path]:
    """A hard kill mid-append can leave a final line with no newline.
    Move those bytes to `<name>.torn` (kept for traceability, never
    deleted) and cut the manifest back to its last complete row, so the
    next append does not glue a new row onto a fragment."""
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    torn_path = path.with_name(path.name + ".torn")
    with open(torn_path, "ab") as f:
        f.write(data[cut:] + b"\n")
        f.flush()
        os.fsync(f.fileno())
    with open(path, "r+b") as f:
        f.truncate(cut)
        f.flush()
        os.fsync(f.fileno())
    return torn_path


def read_manifest_jsonl(path: Path) -> list:
    """Complete rows only; an unterminated final line is an in-progress or
    torn append and is not a recorded trial."""
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_manifest_csv(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
