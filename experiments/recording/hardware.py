"""
Device enumeration, capability probing, and the microphone test (§54-55).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from . import quality
from .recorder import AudioBackend, ContinuousRecorder, DeviceInfo

# Any real analog input carries at least a few LSBs of noise; a peak this
# small means the stream is digital silence.
NO_SIGNAL_PEAK = 1.0 / quality.PCM16_FULL_SCALE


@dataclasses.dataclass
class MicTestResult:
    device: DeviceInfo
    sample_rate: int
    channels: int
    duration_s: float
    peak: float
    rms: float
    clipping_ratio: float
    overflow: bool
    passed: bool
    reasons: list

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["device"] = dataclasses.asdict(self.device)
        return d


def run_mic_test(
    backend: AudioBackend,
    device,
    sample_rate: int,
    channels: int,
    duration_s: float = 3.0,
) -> MicTestResult:
    recorder = ContinuousRecorder(backend, device, sample_rate, channels, buffer_seconds=duration_s + 5)
    recorder.start()
    try:
        target_frames = int(sample_rate * duration_s)
        recorder.wait_until(target_frames, timeout_s=duration_s + 10)
        segment = recorder.buffer.read_segment(0, target_frames)
        overflow = recorder.buffer.overlaps_overflow(0, target_frames)
    finally:
        recorder.stop()

    metrics = quality.measure(segment)
    reasons = []
    if metrics.peak <= NO_SIGNAL_PEAK:
        reasons.append(
            f"no signal: peak {metrics.peak:.6f} is at most one LSB — microphone muted, "
            "disconnected, or the wrong input device is selected"
        )
    if overflow:
        reasons.append("audio backend reported overflow during the test")
    if metrics.clipping_ratio > 0:
        reasons.append(f"clipping detected (ratio={metrics.clipping_ratio:.4f})")

    return MicTestResult(
        device=recorder.device_info,
        sample_rate=sample_rate,
        channels=channels,
        duration_s=duration_s,
        peak=metrics.peak,
        rms=metrics.rms,
        clipping_ratio=metrics.clipping_ratio,
        overflow=overflow,
        passed=not reasons,
        reasons=reasons,
    )


def estimate_bytes(total_trials: int, stored_ms: float, sample_rate: int, channels: int, bytes_per_sample: int = 2, safety_factor: float = 1.2) -> int:
    """REQ-55.1."""
    raw = total_trials * (stored_ms / 1000.0) * sample_rate * channels * bytes_per_sample
    return int(raw * safety_factor)


def nearest_existing_dir(path) -> Path:
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def free_bytes(path) -> int:
    """Free space on the filesystem that will hold `path`, without
    creating anything (a dry run must not write, REQ-59.1)."""
    return shutil.disk_usage(str(nearest_existing_dir(path))).free
