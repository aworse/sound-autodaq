"""
Device enumeration, capability probing, and the microphone test (§54-55).
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

from . import quality
from .recorder import AudioBackend, ContinuousRecorder, DeviceInfo


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


def free_bytes(path) -> int:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(str(path)).free


def check_disk_space(output_dir, required_bytes: int) -> bool:
    return free_bytes(output_dir) >= required_bytes
