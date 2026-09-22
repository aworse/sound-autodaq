"""
Audio device lifecycle, continuous stream, ring buffer, and segment
extraction (Part IV of the spec).

Owns: device lifecycle, continuous stream, buffer, segment extraction.
Never: disk I/O in the callback, Mel processing, classification (REQ-57).

The real PortAudio-backed device and a hardware-free synthetic source both
implement `AudioBackend`, so the whole pipeline is testable without
hardware (REQ-61.1/61.2) while the production path and the test path share
every line of buffering / segmentation / quality code.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable, Optional

import numpy as np

from .errors import AudioDeviceError, AudioStreamError

AudioCallback = Callable[[np.ndarray, bool], None]
# callback(frames: np.ndarray[int16, shape=(n, channels)], overflow: bool) -> None


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    name: str
    index: int
    host_api: str
    max_input_channels: int
    default_sample_rate: float
    input_latency_s: float


class AudioBackend:
    """Interface both the real device backend and the synthetic test
    backend implement."""

    def list_devices(self) -> list:  # list[DeviceInfo]
        raise NotImplementedError

    def resolve_device(self, device) -> DeviceInfo:
        raise NotImplementedError

    def supports_sample_rate(self, device_info: DeviceInfo, sample_rate: int, channels: int) -> bool:
        raise NotImplementedError

    def start(self, device_info: DeviceInfo, sample_rate: int, channels: int, callback: AudioCallback) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real backend (sounddevice / PortAudio)
# ---------------------------------------------------------------------------


class SoundDeviceBackend(AudioBackend):
    def __init__(self):
        self._stream = None

    def _sd(self):
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - hardware-only path
            raise AudioDeviceError(f"sounddevice / PortAudio not available: {exc}") from exc
        return sd

    def list_devices(self) -> list:
        sd = self._sd()
        hostapis = sd.query_hostapis()
        devices = []
        for idx, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0:
                continue
            devices.append(
                DeviceInfo(
                    name=d["name"],
                    index=idx,
                    host_api=hostapis[d["hostapi"]]["name"],
                    max_input_channels=d["max_input_channels"],
                    default_sample_rate=d["default_samplerate"],
                    input_latency_s=d["default_low_input_latency"],
                )
            )
        return devices

    def resolve_device(self, device) -> DeviceInfo:
        sd = self._sd()
        try:
            if device is None:
                idx = sd.default.device[0]
                if idx is None or idx < 0:
                    raise AudioDeviceError("no default input device is available")
            elif isinstance(device, int):
                idx = device
            else:
                idx = sd.query_devices(device, "input")["index"] if hasattr(
                    sd.query_devices(device, "input"), "get"
                ) else None
                # sounddevice returns a dict; resolve index by name lookup.
                idx = None
                for d in self.list_devices():
                    if d.name == device:
                        idx = d.index
                        break
                if idx is None:
                    raise AudioDeviceError(f"audio device not found: {device!r}")
            d = sd.query_devices(idx)
        except Exception as exc:
            if isinstance(exc, AudioDeviceError):
                raise
            raise AudioDeviceError(f"audio device not found: {device!r} ({exc})") from exc
        hostapis = sd.query_hostapis()
        return DeviceInfo(
            name=d["name"],
            index=idx,
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            default_sample_rate=d["default_samplerate"],
            input_latency_s=d["default_low_input_latency"],
        )

    def supports_sample_rate(self, device_info: DeviceInfo, sample_rate: int, channels: int) -> bool:
        sd = self._sd()
        try:
            sd.check_input_settings(
                device=device_info.index, channels=channels, samplerate=sample_rate, dtype="int16"
            )
            return True
        except Exception:
            return False

    def start(self, device_info: DeviceInfo, sample_rate: int, channels: int, callback: AudioCallback) -> None:
        sd = self._sd()

        def _sd_callback(indata, frames, time_info, status):
            overflow = bool(status.input_overflow) if status else False
            if status and status.input_underflow:
                # Underflow on input is unusual; surface via overflow path
                # so the trial is never silently trusted (REQ-21.4).
                overflow = True
            callback(indata.copy(), overflow)

        try:
            self._stream = sd.InputStream(
                device=device_info.index,
                channels=channels,
                samplerate=sample_rate,
                dtype="int16",
                callback=_sd_callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioStreamError(f"failed to open audio input stream: {exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ---------------------------------------------------------------------------
# Synthetic backend (hardware-free, for tests and CI — REQ-61.1/61.2)
# ---------------------------------------------------------------------------


class SyntheticBackend(AudioBackend):
    """Generates a deterministic PCM16 signal, value == (sample_index %
    2000) - 1000, so a test can verify `segment_start_sample` alignment by
    recomputing the expected samples directly from the index (REQ-61.3),
    with no dependency on real time or hardware.

    Runs its generator in a background thread that produces frames as
    fast as the consumer's callback returns, with no artificial sleep —
    a full 19,000-trial synthetic session runs in test time, not wall
    time.
    """

    def __init__(self, overflow_at_samples: Optional[set] = None):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._overflow_at = overflow_at_samples or set()
        self.frames_produced = 0
        self._lock = threading.Lock()

    def list_devices(self) -> list:
        return [
            DeviceInfo(
                name="Synthetic Test Source",
                index=0,
                host_api="synthetic",
                max_input_channels=8,
                default_sample_rate=48000.0,
                input_latency_s=0.0,
            )
        ]

    def resolve_device(self, device) -> DeviceInfo:
        return self.list_devices()[0]

    def supports_sample_rate(self, device_info: DeviceInfo, sample_rate: int, channels: int) -> bool:
        return True

    def start(self, device_info: DeviceInfo, sample_rate: int, channels: int, callback: AudioCallback) -> None:
        self._stop_event.clear()
        chunk = max(1, sample_rate // 100)  # ~10ms chunks

        chunk_period_s = chunk / sample_rate

        def _run():
            idx = 0
            next_tick = time.monotonic()
            while not self._stop_event.is_set():
                sample_indices = np.arange(idx, idx + chunk, dtype=np.int64)
                values = ((sample_indices % 2000) - 1000).astype(np.int16)
                block = np.tile(values.reshape(-1, 1), (1, channels))
                overflow = bool(self._overflow_at & set(sample_indices.tolist()))
                callback(block, overflow)
                idx += chunk
                with self._lock:
                    self.frames_produced = idx
                # Pace to real time, like a real device, so trial phase
                # sleeps and captured sample counts stay consistent.
                next_tick += chunk_period_s
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


# ---------------------------------------------------------------------------
# Ring buffer + segment extraction (REQ-15)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OverflowEvent:
    start_sample: int
    end_sample: int


class RingBuffer:
    """Timestamped circular buffer of continuously captured audio.

    Frames are appended by the (audio-thread-driven) callback via
    `write()`; trial segments are pulled out later by absolute sample
    index via `read_segment()` (REQ-15.4). Sized to comfortably outlive
    the writer thread's worst-case lag (REQ-15.5).
    """

    def __init__(self, capacity_frames: int, channels: int):
        self.capacity = capacity_frames
        self.channels = channels
        self._buf = np.zeros((capacity_frames, channels), dtype=np.int16)
        self._lock = threading.Lock()
        self.total_written = 0
        self._overflow_events: list = []

    def write(self, frames: np.ndarray, overflow: bool) -> None:
        n = frames.shape[0]
        start = self.total_written
        with self._lock:
            pos = start % self.capacity
            end_pos = pos + n
            if end_pos <= self.capacity:
                self._buf[pos:end_pos] = frames
            else:
                first = self.capacity - pos
                self._buf[pos:] = frames[:first]
                self._buf[: end_pos - self.capacity] = frames[first:]
            self.total_written += n
            if overflow:
                self._overflow_events.append(OverflowEvent(start, start + n))

    def read_segment(self, start_sample: int, end_sample: int) -> np.ndarray:
        if start_sample < 0 or end_sample < start_sample:
            raise AudioStreamError(f"invalid segment range [{start_sample}, {end_sample})")
        with self._lock:
            total = self.total_written
            oldest_available = max(0, total - self.capacity)
            if start_sample < oldest_available:
                raise AudioStreamError(
                    f"segment [{start_sample},{end_sample}) was overwritten before extraction "
                    f"(buffer capacity {self.capacity} frames); increase the ring buffer size"
                )
            if end_sample > total:
                raise AudioStreamError(
                    f"segment [{start_sample},{end_sample}) extends beyond captured audio "
                    f"({total} frames written)"
                )
            n = end_sample - start_sample
            out = np.empty((n, self.channels), dtype=np.int16)
            pos = start_sample % self.capacity
            end_pos = pos + n
            if end_pos <= self.capacity:
                out[:] = self._buf[pos:end_pos]
            else:
                first = self.capacity - pos
                out[:first] = self._buf[pos:]
                out[first:] = self._buf[: end_pos - self.capacity]
            return out

    def overlaps_overflow(self, start_sample: int, end_sample: int) -> bool:
        with self._lock:
            events = list(self._overflow_events)
        return any(not (end_sample <= e.start_sample or start_sample >= e.end_sample) for e in events)


class ContinuousRecorder:
    """Session-lifetime audio capture: one stream open per session
    (REQ-15.1/15.2), feeding a RingBuffer sized per REQ-15.5.
    """

    def __init__(
        self,
        backend: AudioBackend,
        device,
        sample_rate: int,
        channels: int,
        buffer_seconds: float = 30.0,
    ):
        self.backend = backend
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_info = backend.resolve_device(device)
        if not backend.supports_sample_rate(self.device_info, sample_rate, channels):
            raise AudioDeviceError(
                f"device {self.device_info.name!r} does not natively support "
                f"{sample_rate} Hz / {channels} ch; implicit resampling is prohibited (REQ-16.3/16.4)"
            )
        capacity = max(sample_rate * 2, int(sample_rate * buffer_seconds))
        self.buffer = RingBuffer(capacity_frames=capacity, channels=channels)
        self._started = False

    def start(self) -> None:
        self.backend.start(self.device_info, self.sample_rate, self.channels, self.buffer.write)
        self._started = True

    def stop(self) -> None:
        if self._started:
            self.backend.stop()
            self._started = False

    @property
    def frames_captured(self) -> int:
        return self.buffer.total_written

    def wait_until(self, sample_count: int, timeout_s: float = 30.0) -> None:
        """Block (via a light poll, never in the audio callback) until the
        buffer has produced at least `sample_count` frames."""
        deadline = time.monotonic() + timeout_s
        while self.buffer.total_written < sample_count:
            if time.monotonic() > deadline:
                raise AudioStreamError("timed out waiting for audio samples; possible device stall")
            time.sleep(0.001)
