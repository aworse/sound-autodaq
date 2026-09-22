"""
Signal quality measurement (§41, §42).

Owns: measurement of recorded audio. Never: modification of recorded
audio (REQ-2.1.4, REQ-57).
"""

from __future__ import annotations

import dataclasses

import numpy as np

PCM16_FULL_SCALE = 32768


@dataclasses.dataclass(frozen=True)
class QualityMetrics:
    peak: float
    rms: float
    clipping_ratio: float


def measure(samples: np.ndarray) -> QualityMetrics:
    """Compute peak / rms / clipping_ratio for a PCM16 int array,
    normalized to [0, 1] (REQ-41.1)."""
    if samples.size == 0:
        return QualityMetrics(peak=0.0, rms=0.0, clipping_ratio=0.0)

    normalized = samples.astype(np.float64) / PCM16_FULL_SCALE
    peak = float(np.max(np.abs(normalized)))
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    clipped = np.abs(samples.astype(np.int64)) >= (PCM16_FULL_SCALE - 1)
    clipping_ratio = float(np.count_nonzero(clipped)) / samples.size
    return QualityMetrics(peak=peak, rms=rms, clipping_ratio=clipping_ratio)


def is_clipping(metrics: QualityMetrics, threshold: float) -> bool:
    return metrics.clipping_ratio > threshold


def is_suspicious_silence(metrics: QualityMetrics, rms_threshold: float) -> bool:
    return metrics.rms < rms_threshold
