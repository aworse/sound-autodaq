import numpy as np
import pytest

from experiments.recording.quality import is_suspicious_silence, measure


def test_full_scale_signal_peak_and_clipping():
    samples = np.full(1000, 32767, dtype=np.int16)
    m = measure(samples)
    assert m.peak == pytest.approx(1.0, rel=1e-3)
    assert m.clipping_ratio == pytest.approx(1.0)


def test_zero_signal_is_silent():
    samples = np.zeros(1000, dtype=np.int16)
    m = measure(samples)
    assert m.peak == 0.0
    assert m.rms == 0.0
    assert m.clipping_ratio == 0.0
    assert is_suspicious_silence(m, rms_threshold=0.001)


def test_known_amplitude_sine_rms():
    n = 48000
    t = np.arange(n) / 48000.0
    amplitude = 16384
    sine = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    m = measure(sine)
    expected_rms = (amplitude / 32768) / (2 ** 0.5)
    assert m.rms == pytest.approx(expected_rms, abs=0.01)
    assert not is_suspicious_silence(m, rms_threshold=0.001)


def test_no_clipping_for_moderate_signal():
    samples = np.full(1000, 1000, dtype=np.int16)
    m = measure(samples)
    assert m.clipping_ratio == 0.0
