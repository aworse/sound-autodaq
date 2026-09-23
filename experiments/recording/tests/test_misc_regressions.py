"""Regression tests for hardware checks, labels, and writer durability."""

import os

import numpy as np
import pytest

from experiments.recording import hardware
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.writer import (
    ManifestWriter,
    read_manifest_jsonl,
    write_wav_atomic,
)

# Dubeolsik: every jamo a single keystroke (with or without Shift) produces.
DUBEOLSIK_SINGLE_KEY = set("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡㅃㅉㄸㄲㅆㅒㅖ")


def test_every_class_is_one_dubeolsik_keystroke():
    from classism.labels import CLASSES

    assert set(CLASSES) <= DUBEOLSIK_SINGLE_KEY
    assert len(CLASSES) == len(set(CLASSES))


def test_mic_test_fails_on_digital_silence():
    result = hardware.run_mic_test(SyntheticBackend(amplitude=0), None, 48000, 1, duration_s=0.2)
    assert not result.passed
    assert any("no signal" in r for r in result.reasons)


def test_mic_test_passes_on_live_signal():
    assert hardware.run_mic_test(SyntheticBackend(), None, 48000, 1, duration_s=0.2).passed


def test_free_bytes_does_not_create_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert hardware.free_bytes(target) > 0
    assert not (tmp_path / "a").exists()


def test_t12_interrupted_write_leaves_nothing_under_final_name(tmp_path, monkeypatch):
    def killed(*args, **kwargs):
        raise OSError("simulated power loss during rename")

    monkeypatch.setattr(os, "replace", killed)
    final = tmp_path / "trial_00000001.wav"
    with pytest.raises(OSError):
        write_wav_atomic(final, np.zeros(100, dtype=np.int16), 48000, 1)
    assert not final.exists()


def _row(i):
    from experiments.recording.tests.test_writer import _record

    return _record(i, file=f"audio/trial_{i:08d}.wav")


def test_torn_manifest_tail_is_set_aside_not_merged(tmp_path):
    with ManifestWriter(tmp_path) as mw:
        mw.append(_row(1))
    with open(tmp_path / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write('{"trial_id": 2, "fi')
    assert [r["trial_id"] for r in read_manifest_jsonl(tmp_path / "manifest.jsonl")] == [1]

    with ManifestWriter(tmp_path) as mw:
        assert mw.torn_tails
        mw.append(_row(2))
    assert [r["trial_id"] for r in read_manifest_jsonl(tmp_path / "manifest.jsonl")] == [1, 2]
    assert '"fi' in (tmp_path / "manifest.jsonl.torn").read_text(encoding="utf-8")
