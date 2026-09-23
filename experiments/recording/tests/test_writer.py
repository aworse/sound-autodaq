import numpy as np
import pytest

from experiments.recording.errors import DuplicateTrialError
from experiments.recording.writer import (
    ManifestWriter,
    TrialRecord,
    read_manifest_csv,
    read_manifest_jsonl,
    read_wav,
    trial_filename,
    write_wav_atomic,
)


def _record(trial_id=1, file="audio/trial_00000001.wav", status="valid"):
    return TrialRecord(
        trial_id=trial_id,
        file=file,
        label="ㄱ",
        scheduled_label="ㄱ",
        observed_label=None,
        participant="P01",
        scenario="S01",
        session="SESSION01",
        repetition=1,
        status=status,
        input_mode="human",
        sample_rate=48000,
        channels=1,
        sample_format="PCM_16",
        num_samples=100,
        duration_ms=2.08,
        segment_start_sample=0,
        segment_end_sample=100,
        trial_start_ns=1,
        input_expected_ns=2,
        input_detected_ns=None,
        trial_end_ns=3,
        wall_clock_utc="2026-09-22T00:00:00.000Z",
        peak=0.5,
        rms=0.1,
        clipping_ratio=0.0,
        overflow=False,
    )


def test_wav_round_trip(tmp_path):
    path = tmp_path / "t.wav"
    samples = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
    write_wav_atomic(path, samples, sample_rate=48000, channels=1)
    read_samples, sr, ch = read_wav(path)
    assert sr == 48000
    assert ch == 1
    assert np.array_equal(read_samples, samples)


def test_atomic_write_never_overwrites(tmp_path):
    path = tmp_path / "t.wav"
    samples = np.zeros(10, dtype=np.int16)
    write_wav_atomic(path, samples, 48000, 1)
    with pytest.raises(DuplicateTrialError):
        write_wav_atomic(path, samples, 48000, 1)


def test_t12_no_partial_file_under_final_name(tmp_path):
    path = tmp_path / "t.wav"
    samples = np.zeros(10, dtype=np.int16)
    write_wav_atomic(path, samples, 48000, 1)
    assert path.exists()
    # No leftover temp file after a successful write.
    assert not (tmp_path / "t.wav.tmp").exists()


def test_manifest_append_and_flush(tmp_path):
    with ManifestWriter(tmp_path) as mw:
        mw.append(_record(1))
        mw.append(_record(2, file="audio/trial_00000002.wav"))

    csv_rows = read_manifest_csv(tmp_path / "manifest.csv")
    jsonl_rows = read_manifest_jsonl(tmp_path / "manifest.jsonl")
    assert len(csv_rows) == 2
    assert len(jsonl_rows) == 2


def test_t11_manifest_csv_jsonl_agree(tmp_path):
    with ManifestWriter(tmp_path) as mw:
        mw.append(_record(1))

    csv_rows = read_manifest_csv(tmp_path / "manifest.csv")
    jsonl_rows = read_manifest_jsonl(tmp_path / "manifest.jsonl")
    assert csv_rows[0]["trial_id"] == str(jsonl_rows[0]["trial_id"])
    assert csv_rows[0]["file"] == jsonl_rows[0]["file"]
    assert csv_rows[0]["label"] == jsonl_rows[0]["label"]
    assert csv_rows[0]["status"] == jsonl_rows[0]["status"]


def test_t10_manifest_rejects_duplicate_trial_id(tmp_path):
    with ManifestWriter(tmp_path) as mw:
        mw.append(_record(1))
        with pytest.raises(DuplicateTrialError):
            mw.append(_record(1))


def test_manifest_resumes_without_duplicating(tmp_path):
    mw = ManifestWriter(tmp_path)
    mw.append(_record(1))
    mw.close()

    mw2 = ManifestWriter(tmp_path)
    assert mw2.has_trial(1)
    mw2.append(_record(2, file="audio/trial_00000002.wav"))
    mw2.close()

    jsonl_rows = read_manifest_jsonl(tmp_path / "manifest.jsonl")
    assert len(jsonl_rows) == 2
    assert {r["trial_id"] for r in jsonl_rows} == {1, 2}


def test_trial_filename_format():
    assert trial_filename(1) == "trial_00000001.wav"
    assert trial_filename(19000) == "trial_00019000.wav"
