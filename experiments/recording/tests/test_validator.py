import numpy as np

from experiments.recording.metadata import write_json_atomic
from experiments.recording.scheduler import generate_schedule, write_schedule
from experiments.recording.validator import validate_session
from experiments.recording.writer import ManifestWriter, TrialRecord, trial_filename, write_wav_atomic

CLASSES = ("A", "B", "C")
REPS = 2
SAMPLE_RATE = 48000
CHANNELS = 1


def _record(trial_id, label, repetition, status="valid", file=None):
    fn = file if file is not None else f"audio/{trial_filename(trial_id)}"
    return TrialRecord(
        trial_id=trial_id,
        file=fn,
        label=label,
        scheduled_label=label,
        observed_label=None,
        participant="P01",
        scenario="S01",
        session="SESSION01",
        repetition=repetition,
        status=status,
        input_mode="human",
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
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
        peak=0.1,
        rms=0.1,
        clipping_ratio=0.0,
        overflow=False,
    )


def _build_clean_session(tmp_path):
    session_dir = tmp_path
    (session_dir / "audio").mkdir(parents=True)
    sched = generate_schedule(CLASSES, REPS, "balanced_random", seed=1, class_definition_version="v1")
    write_schedule(sched, session_dir / "schedule.json")
    write_json_atomic(session_dir / "session.json", {"sample_rate": SAMPLE_RATE, "channels": CHANNELS})

    mw = ManifestWriter(session_dir)
    for t in sched.trials:
        write_wav_atomic(
            session_dir / "audio" / trial_filename(t.trial_id),
            np.zeros(100, dtype=np.int16),
            SAMPLE_RATE,
            CHANNELS,
        )
        mw.append(_record(t.trial_id, t.label, t.repetition))
    mw.close()
    return session_dir, sched


def test_clean_session_passes(tmp_path):
    session_dir, _ = _build_clean_session(tmp_path)
    report = validate_session(session_dir)
    assert report.passed
    assert not report.missing_audio
    assert not report.orphan_audio
    assert not report.duplicate_trial
    assert not report.schedule_drift
    assert not report.format_mismatch


def test_orphan_audio_detected(tmp_path):
    session_dir, _ = _build_clean_session(tmp_path)
    write_wav_atomic(session_dir / "audio" / "trial_00099999.wav", np.zeros(10, dtype=np.int16), SAMPLE_RATE, CHANNELS)
    report = validate_session(session_dir)
    assert not report.passed
    assert "trial_00099999.wav" in report.orphan_audio


def test_missing_audio_detected(tmp_path):
    session_dir, sched = _build_clean_session(tmp_path)
    (session_dir / "audio" / trial_filename(sched.trials[0].trial_id)).unlink()
    report = validate_session(session_dir)
    assert not report.passed
    assert sched.trials[0].trial_id in report.missing_audio


def test_duplicate_trial_detected(tmp_path):
    session_dir, sched = _build_clean_session(tmp_path)
    tid = sched.trials[0].trial_id
    with open(session_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        import json

        f.write(json.dumps(_record(tid, sched.trials[0].label, sched.trials[0].repetition).to_dict(), ensure_ascii=False) + "\n")
    report = validate_session(session_dir)
    assert not report.passed
    assert tid in report.duplicate_trial


def test_schedule_drift_detected(tmp_path):
    session_dir, sched = _build_clean_session(tmp_path)
    import json

    lines = (session_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    wrong_label = next(l for l in CLASSES if l != first["label"])
    first["label"] = wrong_label
    first["scheduled_label"] = wrong_label
    lines[0] = json.dumps(first, ensure_ascii=False)
    (session_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_session(session_dir)
    assert not report.passed
    assert first["trial_id"] in report.schedule_drift


def test_format_mismatch_detected(tmp_path):
    session_dir, sched = _build_clean_session(tmp_path)
    tid = sched.trials[0].trial_id
    path = session_dir / "audio" / trial_filename(tid)
    path.unlink()
    write_wav_atomic(path, np.zeros(100, dtype=np.int16), sample_rate=44100, channels=CHANNELS)
    report = validate_session(session_dir)
    assert not report.passed
    assert tid in report.format_mismatch


def test_balance_failure_detected(tmp_path):
    session_dir, sched = _build_clean_session(tmp_path)
    import json

    lines = (session_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["status"] = "invalid"
    lines[0] = json.dumps(first, ensure_ascii=False)
    (session_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_session(session_dir)
    assert report.class_balance_warnings
