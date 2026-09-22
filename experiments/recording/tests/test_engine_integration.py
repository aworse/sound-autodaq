"""
Full-pipeline integration tests using the hardware-free SyntheticBackend
(REQ-61): synthetic audio -> recorder -> trial engine -> segmentation ->
WAV -> manifest -> validator, with no PortAudio device required.
"""

from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.errors import PreflightError
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.ui import ControlSource, QueueControlSource
from experiments.recording.validator import validate_session
from experiments.recording.writer import read_manifest_jsonl

FAST_TRIAL = {
    "countdown_ms": 0,
    "pre_roll_ms": 5,
    "input_window_ms": 5,
    "post_roll_ms": 5,
    "inter_trial_ms": 0,
}


def _config(tmp_path, **overrides):
    overrides.setdefault("output", {})
    overrides["output"].setdefault("root", str(tmp_path / "data"))
    overrides.setdefault("trial", {})
    trial_overrides = {"repetitions_per_class": 1, **FAST_TRIAL}
    trial_overrides.update(overrides["trial"])
    overrides["trial"] = trial_overrides
    d = base_config_dict(**overrides)
    return config_from_dict(d)


def test_full_synthetic_session_completes_and_validates(tmp_path):
    config = _config(tmp_path)
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary = engine.run(resume=False, control_source=QueueControlSource())

    assert summary.attempted_trials == summary.expected_trials
    assert summary.valid_trials == summary.expected_trials
    assert summary.validation_result == "PASS"

    report = validate_session(engine.session_dir)
    assert report.passed


def test_segment_alignment_matches_synthetic_signal(tmp_path):
    """REQ-61.3: verify segment content matches the expected sample range
    of the synthetic stream, not merely that files exist."""
    import numpy as np

    from experiments.recording.writer import read_wav

    config = _config(tmp_path)
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    engine.run(resume=False, control_source=QueueControlSource())

    records = read_manifest_jsonl(engine.session_dir / "manifest.jsonl")
    assert records
    for r in records:
        if r["status"] != "valid":
            continue
        samples, sr, ch = read_wav(engine.session_dir / r["file"])
        start = r["segment_start_sample"]
        expected_indices = np.arange(start, start + len(samples))
        expected = ((expected_indices % 2000) - 1000).astype(np.int16)
        assert np.array_equal(samples, expected)


class _QuitAfterN(ControlSource):
    """Returns 'quit' once N trials have already been polled through, to
    simulate abnormal termination at a specific point (T-8)."""

    def __init__(self, n: int):
        self.n = n
        self.count = 0

    def poll(self):
        self.count += 1
        if self.count > self.n:
            return "quit"
        return None


def test_t8_resume_position_no_gaps_no_reredo(tmp_path):
    config = _config(tmp_path)
    engine1 = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary1 = engine1.run(resume=False, control_source=_QuitAfterN(5))
    assert summary1.attempted_trials == 5
    assert summary1.validation_result is None  # not validated on early quit

    records_before = read_manifest_jsonl(engine1.session_dir / "manifest.jsonl")
    ids_before = sorted(r["trial_id"] for r in records_before)
    assert ids_before == list(range(1, 6))

    engine2 = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary2 = engine2.run(resume=True, control_source=QueueControlSource())

    records_after = read_manifest_jsonl(engine2.session_dir / "manifest.jsonl")
    ids_after = sorted(r["trial_id"] for r in records_after)
    # No duplicates, no gaps, continues where it left off.
    assert ids_after == list(range(1, len(ids_after) + 1))
    assert len(ids_after) == len(set(ids_after))
    assert summary2.expected_trials == summary1.expected_trials
    assert summary2.attempted_trials >= summary1.attempted_trials


def test_t9_resume_refused_on_sample_rate_change(tmp_path):
    config = _config(tmp_path)
    engine1 = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    engine1.run(resume=False, control_source=_QuitAfterN(2))

    changed = _config(tmp_path, recording={"sample_rate": 44100})
    engine2 = SessionEngine(changed, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    try:
        engine2.run(resume=True, control_source=QueueControlSource())
        assert False, "expected ResumeError"
    except PreflightError as exc:
        assert "sample_rate" in str(exc)


def test_t9_resume_refused_on_repetitions_change(tmp_path):
    config = _config(tmp_path)
    engine1 = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    engine1.run(resume=False, control_source=_QuitAfterN(2))

    changed = _config(tmp_path, trial={"repetitions_per_class": 2})
    engine2 = SessionEngine(changed, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    try:
        engine2.run(resume=True, control_source=QueueControlSource())
        assert False, "expected ResumeError"
    except PreflightError as exc:
        assert "repetitions_per_class" in str(exc)
