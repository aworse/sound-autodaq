"""
Full-pipeline integration tests using the hardware-free SyntheticBackend
(REQ-61): synthetic audio -> recorder -> trial engine -> segmentation ->
WAV -> manifest -> validator, with no PortAudio device required.
"""

import json
import os
import signal

import numpy as np
import pytest

from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.errors import AudioStreamError, DuplicateTrialError, PreflightError
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.ui import ControlSource, Display, QueueControlSource
from experiments.recording.validator import validate_session
from experiments.recording.writer import read_manifest_jsonl, read_wav, trial_filename

FAST_TRIAL = {
    "countdown_ms": 0,
    "pre_roll_ms": 10,
    "input_window_ms": 10,
    "post_roll_ms": 10,
    "inter_trial_ms": 0,
}

def _config(tmp_path, **overrides):
    overrides.setdefault("output", {})
    overrides["output"].setdefault("root", str(tmp_path / "data"))
    trial = {"repetitions_per_class": 1, **FAST_TRIAL}
    trial.update(overrides.pop("trial", {}))
    overrides["trial"] = trial
    return config_from_dict(base_config_dict(**overrides))


def _engine(config, backend=None):
    return SessionEngine(config, backend=backend or SyntheticBackend(), mic_test_duration_s=0.2)


def _rows(engine):
    return read_manifest_jsonl(engine.session_dir / "manifest.jsonl")


class ControlsAtTrial(ControlSource):
    """Delivers the given controls while trial number k (1-based, in the
    order trials are run) is on screen. The engine polls once per trial
    when nothing is pending, so poll k belongs to trial k."""

    def __init__(self, plan: dict, interactive=False):
        self.plan = {k: list(v) for k, v in plan.items()}
        self.trial = 0
        self.buffer: list = []
        self.interactive = interactive

    def poll(self):
        if self.buffer:
            return self.buffer.pop(0)
        self.trial += 1
        self.buffer = self.plan.pop(self.trial, [])
        return self.buffer.pop(0) if self.buffer else None


def _quit_after(n):
    return ControlsAtTrial({n: ["quit"]})


class OverflowInSession(SyntheticBackend):
    """Reports overflow on the given session-stream samples only, so the
    pre-flight microphone test (the first stream opened) stays clean."""

    def __init__(self, samples):
        super().__init__()
        self._samples = set(samples)
        self.starts = 0

    def start(self, device_info, sample_rate, channels, callback):
        self.starts += 1
        self._overflow_at = self._samples if self.starts >= 2 else set()
        super().start(device_info, sample_rate, channels, callback)


# -- the happy path -----------------------------------------------------------


def test_full_synthetic_session_completes_and_validates(tmp_path):
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=QueueControlSource())

    assert summary.completed is True
    assert summary.stop_reason is None
    assert summary.attempted_trials == summary.expected_trials == summary.valid_trials
    assert summary.validation_result == "PASS"
    assert validate_session(engine.session_dir).passed
    for name in ("schedule.json", "session.json", "progress.json", "session_summary.json",
                 "manifest.csv", "manifest.jsonl", "experiment.log"):
        assert (engine.session_dir / name).exists(), name


def test_segment_alignment_matches_synthetic_signal(tmp_path):
    """REQ-61.3: the audio in each file is exactly the stream samples its
    manifest row says it is."""
    engine = _engine(_config(tmp_path))
    engine.run(resume=False, control_source=QueueControlSource())
    for r in _rows(engine):
        samples, _, _ = read_wav(engine.session_dir / r["file"])
        idx = np.arange(r["segment_start_sample"], r["segment_start_sample"] + len(samples))
        assert np.array_equal(samples, ((idx % 2000) - 1000).astype(np.int16))


# -- finding 1: the screen must show the trial being recorded -----------------


class _ScreenLog(Display):
    def __init__(self):
        super().__init__(enabled=True, interactive=False)
        self.press = []

    def show(self, text):
        if "PRESS" in text:
            trial_id = int(text.split("Trial ")[1].split()[0])
            big_target = text.split("target:")[1].split()[0]
            self.press.append((trial_id, big_target, text.split("PRESS")[1].split()[0]))

    def line(self, text):
        pass


def test_screen_shows_the_trial_being_recorded(tmp_path):
    engine = _engine(_config(tmp_path))
    screen = _ScreenLog()
    engine.run(resume=False, control_source=QueueControlSource(), display=screen)

    recorded = {r["trial_id"]: r["label"] for r in _rows(engine)}
    assert [s[0] for s in screen.press] == sorted(recorded)
    for trial_id, big_target, press_label in screen.press:
        assert big_target == press_label == recorded[trial_id]


# -- finding 2: control keys never collide with target keys -------------------


def test_control_keys_are_not_dubeolsik_letters_or_classes():
    from classism.labels import CLASSES
    from experiments.recording.ui import KEYMAP

    for key in KEYMAP:
        assert key.isdigit(), key
        assert key not in CLASSES


# -- finding 3: failures cannot loop forever; a dead mic fails pre-flight -----


def test_persistent_silence_stops_after_max_consecutive_failures(tmp_path):
    config = _config(tmp_path, quality={"silence_rms_threshold": 0.5, "max_consecutive_failures": 3})
    engine = _engine(config)
    summary = engine.run(resume=False, control_source=QueueControlSource())

    assert summary.completed is False
    assert "3 consecutive" in summary.stop_reason
    assert [r["status"] for r in _rows(engine)] == ["suspicious_silence"] * 3


def test_dead_microphone_fails_preflight(tmp_path):
    engine = _engine(_config(tmp_path), backend=SyntheticBackend(amplitude=0))
    with pytest.raises(PreflightError, match="no signal"):
        engine.run(resume=False, control_source=QueueControlSource())
    assert not (engine.session_dir / "schedule.json").exists()


# -- finding 4: a requeued trial does not make validation fail ----------------


def test_requeued_overflow_session_completes_and_passes(tmp_path):
    backend = OverflowInSession(range(0, 1000))  # the first trial's audio
    engine = _engine(_config(tmp_path), backend)
    summary = engine.run(resume=False, control_source=QueueControlSource())

    rows = _rows(engine)
    assert summary.overflow_trials >= 1
    assert summary.completed is True, summary
    assert summary.valid_trials == summary.expected_trials
    report = validate_session(engine.session_dir)
    assert report.passed and not report.count_mismatch
    failed = [r for r in rows if r["status"] == "audio_overflow"]
    retried = {r["trial_id"]: r for r in rows}
    for r in failed:
        retry = retried[r["superseded_by"]]
        assert (retry["label"], retry["repetition"]) == (r["label"], r["repetition"])


# -- findings 5, 9: resume ------------------------------------------------------


def test_t8_resume_position_no_gaps_no_rerecord(tmp_path):
    config = _config(tmp_path)
    summary1 = _engine(config).run(resume=False, control_source=_quit_after(5))
    assert summary1.attempted_trials == 5
    assert summary1.stop_reason == "operator quit"
    assert summary1.completed is False

    engine2 = _engine(config)
    summary2 = engine2.run(resume=True, control_source=QueueControlSource())
    ids = [r["trial_id"] for r in _rows(engine2)]
    assert ids == list(range(1, summary2.expected_trials + 1))
    assert summary2.completed is True


@pytest.mark.parametrize("change,field", [
    ({"recording": {"sample_rate": 44100}}, "sample_rate"),
    ({"trial": {"repetitions_per_class": 2}}, "repetitions_per_class"),
    ({"trial": {"post_roll_ms": 20}}, "post_roll_ms"),
])
def test_t9_resume_refused_on_changed_parameters(tmp_path, change, field):
    _engine(_config(tmp_path)).run(resume=False, control_source=_quit_after(2))
    with pytest.raises(PreflightError, match=field):
        _engine(_config(tmp_path, **change)).run(resume=True, control_source=QueueControlSource())


def test_requeued_pair_survives_resume(tmp_path):
    backend = OverflowInSession(range(0, 1000))
    config = _config(tmp_path)
    engine1 = _engine(config, backend)
    engine1.run(resume=False, control_source=_quit_after(6))
    failed = {(r["label"], r["repetition"]) for r in _rows(engine1) if r["status"] == "audio_overflow"}
    assert failed, "fixture should produce at least one overflowed trial"

    engine2 = _engine(config)
    summary = engine2.run(resume=True, control_source=QueueControlSource())
    valid = {(r["label"], r["repetition"]) for r in _rows(engine2) if r["status"] == "valid"}
    assert failed <= valid
    assert summary.completed is True
    assert summary.valid_trials == summary.expected_trials


def test_null_seed_is_recorded_and_session_resumes(tmp_path):
    config = _config(tmp_path, randomization={"seed": None})
    engine1 = _engine(config)
    engine1.run(resume=False, control_source=_quit_after(3))
    schedule_seed = json.loads((engine1.session_dir / "schedule.json").read_text())["seed"]
    session = json.loads((engine1.session_dir / "session.json").read_text())
    progress = json.loads((engine1.session_dir / "progress.json").read_text())
    assert isinstance(schedule_seed, int)
    assert session["random_seed"] == progress["random_seed"] == schedule_seed
    assert session["resolved_config"]["randomization"]["seed"] == schedule_seed
    assert session["seed_generated"] is True

    summary = _engine(config).run(resume=True, control_source=QueueControlSource())
    assert summary.completed is True


# -- findings 6, 7: never overwrite, dry run never writes ---------------------


def test_new_run_on_finished_session_is_refused_without_touching_it(tmp_path):
    config = _config(tmp_path)
    engine = _engine(config)
    engine.run(resume=False, control_source=QueueControlSource())
    before = {p.name: p.read_bytes() for p in engine.session_dir.iterdir() if p.is_file()}

    with pytest.raises(PreflightError, match="complete session"):
        _engine(_config(tmp_path, randomization={"seed": 999})).run(control_source=QueueControlSource())
    after = {p.name: p.read_bytes() for p in engine.session_dir.iterdir() if p.is_file()}
    assert before == after


def test_dry_run_writes_nothing(tmp_path):
    engine = _engine(_config(tmp_path))
    report = engine.preflight(resume=False, dry_run=True)
    assert report.passed
    assert not (tmp_path / "data").exists()


# -- finding 11: operator controls do what they say -----------------------------


def test_skip_keeps_audio_and_is_not_requeued(tmp_path):
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=ControlsAtTrial({1: ["skip"]}))
    first = _rows(engine)[0]
    assert first["status"] == "skipped" and first["superseded_by"] is None
    assert (engine.session_dir / first["file"]).exists()
    assert summary.skipped_trials == 1 and summary.attempted_trials == summary.expected_trials
    assert validate_session(engine.session_dir).passed


def test_invalid_is_requeued_at_the_end(tmp_path):
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=ControlsAtTrial({1: ["invalid"]}))
    rows = _rows(engine)
    assert rows[0]["status"] == "operator_marked_invalid" and rows[0]["requeue"] == "end"
    assert rows[-1]["trial_id"] == rows[0]["superseded_by"]
    assert (rows[-1]["label"], rows[-1]["repetition"], rows[-1]["status"]) == (
        rows[0]["label"], rows[0]["repetition"], "valid")
    assert summary.completed is True


def test_repeat_re_records_immediately_with_new_id(tmp_path):
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=ControlsAtTrial({1: ["repeat"]}))
    rows = _rows(engine)
    assert rows[0]["status"] == "operator_marked_invalid"
    assert rows[1]["trial_id"] == rows[0]["superseded_by"] > summary.expected_trials
    assert (rows[1]["label"], rows[1]["repetition"]) == (rows[0]["label"], rows[0]["repetition"])
    assert (engine.session_dir / rows[0]["file"]).exists() and (engine.session_dir / rows[1]["file"]).exists()
    assert summary.completed is True


def test_pause_discard_current_reruns_trial_and_records_break(tmp_path):
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=ControlsAtTrial({1: ["pause", "pause"]}))
    rows = _rows(engine)
    assert rows[0]["status"] == "interrupted" and rows[1]["trial_id"] == rows[0]["superseded_by"]
    breaks = json.loads((engine.session_dir / "session.json").read_text())["breaks"]
    assert [b["break_type"] for b in breaks] == ["operator"]
    assert breaks[0]["break_before_trial"] == rows[1]["trial_id"]
    assert summary.completed is True


def test_pause_continue_current_keeps_trial(tmp_path):
    config = _config(tmp_path, controls={"resume_policy": "continue_current"})
    engine = _engine(config)
    engine.run(resume=False, control_source=ControlsAtTrial({1: ["pause", "pause"]}))
    assert _rows(engine)[0]["status"] == "valid"


def test_disabled_skip_is_ignored(tmp_path):
    engine = _engine(_config(tmp_path, controls={"allow_skip": False}))
    engine.run(resume=False, control_source=ControlsAtTrial({1: ["skip"]}))
    assert _rows(engine)[0]["status"] == "valid"


# -- finding 12: breaks actually happen, are measured, and are persisted ------


def test_automatic_break_is_measured_and_persisted_without_changing_order(tmp_path):
    config = _config(tmp_path, **{"break": {"enabled": True, "every_trials": 10, "duration_seconds": 0.3}})
    engine = _engine(config)
    summary = engine.run(resume=False, control_source=QueueControlSource())
    breaks = json.loads((engine.session_dir / "session.json").read_text())["breaks"]
    assert [b["break_before_trial"] for b in breaks] == [11, 21, 31]
    assert all(b["break_type"] == "automatic" and b["break_duration_s"] >= 0.3 for b in breaks)
    assert summary.total_break_s >= 0.9
    schedule = json.loads((engine.session_dir / "schedule.json").read_text())["trials"]
    assert [r["label"] for r in _rows(engine)] == [t["label"] for t in schedule]


# -- finding 13: stream failures stop the session safely ----------------------


class _StallingBackend(SyntheticBackend):
    """Stops delivering audio after `after` samples, like an unplugged mic."""

    def __init__(self, after):
        super().__init__()
        self.after = after

    def start(self, device_info, sample_rate, channels, callback):
        limit = self.after

        def gated(block, overflow):
            if self.frames_produced < limit:
                callback(block, overflow)

        super().start(device_info, sample_rate, channels, gated)


def test_stream_stall_stops_safely_and_is_resumable(tmp_path):
    config = _config(tmp_path)
    # Each stream counts samples from 0: the 0.1 s mic test stays under the
    # limit, the session stream goes silent-dead ~0.2 s in (a few trials).
    engine = _engine(config, _StallingBackend(after=int(0.2 * 48000)))
    engine.mic_test_duration_s = 0.1
    with pytest.raises(AudioStreamError, match="stalled"):
        engine.run(resume=False, control_source=QueueControlSource())

    summary = json.loads((engine.session_dir / "session_summary.json").read_text())
    assert summary["completed"] is False and "stalled" in summary["stop_reason"]
    assert _rows(engine)[-1]["status"] == "interrupted"

    resumed = _engine(config).run(resume=True, control_source=QueueControlSource())
    assert resumed.completed is True


def test_backend_error_stops_session(tmp_path):
    backend = SyntheticBackend()
    engine = _engine(_config(tmp_path), backend)

    class Trip(ControlSource):
        n = 0

        def poll(self):
            self.n += 1
            if self.n == 3:
                backend.error = "PortAudio reported input underflow"
            return None

    with pytest.raises(AudioStreamError, match="underflow"):
        engine.run(resume=False, control_source=Trip())
    assert _rows(engine)[-1]["status"] == "interrupted"


# -- REQ-49 orphan audio, torn manifest ---------------------------------------


def _orphan_after_crash(tmp_path, policy):
    """Simulate a hard kill between the WAV rename and the manifest append
    of the trial after trial 3."""
    config = _config(tmp_path, output={"duplicate_policy": policy})
    engine = _engine(config)
    engine.run(resume=False, control_source=_quit_after(3))
    schedule = json.loads((engine.session_dir / "schedule.json").read_text())["trials"]
    from experiments.recording.writer import write_wav_atomic

    write_wav_atomic(engine.session_dir / "audio" / trial_filename(4), np.full(1440, 500, np.int16), 48000, 1)
    return config, engine, schedule[3]


def test_orphan_wav_with_error_policy_stops_without_overwriting(tmp_path):
    config, engine, _ = _orphan_after_crash(tmp_path, "error")
    orphan = engine.session_dir / "audio" / trial_filename(4)
    before = orphan.read_bytes()
    with pytest.raises(DuplicateTrialError, match="will not be overwritten"):
        _engine(config).run(resume=True, control_source=QueueControlSource())
    assert orphan.read_bytes() == before
    assert (engine.session_dir / "session_summary.json").exists()


def test_orphan_wav_with_new_id_policy_keeps_it_and_rerecords(tmp_path):
    config, engine, sched4 = _orphan_after_crash(tmp_path, "new_id")
    summary = _engine(config).run(resume=True, control_source=QueueControlSource())
    rows = {r["trial_id"]: r for r in _rows(engine)}
    assert rows[4]["status"] == "interrupted" and rows[4]["file"] == f"audio/{trial_filename(4)}"
    retry = rows[rows[4]["superseded_by"]]
    assert (retry["label"], retry["repetition"], retry["status"]) == (sched4["label"], sched4["repetition"], "valid")
    assert summary.completed is True


def test_torn_manifest_tail_is_set_aside_on_resume(tmp_path):
    config = _config(tmp_path)
    engine = _engine(config)
    engine.run(resume=False, control_source=_quit_after(3))
    with open(engine.session_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write('{"trial_id": 4, "fil')
    summary = _engine(config).run(resume=True, control_source=QueueControlSource())
    assert (engine.session_dir / "manifest.jsonl.torn").exists()
    assert summary.completed is True


# -- REQ-40.3 integrity failure needs an explicit decision --------------------


def _corrupt_once(monkeypatch):
    from experiments.recording import engine as engine_mod

    original = engine_mod.SessionEngine._integrity_problem
    calls = {"n": 0}

    def fake(self, path, n):
        calls["n"] += 1
        return "simulated header damage" if calls["n"] == 2 else original(self, path, n)

    monkeypatch.setattr(engine_mod.SessionEngine, "_integrity_problem", fake)


def test_corrupted_write_without_operator_stops_safely(tmp_path, monkeypatch):
    _corrupt_once(monkeypatch)
    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=QueueControlSource(interactive=False))
    assert summary.stop_reason == "stopped after integrity failure"
    assert _rows(engine)[-1]["status"] == "corrupted"


def test_corrupted_write_retry_decision_re_records(tmp_path, monkeypatch):
    _corrupt_once(monkeypatch)
    engine = _engine(_config(tmp_path))
    # Polls 1 and 2 drain trials 1 and 2; poll 3 is the retry/continue/stop prompt.
    summary = engine.run(resume=False, control_source=ControlsAtTrial({3: ["repeat"]}, interactive=True))
    rows = _rows(engine)
    assert rows[1]["status"] == "corrupted" and rows[2]["trial_id"] == rows[1]["superseded_by"]
    assert "retry" in rows[1]["notes"]
    assert summary.completed is True


# -- misc -------------------------------------------------------------------


def test_automated_mode_is_refused_rather_than_mislabelled(tmp_path):
    with pytest.raises(PreflightError, match="not implemented"):
        _engine(_config(tmp_path, input={"mode": "automated"})).run(control_source=QueueControlSource())


def test_ctrl_c_is_a_safe_stop(tmp_path):
    """SIGINT sent while trial 2 is running (from inside the session, so
    the handler is installed regardless of machine speed)."""

    class CtrlCDuringTrial2(ControlSource):
        n = 0

        def poll(self):
            self.n += 1
            if self.n == 2:
                os.kill(os.getpid(), signal.SIGINT)
            return None

    engine = _engine(_config(tmp_path))
    summary = engine.run(resume=False, control_source=CtrlCDuringTrial2())
    assert summary.stop_reason == "operator quit (Ctrl+C)"
    assert summary.attempted_trials == 2
    assert validate_session(engine.session_dir).count_mismatch is False
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
