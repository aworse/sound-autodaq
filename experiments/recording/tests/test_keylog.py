"""Keystroke verification (Appendix E.1/E.2): pure judging rules, then the
full engine with key events injected when the participant is told to press."""

import json

import pytest

from experiments.recording import keylog
from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.errors import PreflightError
from experiments.recording.keylog import KeyEvent, judge
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.trial import Status
from experiments.recording.ui import KEYMAP, Display, QueueControlSource
from experiments.recording.validator import validate_session
from experiments.recording.writer import read_manifest_jsonl

MS = 1_000_000
# segment: pre-roll 0..100 ms, input window 100..200 ms, post-roll 200..300 ms
SEG = dict(segment_start_ns=0, segment_end_ns=300 * MS, input_expected_ns=100 * MS, control_keys=set(KEYMAP))
KEY_FOR = {jamo: key for key, jamo in {**keylog._DUBEOLSIK, **keylog._DUBEOLSIK_SHIFT}.items()}


def test_dubeolsik_mapping_including_shift():
    assert keylog.key_to_jamo("r") == "ㄱ"
    assert keylog.key_to_jamo("R") == "ㄲ"
    assert keylog.key_to_jamo("O") == "ㅒ"
    assert keylog.key_to_jamo("K") == "ㅏ"  # Shift does not change ㅏ
    assert keylog.key_to_jamo(" ") is None
    from classism.labels import CLASSES

    assert set(CLASSES) <= set(KEY_FOR), "every class must be typeable"


def test_correct_key_in_window_has_no_objection():
    v = judge([KeyEvent("r", 150 * MS)], "ㄱ", **SEG)
    assert v.status is None
    assert (v.observed_label, v.observed_key, v.input_detected_ns, v.keystrokes) == ("ㄱ", "r", 150 * MS, 1)


def test_wrong_key_is_mismatch():
    v = judge([KeyEvent("s", 150 * MS)], "ㄱ", **SEG)
    assert v.status == Status.MISMATCH and v.observed_label == "ㄴ"


def test_shift_confusion_hints_at_caps_lock():
    v = judge([KeyEvent("R", 150 * MS)], "ㄱ", **SEG)
    assert v.status == Status.MISMATCH and "Caps Lock" in v.note


def test_no_key_is_invalid():
    v = judge([], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "no keystroke" in v.note


def test_key_outside_recorded_audio_is_invalid():
    v = judge([KeyEvent("r", -200 * MS)], "ㄱ", **SEG)  # pressed during the countdown
    assert v.status == Status.INVALID and "outside the recorded audio" in v.note
    assert v.observed_label == "ㄱ" and v.keystrokes == 0


def test_key_in_pre_roll_is_kept_with_timestamp():
    v = judge([KeyEvent("r", 50 * MS)], "ㄱ", **SEG)
    assert v.status is None and v.input_detected_ns == 50 * MS


def test_two_keys_in_one_recording_is_invalid():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("r", 250 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and v.keystrokes == 2


def test_hangul_ime_is_invalid_with_hint():
    v = judge([KeyEvent("ㄱ", 150 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "English" in v.note


def test_non_jamo_key_is_invalid():
    v = judge([KeyEvent(" ", 150 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "SPACE" in v.note


def test_operator_digit_during_recording_is_invalid():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("0", 250 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "operator key" in v.note


def test_operator_digit_after_recording_is_fine():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("0", 400 * MS)], "ㄱ", **SEG)
    assert v.status is None


# -- engine ----------------------------------------------------------------


def _config(tmp_path, **overrides):
    trial = {"repetitions_per_class": 1, "countdown_ms": 0, "pre_roll_ms": 20, "input_window_ms": 20,
             "post_roll_ms": 20, "inter_trial_ms": 0}
    trial.update(overrides.pop("trial", {}))
    overrides["trial"] = trial
    overrides.setdefault("output", {})["root"] = str(tmp_path / "data")
    overrides.setdefault("input", {}).setdefault("key_detection", "terminal")
    return config_from_dict(base_config_dict(**overrides))


class Participant(Display):
    """Types when the screen says PRESS, into the same control source the
    terminal would feed. `answer(trial_number, target)` returns the key(s)
    to type, or None to type nothing."""

    def __init__(self, source, answer):
        super().__init__(enabled=True, interactive=False)
        self.source = source
        self.answer = answer
        self.n = 0

    def show(self, text):
        if "PRESS" in text:
            self.n += 1
            target = text.split("PRESS")[1].split()[0]
            for key in self.answer(self.n, target) or []:
                self.source.push_key(key)

    def line(self, text):
        pass


def _run(config, answer):
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    source = QueueControlSource()
    summary = engine.run(control_source=source, display=Participant(source, answer))
    return engine, summary, read_manifest_jsonl(engine.session_dir / "manifest.jsonl")


def test_correct_typing_records_observed_label_and_time(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n, target: [KEY_FOR[target]])
    assert summary.completed is True
    for r in rows:
        assert r["status"] == "valid"
        assert r["observed_label"] == r["label"] and r["keystrokes"] == 1
        assert r["trial_start_ns"] <= r["input_detected_ns"] <= r["trial_end_ns"]
    session = json.loads((engine.session_dir / "session.json").read_text())
    assert session["implementation_decisions"]["keystroke_detection"].startswith("terminal")
    assert validate_session(engine.session_dir).passed


def test_wrong_key_is_mismatch_and_re_recorded(tmp_path):
    def answer(n, target):
        if n == 1:
            return [KEY_FOR["ㅎ" if target != "ㅎ" else "ㅁ"]]
        return [KEY_FOR[target]]

    engine, summary, rows = _run(_config(tmp_path), answer)
    first = rows[0]
    assert first["status"] == "mismatch" and first["observed_label"] != first["label"]
    retry = next(r for r in rows if r["trial_id"] == first["superseded_by"])
    assert retry["status"] == "valid" and retry["observed_label"] == first["label"]
    assert summary.completed is True


def test_unfocused_terminal_stops_with_the_reason(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n, target: None)
    assert summary.completed is False
    assert "no keystroke detected" in summary.stop_reason
    assert [r["status"] for r in rows] == ["invalid"] * 3


def test_hangul_ime_stops_with_the_reason(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n, target: [target])
    assert "English" in summary.stop_reason
    assert all(r["status"] == "invalid" for r in rows)


def test_resume_refuses_a_key_detection_change(tmp_path):
    from experiments.recording.tests.test_engine_integration import ControlsAtTrial

    config = _config(tmp_path, input={"key_detection": "none"})
    SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2).run(
        control_source=ControlsAtTrial({2: ["quit"]}))
    changed = _config(tmp_path, input={"key_detection": "terminal"})
    with pytest.raises(PreflightError, match="key_detection"):
        SessionEngine(changed, backend=SyntheticBackend(), mic_test_duration_s=0.2).run(
            resume=True, control_source=QueueControlSource())


def test_terminal_reader_timestamps_keys_and_maps_digits(monkeypatch):
    """The real TerminalControlSource, driven through a pseudo-terminal."""
    import os
    import pty
    import sys
    import time

    from experiments.recording.ui import TerminalControlSource

    master, slave = pty.openpty()
    fake_stdin = os.fdopen(slave, "r")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    source = TerminalControlSource()
    try:
        assert source.interactive
        before = time.monotonic_ns()
        os.write(master, "r".encode())
        time.sleep(0.3)
        os.write(master, "1".encode())
        time.sleep(0.3)
        keys = source.poll_keys()
        assert [k.key for k in keys] == ["r", "1"]
        assert before <= keys[0].t_ns < keys[1].t_ns <= time.monotonic_ns()
        assert source.poll() == "repeat" and source.poll() is None
    finally:
        source.close()
        fake_stdin.close()
        os.close(master)


def test_resuming_a_session_recorded_before_key_detection(tmp_path):
    from experiments.recording.cli import _config_for_resume
    from experiments.recording.metadata import write_json_atomic

    resolved = base_config_dict()
    del resolved["input"]["key_detection"]
    write_json_atomic(tmp_path / "session.json", {"resolved_config": resolved})
    assert _config_for_resume(tmp_path).input.key_detection == "none"
