"""Keystroke verification (Appendix E.1/E.2): the judging rules, the
keyboard-hook and terminal key normalization, then the full engine with
key events injected the way a participant would type them."""

import json
import sys

import pytest

from classism.labels import CLASSES, JAMO
from experiments.recording import clock, keylog
from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.errors import PreflightError
from experiments.recording.keyhook import HookState, normalize_key
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


def TYPE_FOR(label, repetition=1):
    """The key-downs a person makes for a target: [(key, shift), ...].
    A tense consonant or ㅒ/ㅖ is a chord: Shift down, then the letter."""
    if label in keylog.KEY_FOR_JAMO:
        key, shift = keylog.KEY_FOR_JAMO[label]
        return [("shift", False), (key, True)] if shift else [(key, False)]
    return [(keylog.prompted_key(label, repetition), False)]


def press(source, label, repetition=1, t_ns=None, gap_ns=20 * MS):
    """Type `label` into a QueueControlSource like a hook would report it;
    returns the time of the main (last) key-down."""
    t = clock.now_ns() if t_ns is None else t_ns
    seq = TYPE_FOR(label, repetition)
    start = t - gap_ns * (len(seq) - 1)
    for i, (key, shift) in enumerate(seq):
        source.push_key(key, start + i * gap_ns, shift=shift)
    return t


def ev(seq, t0=150 * MS, gap=20 * MS):
    return [KeyEvent(k, t0 + i * gap, s) for i, (k, s) in enumerate(seq)]


# -- mapping ------------------------------------------------------------------


def test_every_class_can_be_typed_and_maps_back_to_itself():
    for label in CLASSES:
        seq = TYPE_FOR(label)
        main = KeyEvent(seq[-1][0], 0, seq[-1][1])
        assert keylog.symbol_of(main, KEYMAP) == label, label


def test_shift_comes_from_the_shift_key_not_from_letter_case():
    assert keylog.symbol_of(KeyEvent("r", 0, shift=True)) == "ㄲ"
    assert keylog.symbol_of(KeyEvent("r", 0, shift=False)) == "ㄱ"
    assert keylog.symbol_of(KeyEvent("k", 0, shift=True)) == "ㅏ"  # Shift does not change ㅏ
    assert keylog.symbol_of(KeyEvent("enter", 0)) == "<other>"
    assert keylog.symbol_of(KeyEvent("1", 0), KEYMAP) is None  # operator digit


def test_other_prompts_cycle_through_non_class_keys():
    keys = [keylog.prompted_key("<other>", rep) for rep in range(1, len(keylog.OTHER_KEYS) + 2)]
    assert keys[: len(keylog.OTHER_KEYS)] == list(keylog.OTHER_KEYS) and keys[-1] == keys[0]
    for key in keylog.OTHER_KEYS:
        assert keylog.symbol_of(KeyEvent(key, 0), KEYMAP) == "<other>"
    assert keylog.prompt_hint("<other>", 1) == "(press Enter)"
    assert keylog.prompt_hint("ㄲ", 1) == "(Shift + ㄱ)"


def test_terminal_characters_normalize_to_key_names():
    assert keylog.normalize_char("R") == ("r", True)
    assert keylog.normalize_char(" ") == ("space", False)
    assert keylog.normalize_char("\x7f") == ("backspace", False)
    assert keylog.normalize_char("!") == ("1", True)
    assert keylog.normalize_char("ㄱ") == ("ㄱ", False)


# -- judging ------------------------------------------------------------------


def test_correct_key_in_window_has_no_objection():
    v = judge(ev([("r", False)]), "ㄱ", **SEG)
    assert v.status is None
    assert (v.observed_label, v.observed_key, v.input_detected_ns, v.keystrokes) == ("ㄱ", "r", 150 * MS, 1)


def test_tense_consonant_chord_is_one_keystroke():
    v = judge(ev([("shift", False), ("r", True)]), "ㄲ", **SEG)
    assert v.status is None and v.observed_label == "ㄲ" and v.keystrokes == 1
    assert v.input_detected_ns == 170 * MS  # the R, not the Shift


def test_tense_target_without_shift_is_mismatch_with_hint():
    v = judge(ev([("r", False)]), "ㄲ", **SEG)
    assert v.status == Status.MISMATCH and "hold Shift" in v.note


def test_plain_target_typed_with_shift_is_an_extra_keystroke():
    v = judge(ev([("shift", False), ("k", True)]), "ㅏ", **SEG)
    assert v.status == Status.INVALID and v.keystrokes == 2


def test_plain_consonant_typed_with_shift_is_mismatch():
    v = judge(ev([("shift", False), ("r", True)]), "ㄱ", **SEG)
    assert v.status == Status.MISMATCH and "without Shift" in v.note


@pytest.mark.parametrize("label,key", [("<sp>", "space"), ("<bs>", "backspace"), ("<caps>", "caps_lock"),
                                       ("<shift>", "shift")])
def test_special_keys(label, key):
    v = judge(ev([(key, False)]), label, **SEG)
    assert v.status is None and v.observed_label == label


def test_shift_target_then_a_letter_is_invalid():
    v = judge(ev([("shift", False), ("r", True)]), "<shift>", **SEG)
    assert v.status == Status.INVALID and v.keystrokes == 2


def test_other_accepts_any_other_key_and_notes_the_difference():
    v = judge(ev([("tab", False)]), "<other>", **SEG, prompted="enter")
    assert v.status is None and v.observed_label == "<other>" and "prompted Enter, pressed Tab" in v.note


def test_wrong_key_is_mismatch():
    v = judge(ev([("s", False)]), "ㄱ", **SEG)
    assert v.status == Status.MISMATCH and v.observed_label == "ㄴ"


def test_no_key_is_invalid():
    v = judge([], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "no keystroke" in v.note


def test_key_outside_recorded_audio_is_invalid():
    v = judge([KeyEvent("r", -200 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "outside the recorded audio" in v.note
    assert v.observed_label == "ㄱ" and v.keystrokes == 0


def test_key_in_pre_roll_is_kept_with_timestamp():
    v = judge([KeyEvent("r", 50 * MS)], "ㄱ", **SEG)
    assert v.status is None and v.input_detected_ns == 50 * MS


def test_two_keys_in_one_recording_is_invalid():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("r", 250 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and v.keystrokes == 2


def test_hangul_from_a_terminal_ime_is_invalid_with_hint():
    v = judge([KeyEvent("ㄱ", 150 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "English" in v.note


def test_operator_digit_during_recording_is_invalid():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("0", 250 * MS)], "ㄱ", **SEG)
    assert v.status == Status.INVALID and "operator key" in v.note


def test_operator_digit_after_recording_is_fine():
    v = judge([KeyEvent("r", 150 * MS), KeyEvent("0", 400 * MS)], "ㄱ", **SEG)
    assert v.status is None


# -- keyboard hook normalization (no pynput needed) ----------------------------


class _Key:
    def __init__(self, char=None, name=None, vk=None):
        self.char, self.name, self.vk = char, name, vk


def _hook():
    events = []
    return HookState(events.append), events


def test_hook_tracks_shift_and_ignores_letter_case():
    state, events = _hook()
    state.press(None, "shift_l", None)
    state.press("R", None, 82)  # Shift held
    state.release("R", None, 82)
    state.release(None, "shift_l", None)
    state.press("R", None, 82)  # Caps Lock on, no Shift: still plain ㄱ
    assert [(e.key, e.shift) for e in events] == [("shift", False), ("r", True), ("r", False)]
    assert keylog.symbol_of(events[1]) == "ㄲ" and keylog.symbol_of(events[2]) == "ㄱ"


def test_hook_drops_auto_repeat():
    state, events = _hook()
    for _ in range(5):
        state.press("r", None, 114)  # held key: the OS repeats the key-down
    state.release("r", None, 114)
    state.press("r", None, 114)
    assert [e.key for e in events] == ["r", "r"]


def test_hook_counts_a_key_whose_release_never_comes_again_after_a_pause():
    """Windows reports no release for 한/영 (VK_HANGUL, 0x15): a second
    tap seconds later is a new keystroke, not auto-repeat."""
    state, events = _hook()
    state.press(None, None, 0x15, t_ns=0)
    state.press(None, None, 0x15, t_ns=3_000 * MS)
    state.press("r", None, 114, t_ns=4_000 * MS)
    for k in range(1, 40):  # held: first repeat after 500 ms, then every 33 ms
        state.press("r", None, 114, t_ns=4_000 * MS + 500 * MS + k * 33 * MS)
    assert [(e.key, e.t_ns) for e in events] == [("vk21", 0), ("vk21", 3_000 * MS), ("r", 4_000 * MS)]
    assert keylog.symbol_of(events[0]) == "<other>"


def test_hook_normalizes_names_symbols_and_hangul_layouts():
    assert normalize_key(None, "shift_r", None) == ("shift", False)
    assert normalize_key(None, "caps_lock", None) == ("caps_lock", False)
    assert normalize_key("!", None, 49) == ("1", False)  # Shift+1 is still the 1 key
    assert normalize_key("ㄲ", None, None) == ("r", True)
    assert normalize_key(None, None, 0x52) == ("r", False)  # Windows VK_R


def test_hook_digit_with_shift_is_still_a_control():
    state, events = _hook()
    state.press(None, "shift", None)
    state.press("!", None, 49)
    assert events[-1].key == "1" and KEYMAP[events[-1].key] == "repeat"


# -- engine ----------------------------------------------------------------


def _config(tmp_path, **overrides):
    trial = {"repetitions_per_class": 1, "countdown_ms": 0, "pre_roll_ms": 30, "input_window_ms": 30,
             "post_roll_ms": 30, "inter_trial_ms": 0}
    trial.update(overrides.pop("trial", {}))
    overrides["trial"] = trial
    overrides.setdefault("output", {})["root"] = str(tmp_path / "data")
    overrides.setdefault("input", {}).setdefault("key_detection", "hook")
    return config_from_dict(base_config_dict(**overrides))


class Participant(Display):
    """Types when the screen says PRESS. `answer(n, target, repetition)`
    returns the label to type (or None for nothing)."""

    def __init__(self, source, answer):
        super().__init__(enabled=True, interactive=False)
        self.source, self.answer = source, answer
        self.n = 0

    def show(self, text):
        if "PRESS" in text:
            self.n += 1
            target = text.split("PRESS")[1].split()[0]
            rep = self.rep_of(text)
            label = self.answer(self.n, target, rep)
            if label == "RAW-HANGUL":
                self.source.push_key(target)
            elif label is not None:
                press(self.source, label, rep)

    @staticmethod
    def rep_of(text):
        # <other> prompts are shown in the hint; recover the repetition from it
        if "(press " in text:
            shown = text.split("(press ")[1].split(")")[0]
            for rep in range(1, len(keylog.OTHER_KEYS) + 1):
                if keylog.display_name(keylog.OTHER_KEYS[rep - 1]) == shown:
                    return rep
        return 1

    def line(self, text):
        pass


def _run(config, answer):
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    source = QueueControlSource()
    summary = engine.run(control_source=source, display=Participant(source, answer))
    return engine, summary, read_manifest_jsonl(engine.session_dir / "manifest.jsonl")


def test_all_38_classes_are_typed_verified_and_recorded(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n, target, rep: target)
    assert summary.completed is True, summary.stop_reason
    assert sorted(r["label"] for r in rows) == sorted(CLASSES)
    for r in rows:
        assert r["status"] == "valid", r
        assert r["observed_label"] == r["label"] and r["keystrokes"] == 1
        assert r["trial_start_ns"] <= r["input_detected_ns"] <= r["trial_end_ns"]
    session = json.loads((engine.session_dir / "session.json").read_text(encoding="utf-8"))
    assert session["implementation_decisions"]["keystroke_detection"].startswith("hook")
    assert "<other>" in session["implementation_decisions"]["other_class"]
    assert validate_session(engine.session_dir).passed


def test_wrong_key_is_mismatch_and_re_recorded(tmp_path):
    def answer(n, target, rep):
        return ("ㅎ" if target != "ㅎ" else "ㅁ") if n == 1 else target

    engine, summary, rows = _run(_config(tmp_path), answer)
    first = rows[0]
    assert first["status"] == "mismatch" and first["observed_label"] != first["label"]
    retry = next(r for r in rows if r["trial_id"] == first["superseded_by"])
    assert retry["status"] == "valid" and retry["observed_label"] == first["label"]
    assert summary.completed is True


def test_no_typing_stops_with_the_reason(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n, target, rep: None)
    assert summary.completed is False
    assert "no keystroke detected" in summary.stop_reason
    assert [r["status"] for r in rows] == ["invalid"] * 3


def test_terminal_detection_is_refused_when_classes_include_shift_or_caps(tmp_path):
    engine = SessionEngine(_config(tmp_path, input={"key_detection": "terminal"}), backend=SyntheticBackend(),
                           mic_test_duration_s=0.2)
    with pytest.raises(PreflightError, match="key_detection: hook"):
        engine.run(control_source=QueueControlSource())


def test_terminal_detection_still_works_for_a_jamo_only_class_list(tmp_path, monkeypatch):
    from experiments.recording import labels as labels_module

    monkeypatch.setattr(labels_module, "load_classes", lambda: (JAMO, "jamo-only-test"))
    config = _config(tmp_path, input={"key_detection": "terminal"})
    engine, summary, rows = _run(config, lambda n, target, rep: target)
    assert summary.completed is True and len(rows) == len(JAMO)


def test_hangul_ime_in_terminal_mode_stops_with_the_hint(tmp_path, monkeypatch):
    from experiments.recording import labels as labels_module

    monkeypatch.setattr(labels_module, "load_classes", lambda: (JAMO, "jamo-only-test"))
    config = _config(tmp_path, input={"key_detection": "terminal"})
    engine, summary, rows = _run(config, lambda n, target, rep: "RAW-HANGUL")
    assert "English" in summary.stop_reason
    assert all(r["status"] == "invalid" for r in rows)


def test_resume_refuses_a_key_detection_change(tmp_path):
    from experiments.recording.tests.test_engine_integration import ControlsAtTrial

    config = _config(tmp_path, input={"key_detection": "none"})
    SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2).run(
        control_source=ControlsAtTrial({2: ["quit"]}))
    changed = _config(tmp_path, input={"key_detection": "hook"})
    with pytest.raises(PreflightError, match="key_detection"):
        SessionEngine(changed, backend=SyntheticBackend(), mic_test_duration_s=0.2).run(
            resume=True, control_source=QueueControlSource())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal")
def test_terminal_reader_timestamps_and_normalizes_keys(monkeypatch):
    """The real TerminalControlSource, driven through a pseudo-terminal."""
    import os
    import pty
    import sys
    import time

    from experiments.recording.ui import TerminalControlSource

    master, slave = pty.openpty()
    fake_stdin = os.fdopen(slave, "r", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    source = TerminalControlSource()
    try:
        assert source.interactive
        before = clock.now_ns()
        os.write(master, "R".encode())
        time.sleep(0.3)
        os.write(master, "1".encode())
        time.sleep(0.3)
        keys = source.poll_keys()
        assert [(k.key, k.shift) for k in keys] == [("r", True), ("1", False)]
        assert before <= keys[0].t_ns < keys[1].t_ns <= clock.now_ns()
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
