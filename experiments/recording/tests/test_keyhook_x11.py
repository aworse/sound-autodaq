"""The real keyboard hook (pynput) end to end, driven by real X11 key
events from xdotool. Runs only with an X display (CI runs the suite under
xvfb-run); skipped elsewhere."""

import os
import shutil
import subprocess
import time

import pytest

from experiments.recording import keylog

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY") or not shutil.which("xdotool"),
    reason="needs an X display and xdotool (run under xvfb-run)",
)

# xdotool keysym names for the physical keys the recorder uses
XDO = {
    "space": "space", "backspace": "BackSpace", "shift": "Shift_L", "caps_lock": "Caps_Lock",
    "enter": "Return", "tab": "Tab", ",": "comma", ".": "period", "/": "slash", ";": "semicolon",
    "'": "apostrophe", "[": "bracketleft", "]": "bracketright", "-": "minus", "=": "equal",
}


def xdo_type(label, repetition=1):
    key = keylog.prompted_key(label, repetition)
    name = XDO.get(key, key)
    chord = label in keylog.SHIFTED_JAMO
    subprocess.run(["xdotool", "key", f"shift+{name}" if chord else name], check=True)


@pytest.fixture
def hook():
    from experiments.recording.keyhook import HookControlSource

    source = HookControlSource()
    time.sleep(0.2)
    yield source
    source.close()
    # leave Caps Lock off for whatever runs next on this display
    if _caps_on():
        subprocess.run(["xdotool", "key", "Caps_Lock"], check=False)


def _caps_on():
    from Xlib import display  # installed with pynput on Linux

    return bool(display.Display().get_keyboard_control().led_mask & 1)


def _drain(source, settle=0.3):
    time.sleep(settle)
    return source.poll_keys()


def test_hook_reports_physical_keys_with_shift_from_the_shift_key(hook):
    subprocess.run(["xdotool", "key", "r", "shift+r", "Caps_Lock", "r", "Caps_Lock", "space", "BackSpace",
                    "Return", "Shift_L"], check=True)
    events = _drain(hook)
    got = [(e.key, e.shift) for e in events]
    assert got == [("r", False), ("shift", False), ("r", True), ("caps_lock", False), ("r", False),
                   ("caps_lock", False), ("space", False), ("backspace", False), ("enter", False),
                   ("shift", False)], got
    assert [keylog.symbol_of(e) for e in events[:5]] == ["ㄱ", "<shift>", "ㄲ", "<caps>", "ㄱ"]
    assert all(a.t_ns <= b.t_ns for a, b in zip(events, events[1:]))


def test_hook_digits_are_controls_even_with_shift(hook):
    subprocess.run(["xdotool", "key", "4", "shift+1"], check=True)
    _drain(hook)
    assert [hook.poll(), hook.poll(), hook.poll()] == ["pause", "repeat", None]


def test_full_session_of_all_38_classes_through_the_real_hook(tmp_path, hook):
    """Every class typed as real X11 key events, recorded with keypress
    capture, verified by the hook — including the trials after the <caps>
    trial has turned Caps Lock on."""
    assert not _caps_on()
    from classism.labels import CLASSES
    from experiments.recording.config import config_from_dict
    from experiments.recording.engine import SessionEngine
    from experiments.recording.recorder import SyntheticBackend
    from experiments.recording.tests.helpers import base_config_dict
    from experiments.recording.tests.test_keylog import Participant
    from experiments.recording.ui import Display
    from experiments.recording.writer import read_manifest_jsonl

    config = config_from_dict(base_config_dict(
        output={"root": str(tmp_path / "data")},
        input={"key_detection": "hook"},
        trial={"capture": "keypress", "repetitions_per_class": 1, "countdown_ms": 0, "pre_roll_ms": 60,
               "input_window_ms": 5000, "post_roll_ms": 60, "inter_trial_ms": 0},
    ))

    class XTypist(Display):
        def __init__(self):
            super().__init__(enabled=True, interactive=False)

        def show(self, text):
            if "PRESS" in text and "too early" not in text:
                xdo_type(text.split("PRESS")[1].split()[0], Participant.rep_of(text))

        def line(self, text):
            pass

    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary = engine.run(control_source=hook, display=XTypist())
    rows = read_manifest_jsonl(engine.session_dir / "manifest.jsonl")
    bad = [(r["label"], r["status"], r["notes"]) for r in rows if r["status"] != "valid"]
    assert summary.completed is True, (summary.stop_reason, bad)
    assert sorted(r["label"] for r in rows) == sorted(CLASSES)
    assert all(r["observed_label"] == r["label"] for r in rows)
