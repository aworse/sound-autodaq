"""trial.capture = keypress: wait for the keystroke, then cut pre_roll_ms
before to post_roll_ms after it out of the continuous ring buffer."""

import threading
import time

import numpy as np
import pytest

from experiments.recording import clock
from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.errors import ConfigError
from experiments.recording.recorder import RingBuffer, SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.tests.test_keylog import press
from experiments.recording.tests.test_keylog import Participant as _Typist
from experiments.recording.ui import Display, QueueControlSource
from experiments.recording.validator import validate_session
from experiments.recording.writer import read_manifest_jsonl, read_wav

SR = 48000
PRE_MS, POST_MS = 100, 100


def _config(tmp_path, **overrides):
    trial = {"capture": "keypress", "repetitions_per_class": 1, "countdown_ms": 0, "pre_roll_ms": PRE_MS,
             "input_window_ms": 3000, "post_roll_ms": POST_MS, "inter_trial_ms": 0}
    trial.update(overrides.pop("trial", {}))
    overrides["trial"] = trial
    overrides.setdefault("output", {})["root"] = str(tmp_path / "data")
    overrides.setdefault("input", {"key_detection": "hook"})
    return config_from_dict(base_config_dict(**overrides))


class ClickBackend(SyntheticBackend):
    """Synthetic stream that also contains a loud click at each moment a
    key is 'pressed', so a test can see where the keystroke landed in the
    saved audio."""

    def __init__(self):
        super().__init__()
        self.clicks: list = []
        self._lock_clicks = threading.Lock()

    def click(self, t_ns):
        with self._lock_clicks:
            self.clicks.append(t_ns)

    def start(self, device_info, sample_rate, channels, callback):
        def with_clicks(block, overflow):
            now = clock.now_ns()
            with self._lock_clicks:
                due = [c for c in self.clicks if c <= now]
                self.clicks = [c for c in self.clicks if c > now]
            if due:
                block = block.copy()
                block[:8] = 20000
            callback(block, overflow)

        super().start(device_info, sample_rate, channels, with_clicks)


class Participant(Display):
    """Presses the target `delay_s(n)` seconds after PRESS appears on trial n
    (None = never). `extra(n, source)` can inject operator input."""

    def __init__(self, source, backend, delay_s, extra=None, early=None):
        super().__init__(enabled=True, interactive=False)
        self.source, self.backend, self.delay_s, self.extra, self.early = source, backend, delay_s, extra, early
        self.n = 0
        self.timers = []
        self.pressed_at: list = []

    def _press(self, label, rep=1):
        """Type the target (a chord for tense consonants / ㅒㅖ); the click
        goes into the audio at the main key-down."""
        t = clock.now_ns()
        if self.backend is not None:
            self.backend.click(t)
        self.pressed_at.append(t)
        press(self.source, label, rep, t_ns=t)

    def show(self, text):
        if text.startswith("=") and ">>> READY" in text and self.early and self.early(self.n + 1):
            self._press(text.split("target:")[1].split()[0])
        if "PRESS" not in text or "too early" in text:
            return
        self.n += 1
        target = text.split("PRESS")[1].split()[0]
        rep = _Typist.rep_of(text)
        if self.extra:
            self.extra(self.n, self.source)
        delay = self.delay_s(self.n)
        if delay is not None:
            timer = threading.Timer(delay, self._press, args=(target, rep))
            self.timers.append(timer)
            timer.start()

    def line(self, text):
        pass


def _run(config, delay_s, backend=None, extra=None, early=None, resume=False):
    backend = backend or ClickBackend()
    engine = SessionEngine(config, backend=backend, mic_test_duration_s=0.2)
    source = QueueControlSource()
    participant = Participant(source, backend, delay_s, extra, early)
    try:
        summary = engine.run(resume=resume, control_source=source, display=participant)
    finally:
        for timer in participant.timers:
            timer.cancel()
    _run.participant = participant
    return engine, summary, read_manifest_jsonl(engine.session_dir / "manifest.jsonl")


def test_recording_is_cut_around_the_keystroke_whenever_it_comes(tmp_path):
    delays = [0.02, 0.25, 0.8, 1.5]

    def quit_after_8(n, source):
        if n == 8:
            source.push("quit")

    engine, summary, rows = _run(_config(tmp_path), lambda n: delays[(n - 1) % len(delays)], extra=quit_after_8)
    assert summary.attempted_trials == 8 and summary.stop_reason == "operator quit"

    pressed_at = _run.participant.pressed_at
    for n, r in enumerate(rows, start=1):
        assert r["status"] == "valid", r
        assert r["observed_label"] == r["label"] and r["keystrokes"] == 1
        # it waited for the participant, however long they took: the
        # recorded keystroke time is exactly when the key was pressed
        assert r["input_detected_ns"] == pressed_at[n - 1]
        assert r["input_detected_ns"] > r["input_expected_ns"]
        # fixed length: pre_roll + post_roll
        assert abs(r["num_samples"] - (PRE_MS + POST_MS) * SR // 1000) <= 1
        # the click (the keystroke) sits pre_roll_ms into the file. The
        # tolerance covers audio-block delivery jitter on a loaded CI
        # machine; a wrong cut would be off by a whole pre-roll (100 ms).
        samples, _, _ = read_wav(engine.session_dir / r["file"])
        click_at = int(np.argmax(np.abs(samples.astype(np.int32))))
        assert abs(click_at - PRE_MS * SR // 1000) <= 20 * SR // 1000, (n, click_at)
        # and the file is exactly the stream samples the row claims
        idx = np.arange(r["segment_start_sample"], r["segment_end_sample"])
        expected = ((idx % 2000) - 1000).astype(np.int16)
        mask = np.abs(samples) < 20000
        assert np.array_equal(samples[mask], expected[mask])


def test_full_keypress_session_completes_and_validates(tmp_path):
    engine, summary, rows = _run(_config(tmp_path), lambda n: 0.03)
    assert summary.completed is True, summary
    assert all(r["status"] == "valid" for r in rows)
    assert validate_session(engine.session_dir).passed


def test_no_keystroke_times_out_and_stops_after_three(tmp_path):
    config = _config(tmp_path, trial={"input_window_ms": 200})
    engine, summary, rows = _run(config, lambda n: None)
    assert summary.completed is False
    assert [r["status"] for r in rows] == ["invalid"] * 3
    assert "no key within" in rows[0]["notes"]


def test_operator_quit_while_waiting_interrupts_and_resume_finishes(tmp_path):
    config = _config(tmp_path)

    def operator_quits_on_trial_2(n, source):
        if n == 2:
            source.push("quit")
            source.push_key("0")

    engine, summary, rows = _run(config, lambda n: 0.03 if n == 1 else None, extra=operator_quits_on_trial_2)
    assert summary.stop_reason == "operator quit"
    assert rows[1]["status"] == "interrupted" and rows[1]["requeue"] == "immediate"
    assert "ended by an operator key" in rows[1]["notes"]

    engine2, summary2, rows2 = _run(config, lambda n: 0.03, resume=True)
    assert summary2.completed is True
    assert rows2[2]["trial_id"] == rows[1]["superseded_by"]


def test_key_pressed_before_press_appears_does_not_trigger(tmp_path):
    config = _config(tmp_path, trial={"countdown_ms": 400})
    engine, summary, rows = _run(
        config, lambda n: 0.05,
        extra=lambda n, source: source.push("quit") if n == 1 else None,
        early=lambda n: n == 1,
    )
    first = rows[0]
    # the trigger is the key after PRESS, not the early one during the countdown
    assert first["input_detected_ns"] > first["input_expected_ns"]
    assert first["status"] == "valid"
    assert "outside the recording" in first["notes"]


@pytest.mark.parametrize("change,match", [
    ({"input": {"key_detection": "none"}}, "key_detection: hook"),
    ({"trial": {"pre_roll_ms": 500, "post_roll_ms": 100, "countdown_ms": 0, "inter_trial_ms": 0}}, "pre_roll_ms"),
    ({"trial": {"post_roll_ms": 0}}, "post_roll_ms"),
])
def test_keypress_config_rules(tmp_path, change, match):
    with pytest.raises(ConfigError, match=match):
        _config(tmp_path, **change)


def test_sample_at_maps_monotonic_time_to_sample_index():
    buf = RingBuffer(capacity_frames=48000, channels=1, sample_rate=48000)
    block = np.zeros((480, 1), dtype=np.int16)
    written = []  # (time just after each block arrived, samples captured by then)
    for _ in range(10):
        buf.write(block, False)
        written.append((clock.now_ns(), buf.total_written))
        time.sleep(0.005)
    for t, count in written:
        # at the moment a block has arrived, the index is that block's end
        # plus however little time passed since (well under 1 ms here)
        assert 0 <= buf.sample_at(t) - count < 48, (buf.sample_at(t), count)
    # between and beyond blocks it advances at the nominal rate
    t_last, count_last = written[-1]
    assert abs(buf.sample_at(t_last + 100_000_000) - buf.sample_at(t_last) - 4800) <= 1


def test_reference_config_waits_without_limit():
    from pathlib import Path

    from experiments.recording.config import load_config

    cfg = load_config(Path(__file__).parents[3] / "configs" / "S01.yaml")
    assert cfg.trial.capture == "keypress" and cfg.trial.input_window_ms == 0


def test_unlimited_wait_accepts_a_slow_keystroke(tmp_path):
    config = _config(tmp_path, trial={"input_window_ms": 0})
    engine, summary, rows = _run(
        config, lambda n: 1.5, extra=lambda n, source: source.push("quit") if n == 2 else None
    )
    assert [r["status"] for r in rows] == ["valid", "valid"]
    for r in rows:
        assert (r["input_detected_ns"] - r["input_expected_ns"]) / 1e9 >= 1.4


def test_unlimited_wait_does_not_hide_a_dead_microphone(tmp_path):
    from experiments.recording.errors import AudioStreamError
    from experiments.recording.tests.test_engine_integration import _StallingBackend

    config = _config(tmp_path, trial={"input_window_ms": 0})
    engine = SessionEngine(config, backend=_StallingBackend(after=int(0.3 * SR)), mic_test_duration_s=0.1)
    outcome = {}

    def run():
        try:
            engine.run(control_source=QueueControlSource())  # nobody ever presses a key
        except AudioStreamError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "the session hung waiting for a key while the microphone was dead"
    assert "stalled" in str(outcome.get("error"))
    rows = read_manifest_jsonl(engine.session_dir / "manifest.jsonl")
    assert rows and rows[-1]["status"] == "interrupted" and rows[-1]["requeue"] == "immediate"


def test_hangul_ime_in_keypress_mode_stops_with_the_hint(tmp_path, monkeypatch):
    """Terminal detection only (a hook never sees IME output): a Hangul IME
    sends jamo characters, which cannot be verified."""
    from classism.labels import JAMO
    from experiments.recording import labels as labels_module

    monkeypatch.setattr(labels_module, "load_classes", lambda: (JAMO, "jamo-only-test"))
    config = _config(tmp_path, trial={"input_window_ms": 0}, input={"key_detection": "terminal"})
    engine = SessionEngine(config, backend=ClickBackend(), mic_test_duration_s=0.2)
    source = QueueControlSource()

    class HangulTyper(Participant):
        def show(self, text):
            if "PRESS" in text and "too early" not in text:
                source.push_key(text.split("PRESS")[1].split()[0])  # the jamo itself, as a Hangul IME sends it

    summary = engine.run(control_source=source, display=HangulTyper(source, None, lambda n: None))
    rows = read_manifest_jsonl(engine.session_dir / "manifest.jsonl")
    assert [r["status"] for r in rows] == ["invalid"] * 3
    assert "English" in summary.stop_reason


def test_key_pressed_the_instant_press_appears_counts(tmp_path):
    """The PRESS timestamp is taken before the screen is drawn, so a key
    that arrives while it is being drawn is on time, not 'too early'."""
    config = _config(tmp_path, trial={"input_window_ms": 2000})
    engine = SessionEngine(config, backend=ClickBackend(), mic_test_duration_s=0.2)
    source = QueueControlSource()

    class Instant(Participant):
        def show(self, text):
            if "PRESS" in text and "too early" not in text:
                self.n += 1
                press(source, text.split("PRESS")[1].split()[0], _Typist.rep_of(text))
                if self.n == 3:
                    source.push("quit")

    engine.run(control_source=source, display=Instant(source, None, lambda n: None))
    rows = read_manifest_jsonl(engine.session_dir / "manifest.jsonl")
    assert [r["status"] for r in rows] == ["valid"] * 3
    assert all(r["input_detected_ns"] >= r["input_expected_ns"] for r in rows)
