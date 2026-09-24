"""
Trial phase state machine and monotonic timing (Part V of the spec).

Owns: phase state machine, monotonic timing. Never: direct file writes
(REQ-57).
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import time
from typing import Optional

from . import clock


class Phase(str, enum.Enum):
    SCHEDULED = "scheduled"
    PREPARE = "prepare"
    COUNTDOWN = "countdown"
    PRE_ROLL = "pre_roll"
    INPUT_WINDOW = "input_window"
    POST_ROLL = "post_roll"
    SAVE = "save"
    INTER_TRIAL = "inter_trial"
    DONE = "done"


class Status(str, enum.Enum):
    """Appendix A status enumeration."""

    VALID = "valid"
    INVALID = "invalid"
    OPERATOR_MARKED_INVALID = "operator_marked_invalid"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"
    CORRUPTED = "corrupted"
    AUDIO_OVERFLOW = "audio_overflow"
    SUSPICIOUS_SILENCE = "suspicious_silence"


class InputMode(str, enum.Enum):
    HUMAN = "human"
    AUTOMATED = "automated"


@dataclasses.dataclass
class TrialDurations:
    countdown_ms: int
    pre_roll_ms: int
    input_window_ms: int
    post_roll_ms: int
    inter_trial_ms: int

    @property
    def stored_ms(self) -> int:
        return self.pre_roll_ms + self.input_window_ms + self.post_roll_ms


@dataclasses.dataclass
class TrialTiming:
    session_start_ns: int
    trial_start_ns: Optional[int] = None
    input_expected_ns: Optional[int] = None
    input_detected_ns: Optional[int] = None
    trial_end_ns: Optional[int] = None
    wall_clock_utc: Optional[str] = None


@dataclasses.dataclass
class TrialResult:
    """The outcome of running one trial through the state machine, before
    segment extraction / writing (which the engine + writer own)."""

    trial_id: int
    scheduled_label: str
    repetition: int
    input_mode: InputMode
    timing: TrialTiming
    status: Status = Status.VALID
    observed_label: Optional[str] = None
    notes: Optional[str] = None
    segment_start_sample: Optional[int] = None
    segment_end_sample: Optional[int] = None
    # capture=keypress: the instant the recording is centred on (the
    # keystroke, or the end of the wait if no key came).
    anchor_ns: Optional[int] = None


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class TrialClock:
    """Injectable clock so tests can run the state machine without real
    sleeps."""

    def monotonic_ns(self) -> int:
        return clock.now_ns()

    def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


# Assumed reaction time for duration estimates only (never for data).
ASSUMED_REACTION_MS = 700


def stored_ms(trial_cfg) -> int:
    """Audio kept per trial."""
    if trial_cfg.capture == "keypress":
        return trial_cfg.pre_roll_ms + trial_cfg.post_roll_ms
    return trial_cfg.pre_roll_ms + trial_cfg.input_window_ms + trial_cfg.post_roll_ms


def estimated_trial_ms(trial_cfg) -> int:
    """Wall time per trial, for pre-session estimates before any trial has
    been measured (keypress capture assumes a reaction time)."""
    if trial_cfg.capture == "keypress":
        return trial_cfg.countdown_ms + ASSUMED_REACTION_MS + trial_cfg.post_roll_ms + trial_cfg.inter_trial_ms
    return trial_cfg.countdown_ms + stored_ms(trial_cfg) + trial_cfg.inter_trial_ms


class TrialStateMachine:
    """Drives one trial through PREPARE -> COUNTDOWN -> PRE_ROLL ->
    INPUT_WINDOW -> POST_ROLL -> SAVE -> INTER_TRIAL, emitting phase
    change callbacks and monotonic timestamps at the boundaries the spec
    requires (REQ-23.2).
    """

    def __init__(self, durations: TrialDurations, session_start_ns: int, clock: Optional[TrialClock] = None):
        self.durations = durations
        self.session_start_ns = session_start_ns
        self.clock = clock or TrialClock()
        self.phase = Phase.SCHEDULED

    def _set_phase(self, phase: Phase, on_phase_change=None) -> None:
        self.phase = phase
        if on_phase_change:
            on_phase_change(phase)

    def run(
        self,
        trial_id: int,
        label: str,
        repetition: int,
        input_mode: InputMode,
        on_phase_change=None,
        on_input_window=None,
        on_countdown=None,
    ) -> TrialResult:
        """Run the full trial timeline.

        `on_input_window` is called with no arguments once the INPUT
        WINDOW phase begins; in HUMAN mode this is when the operator
        should be permitted to type. Returns a TrialResult with all
        REQ-23.2 timestamps populated (`input_detected_ns` remains None
        unless a caller sets it via the returned object post hoc, e.g.
        from a key-log hook — see Open Point E.1).
        """
        timing = TrialTiming(session_start_ns=self.session_start_ns)

        # Stamp first: the PREPARE callback draws the screen, and anything
        # that happens from then on belongs to this trial (REQ-23.2).
        timing.trial_start_ns = self.clock.monotonic_ns()
        timing.wall_clock_utc = utc_now_iso()
        self._set_phase(Phase.PREPARE, on_phase_change)

        self._set_phase(Phase.COUNTDOWN, on_phase_change)
        remaining = self.durations.countdown_ms
        while remaining > 0:
            if on_countdown:
                on_countdown(-(-remaining // 1000))  # whole seconds left, rounded up
            step = min(1000, remaining)
            self.clock.sleep_ms(step)
            remaining -= step

        self._set_phase(Phase.PRE_ROLL, on_phase_change)
        self.clock.sleep_ms(self.durations.pre_roll_ms)

        # Stamp before the callback that draws PRESS: a key pressed the
        # moment it appears must count as on time, not early.
        timing.input_expected_ns = self.clock.monotonic_ns()
        self._set_phase(Phase.INPUT_WINDOW, on_phase_change)
        if on_input_window:
            on_input_window()
        self.clock.sleep_ms(self.durations.input_window_ms)

        self._set_phase(Phase.POST_ROLL, on_phase_change)
        self.clock.sleep_ms(self.durations.post_roll_ms)

        timing.trial_end_ns = self.clock.monotonic_ns()
        self._set_phase(Phase.SAVE, on_phase_change)

        return TrialResult(
            trial_id=trial_id,
            scheduled_label=label,
            repetition=repetition,
            input_mode=input_mode,
            timing=timing,
        )

    def run_keypress(
        self,
        trial_id: int,
        label: str,
        repetition: int,
        input_mode: InputMode,
        wait_for_key,
        on_phase_change=None,
        on_countdown=None,
    ) -> TrialResult:
        """capture=keypress: PREPARE -> COUNTDOWN -> INPUT_WINDOW (open until
        `wait_for_key(press_shown_ns)` returns the keystroke's monotonic time,
        or None) -> POST_ROLL (post_roll_ms after that instant) -> SAVE.

        There is no timed PRE_ROLL phase: the audio before the keystroke is
        already in the continuous ring buffer and is cut out afterwards.
        """
        timing = TrialTiming(session_start_ns=self.session_start_ns)

        # Stamp first: the PREPARE callback draws the screen, and anything
        # that happens from then on belongs to this trial (REQ-23.2).
        timing.trial_start_ns = self.clock.monotonic_ns()
        timing.wall_clock_utc = utc_now_iso()
        self._set_phase(Phase.PREPARE, on_phase_change)

        self._set_phase(Phase.COUNTDOWN, on_phase_change)
        remaining = self.durations.countdown_ms
        while remaining > 0:
            if on_countdown:
                on_countdown(-(-remaining // 1000))
            step = min(1000, remaining)
            self.clock.sleep_ms(step)
            remaining -= step

        timing.input_expected_ns = self.clock.monotonic_ns()
        self._set_phase(Phase.INPUT_WINDOW, on_phase_change)
        key_ns = wait_for_key(timing.input_expected_ns)
        anchor = key_ns if key_ns is not None else self.clock.monotonic_ns()
        timing.input_detected_ns = key_ns

        self._set_phase(Phase.POST_ROLL, on_phase_change)
        end_ns = anchor + self.durations.post_roll_ms * 1_000_000
        wait_ms = (end_ns - self.clock.monotonic_ns()) / 1e6
        self.clock.sleep_ms(max(0, wait_ms))

        timing.trial_end_ns = end_ns
        self._set_phase(Phase.SAVE, on_phase_change)
        return TrialResult(
            trial_id=trial_id,
            scheduled_label=label,
            repetition=repetition,
            input_mode=input_mode,
            timing=timing,
            anchor_ns=anchor,
        )
