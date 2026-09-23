"""
Independent keystroke observation (Appendix E.1/E.2, REQ-14.3, §25).

With input.key_detection = "terminal", every key the participant types
into the recorder's terminal is timestamped on the same monotonic clock
as the trial phases. This module turns one trial's key events into an
observed label and a verdict. It never touches audio or files.

Decisions recorded here (E.1/E.2):
- The IME must be in Latin (English) mode: Latin keys arrive as single
  characters the moment they are pressed. Korean IME composition delays
  and merges jamo into syllables (ㄱ then ㅏ becomes 가), so a Hangul
  character in the window marks the trial invalid, with a hint.
- A keystroke must fall inside the recorded segment (pre-roll start to
  post-roll end), i.e. be present in the audio. Keystrokes elsewhere in
  the pre-roll or post-roll are kept and timestamped for downstream
  filtering. No new status value is introduced (so no schema bump):
  rule violations use the existing `invalid` status.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from .trial import Status

# Standard dubeolsik layout. Shift yields the tense consonants on q/w/e/r/t
# and ㅒ/ㅖ on o/p; on every other key Shift gives the same jamo.
_DUBEOLSIK = {
    "q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ", "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
    "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ", "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
    "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
}
_DUBEOLSIK_SHIFT = {"Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ", "P": "ㅖ"}

# Pairs that differ only by Shift: confusing them usually means Caps Lock.
_SHIFT_PAIRS = {("ㄱ", "ㄲ"), ("ㄷ", "ㄸ"), ("ㅂ", "ㅃ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅉ"), ("ㅐ", "ㅒ"), ("ㅔ", "ㅖ")}

KEY_NAMES = {" ": "SPACE", "\t": "TAB", "\n": "ENTER", "\r": "ENTER", "\x7f": "BACKSPACE", "\x08": "BACKSPACE"}


def key_to_jamo(key: str) -> Optional[str]:
    if key in _DUBEOLSIK_SHIFT:
        return _DUBEOLSIK_SHIFT[key]
    if len(key) == 1 and key.isascii() and key.isalpha():
        return _DUBEOLSIK[key.lower()]
    return None


def key_name(key: str) -> str:
    if key in KEY_NAMES:
        return KEY_NAMES[key]
    if key.startswith("\x1b"):
        return "SPECIAL"
    return key


def _is_hangul(key: str) -> bool:
    return any("ᄀ" <= ch <= "ᇿ" or "㄰" <= ch <= "㆏" or "가" <= ch <= "힣" for ch in key)


@dataclasses.dataclass(frozen=True)
class KeyEvent:
    key: str
    t_ns: int


@dataclasses.dataclass
class KeyVerdict:
    status: Optional[Status]  # None: the keystroke evidence has no objection
    observed_label: Optional[str]
    observed_key: Optional[str]
    input_detected_ns: Optional[int]
    keystrokes: int  # target (non-control) keys pressed inside the recorded segment
    note: Optional[str]


def judge(
    events: list,
    scheduled_label: str,
    segment_start_ns: int,
    segment_end_ns: int,
    input_expected_ns: int,
    control_keys,
) -> KeyVerdict:
    """Judge one trial's key events (already limited to this trial's key
    window by the caller)."""

    def rel_ms(t):
        return (t - input_expected_ns) / 1e6

    targets = [e for e in events if e.key not in control_keys]
    in_segment = [e for e in targets if segment_start_ns <= e.t_ns <= segment_end_ns]
    control_in_segment = [e for e in events if e.key in control_keys and segment_start_ns <= e.t_ns <= segment_end_ns]

    if any(_is_hangul(e.key) for e in targets):
        return KeyVerdict(
            Status.INVALID, None, targets[0].key, targets[0].t_ns, len(in_segment),
            "Hangul IME is on: switch the input method to English so keystrokes can be verified",
        )

    if not in_segment:
        if targets:
            first = targets[0]
            return KeyVerdict(
                Status.INVALID, key_to_jamo(first.key), key_name(first.key), first.t_ns, 0,
                f"keystroke {key_name(first.key)!r} at {rel_ms(first.t_ns):+.0f} ms from the input window "
                "is outside the recorded audio",
            )
        return KeyVerdict(
            Status.INVALID, None, None, None, 0,
            "no keystroke detected in the recorder terminal (is its window focused?)",
        )

    first = in_segment[0]
    observed = key_to_jamo(first.key)
    base = dict(observed_label=observed, observed_key=key_name(first.key), input_detected_ns=first.t_ns,
                keystrokes=len(in_segment))

    if len(in_segment) > 1:
        keys = ", ".join(key_name(e.key) for e in in_segment)
        return KeyVerdict(Status.INVALID, note=f"{len(in_segment)} keystrokes in one recording ({keys})", **base)
    if observed is None:
        return KeyVerdict(Status.INVALID, note=f"pressed {key_name(first.key)!r}, which is not a jamo key", **base)
    if observed != scheduled_label:
        note = f"pressed {observed}, target was {scheduled_label}"
        if (observed, scheduled_label) in _SHIFT_PAIRS or (scheduled_label, observed) in _SHIFT_PAIRS:
            note += " (check Shift / Caps Lock)"
        return KeyVerdict(Status.MISMATCH, note=note, **base)
    if control_in_segment:
        return KeyVerdict(
            Status.INVALID,
            note=f"operator key {control_in_segment[0].key!r} was pressed during the recording",
            **base,
        )

    extra = [e for e in targets if e not in in_segment]
    note = f"{len(extra)} other key(s) pressed outside the recording" if extra else None
    return KeyVerdict(None, note=note, **base)
