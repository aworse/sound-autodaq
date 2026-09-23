"""
Independent keystroke observation (Appendix E.1/E.2, REQ-14.3, §25).

Key sources (a keyboard hook, or the recorder's terminal) deliver
`KeyEvent`s: a normalized physical-key name, the monotonic time it went
down, and whether Shift was held. This module maps those to the 38-class
label space and judges one trial's events. It never touches audio or files.

Normalized key names: "a".."z" for letter keys (the physical key, never
upper case), "0".."9", base punctuation ("," "." "/" ";" "'" "[" "]" "-"
"=" "`" "\\"), and named keys: "space", "backspace", "enter", "tab",
"shift", "caps_lock", "esc", plus whatever else a source reports (arrows,
ctrl, alt, function keys, ...). A Hangul character is kept as-is: it can
only come from a terminal with the Korean IME on.

Decisions recorded here (E.1/E.2):
- Shift chords: a tense consonant or ㅒ/ㅖ is Shift + base key. The Shift
  press that belongs to the chord is expected; any other Shift press in
  the recording is an extra keystroke.
- `<other>` is every key that is not a jamo key, Space, Backspace, Shift,
  Caps Lock or an operator digit. Trials prompt a specific one, cycling
  through OTHER_KEYS by repetition; any `<other>` key counts as the label.
- A keystroke must fall inside the recorded segment (pre-roll start to
  post-roll end). Keystrokes in the pre-roll or post-roll are kept with
  their timestamp. No new status value: rule violations are `invalid`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from .trial import Status

# Standard dubeolsik layout. Shift yields the tense consonants on q/w/e/r/t
# and ㅒ/ㅖ on o/p; on every other key Shift gives the same jamo.
DUBEOLSIK = {
    "q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ", "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
    "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ", "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
    "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
}
DUBEOLSIK_SHIFT = {"q": "ㅃ", "w": "ㅉ", "e": "ㄸ", "r": "ㄲ", "t": "ㅆ", "o": "ㅒ", "p": "ㅖ"}
SHIFTED_JAMO = frozenset(DUBEOLSIK_SHIFT.values())

# jamo -> (physical key, shift): how to type each jamo target
KEY_FOR_JAMO = {j: (k, False) for k, j in DUBEOLSIK.items()}
KEY_FOR_JAMO.update({j: (k, True) for k, j in DUBEOLSIK_SHIFT.items()})

SPECIAL_KEYS = {"space": "<sp>", "backspace": "<bs>", "shift": "<shift>", "caps_lock": "<caps>"}

# Keys prompted for `<other>` trials, cycled by repetition. None of them
# is a jamo key, a special class key, or an operator digit.
OTHER_KEYS = ("enter", "tab", ",", ".", "/", ";", "'", "[", "]", "-", "=")

DISPLAY_NAMES = {
    "space": "Space", "backspace": "Backspace", "shift": "Shift", "caps_lock": "Caps Lock",
    "enter": "Enter", "tab": "Tab", "esc": "Esc",
}

# Pairs that differ only by Shift.
_SHIFT_PAIRS = {(DUBEOLSIK[k], DUBEOLSIK_SHIFT[k]) for k in DUBEOLSIK_SHIFT}


@dataclasses.dataclass(frozen=True)
class KeyEvent:
    key: str
    t_ns: int
    shift: bool = False


# US layout: the character Shift produces -> the physical key.
SHIFTED_SYMBOLS = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "-", "+": "=", "{": "[", "}": "]", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`", "|": "\\",
}
_CHAR_NAMES = {" ": "space", "\t": "tab", "\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace",
               "\x1b": "esc"}


def normalize_char(ch: str) -> tuple:
    """(key, shift) for a character as a terminal delivers it. Upper case
    and shifted symbols imply Shift. Hangul is returned unchanged."""
    if ch in _CHAR_NAMES:
        return _CHAR_NAMES[ch], False
    if ch.startswith("\x1b"):
        return "special", False
    if len(ch) == 1 and ch.isascii() and ch.isalpha():
        return ch.lower(), ch.isupper()
    if ch in SHIFTED_SYMBOLS:
        return SHIFTED_SYMBOLS[ch], True
    return ch, False


def is_hangul(key: str) -> bool:
    return any("ᄀ" <= ch <= "ᇿ" or "㄰" <= ch <= "㆏" or "가" <= ch <= "힣" for ch in key)


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def symbol_of(event: KeyEvent, control_keys=()) -> Optional[str]:
    """The class a key event belongs to, or None for an operator digit or
    a Hangul character (which only a terminal with the Korean IME sends)."""
    key = event.key
    if key in control_keys or is_hangul(key):
        return None
    if key in DUBEOLSIK:
        if event.shift and key in DUBEOLSIK_SHIFT:
            return DUBEOLSIK_SHIFT[key]
        return DUBEOLSIK[key]
    if key in SPECIAL_KEYS:
        return SPECIAL_KEYS[key]
    return "<other>"


def prompted_key(label: str, repetition: int) -> Optional[str]:
    """The physical key a trial asks for: a jamo's key, a special key, or
    for `<other>` the key from OTHER_KEYS this repetition rotates to."""
    if label in KEY_FOR_JAMO:
        return KEY_FOR_JAMO[label][0]
    for key, symbol in SPECIAL_KEYS.items():
        if symbol == label:
            return key
    if label == "<other>":
        return OTHER_KEYS[(repetition - 1) % len(OTHER_KEYS)]
    return None


def prompt_hint(label: str, repetition: int) -> str:
    """Words shown next to the target so nobody has to guess the key."""
    if label in SHIFTED_JAMO:
        return f"(Shift + {DUBEOLSIK[KEY_FOR_JAMO[label][0]]})"
    if label == "<shift>":
        return "(tap Shift on its own)"
    if label in ("<sp>", "<bs>", "<caps>"):
        return f"({display_name(prompted_key(label, repetition))})"
    if label == "<other>":
        return f"(press {display_name(prompted_key(label, repetition))})"
    return ""


def is_trigger(event: KeyEvent, target: str, control_keys) -> bool:
    """Whether a key-down should end the wait in keypress capture: the
    first real key, where Shift only counts when Shift itself is the target
    (otherwise it is the modifier of the chord that follows)."""
    if event.key in control_keys:
        return False
    if event.key == "shift":
        return target == "<shift>"
    return True


@dataclasses.dataclass
class KeyVerdict:
    status: Optional[Status]  # None: the keystroke evidence has no objection
    observed_label: Optional[str]
    observed_key: Optional[str]
    input_detected_ns: Optional[int]
    keystrokes: int  # keys pressed inside the recorded segment, chord Shift not counted
    note: Optional[str]


def judge(
    events: list,
    scheduled_label: str,
    segment_start_ns: int,
    segment_end_ns: int,
    input_expected_ns: int,
    control_keys,
    prompted: Optional[str] = None,
) -> KeyVerdict:
    """Judge one trial's key events (already limited to this trial's key
    window by the caller)."""

    def rel_ms(t):
        return (t - input_expected_ns) / 1e6

    def in_segment(e):
        return segment_start_ns <= e.t_ns <= segment_end_ns

    keys = [e for e in events if e.key not in control_keys]
    controls_in_segment = [e for e in events if e.key in control_keys and in_segment(e)]

    hangul = [e for e in keys if is_hangul(e.key)]
    if hangul:
        return KeyVerdict(
            Status.INVALID, None, hangul[0].key, hangul[0].t_ns, 0,
            "Hangul IME is on: switch the input method to English so keystrokes can be verified",
        )

    # The main keystroke: the first key in the recording that is not a
    # chord modifier (Shift counts only when Shift itself is the target).
    seg = [e for e in keys if in_segment(e)]
    candidates = [e for e in seg if is_trigger(e, scheduled_label, control_keys)]
    if not candidates:
        outside = [e for e in keys if is_trigger(e, scheduled_label, control_keys)]
        if outside:
            first = outside[0]
            return KeyVerdict(
                Status.INVALID, symbol_of(first, control_keys), display_name(first.key), first.t_ns, 0,
                f"keystroke {display_name(first.key)!r} at {rel_ms(first.t_ns):+.0f} ms from PRESS "
                "is outside the recorded audio",
            )
        return KeyVerdict(Status.INVALID, None, None, None, 0, "no keystroke detected")

    main = candidates[0]
    observed = symbol_of(main, control_keys)

    # A Shift press before the main key is the chord's modifier when the key
    # typed is a Shift jamo (ㄲ ... ㅖ). Judged on what was typed, not on the
    # target, so Shift+R for ㄱ reads as "typed ㄲ" rather than "two keys".
    chord_shift = observed in SHIFTED_JAMO and main.shift
    extras = [
        e for e in seg
        if e is not main and not (chord_shift and e.key == "shift" and e.t_ns <= main.t_ns)
    ]
    base = dict(
        observed_label=observed, observed_key=display_name(main.key), input_detected_ns=main.t_ns,
        keystrokes=1 + len(extras),
    )

    if extras:
        names = ", ".join(display_name(e.key) for e in [main] + extras)
        return KeyVerdict(Status.INVALID, note=f"{1 + len(extras)} keystrokes in one recording ({names})", **base)
    if observed != scheduled_label:
        note = f"pressed {observed} ({display_name(main.key)}), target was {scheduled_label}"
        if (observed, scheduled_label) in _SHIFT_PAIRS:
            note += " — hold Shift"
        elif (scheduled_label, observed) in _SHIFT_PAIRS:
            note += " — without Shift"
        return KeyVerdict(Status.MISMATCH, note=note, **base)
    if controls_in_segment:
        return KeyVerdict(
            Status.INVALID,
            note=f"operator key {controls_in_segment[0].key!r} was pressed during the recording",
            **base,
        )

    notes = []
    if prompted and scheduled_label == "<other>" and main.key != prompted:
        notes.append(f"prompted {display_name(prompted)}, pressed {display_name(main.key)}")
    outside = [e for e in keys if not in_segment(e)]
    if outside:
        notes.append(f"{len(outside)} other key(s) pressed outside the recording")
    return KeyVerdict(None, note="; ".join(notes) or None, **base)
