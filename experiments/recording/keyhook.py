"""
OS-level keyboard hook (input.key_detection = hook).

Physical key-downs are read from the operating system via pynput, so the
recorder sees Shift and Caps Lock (which a terminal never receives), sees
keys regardless of which window has focus, and is not affected by the
input method (Hangul/English) or the Caps Lock state.

- Shift for a jamo chord comes from tracking the Shift key itself, never
  from the character's case (Caps Lock would make every letter upper
  case).
- OS auto-repeat is dropped: a held key makes one sound, so it is one
  event.
- Digits are operator controls (ui.KEYMAP), read from the same hook.

pynput is imported only when the hook starts: it needs a desktop session
(an X server on Linux, Accessibility permission on macOS).
"""

from __future__ import annotations

import queue
from typing import Optional

from .errors import RecorderError
from .keylog import KEY_FOR_JAMO, SHIFTED_SYMBOLS, KeyEvent
from .ui import KEYMAP, ControlSource, TerminalControlSource
from . import clock

_NAMED = {
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
    "space": "space", "backspace": "backspace", "enter": "enter", "tab": "tab",
    "caps_lock": "caps_lock", "esc": "esc",
}


class KeyHookError(RecorderError):
    """The keyboard hook could not be started."""


def normalize_key(char: Optional[str], name: Optional[str], vk: Optional[int]) -> tuple:
    """(key, implied_shift) for one pynput key. `char`/`name`/`vk` are the
    attributes pynput exposes (KeyCode.char, Key.name, KeyCode.vk)."""
    if name:
        return _NAMED.get(name, name), False
    if char:
        if char in KEY_FOR_JAMO:  # a Hangul layout reporting the jamo itself
            return KEY_FOR_JAMO[char]
        if len(char) == 1 and char.isascii() and char.isalpha():
            return char.lower(), False  # case says nothing: Caps Lock changes it
        if char in SHIFTED_SYMBOLS:
            return SHIFTED_SYMBOLS[char], False  # Shift is tracked separately
        return char, False
    if vk is not None:
        if 0x41 <= vk <= 0x5A:  # Windows virtual-key codes for A-Z
            return chr(vk).lower(), False
        if 0x30 <= vk <= 0x39:
            return chr(vk), False
        return f"vk{vk}", False
    return "unknown", False


# OS auto-repeat sends a held key again every 30-500 ms (1 s at most
# before the first repeat). A press of a key that is still down counts as
# a new keystroke only after a longer gap: some keys (Windows 한/영 and
# 한자, macOS Caps Lock) never report their release.
AUTOREPEAT_GAP_NS = 1_200_000_000


class HookState:
    """Turns raw press/release callbacks into KeyEvents: tracks Shift and
    drops auto-repeat. Separate from pynput so it can be tested directly."""

    def __init__(self, emit):
        self._emit = emit
        self._down: dict = {}  # key -> time of its last press or auto-repeat

    def press(self, char, name, vk, t_ns: Optional[int] = None) -> None:
        key, implied_shift = normalize_key(char, name, vk)
        t = clock.now_ns() if t_ns is None else t_ns
        last = self._down.get(key)
        self._down[key] = t
        # Caps Lock is exempt: some systems report its release late or never.
        if last is not None and t - last < AUTOREPEAT_GAP_NS and key != "caps_lock":
            return
        shift = implied_shift or ("shift" in self._down and key != "shift")
        self._emit(KeyEvent(key, t, shift))

    def release(self, char, name, vk) -> None:
        key, _ = normalize_key(char, name, vk)
        self._down.pop(key, None)


def _attrs(k):
    return getattr(k, "char", None), getattr(k, "name", None), getattr(k, "vk", None)


class HookControlSource(ControlSource):
    """ControlSource backed by the keyboard hook: poll() gives operator
    controls (digit keys), poll_keys() every key-down. While it runs, the
    terminal's own input is swallowed so a digit is never acted on twice."""

    interactive = True

    def __init__(self):
        self._keymap = KEYMAP
        self._keys: "queue.Queue[KeyEvent]" = queue.Queue()
        self._controls: "queue.Queue[str]" = queue.Queue()
        self._state = HookState(self._on_event)
        try:
            from pynput import keyboard
        except Exception as exc:
            raise KeyHookError(
                "input.key_detection is 'hook' but the keyboard hook cannot start "
                f"({type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else exc!r}). "
                "Run the recorder in a desktop session: on Linux an X11 session with DISPLAY set, on macOS "
                "grant the terminal Input Monitoring and Accessibility permission, on Windows no setup is needed. Or use "
                "input.key_detection: terminal for a jamo-only class list."
            ) from exc
        self._listener = keyboard.Listener(
            on_press=lambda k: self._state.press(*_attrs(k)),
            on_release=lambda k: self._state.release(*_attrs(k)),
        )
        self._listener.start()
        self._listener.wait()
        self._terminal = TerminalControlSource(swallow=True)

    def _on_event(self, event: KeyEvent) -> None:
        self._keys.put(event)
        control = self._keymap.get(event.key)
        if control:
            self._controls.put(control)

    def poll(self) -> Optional[str]:
        try:
            return self._controls.get_nowait()
        except queue.Empty:
            return None

    def poll_keys(self) -> list:
        out = []
        while True:
            try:
                out.append(self._keys.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._listener.stop()
        self._terminal.close()
