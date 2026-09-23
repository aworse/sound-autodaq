"""
Terminal rendering and operator control input (§27-28).

Owns: rendering and control input. Never: experimental decisions
(labels, order, status semantics) (REQ-57).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Optional

from . import clock
from .keylog import KeyEvent, normalize_char

# REQ-28.1 names SPACE/R/S/I/Q, but on a dubeolsik keyboard R, S, I and Q
# are the keys for ㄱ, ㄴ, ㅑ and ㅂ — target classes. A participant typing
# ㅂ with the IME in Latin mode would end the session. REQ-28.2 (no
# collision) wins under the §2.7 priority order, so every control is a
# digit: digits are not jamo keys, not class keys, and come through
# unchanged in either IME mode. Other keys never act as controls; with key
# detection on they are timestamped to verify the participant's keystroke.
KEYMAP = {
    "1": "repeat",
    "2": "skip",
    "3": "invalid",
    "4": "pause",
    "0": "quit",
}

CONTROL_HELP = "[1] Repeat  [2] Skip  [3] Invalid  [4] Pause/Resume  [0] Quit"

CONTROL_POLICY_TEXT = (
    "Controls are digit keys; typing the target never triggers a control. "
    "A control applies to the trial on screen."
)

KEY_DETECTION_TEXT = {
    "terminal": "Key check ON: type the target in THIS window, input method in English, Caps Lock off.",
    "hook": "Key check ON (keyboard hook): type on the keyboard, any window, any input method.",
}


class ControlSource:
    """poll() returns one of 'repeat'/'skip'/'invalid'/'pause'/'quit', or
    None if nothing is pending. Never blocks. `interactive` says whether a
    person is there to answer a prompt (retry/continue/stop)."""

    interactive = False

    def poll(self) -> Optional[str]:
        raise NotImplementedError

    def poll_keys(self) -> list:
        """Every key pressed since the last call, as KeyEvent(key, t_ns)
        stamped with clock.now_ns() when read — the clock the trial
        phases use. Includes control digits."""
        return []

    def close(self) -> None:
        pass


class QueueControlSource(ControlSource):
    """Programmatic control source for tests and AUTOMATED mode: controls
    pushed onto the queue are delivered in order."""

    def __init__(self, interactive: bool = False):
        self.events: "queue.Queue[str]" = queue.Queue()
        self.keys: "queue.Queue[KeyEvent]" = queue.Queue()
        self.interactive = interactive

    def push(self, control: str) -> None:
        self.events.put(control)

    def push_key(self, key: str, t_ns: Optional[int] = None, shift: bool = False) -> None:
        """Deliver a normalized key-down (see keylog), like a hook would."""
        self.keys.put(KeyEvent(key, clock.now_ns() if t_ns is None else t_ns, shift))

    def poll_keys(self) -> list:
        return _drain(self.keys)

    def poll(self) -> Optional[str]:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None


class TerminalControlSource(ControlSource):
    """Reads keypresses from stdin in a background thread without blocking
    the engine. Inert (always None, not interactive) when stdin is not a
    TTY, e.g. under CI. close() restores the terminal mode.

    With swallow=True it only keeps the terminal quiet (no echo) and
    discards what is typed; the keyboard hook is then the input source."""

    def __init__(self, swallow: bool = False):
        self._swallow = swallow
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._keys: "queue.Queue[KeyEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_attrs = None
        self._fd = None
        self.interactive = sys.stdin.isatty()
        if self.interactive:
            if os.name == "nt":
                target = self._run_windows
            else:
                import termios
                import tty

                self._fd = sys.stdin.fileno()
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                target = self._run_posix
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()

    def _handle(self, chars: list, t_ns: int) -> None:
        if self._swallow:
            return
        for ch in chars:
            key, shift = normalize_char(ch)
            self._keys.put(KeyEvent(key, t_ns, shift))
            control = KEYMAP.get(key)
            if control:
                self._queue.put(control)

    def _run_posix(self) -> None:
        import select

        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            data = os.read(self._fd, 64)
            t_ns = clock.now_ns()
            chunk = data.decode("utf-8", errors="replace")
            # An escape sequence (arrow keys, F-keys) is one keypress.
            self._handle([chunk] if chunk.startswith("\x1b") else list(chunk), t_ns)

    def _run_windows(self) -> None:
        # The Windows console has no cbreak mode or select() on stdin:
        # msvcrt reads one key at a time without echo. Polled every 2 ms,
        # which bounds the timestamp delay of terminal key detection.
        import msvcrt

        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.002)
                continue
            t_ns = clock.now_ns()
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow / function key: a second code follows
                msvcrt.getwch()
                ch = "\x1b["  # one "special" key, like a POSIX escape sequence
            self._handle([ch], t_ns)

    def poll(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def poll_keys(self) -> list:
        return _drain(self._keys)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._old_attrs is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_trial_screen(
    participant: str,
    scenario: str,
    session: str,
    trial_id: int,
    overall_completed: int,
    overall_total: int,
    current_label: str,
    class_completed: int,
    class_total: int,
    next_label: Optional[str],
    status: str,
    elapsed_s: float,
    remaining_s: Optional[float],
    notice: Optional[str] = None,
    key_detection: str = "none",
    hint: str = "",
) -> str:
    """The trial on screen is always the one being recorded (REQ-27.3):
    this is rendered before and during a trial, never after it."""
    pct = (100.0 * overall_completed / overall_total) if overall_total else 0.0
    remaining = f"~{fmt_hms(remaining_s)}  (approx.)" if remaining_s is not None else "estimating..."
    lines = [
        "====================================",
        " CLASSISM DATA COLLECTION",
        "====================================",
        "",
        f"Participant : {participant}",
        f"Scenario    : {scenario}",
        f"Session     : {session}",
        "",
        "Overall:",
        f"{overall_completed} / {overall_total}    {pct:.1f}%",
        "",
        f"Trial {trial_id}   target:",
        "",
        f"        {current_label}   {hint}".rstrip(),
        "",
        f"Class: {class_completed} / {class_total}      Next: {next_label or '-'}",
        "",
        f">>> {status}",
        "",
        f"Elapsed   : {fmt_hms(elapsed_s)}",
        f"Remaining : {remaining}",
    ]
    if notice:
        lines += ["", f"!! {notice}"]
    lines += [
        "",
        CONTROL_HELP,
        CONTROL_POLICY_TEXT,
    ] + ([KEY_DETECTION_TEXT[key_detection]] if key_detection in KEY_DETECTION_TEXT else []) + [
        "====================================",
        "",
    ]
    return "\n".join(lines)


def _enable_windows_ansi() -> None:
    """Let the classic Windows console (conhost) interpret the escape codes
    used for the clear-screen redraw; Windows Terminal already does."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class Display:
    """`enabled` gates all output; `interactive` gates the clear-screen
    redraw (a non-TTY gets plain lines instead of escape codes)."""

    def __init__(self, enabled: bool = True, interactive: bool = False):
        self.enabled = enabled
        self.interactive = interactive
        if interactive and os.name == "nt":
            _enable_windows_ansi()

    def show(self, text: str) -> None:
        if not self.enabled:
            return
        if self.interactive:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(text)
            sys.stdout.flush()

    def line(self, text: str) -> None:
        if self.enabled:
            print(text, flush=True)
