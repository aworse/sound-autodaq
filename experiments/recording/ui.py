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
from typing import Optional

# REQ-28.1 names SPACE/R/S/I/Q, but on a dubeolsik keyboard R, S, I and Q
# are the keys for ㄱ, ㄴ, ㅑ and ㅂ — target classes. A participant typing
# ㅂ with the IME in Latin mode would end the session. REQ-28.2 (no
# collision) wins under the §2.7 priority order, so every control is a
# digit: digits are not jamo keys and come through unchanged in either IME
# mode. Letters, jamo and space typed into the terminal are ignored.
KEYMAP = {
    "1": "repeat",
    "2": "skip",
    "3": "invalid",
    "4": "pause",
    "0": "quit",
}

CONTROL_HELP = "[1] Repeat  [2] Skip  [3] Invalid  [4] Pause/Resume  [0] Quit"

CONTROL_POLICY_TEXT = (
    "Controls are digit keys typed into this terminal. Jamo/letter keys are "
    "ignored, so typing the target never triggers a control. A control "
    "applies to the trial on screen."
)


class ControlSource:
    """poll() returns one of 'repeat'/'skip'/'invalid'/'pause'/'quit', or
    None if nothing is pending. Never blocks. `interactive` says whether a
    person is there to answer a prompt (retry/continue/stop)."""

    interactive = False

    def poll(self) -> Optional[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class QueueControlSource(ControlSource):
    """Programmatic control source for tests and AUTOMATED mode: controls
    pushed onto the queue are delivered in order."""

    def __init__(self, interactive: bool = False):
        self.events: "queue.Queue[str]" = queue.Queue()
        self.interactive = interactive

    def push(self, control: str) -> None:
        self.events.put(control)

    def poll(self) -> Optional[str]:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None


class TerminalControlSource(ControlSource):
    """Reads keypresses from stdin in a background thread without blocking
    the engine. Inert (always None, not interactive) when stdin is not a
    TTY, e.g. under CI. close() restores the terminal mode."""

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_attrs = None
        self._fd = None
        self.interactive = sys.stdin.isatty()
        if self.interactive:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        import select

        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            chunk = os.read(self._fd, 64).decode("utf-8", errors="ignore")
            for ch in chunk:
                control = KEYMAP.get(ch)
                if control:
                    self._queue.put(control)

    def poll(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._old_attrs is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None


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
        f"        {current_label}",
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
        "====================================",
        "",
    ]
    return "\n".join(lines)


class Display:
    """`enabled` gates all output; `interactive` gates the clear-screen
    redraw (a non-TTY gets plain lines instead of escape codes)."""

    def __init__(self, enabled: bool = True, interactive: bool = False):
        self.enabled = enabled
        self.interactive = interactive

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
