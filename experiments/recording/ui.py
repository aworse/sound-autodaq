"""
Terminal rendering and operator control input (§27-28).

Owns: rendering and control input. Never: experimental decisions
(labels, order, status semantics) (REQ-57).
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

CONTROL_POLICY_TEXT = (
    "Control policy: this terminal is the sole control surface. Key "
    "presses here drive [SPACE]/[R]/[S]/[I]/[Q]; the target keystroke "
    "itself is performed on the physical keyboard under test, which is "
    "not read by this program in HUMAN mode unless a key-log hook is "
    "configured (REQ-28.2/28.3)."
)

_KEYMAP = {
    " ": "pause",
    "r": "repeat",
    "R": "repeat",
    "s": "skip",
    "S": "skip",
    "i": "invalid",
    "I": "invalid",
    "q": "quit",
    "Q": "quit",
}


class ControlSource:
    """Interface: poll() returns one of 'pause'/'repeat'/'skip'/'invalid'/
    'quit', or None if nothing is pending. Never blocks."""

    def poll(self) -> Optional[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class QueueControlSource(ControlSource):
    """Programmatic control source for AUTOMATED mode and tests
    (REQ-24.3): push control names onto `events` and they are delivered
    on the next poll()."""

    def __init__(self):
        self.events: "queue.Queue[str]" = queue.Queue()

    def push(self, control: str) -> None:
        self.events.put(control)

    def poll(self) -> Optional[str]:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None


class TerminalControlSource(ControlSource):
    """Reads single keypresses from stdin in a background thread, without
    blocking the trial engine. A no-op (always returns None) when stdin
    is not a TTY, e.g. under CI."""

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if sys.stdin.isatty():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop.is_set():
                    ch = sys.stdin.read(1)
                    control = _KEYMAP.get(ch)
                    if control:
                        self._queue.put(control)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            return

    def poll(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()


def render_trial_screen(
    participant: str,
    scenario: str,
    session: str,
    overall_completed: int,
    overall_total: int,
    current_label: str,
    class_completed: int,
    class_total: int,
    next_label: Optional[str],
    status: str,
    elapsed_s: float,
    remaining_s: Optional[float],
) -> str:
    pct = (100.0 * overall_completed / overall_total) if overall_total else 0.0

    def fmt_hms(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    remaining_str = f"~{fmt_hms(remaining_s)}  (approx.)" if remaining_s is not None else "unknown"

    return (
        "====================================\n"
        " CLASSISM DATA COLLECTION\n"
        "====================================\n"
        "\n"
        f"Participant : {participant}\n"
        f"Scenario    : {scenario}\n"
        f"Session     : {session}\n"
        "\n"
        "Overall:\n"
        f"{overall_completed} / {overall_total}    {pct:.1f}%\n"
        "\n"
        "Current class:\n"
        f"{current_label}\n"
        "\n"
        "Class:\n"
        f"{class_completed} / {class_total}\n"
        "\n"
        "Next:\n"
        f"{next_label if next_label else '-'}\n"
        "\n"
        "Status:\n"
        f"{status}\n"
        "\n"
        f"Elapsed   : {fmt_hms(elapsed_s)}\n"
        f"Remaining : {remaining_str}\n"
        "\n"
        "[SPACE] Pause  [R] Repeat  [S] Skip\n"
        "[I] Invalid    [Q] Quit\n"
        "====================================\n"
    )


class Display:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def show(self, text: str) -> None:
        if self.enabled:
            sys.stdout.write("\x1b[2J\x1b[H")  # clear screen, home cursor
            sys.stdout.write(text)
            sys.stdout.flush()

    def line(self, text: str) -> None:
        if self.enabled:
            print(text)
