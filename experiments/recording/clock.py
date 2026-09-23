"""
The one clock for keystroke and audio timestamps.

Trial phases, key events and audio block arrivals are all stamped with
`now_ns()` so a keystroke time can be turned into a sample index. It is
`time.perf_counter_ns`: monotonic on every platform, and high resolution
on Windows, where `time.monotonic` ticks in 15.6 ms steps before Python
3.13. On Linux and macOS both read the same OS clock.
"""

import contextlib
import sys
import time

now_ns = time.perf_counter_ns
now_s = time.perf_counter


@contextlib.contextmanager
def fine_timers():
    """Windows wakes sleeping threads on a 15.6 ms system timer tick by
    default, so short sleeps (trial phases, polling) overshoot by up to
    that much. Ask for 1 ms ticks while a session runs, as audio software
    does. No effect elsewhere."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)
