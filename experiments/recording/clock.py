"""
The one clock for keystroke and audio timestamps.

Trial phases, key events and audio block arrivals are all stamped with
`now_ns()` so a keystroke time can be turned into a sample index. It is
`time.perf_counter_ns`: monotonic on every platform, and high resolution
on Windows, where `time.monotonic` ticks in 15.6 ms steps before Python
3.13. On Linux and macOS both read the same OS clock.
"""

import time

now_ns = time.perf_counter_ns
now_s = time.perf_counter
