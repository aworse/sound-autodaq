# sound-autodaq

Implementation of `CLASSISM-SPEC-REC`: an automated acoustic data
acquisition system for collecting labelled keystroke-acoustic recordings
for the `classism` keyboard-jamo classifier.

## Layout

```
classism/                  project-level single source of truth
  labels.py                CLASSES, CLASS_DEFINITION_VERSION
  spec.md                  audio format contract referenced by the recorder

experiments/recording/     the recorder (this spec's normative subject)
  config.py                YAML schema, validation, resolution
  labels.py                thin re-export of classism.labels
  scheduler.py              trial pool, randomization strategies, schedule.json
  recorder.py               audio backends (PortAudio + hardware-free synthetic),
                             ring buffer, segment extraction
  trial.py                  trial phase state machine, monotonic timing
  writer.py                 atomic WAV writes, manifest.csv/.jsonl
  quality.py                peak / RMS / clipping / silence measurement
  metadata.py                session.json, session_summary.json, versioning
  progress.py                progress.json, resume support
  hardware.py                device probing, mic test, disk estimate
  ui.py                      terminal display + operator controls
  validator.py                read-only dataset validation
  engine.py                   session orchestration
  cli.py, __main__.py         command-line interface
  webui/                       optional browser dashboard (read-only status viewer)
  tests/                      unit + hardware-free integration tests
```

## Usage

```bash
pip install -r requirements.txt

# new session
python -m experiments.recording --config configs/S01.yaml

# dry run (no device opened, no files written)
python -m experiments.recording --config configs/S01.yaml --dry-run

# resume an interrupted session
python -m experiments.recording --resume data/raw/P01/S01/SESSION01

# microphone test
python -m experiments.recording --mic-test

# list input devices
python -m experiments.recording --list-devices

# validate a recorded dataset (read-only, no hardware needed)
python -m experiments.recording --validate data/raw/P01/S01/SESSION01

# browser dashboard: live status view of a session directory
# (read-only, polls the same files the recorder writes; run alongside
# --config/--resume in another terminal, or point it at a finished session)
python -m experiments.recording --dashboard data/raw/P01/S01/SESSION01 --port 8765
# then open http://127.0.0.1:8765
```

Exit status: `0` when the session completes and validates, and also
when the operator stops it with `0`/Ctrl+C (REQ-29.3). In that case the
console says `SESSION INCOMPLETE` and prints the resume command; a stop
is never reported as a complete session (REQ-46.2). `1` for errors and
validation failures.

## Operator controls

Press these digits (with `key_detection: hook` from any window; with
`terminal`, into the recorder's terminal):

| key | action |
| --- | --- |
| `1` | repeat: mark this trial invalid and re-record it immediately under a new trial id |
| `2` | skip: keep the audio, mark `skipped`, do not re-record |
| `3` | invalid: mark `operator_marked_invalid`, re-record at the end of the session |
| `4` | pause / resume |
| `0` | quit safely after the current trial |

A key applies to the trial on screen: the one in countdown, recording,
or the gap right after it.

**Deviation from REQ-28.1 (written justification):** the spec names
SPACE/R/S/I/Q. On a dubeolsik keyboard R, S, I and Q are the keys for ㄱ,
ㄴ, ㅑ and ㅂ. Those are target classes, so a participant typing ㅂ with
the IME in Latin mode would end the session. REQ-28.2 (controls must not
collide with experimental keystrokes) takes precedence under the §2.7
priority order, so the controls are digits. Letter, jamo and space keys
never act as controls. With key detection on, they are used to verify
the keystroke (see below). The digit row is never a target: `<other>`
trials prompt other keys.

## Recording on the keystroke (`trial.capture: keypress`)

This is the reference config's setting. There is no reaction-time
pressure. After the countdown the screen shows `PRESS ㄱ (take your
time)` and **waits for the key as long as it takes**
(`input_window_ms: 0`; a positive value sets a time limit instead).
Then:

```
microphone: ──────────── always recording into the ring buffer ────────────
you:                           ...thinking...   press ㄱ
saved file:                          |← pre_roll →|← post_roll →|
                                     cut out of the buffer afterwards
```

The audio *before* the keystroke is not recorded in advance: the
microphone never stops, so those samples are already in memory. The
recorder converts the keystroke's timestamp to a sample number and cuts
`pre_roll_ms` before to `post_roll_ms` after it. Every file therefore
has the same length (`pre_roll_ms + post_roll_ms`), with the keystroke
at the same offset. In the synthetic end-to-end test, the keystroke
lands within a few ms of `pre_roll_ms`, bounded by the audio block size.

For the participant: press once when `PRESS` appears, release, and stay
still until `saved`. A key pressed during the countdown does not count
(the screen says "too early"). A digit key while waiting (for example
`0` to quit, `4` to pause) ends the wait: the trial is marked
`interrupted` and re-recorded later, not counted as a failure.

Because the wait has no time limit:
- **With `key_detection: terminal`, if the terminal window loses
  focus**, keypresses never reach the recorder, so the screen simply
  stays at `PRESS`. There is no failure count and no automatic stop.
  Click the recorder window and press again. (With the input method in
  Hangul mode, keys *do* arrive, often one keystroke late because of
  syllable composition. Those trials are marked `invalid` with a
  "switch the input method to English" hint, and three in a row stop
  the session.) The hook has neither problem: it sees physical keys
  regardless of focus and input method.
- **A dead microphone does not leave the session hanging.** If the
  audio backend reports an error, or no audio arrives for one second
  while waiting, the trial is marked `interrupted` and the session stops
  safely, ready to resume.

With a positive `input_window_ms`, a trial that gets no key in time is
instead marked `invalid` and re-recorded, and a run of those stops the
session.

This needs `input.key_detection: hook` (or `terminal`). It also needs
`post_roll_ms + inter_trial_ms + countdown_ms >= pre_roll_ms`, so that
the previous keystroke can never fall inside this one's pre-roll; the
config validator enforces this. `trial.capture: scheduled` keeps the
spec's fixed REQ-22 timeline. The capture mode is part of the session: a
resume that changes it is refused. This mode deviates from REQ-22's
fixed timeline, and the deviation is recorded in every `session.json`
under `implementation_decisions.capture`.

## Keystroke verification (`input.key_detection`)

Each trial gets an independent `observed_label` and `input_detected_ns`
(REQ-14.3), alongside `observed_key` and `keystrokes`. Every key is
timestamped on the same monotonic clock as the trial phases and mapped
through the dubeolsik layout to a class.

- **`hook`** (the reference config's setting): an OS keyboard hook
  (`pynput`) sees **physical keys** in any window, whatever the input
  method or Caps Lock state, including a bare Shift or Caps Lock press.
  Shift is tracked from the Shift key itself, not from letter case, so
  ㄱ stays ㄱ after the `<caps>` trial turns Caps Lock on. Held keys
  that auto-repeat count once. While the hook runs, the recorder's
  terminal swallows typed characters so they do not pile up on screen.
- **`terminal`**: the participant types into the recorder's terminal
  window. A terminal only sees characters, so it **cannot see a bare
  Shift or Caps Lock**; pre-flight refuses it when the class list
  contains `<shift>` or `<caps>`. Focus, English input mode and Caps
  Lock off are all required.
- **`none`**: for a keyboard that is not connected to the recording
  computer. The input window is purely time-based and `observed_label`
  stays null.

| what happened | status |
| --- | --- |
| the target key, once, inside the recorded audio | `valid` |
| a different class (e.g. ㄱ for ㄲ: "hold Shift") | `mismatch`, then re-recorded |
| no key / key outside the recorded audio / two keys in one recording / Hangul IME (terminal) / operator digit during the recording | `invalid`, then re-recorded |

If the setup is wrong, every trial fails the check, and the session
stops after `quality.max_consecutive_failures` trials with the reason
on screen. It does not loop. The mode is part of the session: a resume
that changes it is refused.

**Hook permissions.** Linux: needs an X11 session (under Wayland the
hook only sees XWayland windows). macOS: grant the terminal app
*Input Monitoring* (and *Accessibility*) in System Settings; how macOS
reports Caps Lock through the hook is not yet verified on hardware, so
check the `<caps>` trials in the pilot. Windows: no setup; run it from
Windows Terminal or cmd, and the Korean IME may stay in 한글 mode. If the
hook cannot start, the recorder exits with the reason before opening
audio.

**Windows specifics.** CI runs the suite on `windows-latest`, including
the hook fed real `SendInput` key events for all 38 classes. Key and
audio timestamps use `time.perf_counter_ns` (`experiments/recording/clock.py`):
before Python 3.13, Windows `time.monotonic` ticks in 15.6 ms steps,
which would move every cut by up to that much. The 한/영 and 한자 keys
report no key release on Windows; the hook still counts each tap (as
`<other>`), because only a press within 1.2 s of the previous one is
treated as auto-repeat. Opening the dashboard while recording is safe:
a status-file update that collides with a read is retried.

## Classes (`classism/labels.py`)

The 38 classes mirror `aworse/classism` (`CLASS_DEFINITION_VERSION =
classism-afe-38@c6606a0`), in its order: 33 dubeolsik jamo, then
`<sp> <bs> <shift> <caps> <other>`.

| class | what the participant presses |
| --- | --- |
| 19 consonants, 14 vowels | the jamo's dubeolsik key; ㄲ ㄸ ㅃ ㅆ ㅉ ㅒ ㅖ as **Shift + key** (the screen says e.g. `ㄲ (Shift + ㄱ)`). The chord's Shift is part of the keystroke, not a second key. |
| `<sp>` `<bs>` | Space, Backspace |
| `<shift>` | Shift on its own, then release |
| `<caps>` | Caps Lock (toggles; the hook does not care) |
| `<other>` | the key on screen, cycled per repetition through Enter, Tab, `,` `.` `/` `;` `'` `[` `]` `-` `=` |

`<other>` had no upstream definition, so this recorder defines it: **any
key that is not a jamo key, Space, Backspace, Shift, Caps Lock or an
operator digit**. The prompt rotates keys so the class covers more than
one sound. Pressing a different `<other>` key still counts as `<other>`;
the manifest notes which key it was.

## Implementation decisions (spec Appendix E)

These are also written into every `session.json` under
`implementation_decisions`:

- **Capture mode:** `trial.capture` (see above), recorded in
  `session.json`.
- **E.1 keystroke detection:** chosen per session by
  `input.key_detection` (see above) and recorded in `session.json`.
- **E.2 out-of-window keystrokes:** no new status value, so no schema
  bump. A keystroke outside the recorded audio is `invalid`. One in the
  pre-roll or post-roll is kept, with its timestamp, for downstream
  filtering.
- **E.3 re-queue placement:** repeat and pause-discard re-record
  immediately. Invalid, silent, overflowed and corrupted trials go to the
  end of the session (when `repeat_on_invalid` is true). Skipped trials
  are never re-recorded. A failure streak of
  `quality.max_consecutive_failures` stops the session safely.
- **E.4 audio backend:** sounddevice/PortAudio at int16. The native
  sample rate is verified before opening, with no software resampling.
- **E.5 manual order file:** a JSON list of `{"label", "repetition"}`
  objects.
- `input.mode: automated` is rejected at pre-flight. This recorder
  cannot synthesize key events, and it will not record human keystrokes
  under an `automated` label.

## Tests

```bash
python -m pytest experiments/recording/tests
# with the real keyboard-hook tests (needs xvfb and xdotool):
xvfb-run -a python -m pytest experiments/recording/tests
```

`test_keyhook_real.py` types all 38 classes as real OS key events
(xdotool under Xvfb, `SendInput` on Windows) and records a
keypress-capture session through the real `pynput` hook. Elsewhere those
tests are skipped.

CI (`.github/workflows/tests.yml`) runs the linter, the full suite and
a no-write dry run on Python 3.10–3.13 (Linux) and 3.10/3.13 (Windows)
for every pull request and every push to `main` (REQ-61.4). A pinned-digest test checks that schedules
are byte-identical across those versions (REQ-2.4.1).

The full pipeline (scheduler -> continuous audio capture -> segmentation
-> WAV -> manifest -> validator) is covered without any physical audio
hardware via a deterministic synthetic audio source
(`experiments.recording.recorder.SyntheticBackend`), per REQ-61.
