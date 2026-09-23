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

Type these digits into the recorder's terminal:

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
the keystroke (next section).

## Keystroke verification (`input.key_detection`)

With `key_detection: terminal` (the reference config's setting), the
participant types each target **into the recorder's terminal window**.
Every key is timestamped on the same monotonic clock as the trial
phases, then mapped through the dubeolsik layout. That gives each trial
an independent `observed_label` and `input_detected_ns` (REQ-14.3),
alongside `observed_key` and `keystrokes`:

| what happened | status |
| --- | --- |
| the target jamo, once, inside the recorded audio | `valid` |
| a different jamo | `mismatch`, then re-recorded |
| no key / key outside the recorded audio / two keys in one recording / non-jamo key / Hangul IME / operator digit during the recording | `invalid`, then re-recorded |

Requirements: the terminal window must have focus, the **input method
must be in English mode**, and **Caps Lock must be off**. A Hangul IME
composes jamo into syllables and delays them, so its characters are
rejected with a hint to switch. If the setup is wrong, every trial
fails the check, and the session stops after
`quality.max_consecutive_failures` trials with the reason on screen
(for example "no keystroke detected… is its window focused?"). It does
not loop.

Use `key_detection: none` when the keyboard under test is not connected
to the recording computer. The input window is then purely time-based
and `observed_label` stays null. The mode is part of the session: a
resume that changes it is refused.

## Class definition: placeholder

`classism/labels.py` is a **stand-in**. The real classism class
definition (38 classes per the spec) was not available here. The
placeholder lists only the 33 jamo a single dubeolsik keystroke can
produce: 19 consonants including the Shift doubles, and 14 vowels
including ㅒ/ㅖ. Compound vowels such as ㅘ need two keystrokes and are
excluded. Drop the project's real `labels.py` in before collecting
training data. Nothing else changes, because the recorder derives the
class count from `len(CLASSES)`.

## Implementation decisions (spec Appendix E)

These are also written into every `session.json` under
`implementation_decisions`:

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
```

CI (`.github/workflows/tests.yml`) runs the linter, the full suite and
a no-write dry run on Python 3.10–3.13 for every pull request and every
push to `main` (REQ-61.4). A pinned-digest test checks that schedules
are byte-identical across those versions (REQ-2.4.1).

The full pipeline (scheduler -> continuous audio capture -> segmentation
-> WAV -> manifest -> validator) is covered without any physical audio
hardware via a deterministic synthetic audio source
(`experiments.recording.recorder.SyntheticBackend`), per REQ-61.
