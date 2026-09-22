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

## Tests

```bash
python -m pytest experiments/recording/tests
```

The full pipeline (scheduler -> continuous audio capture -> segmentation
-> WAV -> manifest -> validator) is covered without any physical audio
hardware via a deterministic synthetic audio source
(`experiments.recording.recorder.SyntheticBackend`), per REQ-61.
