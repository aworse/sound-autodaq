"""--export-classism: a recorded session as classism training data."""

import csv

import pytest

from classism.labels import CLASSES, JAMO
from experiments.recording import cli
from experiments.recording.export_classism import (
    EXTRA_COLUMNS,
    LABEL_COLUMNS,
    ExportError,
    export_session,
)
from experiments.recording.tests.test_engine_integration import ControlsAtTrial
from experiments.recording.tests.test_keylog import _config, _run
from experiments.recording.writer import read_manifest_jsonl

UID = "P01_S01_SESSION01"
SPECIAL_OF_KEYTYPE = {"space": "<sp>", "backspace": "<bs>", "shift": "<shift>", "caps": "<caps>", "other": "<other>"}


def classism_symbol(row):
    """classism/src/labels.py symbol_of: the label its loader trains on."""
    if row["keytype"] == "normal":
        assert row["jamo"] in JAMO, row
        return row["jamo"]
    return SPECIAL_OF_KEYTYPE[row["keytype"]]


def _read(out):
    with open(out / "labels" / f"{UID}.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _keypress_session(tmp_path):
    config = _config(tmp_path, trial={"capture": "keypress", "pre_roll_ms": 60, "input_window_ms": 5000,
                                      "post_roll_ms": 110})
    engine, summary, rows = _run(config, lambda n, target, rep: target)
    assert summary.completed is True, summary.stop_reason
    return engine, rows


def test_all_38_classes_export_in_the_classism_label_table(tmp_path):
    engine, manifest = _keypress_session(tmp_path)
    out = tmp_path / "classism"
    result = export_session(engine.session_dir, out)
    header, rows = _read(out)

    assert result.exported == 38 and not result.skipped and not result.warnings
    assert header == LABEL_COLUMNS + EXTRA_COLUMNS
    by_trial = {r["trial_id"]: r for r in manifest}
    for row in rows:
        trial = by_trial[int(row["trial_id"])]
        assert classism_symbol(row) == trial["label"]
        assert row["shift"] == ("1" if trial["label"] in "ㄲㄸㅃㅆㅉㅒㅖ" or trial["label"] == "<shift>" else "0")
        assert (row["participant"], row["scenario"]) == ("P01", "S01")
        assert row["clip_id"] == f"{UID}_{trial['trial_id']:08d}"
        # keypress capture: the keystroke sits exactly pre_roll into the file
        assert (row["wav_onset_s"], row["onset_source"]) == ("0.0600", "keystroke")
        clip = out / "audio" / f"{row['clip_id']}.wav"
        assert clip.read_bytes() == (engine.session_dir / trial["file"]).read_bytes()
    assert sorted(classism_symbol(r) for r in rows) == sorted(CLASSES)
    onsets = [float(r["onset_s"]) for r in rows]
    assert onsets[0] >= 0 and onsets == sorted(onsets)


def test_scheduled_capture_places_the_onset_where_the_key_was_typed(tmp_path):
    config = _config(tmp_path, trial={"pre_roll_ms": 30, "input_window_ms": 30, "post_roll_ms": 120})
    engine, summary, _ = _run(config, lambda n, target, rep: target)
    out = tmp_path / "classism"
    export_session(engine.session_dir, out)
    _, rows = _read(out)
    assert len(rows) == 38
    for row in rows:
        assert row["onset_source"] == "keystroke"
        assert 0.030 <= float(row["wav_onset_s"]) <= 0.060  # typed within the 30 ms input window


def test_without_key_detection_the_prompt_is_the_onset_and_a_warning_says_so(tmp_path):
    config = _config(tmp_path, input={"key_detection": "none"},
                     trial={"pre_roll_ms": 10, "input_window_ms": 10, "post_roll_ms": 100})
    engine, _, _ = _run(config, lambda n, target, rep: None)
    result = export_session(engine.session_dir, tmp_path / "classism")
    _, rows = _read(tmp_path / "classism")
    assert result.exported == 38 and {r["onset_source"] for r in rows} == {"prompt"}
    assert any("acoustically" in w for w in result.warnings)


def test_a_clip_too_short_for_the_100_ms_window_is_skipped_with_the_reason(tmp_path):
    config = _config(tmp_path, input={"key_detection": "none"},
                     trial={"pre_roll_ms": 10, "input_window_ms": 10, "post_roll_ms": 10})
    engine, _, _ = _run(config, lambda n, target, rep: None)
    result = export_session(engine.session_dir, tmp_path / "classism")
    assert result.exported == 0 and len(result.skipped) == 38
    assert "100 ms classism window" in result.skipped[0][1]


def test_an_incomplete_session_exports_its_valid_trials(tmp_path):
    from experiments.recording.engine import SessionEngine
    from experiments.recording.recorder import SyntheticBackend

    config = _config(tmp_path, input={"key_detection": "none"},
                     trial={"pre_roll_ms": 10, "input_window_ms": 10, "post_roll_ms": 100})
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    engine.run(control_source=ControlsAtTrial({4: ["quit"]}))
    valid = [r for r in read_manifest_jsonl(engine.session_dir / "manifest.jsonl") if r["status"] == "valid"]

    result = export_session(engine.session_dir, tmp_path / "classism")
    assert result.exported == len(valid) == 4
    assert result.still_to_record == 38 - 4


def test_a_damaged_session_is_refused(tmp_path):
    engine, manifest = _keypress_session(tmp_path)
    (engine.session_dir / manifest[0]["file"]).unlink()
    with pytest.raises(ExportError, match="missing audio"):
        export_session(engine.session_dir, tmp_path / "classism")


def test_reexport_replaces_only_this_sessions_clips(tmp_path):
    engine, _ = _keypress_session(tmp_path)
    out = tmp_path / "classism"
    export_session(engine.session_dir, out)
    stale = out / "audio" / f"{UID}_99999999.wav"
    other_session = out / "audio" / f"{UID}_2_00000001.wav"
    stale.write_bytes(b"old")
    other_session.write_bytes(b"other")

    result = export_session(engine.session_dir, out)
    assert result.exported == 38
    assert not stale.exists() and other_session.exists()
    assert len(list((out / "audio").glob("*.wav"))) == 39


def test_cli_export(tmp_path, capsys):
    engine, _ = _keypress_session(tmp_path)
    out = tmp_path / "classism"
    assert cli.main(["--export-classism", str(engine.session_dir), "--out", str(out)]) == 0
    assert "Exported 38 clips" in capsys.readouterr().out
    assert cli.main(["--export-classism", str(engine.session_dir)]) == 1
