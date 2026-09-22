from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.ui import QueueControlSource
from experiments.recording.webui.status import read_status


def _run_session(tmp_path):
    d = base_config_dict(
        output={"root": str(tmp_path / "data")},
        trial={
            "repetitions_per_class": 1,
            "countdown_ms": 0,
            "pre_roll_ms": 5,
            "input_window_ms": 5,
            "post_roll_ms": 5,
            "inter_trial_ms": 0,
        },
    )
    config = config_from_dict(d)
    engine = SessionEngine(config, backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary = engine.run(resume=False, control_source=QueueControlSource())
    return engine, summary


def test_status_missing_session_dir_reports_not_found(tmp_path):
    status = read_status(tmp_path / "nope")
    assert status["found"] is False


def test_status_reflects_completed_session(tmp_path):
    engine, summary = _run_session(tmp_path)
    status = read_status(engine.session_dir)

    assert status["found"] is True
    assert status["participant"] == "P01"
    assert status["scenario"] == "S01"
    assert status["session"] == "SESSION01"
    assert status["overall_total"] == summary.expected_trials
    assert status["overall_completed"] == summary.attempted_trials
    assert status["overall_valid"] == summary.valid_trials
    assert status["is_complete"] is True
    assert status["validation_result"] == "PASS"
    assert status["remaining_s"] in (None, 0) or status["remaining_s"] >= 0
    assert 0.0 <= status["completion_rate"] <= 1.0
    assert status["current_label"] is not None
