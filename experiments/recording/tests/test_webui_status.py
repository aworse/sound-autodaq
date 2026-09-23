from experiments.recording.config import config_from_dict
from experiments.recording.engine import SessionEngine
from experiments.recording.recorder import SyntheticBackend
from experiments.recording.tests.helpers import base_config_dict
from experiments.recording.ui import Display, QueueControlSource
from experiments.recording.webui.status import read_status


def _config(tmp_path):
    d = base_config_dict(
        output={"root": str(tmp_path / "data")},
        trial={
            "repetitions_per_class": 1,
            "countdown_ms": 0,
            "pre_roll_ms": 10,
            "input_window_ms": 10,
            "post_roll_ms": 10,
            "inter_trial_ms": 0,
        },
    )
    return config_from_dict(d)


def test_status_missing_session_dir_reports_not_found(tmp_path):
    assert read_status(tmp_path / "nope")["found"] is False


def test_status_reflects_completed_session(tmp_path):
    engine = SessionEngine(_config(tmp_path), backend=SyntheticBackend(), mic_test_duration_s=0.2)
    summary = engine.run(resume=False, control_source=QueueControlSource())
    status = read_status(engine.session_dir)

    assert status["found"] is True
    assert (status["participant"], status["scenario"], status["session"]) == ("P01", "S01", "SESSION01")
    assert status["overall_total"] == summary.expected_trials
    assert status["overall_completed"] == summary.expected_trials
    assert status["overall_valid"] == summary.valid_trials
    assert status["is_complete"] is True
    assert status["validation_result"] == "PASS"
    # Nothing is being recorded once the session is complete.
    assert status["current_label"] is None
    assert status["remaining_s"] is None


class _ProbeDisplay(Display):
    """At the moment the participant is told to press a key, ask the
    dashboard what it is showing."""

    def __init__(self, session_dir):
        super().__init__(enabled=True, interactive=False)
        self.session_dir = session_dir
        self.samples = []

    def show(self, text):
        if "PRESS" in text:
            trial_id = int(text.split("Trial ")[1].split()[0])
            label = text.split("PRESS")[1].split()[0]
            status = read_status(self.session_dir)
            self.samples.append((trial_id, label, status["current_trial_id"], status["current_label"]))

    def line(self, text):
        pass


def test_dashboard_shows_the_trial_being_recorded_not_the_previous_one(tmp_path):
    engine = SessionEngine(_config(tmp_path), backend=SyntheticBackend(), mic_test_duration_s=0.2)
    probe = _ProbeDisplay(engine.session_dir)
    engine.run(resume=False, control_source=QueueControlSource(), display=probe)

    assert len(probe.samples) == engine._sched.total_trials
    for trial_id, label, dash_id, dash_label in probe.samples:
        assert (dash_id, dash_label) == (trial_id, label)
