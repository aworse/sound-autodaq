"""
Session orchestration: ties config, scheduler, recorder, trial state
machine, writer, quality, progress, and metadata together into the full
trial lifecycle (Part I §1.4, Part V, Part IX).

This module is the one place allowed to coordinate across those modules;
it does not itself own scheduling, audio capture, WAV encoding, or
quality measurement (REQ-57) — it calls into the modules that do.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from . import hardware, labels as labels_module, metadata, progress as progress_mod
from . import quality, scheduler, ui
from .config import Config
from .errors import PreflightError, ResumeError, ScheduleError
from .recorder import AudioBackend, ContinuousRecorder, SoundDeviceBackend
from .trial import (
    InputMode,
    Phase,
    Status,
    TrialClock,
    TrialDurations,
    TrialStateMachine,
    utc_now_iso,
)
from .writer import ManifestWriter, TrialRecord, read_wav, trial_filename, write_wav_atomic

SESSION_FILES = (
    "schedule.json",
    "session.json",
    "progress.json",
    "session_summary.json",
    "manifest.csv",
    "manifest.jsonl",
    "experiment.log",
)


def session_dir_for(config: Config) -> Path:
    return Path(config.output.root) / config.participant.id / config.scenario.id / config.session.id


def setup_logger(session_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"classism.recording.{id(session_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(session_dir / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class PreflightReport:
    def __init__(self):
        self.items: list = []  # list[(name, bool, str)]

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def render(self) -> str:
        lines = []
        for name, ok, detail in self.items:
            mark = "x" if ok else " "
            lines.append(f"[{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))
        return "\n".join(lines)


class SessionEngine:
    def __init__(self, config: Config, backend: Optional[AudioBackend] = None, mic_test_duration_s: float = 3.0):
        self.config = config
        self.backend = backend or SoundDeviceBackend()
        self.session_dir = session_dir_for(config)
        self.audio_dir = self.session_dir / "audio"
        self.classes, self.class_definition_version = labels_module.load_classes()
        self.num_classes = len(self.classes)
        self.mic_test_duration_s = mic_test_duration_s

    # -- schedule -----------------------------------------------------

    def build_or_load_schedule(self, resume: bool) -> scheduler.Schedule:
        schedule_path = self.session_dir / "schedule.json"
        seed = self.config.randomization.seed
        if seed is None:
            import random as _random

            seed = _random.SystemRandom().randrange(1, 2**31 - 1)

        if resume:
            if not schedule_path.exists():
                raise ResumeError(f"cannot resume: no schedule.json in {self.session_dir}")
            sched = scheduler.load_schedule(schedule_path)
            self._verify_resume_consistency(sched)
            return sched

        sched = scheduler.generate_schedule(
            classes=self.classes,
            repetitions_per_class=self.config.trial.repetitions_per_class,
            strategy=self.config.randomization.strategy,
            seed=seed,
            class_definition_version=self.class_definition_version,
            block_size=self.config.randomization.block_size,
            manual_order_file=self.config.randomization.manual_order_file,
        )
        return sched

    def _verify_resume_consistency(self, sched: scheduler.Schedule) -> None:
        session_json_path = self.session_dir / "session.json"
        if not session_json_path.exists():
            return
        prior = metadata.read_json(session_json_path)
        checks = [
            ("class_definition_version", prior.get("class_definition_version"), self.class_definition_version),
            ("sample_rate", prior.get("sample_rate"), self.config.recording.sample_rate),
            ("channels", prior.get("channels"), self.config.recording.channels),
            ("sample_format", prior.get("sample_format"), self.config.recording.format),
            ("randomization_strategy", prior.get("randomization_strategy"), self.config.randomization.strategy),
            ("random_seed", prior.get("random_seed"), sched.seed),
            ("repetitions_per_class", prior.get("repetitions_per_class"), self.config.trial.repetitions_per_class),
        ]
        for field, old, new in checks:
            if old != new:
                raise ResumeError(
                    f"resume refused: {field} differs (session.json has {old!r}, "
                    f"current configuration has {new!r})"
                )

    # -- preflight ------------------------------------------------------

    def preflight(self, resume: bool, dry_run: bool = False) -> PreflightReport:
        report = PreflightReport()
        cfg = self.config

        report.add("configuration schema valid", True)
        report.add("participant id set", bool(cfg.participant.id))
        report.add("scenario id set", bool(cfg.scenario.id))
        report.add("session id set", bool(cfg.session.id))

        try:
            self.classes, self.class_definition_version = labels_module.load_classes()
            report.add("class definition loaded, version recorded", True, self.class_definition_version)
        except Exception as exc:
            report.add("class definition loaded, version recorded", False, str(exc))
            return report

        expected_total = len(self.classes) * cfg.trial.repetitions_per_class
        report.add(
            "trial count valid and consistent with classes x repetitions",
            expected_total > 0,
            f"{len(self.classes)} classes x {cfg.trial.repetitions_per_class} reps = {expected_total}",
        )

        try:
            sched = self.build_or_load_schedule(resume=resume)
            report.add("class balance check passed", True)
            seed_ok = sched.seed is not None
            report.add("random seed set and recorded", seed_ok, str(sched.seed))
            if not dry_run:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                if not resume:
                    scheduler.write_schedule(sched, self.session_dir / "schedule.json")
            report.add("schedule generated (or loaded) and written to disk", True)
            self._schedule = sched
        except (ScheduleError, ResumeError) as exc:
            report.add("class balance check passed / schedule ready", False, str(exc))
            return report

        if dry_run:
            report.add("audio device detected", True, "skipped (dry run)")
            report.add("configured sample rate natively supported", True, "skipped (dry run)")
            report.add("configured channel count supported", True, "skipped (dry run)")
            report.add("microphone test passed", True, "skipped (dry run)")
        else:
            try:
                device_info = self.backend.resolve_device(cfg.recording.device)
                report.add("audio device detected", True, device_info.name)
                supported = self.backend.supports_sample_rate(
                    device_info, cfg.recording.sample_rate, cfg.recording.channels
                )
                report.add("configured sample rate natively supported", supported)
                report.add("configured channel count supported", supported)
                if not supported:
                    return report
            except Exception as exc:
                report.add("audio device detected", False, str(exc))
                return report

            try:
                mic_result = hardware.run_mic_test(
                    self.backend,
                    cfg.recording.device,
                    cfg.recording.sample_rate,
                    cfg.recording.channels,
                    duration_s=self.mic_test_duration_s,
                )
                self._mic_test_result = mic_result
                report.add("microphone test passed", mic_result.passed, "; ".join(mic_result.reasons))
                if not mic_result.passed:
                    return report
            except Exception as exc:
                report.add("microphone test passed", False, str(exc))
                return report

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / ".write_test").write_text("ok")
            (self.session_dir / ".write_test").unlink()
            report.add("output directory exists and is writable", True)
        except OSError as exc:
            report.add("output directory exists and is writable", False, str(exc))
            return report

        conflict = self.session_dir.exists() and any((self.session_dir / f).exists() for f in ("manifest.jsonl",)) and not resume
        report.add("no conflicting non-empty session directory (unless resuming)", not conflict)
        if conflict:
            return report

        required_bytes = hardware.estimate_bytes(
            total_trials=expected_total,
            stored_ms=cfg.trial.pre_roll_ms + cfg.trial.input_window_ms + cfg.trial.post_roll_ms,
            sample_rate=cfg.recording.sample_rate,
            channels=cfg.recording.channels,
        )
        disk_ok = dry_run or hardware.check_disk_space(cfg.output.root, required_bytes)
        report.add("estimated disk requirement satisfied", disk_ok, f"need ~{required_bytes} bytes")

        return report

    # -- main run -------------------------------------------------------

    def run(
        self,
        resume: bool = False,
        control_source: Optional[ui.ControlSource] = None,
        display: Optional[ui.Display] = None,
    ) -> metadata.SessionSummary:
        cfg = self.config
        report = self.preflight(resume=resume, dry_run=False)
        if not report.passed:
            raise PreflightError("pre-flight checklist failed:\n" + report.render())

        sched = self._schedule
        logger = setup_logger(self.session_dir)
        logger.info("session start uid=%s", cfg.session_uid)
        logger.info("configuration loaded: %s", cfg.resolved_dict())
        logger.info(
            "class definition loaded: version=%s count=%d", self.class_definition_version, self.num_classes
        )
        logger.info(
            "schedule ready: seed=%s strategy=%s count=%d", sched.seed, sched.strategy, sched.total_trials
        )

        progress_path = self.session_dir / "progress.json"
        session_json_path = self.session_dir / "session.json"
        started_utc = utc_now_iso()

        prog = progress_mod.read_progress(progress_path) if resume else None
        if prog is None:
            prog = progress_mod.new_progress(cfg.session_uid, "schedule.json", sched.seed, sched.total_trials)

        recorder = ContinuousRecorder(
            self.backend, cfg.recording.device, cfg.recording.sample_rate, cfg.recording.channels
        )
        recorder.start()
        logger.info("audio device initialized: %s", recorder.device_info)

        session_meta = metadata.build_session_metadata(
            config=cfg,
            class_definition_version=self.class_definition_version,
            num_classes=self.num_classes,
            total_trials=sched.total_trials,
            device_info_dict={
                "audio_device_name": recorder.device_info.name,
                "audio_device_index": recorder.device_info.index,
                "host_api": recorder.device_info.host_api,
                "sample_rate": cfg.recording.sample_rate,
                "channel_count": cfg.recording.channels,
                "input_latency_s": recorder.device_info.input_latency_s,
            },
            keyboard_dict=cfg.hardware.keyboard.__dict__,
            microphone_dict=cfg.hardware.microphone.__dict__,
            environment_dict=cfg.environment.__dict__,
            started_utc=started_utc,
            mic_test_result=getattr(self, "_mic_test_result", None) and self._mic_test_result.to_dict(),
        )
        if resume and session_json_path.exists():
            prior = metadata.read_json(session_json_path)
            resumes = prior.get("resumes", [])
            resumes.append({"resumed_utc": utc_now_iso(), "git_commit": session_meta["git_commit"]})
            session_meta["resumes"] = resumes
            session_meta["started_utc"] = prior.get("started_utc", started_utc)
            session_meta["breaks"] = prior.get("breaks", [])
        metadata.write_json_atomic(session_json_path, session_meta)

        control_source = control_source or ui.QueueControlSource()
        display = display or ui.Display(enabled=False)

        durations = TrialDurations(
            countdown_ms=cfg.trial.countdown_ms,
            pre_roll_ms=cfg.trial.pre_roll_ms,
            input_window_ms=cfg.trial.input_window_ms,
            post_roll_ms=cfg.trial.post_roll_ms,
            inter_trial_ms=cfg.trial.inter_trial_ms,
        )
        input_mode = InputMode(cfg.input.mode)

        run_start = time.monotonic()
        consecutive_failures = 0
        total_break_s = 0.0
        quit_requested = False

        pending = [t for t in sched.trials if t.trial_id >= prog.next_trial_id]
        next_extra_trial_id = sched.total_trials + 1
        class_valid_counts: dict = {}
        for r in self._read_all_records():
            if r["status"] == Status.VALID.value:
                class_valid_counts[r["label"]] = class_valid_counts.get(r["label"], 0) + 1

        manifest = ManifestWriter(self.session_dir)
        try:
            idx = 0
            while idx < len(pending) and not quit_requested:
                scheduled = pending[idx]
                idx += 1
                if manifest.has_trial(scheduled.trial_id):
                    continue

                control = control_source.poll()
                if control == "quit":
                    quit_requested = True
                    break
                if control == "pause":
                    logger.info("pause requested before trial %d", scheduled.trial_id)
                    while True:
                        c = control_source.poll()
                        if c == "pause":
                            break
                        if c == "quit":
                            quit_requested = True
                            break
                        time.sleep(0.01)
                    logger.info("resumed")
                    if quit_requested:
                        break
                    control = None

                status = Status.VALID
                notes = None
                wav_relpath = None
                metrics = quality.QualityMetrics(peak=0.0, rms=0.0, clipping_ratio=0.0)
                overflow = False
                num_samples = 0
                segment_start = recorder.frames_captured
                segment_end = segment_start
                result = None

                if control == "skip":
                    status = Status.SKIPPED
                    result = _interrupted_result(scheduled, input_mode, run_start)
                elif control == "invalid":
                    status = Status.OPERATOR_MARKED_INVALID
                    result = _interrupted_result(scheduled, input_mode, run_start)
                else:
                    machine = TrialStateMachine(
                        durations, session_start_ns=run_start_ns(run_start), clock=TrialClock()
                    )
                    sample_offsets = {}

                    def on_phase_change(phase: Phase, _offsets=sample_offsets, _rec=recorder) -> None:
                        if phase in (Phase.PRE_ROLL, Phase.SAVE):
                            _offsets[phase] = _rec.frames_captured

                    logger.info(
                        "trial start id=%d label=%s rep=%d", scheduled.trial_id, scheduled.label, scheduled.repetition
                    )
                    result = machine.run(
                        trial_id=scheduled.trial_id,
                        label=scheduled.label,
                        repetition=scheduled.repetition,
                        input_mode=input_mode,
                        on_phase_change=on_phase_change,
                    )
                    segment_start = sample_offsets.get(Phase.PRE_ROLL, recorder.frames_captured)
                    segment_end = sample_offsets.get(Phase.SAVE, recorder.frames_captured)
                    num_samples = max(0, segment_end - segment_start)

                if status == Status.VALID:
                    try:
                        recorder.wait_until(segment_end, timeout_s=5)
                        segment = recorder.buffer.read_segment(segment_start, segment_end)
                        overflow = recorder.buffer.overlaps_overflow(segment_start, segment_end)
                        metrics = quality.measure(segment)

                        filename = trial_filename(scheduled.trial_id)
                        wav_path = self.audio_dir / filename
                        write_wav_atomic(wav_path, segment, cfg.recording.sample_rate, cfg.recording.channels)

                        written_samples, written_rate, written_channels = read_wav(wav_path)
                        integrity_ok = (
                            wav_path.exists()
                            and wav_path.stat().st_size > 0
                            and written_rate == cfg.recording.sample_rate
                            and written_channels == cfg.recording.channels
                            and abs(len(written_samples) - num_samples) <= 1
                        )
                        if not integrity_ok:
                            status = Status.CORRUPTED
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0
                            wav_relpath = f"audio/{filename}"
                            if overflow:
                                status = Status.AUDIO_OVERFLOW
                            elif quality.is_suspicious_silence(metrics, cfg.quality.silence_rms_threshold):
                                status = Status.SUSPICIOUS_SILENCE
                    except Exception as exc:
                        logger.error("trial %d failed: %s", scheduled.trial_id, exc)
                        status = Status.CORRUPTED
                        notes = str(exc)
                        consecutive_failures += 1

                record = TrialRecord(
                    trial_id=scheduled.trial_id,
                    file=wav_relpath,
                    label=scheduled.label,
                    scheduled_label=scheduled.label,
                    observed_label=None,
                    participant=cfg.participant.id,
                    scenario=cfg.scenario.id,
                    session=cfg.session.id,
                    repetition=scheduled.repetition,
                    status=status.value,
                    input_mode=input_mode.value,
                    sample_rate=cfg.recording.sample_rate,
                    channels=cfg.recording.channels,
                    sample_format=cfg.recording.format,
                    num_samples=num_samples,
                    duration_ms=1000.0 * num_samples / cfg.recording.sample_rate if cfg.recording.sample_rate else 0.0,
                    segment_start_sample=segment_start,
                    segment_end_sample=segment_end,
                    trial_start_ns=result.timing.trial_start_ns,
                    input_expected_ns=result.timing.input_expected_ns,
                    input_detected_ns=result.timing.input_detected_ns,
                    trial_end_ns=result.timing.trial_end_ns,
                    wall_clock_utc=result.timing.wall_clock_utc,
                    peak=metrics.peak,
                    rms=metrics.rms,
                    clipping_ratio=metrics.clipping_ratio,
                    overflow=overflow,
                    notes=notes,
                )
                manifest.append(record)
                logger.info("trial complete id=%d status=%s", scheduled.trial_id, status.value)
                if status == Status.VALID:
                    class_valid_counts[scheduled.label] = class_valid_counts.get(scheduled.label, 0) + 1

                prog = progress_mod.advance(prog, scheduled.trial_id)
                if prog.completed_trials % cfg.output.progress_flush_every == 0:
                    progress_mod.write_progress(progress_path, prog)

                if status != Status.VALID and cfg.controls.repeat_on_invalid and status not in (
                    Status.SKIPPED,
                ):
                    pending.append(
                        scheduler.ScheduledTrial(
                            trial_id=next_extra_trial_id, label=scheduled.label, repetition=scheduled.repetition
                        )
                    )
                    next_extra_trial_id += 1

                if consecutive_failures >= cfg.quality.max_consecutive_failures:
                    logger.error("safe stop: %d consecutive integrity failures", consecutive_failures)
                    quit_requested = True

                if durations.inter_trial_ms > 0:
                    time.sleep(durations.inter_trial_ms / 1000.0)

                if (
                    cfg.break_.enabled
                    and prog.completed_trials % cfg.break_.every_trials == 0
                    and idx < len(pending)
                    and not quit_requested
                ):
                    logger.info("automatic break before trial %d", prog.next_trial_id)
                    break_start = time.monotonic()
                    time.sleep(0)  # breaks are logged instantly in headless runs; real UI may extend this
                    break_elapsed = time.monotonic() - break_start
                    total_break_s += break_elapsed
                    session_meta.setdefault("breaks", []).append(
                        {
                            "break_before_trial": prog.next_trial_id,
                            "break_duration_s": cfg.break_.duration_seconds,
                            "break_type": "automatic",
                        }
                    )

                display.show(
                    ui.render_trial_screen(
                        participant=cfg.participant.id,
                        scenario=cfg.scenario.id,
                        session=cfg.session.id,
                        overall_completed=prog.completed_trials,
                        overall_total=sched.total_trials,
                        current_label=scheduled.label,
                        class_completed=class_valid_counts.get(scheduled.label, 0),
                        class_total=cfg.trial.repetitions_per_class,
                        next_label=pending[idx].label if idx < len(pending) else None,
                        status=status.value,
                        elapsed_s=time.monotonic() - run_start,
                        remaining_s=None,
                    )
                )

            progress_mod.write_progress(progress_path, prog)
        finally:
            recorder.stop()
            manifest.close()

        session_meta["ended_utc"] = utc_now_iso()
        metadata.write_json_atomic(session_json_path, session_meta)

        records = self._read_all_records()
        summary = metadata.build_summary(
            session_uid=cfg.session_uid,
            expected_trials=sched.total_trials,
            records=records,
            duration_s=time.monotonic() - run_start,
            total_break_s=total_break_s,
        )
        if not quit_requested:
            from .validator import validate_session

            try:
                report_v = validate_session(self.session_dir)
                summary.validated = True
                summary.validation_result = "PASS" if report_v.passed else "FAIL"
            except Exception as exc:
                summary.validated = False
                summary.validation_result = f"ERROR: {exc}"

        metadata.write_json_atomic(self.session_dir / "session_summary.json", summary.to_dict())
        logger.info("session end uid=%s completed=%d", cfg.session_uid, prog.completed_trials)
        return summary

    def _read_all_records(self) -> list:
        from .writer import read_manifest_jsonl

        return read_manifest_jsonl(self.session_dir / "manifest.jsonl")


def run_start_ns(run_start_monotonic: float) -> int:
    return int(run_start_monotonic * 1e9)


def _interrupted_result(scheduled, input_mode, run_start: float):
    """A minimal TrialResult for a trial that was skipped/invalidated
    before its audio phases ran (REQ-28.4: still produces a metadata row,
    with `file` left null)."""
    from .trial import TrialResult, TrialTiming

    now_ns = int(time.monotonic() * 1e9)
    timing = TrialTiming(
        session_start_ns=run_start_ns(run_start),
        trial_start_ns=now_ns,
        input_expected_ns=now_ns,
        input_detected_ns=None,
        trial_end_ns=now_ns,
        wall_clock_utc=utc_now_iso(),
    )
    return TrialResult(
        trial_id=scheduled.trial_id,
        scheduled_label=scheduled.label,
        repetition=scheduled.repetition,
        input_mode=input_mode,
        timing=timing,
    )
