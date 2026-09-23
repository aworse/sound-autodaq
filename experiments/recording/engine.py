"""
Session orchestration: ties config, scheduler, recorder, trial state
machine, writer, quality, progress, and metadata together into the full
trial lifecycle (Part I §1.4, Part V, Part IX).

This module coordinates across those modules; it does not itself own
scheduling, audio capture, WAV encoding, or quality measurement (REQ-57).
"""

from __future__ import annotations

import datetime
import errno
import logging
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Optional

from . import hardware, labels as labels_module, metadata, progress as progress_mod
from . import quality, scheduler, ui
from .config import Config
from .errors import (
    AudioStreamError,
    DiskSpaceError,
    DuplicateTrialError,
    PreflightError,
    RecorderError,
    ResumeError,
    ScheduleError,
)
from .progress import REQUEUE_END, REQUEUE_IMMEDIATE
from .recorder import AudioBackend, ContinuousRecorder, SoundDeviceBackend
from .scheduler import ScheduledTrial
from .trial import (
    InputMode,
    Phase,
    Status,
    TrialClock,
    TrialDurations,
    TrialStateMachine,
    utc_now_iso,
)
from .writer import (
    ManifestWriter,
    TrialRecord,
    read_manifest_jsonl,
    read_wav,
    trial_filename,
    wav_params,
    write_wav_atomic,
)

# Failures of the capture itself (not operator decisions); this many in a
# row triggers a safe stop (REQ-40.4), which also bounds re-queueing.
SYSTEM_FAILURES = {Status.CORRUPTED, Status.AUDIO_OVERFLOW, Status.SUSPICIOUS_SILENCE}

# A trial that captured less than this fraction of its configured audio
# means the stream stopped delivering samples (device stall/unplug).
STALL_FRACTION = 0.5

_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}


def session_dir_for(config: Config) -> Path:
    return Path(config.output.root) / config.participant.id / config.scenario.id / config.session.id


class _IsoFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


def setup_logger(session_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"classism.recording.{Path(session_dir).resolve()}")
    logger.setLevel(logging.INFO)
    close_logger(logger)
    handler = logging.FileHandler(Path(session_dir) / "experiment.log", encoding="utf-8")
    handler.setFormatter(_IsoFormatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


class PreflightReport:
    def __init__(self):
        self.items: list = []  # list[(name, ok, detail)]

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def render(self) -> str:
        return "\n".join(
            f"[{'x' if ok else ' '}] {name}" + (f" — {detail}" if detail else "")
            for name, ok, detail in self.items
        )


class SessionEngine:
    def __init__(self, config: Config, backend: Optional[AudioBackend] = None, mic_test_duration_s: float = 3.0):
        self.config = config
        self.backend = backend or SoundDeviceBackend()
        self.session_dir = session_dir_for(config)
        self.audio_dir = self.session_dir / "audio"
        self.classes, self.class_definition_version = labels_module.load_classes()
        self.num_classes = len(self.classes)
        self.mic_test_duration_s = mic_test_duration_s
        # REQ-11.2: never run with an unrecorded seed. A null seed is
        # replaced here, once, and this value is what every file records.
        seed = config.randomization.seed
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
        self.seed_generated = seed is None
        self._schedule: Optional[scheduler.Schedule] = None
        self._mic_test_result = None

    # -- schedule -----------------------------------------------------

    def build_or_load_schedule(self, resume: bool) -> scheduler.Schedule:
        schedule_path = self.session_dir / "schedule.json"
        if resume:
            if not schedule_path.exists():
                raise ResumeError(f"cannot resume: no schedule.json in {self.session_dir}")
            sched = scheduler.load_schedule(schedule_path)
            self._verify_resume_consistency(sched)
            self.seed = sched.seed
            self.seed_generated = False
            return sched

        return scheduler.generate_schedule(
            classes=self.classes,
            repetitions_per_class=self.config.trial.repetitions_per_class,
            strategy=self.config.randomization.strategy,
            seed=self.seed,
            class_definition_version=self.class_definition_version,
            block_size=self.config.randomization.block_size,
            manual_order_file=self.config.randomization.manual_order_file,
        )

    def _verify_resume_consistency(self, sched: scheduler.Schedule) -> None:
        cfg = self.config
        checks = [
            ("schedule.json class_definition_version", sched.class_definition_version, self.class_definition_version),
            ("schedule.json repetitions_per_class", sched.repetitions_per_class, cfg.trial.repetitions_per_class),
            ("schedule.json strategy", sched.strategy, cfg.randomization.strategy),
        ]
        if cfg.randomization.seed is not None:
            checks.append(("schedule.json seed", sched.seed, cfg.randomization.seed))
        session_json_path = self.session_dir / "session.json"
        if session_json_path.exists():
            prior = metadata.read_json(session_json_path)
            checks += [
                ("class_definition_version", prior.get("class_definition_version"), self.class_definition_version),
                ("sample_rate", prior.get("sample_rate"), cfg.recording.sample_rate),
                ("channels", prior.get("channels"), cfg.recording.channels),
                ("sample_format", prior.get("sample_format"), cfg.recording.format),
                ("input_mode", prior.get("input_mode"), cfg.input.mode),
                ("randomization_strategy", prior.get("randomization_strategy"), cfg.randomization.strategy),
                ("random_seed", prior.get("random_seed"), sched.seed),
                ("repetitions_per_class", prior.get("repetitions_per_class"), cfg.trial.repetitions_per_class),
            ]
            for key in ("countdown_ms", "pre_roll_ms", "input_window_ms", "post_roll_ms", "inter_trial_ms"):
                checks.append((key, prior.get(key), getattr(cfg.trial, key)))
        for field, old, new in checks:
            if old != new:
                raise ResumeError(
                    f"resume refused: {field} differs (on disk {old!r}, current configuration {new!r}); "
                    "continuing under different acquisition parameters in one session is not allowed (REQ-48.4)"
                )
        try:
            scheduler.validate_balance(sched, self.classes, cfg.trial.repetitions_per_class)
        except ScheduleError as exc:
            raise ResumeError(f"resume refused: schedule.json no longer matches the class definition: {exc}") from exc

    # -- preflight ------------------------------------------------------

    def preflight(self, resume: bool, dry_run: bool = False) -> PreflightReport:
        """Pre-flight checklist (§53). Nothing is written until every
        other item has passed, so a failed check never leaves a
        half-initialised session directory behind; a dry run writes
        nothing at all (REQ-59.1)."""
        report = PreflightReport()
        cfg = self.config

        report.add("configuration schema valid", True)
        report.add("participant id set", bool(cfg.participant.id))
        report.add("scenario id set", bool(cfg.scenario.id))
        report.add("session id set", bool(cfg.session.id))
        if cfg.input.mode != InputMode.HUMAN.value:
            report.add(
                "input mode supported",
                False,
                f"input.mode={cfg.input.mode!r} is not implemented (this recorder does not synthesize "
                "OS key events); use 'human'",
            )
            return report

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

        state = progress_mod.session_dir_state(self.session_dir)
        if resume:
            ok = state == "incomplete"
            detail = {"absent": "nothing to resume here", "complete": "session is already complete"}.get(state, "")
            report.add("incomplete session present to resume", ok, detail or str(self.session_dir))
        else:
            ok = state == "absent"
            report.add(
                "no conflicting non-empty session directory (unless resuming)",
                ok,
                "" if ok else f"{self.session_dir} already holds a {state} session; resume it or choose a new session.id",
            )
        if not ok:
            return report

        try:
            sched = self.build_or_load_schedule(resume=resume)
        except (ScheduleError, ResumeError) as exc:
            report.add("class balance check passed / schedule ready", False, str(exc))
            return report
        report.add("class balance check passed", True)
        seed_detail = f"{sched.seed} (generated, config had none)" if self.seed_generated else str(sched.seed)
        report.add("random seed set and recorded", True, seed_detail)

        if dry_run:
            parent = hardware.nearest_existing_dir(self.session_dir)
            writable = os.access(parent, os.W_OK)
            report.add("output directory exists and is writable", writable, f"checked {parent} (dry run: not created)")
        else:
            try:
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                probe = self.session_dir / ".write_test"
                probe.write_text("ok")
                probe.unlink()
                report.add("output directory exists and is writable", True)
            except OSError as exc:
                report.add("output directory exists and is writable", False, str(exc))
                return report

        if dry_run:
            for item in (
                "audio device detected",
                "configured sample rate natively supported",
                "configured channel count supported",
                "microphone test passed",
            ):
                report.add(item, True, "skipped (dry run)")
        else:
            try:
                device_info = self.backend.resolve_device(cfg.recording.device)
                report.add("audio device detected", True, device_info.name)
                supported = self.backend.supports_sample_rate(
                    device_info, cfg.recording.sample_rate, cfg.recording.channels
                )
                detail = "" if supported else (
                    f"{cfg.recording.sample_rate} Hz / {cfg.recording.channels} ch not natively supported by "
                    f"{device_info.name!r}; resampling is prohibited"
                )
                report.add("configured sample rate natively supported", supported, detail)
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

        required_bytes = hardware.estimate_bytes(
            total_trials=expected_total,
            stored_ms=cfg.trial.pre_roll_ms + cfg.trial.input_window_ms + cfg.trial.post_roll_ms,
            sample_rate=cfg.recording.sample_rate,
            channels=cfg.recording.channels,
        )
        free = hardware.free_bytes(self.session_dir)
        report.add(
            "estimated disk requirement satisfied",
            free >= required_bytes,
            f"need ~{required_bytes / 1e9:.2f} GB, {free / 1e9:.2f} GB free",
        )
        if free < required_bytes:
            return report

        if not dry_run and not resume:
            try:
                scheduler.write_schedule(sched, self.session_dir / "schedule.json")
            except OSError as exc:
                report.add("schedule generated and written to disk", False, str(exc))
                return report
        report.add(
            "schedule generated (or loaded, on resume) and written to disk",
            True,
            "dry run: not written" if dry_run else "",
        )
        self._schedule = sched
        return report

    # -- main run -------------------------------------------------------

    def run(
        self,
        resume: bool = False,
        control_source: Optional[ui.ControlSource] = None,
        display: Optional[ui.Display] = None,
    ) -> metadata.SessionSummary:
        cfg = self.config
        self._controls = control_source or ui.QueueControlSource()
        self._display = display or ui.Display(enabled=False)

        report = self.preflight(resume=resume, dry_run=False)
        self._display.line(report.render())
        if not report.passed:
            raise PreflightError("pre-flight checklist failed:\n" + report.render())

        sched = self._schedule
        self._sched = sched
        log = self._log = setup_logger(self.session_dir)
        log.info("session %s uid=%s", "resume" if resume else "start", cfg.session_uid)
        log.info("configuration loaded (resolved): %s", cfg.resolved_dict())
        log.info("class definition loaded: version=%s count=%d", self.class_definition_version, self.num_classes)
        log.info("schedule %s: seed=%s strategy=%s count=%d",
                 "loaded" if resume else "generated", sched.seed, sched.strategy, sched.total_trials)
        for line in report.render().splitlines():
            log.info("pre-flight %s", line)
        if self._mic_test_result is not None:
            log.info("microphone test: %s", self._mic_test_result.to_dict())

        self._meta = self._write_initial_session_json(resume)

        manifest = ManifestWriter(self.session_dir)
        for torn in manifest.torn_tails:
            log.warning("manifest had a torn final line from an abrupt stop; set aside in %s", torn.name)
        rows = read_manifest_jsonl(self.session_dir / "manifest.jsonl")
        self._rows_count = len(rows)
        self._class_valid: dict = {}
        for r in rows:
            if r["status"] == Status.VALID.value:
                self._class_valid[r["label"]] = self._class_valid.get(r["label"], 0) + 1
        queue, self._next_id = progress_mod.reconstruct_queue(sched, rows)
        self._queue = queue

        self._progress_path = self.session_dir / "progress.json"
        self._progress = progress_mod.Progress(
            session_uid=cfg.session_uid,
            schedule_file="schedule.json",
            random_seed=sched.seed,
            total_trials=sched.total_trials,
            completed_trials=len(rows),
            last_trial_id=max((r["trial_id"] for r in rows), default=None),
            next_trial_id=queue[0].trial_id if queue else None,
            updated_utc=utc_now_iso(),
        )
        self._save_progress(progress_mod.STATE_RUNNING)

        self._durations = TrialDurations(
            countdown_ms=cfg.trial.countdown_ms,
            pre_roll_ms=cfg.trial.pre_roll_ms,
            input_window_ms=cfg.trial.input_window_ms,
            post_roll_ms=cfg.trial.post_roll_ms,
            inter_trial_ms=cfg.trial.inter_trial_ms,
        )
        self._trial_seconds: list = []
        self._total_break_s = 0.0
        self._consecutive_failures = 0
        self._notice: Optional[str] = None
        self._sigint = False
        self._run_start = time.monotonic()

        self._recorder = None
        prev_sigint = None
        if threading.current_thread() is threading.main_thread():
            prev_sigint = signal.signal(signal.SIGINT, self._on_sigint)

        stop_reason: Optional[str] = None
        error: Optional[BaseException] = None
        summary = None
        try:
            self._recorder = ContinuousRecorder(
                self.backend, cfg.recording.device, cfg.recording.sample_rate, cfg.recording.channels
            )
            self._recorder.start()
            log.info("audio stream opened: %s", self._recorder.device_info)
            while self._queue and stop_reason is None:
                stop_reason = self._run_trial(self._queue.pop(0), manifest)
        except RecorderError as exc:
            error, stop_reason = exc, f"error: {exc}"
            log.exception("stopping on error")
        except OSError as exc:
            error = (
                DiskSpaceError(f"disk full while writing {getattr(exc, 'filename', '')}: {exc}")
                if exc.errno in _DISK_FULL_ERRNOS
                else RecorderError(f"I/O error: {exc}")
            )
            stop_reason = f"error: {error}"
            log.exception("stopping on I/O error")
        except Exception as exc:
            error, stop_reason = exc, f"error: {exc!r}"
            log.exception("stopping on unexpected exception")
        finally:
            # Safe-stop sequence (REQ-29.3): the current trial has already
            # been written; flush, record, then close the stream last.
            try:
                log.info("shutdown: closing manifest")
                manifest.close()
                summary = self._finalize(stop_reason)
            finally:
                log.info("shutdown: closing audio stream")
                if self._recorder is not None:
                    self._recorder.stop()
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
                log.info("session end uid=%s completed=%s stop_reason=%s",
                         cfg.session_uid, summary.completed if summary else None, stop_reason)
                close_logger(log)

        if error is not None:
            if isinstance(error, RecorderError):
                raise error
            raise RecorderError(f"session stopped on an unexpected error: {error!r}") from error
        return summary

    # -- one trial --------------------------------------------------------

    def _run_trial(self, t: ScheduledTrial, manifest: ManifestWriter) -> Optional[str]:
        """Record one trial end to end and append its manifest row.
        Returns a stop reason, or None to continue."""
        cfg = self.config
        log = self._log

        if (self.audio_dir / trial_filename(t.trial_id)).exists():
            self._handle_orphan_wav(t, manifest)
            return None

        trial_wall_start = time.monotonic()
        offsets: dict = {}

        def on_phase_change(phase: Phase) -> None:
            if phase in (Phase.PRE_ROLL, Phase.SAVE):
                offsets[phase] = self._recorder.frames_captured
            text = {
                Phase.PREPARE: "READY",
                Phase.PRE_ROLL: "get ready...",
                Phase.INPUT_WINDOW: f"PRESS   {t.label}   NOW",
                Phase.POST_ROLL: "recording... hold still",
                Phase.SAVE: "saved",
            }.get(phase)
            if text:
                self._render(t, text)

        def on_countdown(seconds_left: int) -> None:
            self._render(t, f"{seconds_left}")

        log.info("trial start id=%d label=%s rep=%d", t.trial_id, t.label, t.repetition)
        machine = TrialStateMachine(self._durations, session_start_ns=int(self._run_start * 1e9), clock=TrialClock())
        result = machine.run(
            trial_id=t.trial_id,
            label=t.label,
            repetition=t.repetition,
            input_mode=InputMode(cfg.input.mode),
            on_phase_change=on_phase_change,
            on_countdown=on_countdown,
        )
        # The inter-trial gap belongs to this trial's key window: a key
        # pressed right after the keystroke still applies to this trial.
        if self._queue:
            self._render(t, f"saved — next: {self._queue[0].label}")
        if self._durations.inter_trial_ms > 0:
            time.sleep(self._durations.inter_trial_ms / 1000.0)
        controls = self._drain_controls()

        start = offsets.get(Phase.PRE_ROLL, self._recorder.frames_captured)
        end = offsets.get(Phase.SAVE, self._recorder.frames_captured)
        num_samples = max(0, end - start)
        expected = self._durations.stored_ms * cfg.recording.sample_rate / 1000.0
        stalled = num_samples < STALL_FRACTION * expected

        status = Status.VALID
        notes: list = []
        wav_rel: Optional[str] = None
        metrics = quality.QualityMetrics(peak=0.0, rms=0.0, clipping_ratio=0.0)
        overflow = False
        integrity_detail = None

        if num_samples > 0:
            segment = self._recorder.buffer.read_segment(start, end)
            overflow = self._recorder.buffer.overlaps_overflow(start, end)
            metrics = quality.measure(segment)
            wav_path = self.audio_dir / trial_filename(t.trial_id)
            try:
                write_wav_atomic(wav_path, segment, cfg.recording.sample_rate, cfg.recording.channels)
            except OSError as exc:
                if exc.errno in _DISK_FULL_ERRNOS:
                    raise DiskSpaceError(f"disk full writing {wav_path}: {exc}") from exc
                integrity_detail = f"WAV write failed: {exc}"
            if integrity_detail is None:
                wav_rel = f"audio/{wav_path.name}"
                integrity_detail = self._integrity_problem(wav_path, num_samples)

        backend_error = self.backend.error
        if stalled or backend_error:
            status = Status.INTERRUPTED
            notes.append(backend_error or f"audio stream stalled: captured {num_samples} of ~{int(expected)} samples")
        elif integrity_detail:
            status = Status.CORRUPTED
            notes.append(integrity_detail)
            log.error("integrity check failed for trial %d: %s", t.trial_id, integrity_detail)
        elif overflow:
            status = Status.AUDIO_OVERFLOW
            self._notice = f"audio overflow on trial {t.trial_id}; it will be recorded again"
            log.warning("overflow in trial %d", t.trial_id)
        elif quality.is_suspicious_silence(metrics, cfg.quality.silence_rms_threshold):
            status = Status.SUSPICIOUS_SILENCE
            self._notice = f"trial {t.trial_id} was silent (rms {metrics.rms:.5f}); it will be recorded again"
            log.warning("suspicious silence in trial %d rms=%.6f", t.trial_id, metrics.rms)
        if metrics.clipping_ratio > cfg.quality.clipping_threshold:
            self._notice = f"clipping {metrics.clipping_ratio:.4f} on trial {t.trial_id} — check input gain"
            log.warning("clipping ratio %.6f on trial %d", metrics.clipping_ratio, t.trial_id)

        op, pause, quit_ = self._interpret_controls(controls, t)
        requeue: Optional[str] = None
        stop_reason: Optional[str] = None

        if status in (Status.INTERRUPTED, Status.CORRUPTED):
            if op:
                notes.append(f"operator pressed {op} (overridden by {status.value})")
        elif op == "skip":
            status = Status.SKIPPED
        elif op in ("invalid", "repeat"):
            status = Status.OPERATOR_MARKED_INVALID
            notes.append("operator requested repeat" if op == "repeat" else "operator marked invalid")
        elif pause and cfg.controls.resume_policy == "discard_current":
            status = Status.INTERRUPTED
            notes.append("discarded by operator pause (resume_policy=discard_current)")

        if status in (Status.VALID, Status.SKIPPED):
            requeue = None
        elif status == Status.INTERRUPTED or op == "repeat":
            requeue = REQUEUE_IMMEDIATE
        elif status == Status.CORRUPTED:
            decision = self._integrity_decision(t, integrity_detail)
            notes.append(f"operator decision after integrity failure: {decision}")
            if decision == "retry":
                requeue = REQUEUE_IMMEDIATE
            elif decision == "continue":
                requeue = REQUEUE_END if cfg.controls.repeat_on_invalid else None
            else:
                requeue = REQUEUE_IMMEDIATE
                stop_reason = "stopped after integrity failure"
        elif cfg.controls.repeat_on_invalid:
            requeue = REQUEUE_END

        superseded_by = None
        if requeue:
            superseded_by = self._next_id
            self._next_id += 1
            retry = ScheduledTrial(trial_id=superseded_by, label=t.label, repetition=t.repetition)
            if requeue == REQUEUE_IMMEDIATE:
                self._queue.insert(0, retry)
            else:
                self._queue.append(retry)

        record = TrialRecord(
            trial_id=t.trial_id,
            file=wav_rel,
            label=t.label,
            scheduled_label=t.label,
            observed_label=None,
            participant=cfg.participant.id,
            scenario=cfg.scenario.id,
            session=cfg.session.id,
            repetition=t.repetition,
            status=status.value,
            input_mode=cfg.input.mode,
            sample_rate=cfg.recording.sample_rate,
            channels=cfg.recording.channels,
            sample_format=cfg.recording.format,
            num_samples=num_samples,
            duration_ms=1000.0 * num_samples / cfg.recording.sample_rate,
            segment_start_sample=start,
            segment_end_sample=end,
            trial_start_ns=result.timing.trial_start_ns,
            input_expected_ns=result.timing.input_expected_ns,
            input_detected_ns=result.timing.input_detected_ns,
            trial_end_ns=result.timing.trial_end_ns,
            wall_clock_utc=result.timing.wall_clock_utc,
            peak=metrics.peak,
            rms=metrics.rms,
            clipping_ratio=metrics.clipping_ratio,
            overflow=overflow,
            notes="; ".join(notes) or None,
            superseded_by=superseded_by,
            requeue=requeue,
        )
        self._append(manifest, record)
        log.info("trial complete id=%d status=%s%s", t.trial_id, status.value,
                 f" requeued as {superseded_by} ({requeue})" if superseded_by else "")
        self._trial_seconds.append(time.monotonic() - trial_wall_start)

        if status in SYSTEM_FAILURES:
            self._consecutive_failures += 1
        elif status == Status.VALID:
            self._consecutive_failures = 0

        if stalled or backend_error:
            raise AudioStreamError(
                f"{notes[0]} during trial {t.trial_id}; check the microphone connection, then resume the session"
            )
        if stop_reason:
            return stop_reason
        if self._consecutive_failures >= cfg.quality.max_consecutive_failures:
            reason = (
                f"{self._consecutive_failures} consecutive quality/integrity failures "
                f"(last: {status.value}); check the microphone and resume"
            )
            log.error("safe stop: %s", reason)
            return reason
        if quit_:
            return "operator quit (Ctrl+C)" if self._sigint else "operator quit"
        if pause:
            reason = self._pause()
            if reason:
                return reason
        if (
            cfg.break_.enabled
            and self._queue
            and self._rows_count % cfg.break_.every_trials == 0
        ):
            return self._automatic_break()
        return None

    def _integrity_problem(self, wav_path: Path, num_samples: int) -> Optional[str]:
        """REQ-40.1 post-write check. Returns a description of the first
        problem found, or None."""
        cfg = self.config
        try:
            if not wav_path.exists():
                return "file missing after write"
            if wav_path.stat().st_size <= 0:
                return "file is empty"
            sr, ch, width, frames = wav_params(wav_path)
            read_wav(wav_path)
        except Exception as exc:
            return f"not readable as WAV: {exc}"
        if sr != cfg.recording.sample_rate:
            return f"sample rate {sr} != {cfg.recording.sample_rate}"
        if ch != cfg.recording.channels:
            return f"channels {ch} != {cfg.recording.channels}"
        if width != 2:
            return f"sample width {width * 8} bit != 16 bit"
        if abs(frames - num_samples) > 1:
            return f"sample count {frames} != {num_samples}"
        return None

    def _handle_orphan_wav(self, t: ScheduledTrial, manifest: ManifestWriter) -> None:
        """audio/trial_N.wav exists but trial N has no manifest row: the
        process died between the WAV rename and the manifest append.
        Never overwrite it (REQ-49.2); apply output.duplicate_policy."""
        cfg = self.config
        path = self.audio_dir / trial_filename(t.trial_id)
        if cfg.output.duplicate_policy == "error":
            self._log.error("duplicate trial file %s with no manifest row; duplicate_policy=error", path)
            raise DuplicateTrialError(
                f"{path} already exists but trial {t.trial_id} has no manifest row (the previous run most likely "
                "stopped between writing the WAV and appending the manifest). It will not be overwritten. Set "
                "output.duplicate_policy: new_id to keep it as an 'interrupted' row and re-record the pair under "
                "a new trial id."
            )
        new_id = self._next_id
        self._next_id += 1
        try:
            samples, _, _ = read_wav(path)
            metrics = quality.measure(samples)
            n = int(samples.shape[0])
        except Exception:
            metrics, n = None, 0
        record = TrialRecord(
            trial_id=t.trial_id,
            file=f"audio/{path.name}",
            label=t.label,
            scheduled_label=t.label,
            observed_label=None,
            participant=cfg.participant.id,
            scenario=cfg.scenario.id,
            session=cfg.session.id,
            repetition=t.repetition,
            status=Status.INTERRUPTED.value,
            input_mode=cfg.input.mode,
            sample_rate=cfg.recording.sample_rate,
            channels=cfg.recording.channels,
            sample_format=cfg.recording.format,
            num_samples=n,
            duration_ms=1000.0 * n / cfg.recording.sample_rate,
            segment_start_sample=None,
            segment_end_sample=None,
            trial_start_ns=None,
            input_expected_ns=None,
            input_detected_ns=None,
            trial_end_ns=None,
            wall_clock_utc=None,
            peak=metrics.peak if metrics else None,
            rms=metrics.rms if metrics else None,
            clipping_ratio=metrics.clipping_ratio if metrics else None,
            overflow=None,
            notes="audio found with no manifest row after an abrupt stop; timing unknown (duplicate_policy=new_id)",
            superseded_by=new_id,
            requeue=REQUEUE_IMMEDIATE,
        )
        self._append(manifest, record)
        self._queue.insert(0, ScheduledTrial(trial_id=new_id, label=t.label, repetition=t.repetition))
        self._log.warning("orphan %s recorded as interrupted; pair re-queued as trial %d", path.name, new_id)

    def _append(self, manifest: ManifestWriter, record: TrialRecord) -> None:
        manifest.append(record)
        self._rows_count += 1
        if record.status == Status.VALID.value:
            self._class_valid[record.label] = self._class_valid.get(record.label, 0) + 1
        p = self._progress
        p.completed_trials = self._rows_count
        p.last_trial_id = record.trial_id
        p.next_trial_id = self._queue[0].trial_id if self._queue else None
        if self._rows_count % self.config.output.progress_flush_every == 0:
            self._save_progress(progress_mod.STATE_RUNNING)

    # -- operator interaction ---------------------------------------------

    def _on_sigint(self, signum, frame) -> None:
        self._sigint = True

    def _drain_controls(self) -> list:
        """Every control key pressed since the last drain, stopping after a
        pause so a following resume key stays queued for the pause loop."""
        out = []
        for _ in range(64):
            c = self._controls.poll()
            if c is None:
                break
            out.append(c)
            if c == "pause":
                break
        if self._sigint:
            out.append("quit")
        return out

    def _interpret_controls(self, controls: list, t: ScheduledTrial) -> tuple:
        """Returns (status_key or None, pause, quit). The last
        status-affecting key wins; disabled controls are ignored and logged."""
        cfg = self.config
        allowed = {
            "repeat": cfg.controls.allow_repeat,
            "skip": cfg.controls.allow_skip,
            "invalid": True,
            "pause": cfg.controls.allow_pause,
            "quit": True,
        }
        op = None
        pause = quit_ = False
        for c in controls:
            if not allowed.get(c, False):
                self._log.info("ignored disabled control %r during trial %d", c, t.trial_id)
                continue
            if c in ("repeat", "skip", "invalid"):
                op = c
            elif c == "pause":
                pause = True
            elif c == "quit":
                quit_ = True
        if op:
            self._log.info("operator %s on trial %d", op, t.trial_id)
        return op, pause, quit_

    def _integrity_decision(self, t: ScheduledTrial, detail: Optional[str]) -> str:
        """REQ-40.3: never silently move on after a corrupted write."""
        if not self._controls.interactive:
            self._log.error("integrity failure on trial %d with no operator attached: stopping safely", t.trial_id)
            return "stop"
        self._render(t, "INTEGRITY CHECK FAILED", notice=f"{detail}.  [1] retry now   [2] continue   [0] stop")
        while True:
            c = self._controls.poll()
            if self._sigint or c == "quit":
                return "stop"
            if c == "repeat":
                return "retry"
            if c == "skip":
                return "continue"
            time.sleep(0.05)

    def _pause(self) -> Optional[str]:
        nxt = self._queue[0] if self._queue else None
        self._log.info("pause (next trial %s)", nxt.trial_id if nxt else None)
        self._save_progress(progress_mod.STATE_PAUSED)
        self._display.show(
            "PAUSED\n\n"
            f"Current trial : {self._completed_pairs()} / {self._sched.total_trials}\n"
            f"Target        : {nxt.label if nxt else '-'}\n\n"
            "[4] resume    [0] quit\n"
        )
        started = time.monotonic()
        reason = None
        while True:
            c = self._controls.poll()
            if self._sigint or c == "quit":
                reason = "operator quit (during pause)"
                break
            if c == "pause":
                break
            time.sleep(0.05)
        self._record_break(nxt, time.monotonic() - started, "operator")
        self._log.info("resume after pause")
        self._save_progress(progress_mod.STATE_RUNNING)
        return reason

    def _automatic_break(self) -> Optional[str]:
        cfg = self.config
        nxt = self._queue[0]
        self._log.info("break start before trial %d (%.0fs)", nxt.trial_id, cfg.break_.duration_seconds)
        self._save_progress(progress_mod.STATE_BREAK)
        started = time.monotonic()
        deadline = started + cfg.break_.duration_seconds
        reason = None
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            self._display.show(f"BREAK\n\nResuming in {int(left) + 1} s\n\nNext target: {nxt.label}\n\n[0] quit\n")
            c = self._controls.poll()
            if self._sigint or c == "quit":
                reason = "operator quit (during break)"
                break
            time.sleep(min(0.2, left))
        self._record_break(nxt, time.monotonic() - started, "automatic")
        self._log.info("break end")
        self._save_progress(progress_mod.STATE_RUNNING)
        return reason

    def _record_break(self, nxt: Optional[ScheduledTrial], duration_s: float, kind: str) -> None:
        """REQ-30.3/30.4: measured, not configured, and persisted now so a
        crash later in the session does not lose it."""
        self._total_break_s += duration_s
        self._meta.setdefault("breaks", []).append(
            {
                "break_before_trial": nxt.trial_id if nxt else None,
                "break_duration_s": round(duration_s, 3),
                "break_type": kind,
                "started_utc": utc_now_iso(),
            }
        )
        metadata.write_json_atomic(self.session_dir / "session.json", self._meta)

    # -- display ------------------------------------------------------------

    def _completed_pairs(self) -> int:
        pending = {(q.label, q.repetition) for q in self._queue}
        return self._sched.total_trials - len(pending)

    def _remaining_s(self) -> Optional[float]:
        """REQ-27.5: from measured trial durations once there are any."""
        if self._trial_seconds:
            recent = self._trial_seconds[-50:]
            per_trial = sum(recent) / len(recent)
        else:
            d = self._durations
            per_trial = (d.countdown_ms + d.stored_ms + d.inter_trial_ms) / 1000.0
        return per_trial * (len(self._queue) + 1)

    def _render(self, t: ScheduledTrial, status: str, notice: Optional[str] = None) -> None:
        cfg = self.config
        pending_pairs = {(q.label, q.repetition) for q in self._queue} | {(t.label, t.repetition)}
        self._display.show(
            ui.render_trial_screen(
                participant=cfg.participant.id,
                scenario=cfg.scenario.id,
                session=cfg.session.id,
                trial_id=t.trial_id,
                overall_completed=self._sched.total_trials - len(pending_pairs),
                overall_total=self._sched.total_trials,
                current_label=t.label,
                class_completed=self._class_valid.get(t.label, 0),
                class_total=cfg.trial.repetitions_per_class,
                next_label=self._queue[0].label if self._queue else None,
                status=status,
                elapsed_s=time.monotonic() - self._run_start,
                remaining_s=self._remaining_s(),
                notice=notice or self._notice,
            )
        )

    # -- files --------------------------------------------------------------

    def _write_initial_session_json(self, resume: bool) -> dict:
        cfg = self.config
        info = self.backend.resolve_device(cfg.recording.device)
        meta = metadata.build_session_metadata(
            config=cfg,
            class_definition_version=self.class_definition_version,
            num_classes=self.num_classes,
            total_trials=self._sched.total_trials,
            device_info_dict={
                "audio_device_name": info.name,
                "audio_device_index": info.index,
                "host_api": info.host_api,
                "sample_rate": cfg.recording.sample_rate,
                "channel_count": cfg.recording.channels,
                "input_latency_s": info.input_latency_s,
            },
            keyboard_dict=dict(cfg.hardware.keyboard.__dict__),
            microphone_dict=dict(cfg.hardware.microphone.__dict__),
            environment_dict=dict(cfg.environment.__dict__),
            started_utc=utc_now_iso(),
            random_seed=self._sched.seed,
            mic_test_result=self._mic_test_result.to_dict() if self._mic_test_result else None,
        )
        meta["seed_generated"] = self.seed_generated
        if meta.get("git_dirty"):
            msg = "WARNING: recording from a git worktree with uncommitted changes (REQ-38.4)"
            self._log.warning(msg)
            self._display.line(msg)

        path = self.session_dir / "session.json"
        if resume and path.exists():
            prior = metadata.read_json(path)
            meta["started_utc"] = prior.get("started_utc", meta["started_utc"])
            meta["breaks"] = prior.get("breaks", [])
            meta["seed_generated"] = prior.get("seed_generated", False)
            first_commit = prior.get("git_commit")
            meta["resumes"] = prior.get("resumes", []) + [
                {"resumed_utc": utc_now_iso(), "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"]}
            ]
            meta["git_commit"] = first_commit
            if first_commit != meta["resumes"][-1]["git_commit"]:
                msg = (
                    f"WARNING: resuming with recorder commit {meta['resumes'][-1]['git_commit']} "
                    f"but the session was started with {first_commit} (REQ-48.5)"
                )
                self._log.warning(msg)
                self._display.line(msg)
        metadata.write_json_atomic(path, meta)
        return meta

    def _save_progress(self, state: str) -> None:
        self._progress.state = state
        nxt = self._queue[0] if self._queue else None
        self._progress.next_trial_id = nxt.trial_id if nxt else None
        self._progress.current_trial_id = nxt.trial_id if nxt else None
        self._progress.current_label = nxt.label if nxt else None
        progress_mod.write_progress(self._progress_path, self._progress)

    def _finalize(self, stop_reason: Optional[str]) -> metadata.SessionSummary:
        from .validator import validate_session

        log = self._log
        cfg = self.config
        rows = read_manifest_jsonl(self.session_dir / "manifest.jsonl")
        processed_all = not self._queue and stop_reason is None

        self._progress.completed_trials = len(rows)
        log.info("shutdown: writing progress.json")
        self._save_progress(progress_mod.STATE_COMPLETE if processed_all else progress_mod.STATE_STOPPED)

        log.info("shutdown: writing session.json")
        self._meta["ended_utc"] = utc_now_iso()
        metadata.write_json_atomic(self.session_dir / "session.json", self._meta)

        summary = metadata.build_summary(
            session_uid=cfg.session_uid,
            expected_trials=self._sched.total_trials,
            records=rows,
            duration_s=time.monotonic() - self._run_start,
            total_break_s=self._total_break_s,
        )
        summary.stop_reason = stop_reason
        if processed_all:
            try:
                report = validate_session(self.session_dir)
                summary.validated = True
                summary.validation_result = "PASS" if report.passed else "FAIL"
                if not report.passed:
                    log.error("validation failed:\n%s", report.render())
            except Exception as exc:
                summary.validation_result = f"ERROR: {exc}"
                log.exception("validation error")
        summary.completed = processed_all and summary.validation_result == "PASS"
        log.info("shutdown: writing session_summary.json")
        metadata.write_json_atomic(self.session_dir / "session_summary.json", summary.to_dict())
        if summary.completed and processed_all:
            self._save_progress(progress_mod.STATE_COMPLETE)
        elif processed_all:
            self._save_progress(progress_mod.STATE_STOPPED)
        return summary
