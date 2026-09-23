"""
Command-line interface: argument parsing and command dispatch (§58-59).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import hardware, metadata, progress
from .config import Config, config_from_dict, load_config
from .engine import SessionEngine, session_dir_for
from .errors import RecorderError
from .recorder import SoundDeviceBackend
from .trial import ASSUMED_REACTION_MS, estimated_trial_ms, stored_ms
from .ui import Display, TerminalControlSource
from .validator import validate_session


def _config_for_resume(session_dir: Path) -> Config:
    session_json_path = session_dir / "session.json"
    if not session_json_path.exists():
        raise RecorderError(f"cannot resume: no session.json in {session_dir}")
    prior = metadata.read_json(session_json_path)
    resolved = prior.get("resolved_config")
    if not resolved:
        raise RecorderError(f"cannot resume: session.json in {session_dir} has no resolved_config")
    # Sessions recorded before key detection / keypress capture existed ran
    # without them; that is a fact about those sessions, not a guessed default.
    resolved.setdefault("input", {}).setdefault("key_detection", "none")
    resolved.setdefault("trial", {}).setdefault("capture", "scheduled")
    return config_from_dict(resolved)


def _next_free_session(config: Config) -> Config:
    """REQ-48.6: declining to resume starts a new session directory next to
    the old one (SESSION01 -> SESSION01_2, _3, ...); nothing is overwritten."""
    k = 2
    while True:
        candidate = dataclasses.replace(config, session=dataclasses.replace(config.session, id=f"{config.session.id}_{k}"))
        if progress.session_dir_state(session_dir_for(candidate)) == "absent":
            return candidate
        k += 1


def _ask_resume(session_dir: Path) -> str:
    """REQ-48.1 prompt. Returns 'y', 'n' or 'q'."""
    prog = progress.read_progress(session_dir / "progress.json")
    print("Found incomplete session.\n")
    print(f"Session   : {session_dir}")
    if prog is not None:
        print(f"Completed : {prog.completed_trials} / {prog.total_trials}")
        print(f"Seed      : {prog.random_seed}")
    print("\nResume?\n  [Y] Resume this session\n  [N] Start a new session\n  [Q] Quit")
    while True:
        answer = input("> ").strip().lower()
        if answer in ("y", "n", "q"):
            return answer


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        config = load_config(args.config)
        resume = False
    else:
        config = _config_for_resume(Path(args.resume))
        resume = True

    if not resume and not args.dry_run:
        state = progress.session_dir_state(session_dir_for(config))
        if state == "complete":
            print(f"ERROR: {session_dir_for(config)} already holds a complete session. "
                  "Choose a new session.id in the configuration.", file=sys.stderr)
            return 1
        if state == "incomplete":
            if not sys.stdin.isatty():
                print(f"ERROR: {session_dir_for(config)} holds an incomplete session. Resume it with "
                      f"--resume {session_dir_for(config)} or choose a new session.id.", file=sys.stderr)
                return 1
            answer = _ask_resume(session_dir_for(config))
            if answer == "q":
                return 0
            if answer == "y":
                resume = True
            else:
                config = _next_free_session(config)
                print(f"Starting a new session in {session_dir_for(config)}")

    engine = SessionEngine(config, backend=SoundDeviceBackend())

    if args.dry_run:
        return _dry_run(engine)

    control_source = TerminalControlSource()
    display = Display(enabled=True, interactive=sys.stdout.isatty())
    try:
        summary = engine.run(resume=resume, control_source=control_source, display=display)
    except RecorderError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if progress.session_dir_state(engine.session_dir) == "incomplete":
            print(f"Data recorded so far is kept in {engine.session_dir}; after fixing the cause, resume with "
                  f"--resume {engine.session_dir}", file=sys.stderr)
        return 1
    finally:
        control_source.close()

    print(f"\n{summary.valid_trials} valid of {summary.expected_trials} expected "
          f"({summary.completion_rate * 100:.2f}%), {summary.attempted_trials} trials recorded.")
    if summary.completed:
        print("SESSION COMPLETE — validation PASS")
        return 0
    if summary.stop_reason:
        # REQ-29.3: a requested stop exits 0; REQ-46.2: but it is not reported as a complete session.
        print(f"SESSION INCOMPLETE — stopped: {summary.stop_reason}")
        print(f"Resume with: python -m experiments.recording --resume {engine.session_dir}")
        return 0
    print(f"SESSION INCOMPLETE — validation {summary.validation_result}. Inspect with "
          f"python -m experiments.recording --validate {engine.session_dir}")
    return 1


def _dry_run(engine: SessionEngine) -> int:
    cfg = engine.config
    report = engine.preflight(resume=False, dry_run=True)
    if not report.passed:
        print("DRY RUN — FAILED")
        print(report.render())
        return 1

    sched = engine._schedule
    est_duration_s = estimated_trial_ms(cfg.trial) / 1000.0 * sched.total_trials
    est_bytes = hardware.estimate_bytes(
        sched.total_trials, stored_ms(cfg.trial), cfg.recording.sample_rate, cfg.recording.channels
    )

    print("DRY RUN — no audio device opened, no files written\n")
    print(report.render(), "\n")
    print("Config          : valid")
    print(f"Class definition: {engine.class_definition_version}")
    print(f"Classes         : {engine.num_classes}")
    print(f"Repetitions     : {cfg.trial.repetitions_per_class}")
    print(f"Total trials    : {sched.total_trials}")
    print(f"Capture         : {cfg.trial.capture}\n")
    print(f"Strategy        : {sched.strategy}")
    print(f"Random seed     : {sched.seed}\n")
    print("Class balance   : ALL PASS")
    print("Schedule        : VALID")
    print("First 10 labels :", " ".join(t.label for t in sched.trials[:10]), "\n")
    assumed = f" (assuming a {ASSUMED_REACTION_MS} ms reaction)" if cfg.trial.capture == "keypress" else ""
    print(f"Estimated duration : ~{est_duration_s / 3600:.1f}h{assumed}")
    print(f"Estimated disk     : {est_bytes / 1e9:.2f} GB\n")
    print("RESULT: PASS")
    return 0


def cmd_mic_test(args: argparse.Namespace) -> int:
    backend = SoundDeviceBackend()
    result = hardware.run_mic_test(backend, args.device, args.sample_rate, args.channels)
    print("MIC TEST\n")
    print(f"Device      : {result.device.name} ({result.device.host_api}, index {result.device.index})")
    print(f"Recording {result.duration_s:.0f} seconds...\n")
    print(f"Sample rate : {result.sample_rate} Hz")
    print(f"Channels    : {result.channels}")
    print(f"Peak        : {result.peak:.2f}")
    print(f"RMS         : {result.rms:.3f}")
    print(f"Clipping    : {result.clipping_ratio:.4f}")
    print(f"Overflow    : {'yes' if result.overflow else 'none'}\n")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    if result.reasons:
        for r in result.reasons:
            print(f"  - {r}")
    return 0 if result.passed else 1


def cmd_list_devices(args: argparse.Namespace) -> int:
    backend = SoundDeviceBackend()
    for d in backend.list_devices():
        print(f"[{d.index}] {d.name}  ({d.host_api}, {d.max_input_channels} ch, "
              f"{d.default_sample_rate:.0f} Hz default, latency {d.input_latency_s * 1000:.1f} ms)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_session(args.validate)
    print(report.render())
    return 0 if report.passed else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .webui.server import serve

    serve(Path(args.dashboard), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.recording")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="path to a session YAML configuration")
    group.add_argument("--resume", help="path to a session directory to resume")
    group.add_argument("--mic-test", action="store_true")
    group.add_argument("--validate", help="path to a session directory to validate")
    group.add_argument("--list-devices", action="store_true")
    group.add_argument("--dashboard", help="path to a session directory to view live in a browser")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1", help="dashboard bind host")
    parser.add_argument("--port", type=int, default=8765, help="dashboard bind port")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mic_test:
            return cmd_mic_test(args)
        if args.list_devices:
            return cmd_list_devices(args)
        if args.validate:
            return cmd_validate(args)
        if args.dashboard:
            return cmd_dashboard(args)
        return cmd_run(args)
    except RecorderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
