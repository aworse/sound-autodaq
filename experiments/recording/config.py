"""
Configuration schema, parsing, and validation (REQ-51 through REQ-52).

Owns: config schema, validation, resolution to a plain dict for embedding
in session.json. Never: audio, files, scheduling (REQ-57).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import ConfigError

CONFIG_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExperimentSection:
    name: str
    config_version: str


@dataclasses.dataclass(frozen=True)
class ParticipantSection:
    id: str


@dataclasses.dataclass(frozen=True)
class ScenarioSection:
    id: str
    name: str
    description: str


@dataclasses.dataclass(frozen=True)
class SessionSection:
    id: str


@dataclasses.dataclass(frozen=True)
class RecordingSection:
    device: Optional[Any]
    sample_rate: int
    channels: int
    format: str
    allow_resample: bool


@dataclasses.dataclass(frozen=True)
class TrialSection:
    # scheduled: fixed pre-roll / input window / post-roll timeline (REQ-22).
    # keypress: wait for the keystroke (up to input_window_ms, 0 = no limit)
    #   and store pre_roll_ms before to post_roll_ms after it, cut from the
    #   continuous ring buffer.
    capture: str
    repetitions_per_class: int
    countdown_ms: int
    pre_roll_ms: int
    input_window_ms: int
    post_roll_ms: int
    inter_trial_ms: int


@dataclasses.dataclass(frozen=True)
class InputSection:
    mode: str  # human | automated
    key_detection: str  # hook | terminal | none


@dataclasses.dataclass(frozen=True)
class RandomizationSection:
    strategy: str
    seed: Optional[int]
    block_size: Optional[int]
    manual_order_file: Optional[str]


@dataclasses.dataclass(frozen=True)
class ControlsSection:
    allow_repeat: bool
    allow_skip: bool
    allow_pause: bool
    repeat_on_invalid: bool
    resume_policy: str  # discard_current | continue_current


@dataclasses.dataclass(frozen=True)
class BreakSection:
    enabled: bool
    every_trials: int
    duration_seconds: float


@dataclasses.dataclass(frozen=True)
class QualitySection:
    clipping_threshold: float
    silence_rms_threshold: float
    max_consecutive_failures: int


@dataclasses.dataclass(frozen=True)
class OutputSection:
    root: str
    duplicate_policy: str  # error | new_id
    progress_flush_every: int


@dataclasses.dataclass(frozen=True)
class KeyboardSection:
    keyboard_id: str
    keyboard_model: str
    keyboard_type: str
    switch_type: str
    layout: str


@dataclasses.dataclass(frozen=True)
class MicrophoneSection:
    microphone_id: str
    microphone_model: str
    placement: str
    distance_cm: Optional[float]
    orientation: str


@dataclasses.dataclass(frozen=True)
class HardwareSection:
    keyboard: KeyboardSection
    microphone: MicrophoneSection


@dataclasses.dataclass(frozen=True)
class EnvironmentSection:
    room_id: str
    background_noise: str
    air_conditioner: bool
    fan: bool
    other_devices: list
    background_db: Optional[float]


@dataclasses.dataclass(frozen=True)
class Config:
    experiment: ExperimentSection
    participant: ParticipantSection
    scenario: ScenarioSection
    session: SessionSection
    recording: RecordingSection
    trial: TrialSection
    input: InputSection
    randomization: RandomizationSection
    controls: ControlsSection
    break_: BreakSection
    quality: QualitySection
    output: OutputSection
    hardware: HardwareSection
    environment: EnvironmentSection

    @property
    def session_uid(self) -> str:
        return f"{self.participant.id}_{self.scenario.id}_{self.session.id}"

    def resolved_dict(self) -> dict:
        """Full resolved configuration as a plain, JSON-serializable dict."""
        d = dataclasses.asdict(self)
        d["break"] = d.pop("break_")
        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_STR = "str"
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_LIST = "list"
_ANY = "any"

_TYPE_MAP = {
    _STR: str,
    _INT: int,
    _FLOAT: (int, float),
    _BOOL: bool,
    _LIST: list,
}


def _check_type(path: str, value: Any, kind: str, nullable: bool) -> None:
    if value is None:
        if nullable:
            return
        raise ConfigError(f"{path}: must not be null")
    if kind == _ANY:
        return
    py_type = _TYPE_MAP[kind]
    if kind == _BOOL and not isinstance(value, bool):
        raise ConfigError(f"{path}: expected bool, got {type(value).__name__}")
    if kind != _BOOL and isinstance(value, bool):
        raise ConfigError(f"{path}: expected {kind}, got bool")
    if not isinstance(value, py_type):
        raise ConfigError(f"{path}: expected {kind}, got {type(value).__name__}")


# Each field: name -> (kind, nullable)
_SECTION_FIELDS = {
    "experiment": {"name": (_STR, False), "config_version": (_STR, False)},
    "participant": {"id": (_STR, False)},
    "scenario": {"id": (_STR, False), "name": (_STR, False), "description": (_STR, False)},
    "session": {"id": (_STR, False)},
    "recording": {
        "device": (_ANY, True),
        "sample_rate": (_INT, False),
        "channels": (_INT, False),
        "format": (_STR, False),
        "allow_resample": (_BOOL, False),
    },
    "trial": {
        "capture": (_STR, False),
        "repetitions_per_class": (_INT, False),
        "countdown_ms": (_INT, False),
        "pre_roll_ms": (_INT, False),
        "input_window_ms": (_INT, False),
        "post_roll_ms": (_INT, False),
        "inter_trial_ms": (_INT, False),
    },
    "input": {"mode": (_STR, False), "key_detection": (_STR, False)},
    "randomization": {
        "strategy": (_STR, False),
        "seed": (_INT, True),
        "block_size": (_INT, True),
        "manual_order_file": (_STR, True),
    },
    "controls": {
        "allow_repeat": (_BOOL, False),
        "allow_skip": (_BOOL, False),
        "allow_pause": (_BOOL, False),
        "repeat_on_invalid": (_BOOL, False),
        "resume_policy": (_STR, False),
    },
    "break": {
        "enabled": (_BOOL, False),
        "every_trials": (_INT, False),
        "duration_seconds": (_FLOAT, False),
    },
    "quality": {
        "clipping_threshold": (_FLOAT, False),
        "silence_rms_threshold": (_FLOAT, False),
        "max_consecutive_failures": (_INT, False),
    },
    "output": {
        "root": (_STR, False),
        "duplicate_policy": (_STR, False),
        "progress_flush_every": (_INT, False),
    },
    "environment": {
        "room_id": (_STR, False),
        "background_noise": (_STR, False),
        "air_conditioner": (_BOOL, False),
        "fan": (_BOOL, False),
        "other_devices": (_LIST, False),
        "background_db": (_FLOAT, True),
    },
}

_HARDWARE_KEYBOARD_FIELDS = {
    "keyboard_id": (_STR, False),
    "keyboard_model": (_STR, False),
    "keyboard_type": (_STR, False),
    "switch_type": (_STR, False),
    "layout": (_STR, False),
}

_HARDWARE_MICROPHONE_FIELDS = {
    "microphone_id": (_STR, False),
    "microphone_model": (_STR, False),
    "placement": (_STR, False),
    "distance_cm": (_FLOAT, True),
    "orientation": (_STR, False),
}

_TOP_LEVEL_SECTIONS = set(_SECTION_FIELDS) | {"hardware"}

_VALID_FORMATS = {"PCM_16"}
_VALID_STRATEGIES = {"random", "balanced_random", "block_random", "manual"}
_VALID_INPUT_MODES = {"human", "automated"}
_VALID_KEY_DETECTION = {"hook", "terminal", "none"}
_VALID_CAPTURE = {"scheduled", "keypress"}
_VALID_RESUME_POLICIES = {"discard_current", "continue_current"}
_VALID_DUPLICATE_POLICIES = {"error", "new_id"}


def _validate_section(section_name: str, raw: Any, fields: dict) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError(f"{section_name}: must be a mapping")
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(f"{section_name}: unknown key(s) {sorted(unknown)}")
    missing = set(fields) - set(raw)
    if missing:
        raise ConfigError(f"{section_name}: missing required key(s) {sorted(missing)}")
    for key, (kind, nullable) in fields.items():
        _check_type(f"{section_name}.{key}", raw[key], kind, nullable)
    return raw


def _positive_int(path: str, value: int) -> None:
    if value <= 0:
        raise ConfigError(f"{path}: must be a positive integer, got {value}")


def validate_config_dict(raw: dict) -> dict:
    """Validate a raw parsed-YAML dict against the schema.

    Raises ConfigError with a specific field name on any violation
    (REQ-51.2, REQ-51.3, T-7). Returns the same dict, section-validated.
    """
    if not isinstance(raw, dict):
        raise ConfigError("top level: configuration must be a mapping")

    unknown = set(raw) - _TOP_LEVEL_SECTIONS
    if unknown:
        raise ConfigError(f"top level: unknown section(s) {sorted(unknown)}")
    missing = _TOP_LEVEL_SECTIONS - set(raw)
    if missing:
        raise ConfigError(f"top level: missing section(s) {sorted(missing)}")

    # The spec's own reference config (§52) writes `config_version: 1`;
    # accept an integer and store it as the string it identifies.
    exp = raw.get("experiment")
    if isinstance(exp, dict):
        v = exp.get("config_version")
        if isinstance(v, int) and not isinstance(v, bool):
            exp["config_version"] = str(v)

    for section_name, fields in _SECTION_FIELDS.items():
        _validate_section(section_name, raw[section_name], fields)

    hw = raw["hardware"]
    if not isinstance(hw, dict) or set(hw) != {"keyboard", "microphone"}:
        raise ConfigError("hardware: must contain exactly 'keyboard' and 'microphone'")
    _validate_section("hardware.keyboard", hw["keyboard"], _HARDWARE_KEYBOARD_FIELDS)
    _validate_section("hardware.microphone", hw["microphone"], _HARDWARE_MICROPHONE_FIELDS)

    # Range / enum checks beyond simple typing.
    _positive_int("recording.sample_rate", raw["recording"]["sample_rate"])
    _positive_int("recording.channels", raw["recording"]["channels"])
    if raw["recording"]["format"] not in _VALID_FORMATS:
        raise ConfigError(
            f"recording.format: must be one of {sorted(_VALID_FORMATS)}, "
            f"got {raw['recording']['format']!r}"
        )
    if raw["recording"]["allow_resample"] is not False:
        raise ConfigError(
            "recording.allow_resample: MUST be false for dataset collection (REQ-52.1)"
        )

    _positive_int("trial.repetitions_per_class", raw["trial"]["repetitions_per_class"])
    for k in ("countdown_ms", "pre_roll_ms", "input_window_ms", "post_roll_ms", "inter_trial_ms"):
        v = raw["trial"][k]
        if v < 0:
            raise ConfigError(f"trial.{k}: must be >= 0, got {v}")

    capture = raw["trial"]["capture"]
    if capture not in _VALID_CAPTURE:
        raise ConfigError(f"trial.capture: must be one of {sorted(_VALID_CAPTURE)}, got {capture!r}")
    if capture == "keypress":
        if raw["input"]["key_detection"] not in ("hook", "terminal"):
            raise ConfigError(
                "trial.capture: keypress needs input.key_detection: hook (or terminal) "
                "(the recording is cut around the detected keystroke)"
            )
        t = raw["trial"]
        if t["post_roll_ms"] <= 0:
            raise ConfigError("trial.post_roll_ms: must be > 0 with capture: keypress (the key release is in it)")
        # The previous keystroke happened at least post_roll + gap + countdown
        # before this PRESS appears; keeping that >= pre_roll guarantees it
        # can never fall inside this trial's pre-roll.
        if t["post_roll_ms"] + t["inter_trial_ms"] + t["countdown_ms"] < t["pre_roll_ms"]:
            raise ConfigError(
                "trial: with capture: keypress, post_roll_ms + inter_trial_ms + countdown_ms "
                f"({t['post_roll_ms'] + t['inter_trial_ms'] + t['countdown_ms']}) must be >= pre_roll_ms "
                f"({t['pre_roll_ms']}), or the previous keystroke can land in this trial's pre-roll"
            )

    if raw["input"]["mode"] not in _VALID_INPUT_MODES:
        raise ConfigError(
            f"input.mode: must be one of {sorted(_VALID_INPUT_MODES)}, "
            f"got {raw['input']['mode']!r}"
        )

    if raw["input"]["key_detection"] not in _VALID_KEY_DETECTION:
        raise ConfigError(
            f"input.key_detection: must be one of {sorted(_VALID_KEY_DETECTION)}, "
            f"got {raw['input']['key_detection']!r}"
        )

    strategy = raw["randomization"]["strategy"]
    if strategy not in _VALID_STRATEGIES:
        raise ConfigError(
            f"randomization.strategy: must be one of {sorted(_VALID_STRATEGIES)}, "
            f"got {strategy!r}"
        )
    if strategy == "block_random" and raw["randomization"]["block_size"] is None:
        raise ConfigError("randomization.block_size: required when strategy=block_random")
    if strategy == "manual" and raw["randomization"]["manual_order_file"] is None:
        raise ConfigError("randomization.manual_order_file: required when strategy=manual")

    if raw["controls"]["resume_policy"] not in _VALID_RESUME_POLICIES:
        raise ConfigError(
            f"controls.resume_policy: must be one of {sorted(_VALID_RESUME_POLICIES)}, "
            f"got {raw['controls']['resume_policy']!r}"
        )

    _positive_int("break.every_trials", raw["break"]["every_trials"])
    if raw["break"]["duration_seconds"] < 0:
        raise ConfigError("break.duration_seconds: must be >= 0")

    for k in ("clipping_threshold", "silence_rms_threshold"):
        v = raw["quality"][k]
        if not (0.0 <= v <= 1.0):
            raise ConfigError(f"quality.{k}: must be in [0, 1], got {v}")
    _positive_int("quality.max_consecutive_failures", raw["quality"]["max_consecutive_failures"])

    if raw["output"]["duplicate_policy"] not in _VALID_DUPLICATE_POLICIES:
        raise ConfigError(
            f"output.duplicate_policy: must be one of {sorted(_VALID_DUPLICATE_POLICIES)}, "
            f"got {raw['output']['duplicate_policy']!r}"
        )
    _positive_int("output.progress_flush_every", raw["output"]["progress_flush_every"])

    return raw


def _build_config(raw: dict) -> Config:
    return Config(
        experiment=ExperimentSection(**raw["experiment"]),
        participant=ParticipantSection(**raw["participant"]),
        scenario=ScenarioSection(**raw["scenario"]),
        session=SessionSection(**raw["session"]),
        recording=RecordingSection(**raw["recording"]),
        trial=TrialSection(**raw["trial"]),
        input=InputSection(**raw["input"]),
        randomization=RandomizationSection(**raw["randomization"]),
        controls=ControlsSection(**raw["controls"]),
        break_=BreakSection(**raw["break"]),
        quality=QualitySection(**raw["quality"]),
        output=OutputSection(**raw["output"]),
        hardware=HardwareSection(
            keyboard=KeyboardSection(**raw["hardware"]["keyboard"]),
            microphone=MicrophoneSection(**raw["hardware"]["microphone"]),
        ),
        environment=EnvironmentSection(**raw["environment"]),
    )


def load_config(path: str | Path) -> Config:
    """Load, parse, and validate a YAML configuration file (REQ-51)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read configuration file {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"configuration file {path} is empty")
    validate_config_dict(raw)
    return _build_config(raw)


def config_from_dict(raw: dict) -> Config:
    """Validate and build a Config from an in-memory dict (used by tests)."""
    validate_config_dict(raw)
    return _build_config(raw)
