"""Error taxonomy for the recorder (REQ-50.1).

Every condition REQ-50.1 requires to be "detected and reported with a
distinct, actionable message" gets its own exception class so callers
(CLI, tests) can distinguish failure modes programmatically as well as by
message text.
"""


class RecorderError(Exception):
    """Base class for all recorder errors."""


class ConfigError(RecorderError):
    """Invalid or unparsable configuration (REQ-51.2, REQ-50.1)."""


class ClassDefinitionError(RecorderError):
    """Missing or unloadable class definition (REQ-4.4, REQ-50.1)."""


class ScheduleError(RecorderError):
    """Class balance / schedule generation or persistence failure."""


class ResumeError(RecorderError):
    """Schedule/configuration mismatch, or other resume failure (REQ-48.4)."""


class AudioDeviceError(RecorderError):
    """Audio device not found / disconnected / unsupported format."""


class AudioStreamError(RecorderError):
    """Overflow / underflow / backend exception during capture."""


class DiskSpaceError(RecorderError):
    """Insufficient disk space, before or during a session (REQ-55)."""


class IntegrityError(RecorderError):
    """Post-write WAV integrity check failure (REQ-40)."""


class DuplicateTrialError(RecorderError):
    """A trial_id / filename collision under duplicate_policy=error (REQ-49.2)."""


class PreflightError(RecorderError):
    """A pre-flight checklist item failed (§53)."""
