# classism — project-level specification (excerpt)

This file is the project-level single source of truth referenced by
`CLASSISM-SPEC-REC` (`experiments/recording`), per its REQ-2.2.1 precedence
order. Only the fields the recording subsystem depends on are listed here.

## Audio format contract

The Mel feature pipeline downstream of the recorder is defined against:

```
sample_rate  : 48000 Hz
channels     : 1 (mono)
bit_depth    : 16-bit signed linear PCM
```

The recorder MUST NOT deviate from these values without a corresponding
update here and a `dataset_schema_version` bump in the recording
subsystem.

## Class definition

The classifier's target classes are defined in `classism/labels.py`
(`CLASSES`, `CLASS_DEFINITION_VERSION`). The recording subsystem imports
this module directly and MUST NOT redefine the class list.
