"""Shared test fixtures: a minimal valid configuration dict."""

import copy


def base_config_dict(**overrides) -> dict:
    d = {
        "experiment": {"name": "test_experiment", "config_version": "1"},
        "participant": {"id": "P01"},
        "scenario": {"id": "S01", "name": "quiet_room", "description": "test scenario"},
        "session": {"id": "SESSION01"},
        "recording": {
            "device": None,
            "sample_rate": 48000,
            "channels": 1,
            "format": "PCM_16",
            "allow_resample": False,
        },
        "trial": {
            "repetitions_per_class": 3,
            "countdown_ms": 0,
            "pre_roll_ms": 10,
            "input_window_ms": 10,
            "post_roll_ms": 10,
            "inter_trial_ms": 0,
        },
        "input": {"mode": "human"},
        "randomization": {
            "strategy": "balanced_random",
            "seed": 42,
            "block_size": None,
            "manual_order_file": None,
        },
        "controls": {
            "allow_repeat": True,
            "allow_skip": True,
            "allow_pause": True,
            "repeat_on_invalid": True,
            "resume_policy": "discard_current",
        },
        "break": {"enabled": False, "every_trials": 500, "duration_seconds": 60},
        "quality": {
            "clipping_threshold": 0.001,
            "silence_rms_threshold": 0.001,
            "max_consecutive_failures": 3,
        },
        "output": {"root": "data/raw", "duplicate_policy": "error", "progress_flush_every": 1},
        "hardware": {
            "keyboard": {
                "keyboard_id": "KB01",
                "keyboard_model": "unknown",
                "keyboard_type": "unknown",
                "switch_type": "unknown",
                "layout": "dubeolsik",
            },
            "microphone": {
                "microphone_id": "MIC01",
                "microphone_model": "unknown",
                "placement": "unknown",
                "distance_cm": None,
                "orientation": "unknown",
            },
        },
        "environment": {
            "room_id": "unknown",
            "background_noise": "unknown",
            "air_conditioner": False,
            "fan": False,
            "other_devices": [],
            "background_db": None,
        },
    }
    _deep_update(d, overrides)
    return d


def _deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
