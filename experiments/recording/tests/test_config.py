import pytest

from experiments.recording.config import config_from_dict
from experiments.recording.errors import ConfigError
from experiments.recording.tests.helpers import base_config_dict


def test_valid_config_loads():
    cfg = config_from_dict(base_config_dict())
    assert cfg.participant.id == "P01"
    assert cfg.session_uid == "P01_S01_SESSION01"


def test_missing_required_field_rejected():
    d = base_config_dict()
    del d["participant"]["id"]
    with pytest.raises(ConfigError, match="participant"):
        config_from_dict(d)


def test_unknown_top_level_key_rejected():
    d = base_config_dict()
    d["bogus_section"] = {}
    with pytest.raises(ConfigError, match="unknown section"):
        config_from_dict(d)


def test_unknown_nested_key_rejected():
    d = base_config_dict()
    d["recording"]["bogus_field"] = 1
    with pytest.raises(ConfigError, match="recording"):
        config_from_dict(d)


def test_out_of_range_value_rejected():
    d = base_config_dict(quality={"clipping_threshold": 5.0})
    with pytest.raises(ConfigError, match="quality.clipping_threshold"):
        config_from_dict(d)


def test_allow_resample_must_be_false():
    d = base_config_dict(recording={"allow_resample": True})
    with pytest.raises(ConfigError, match="allow_resample"):
        config_from_dict(d)


def test_invalid_strategy_rejected():
    d = base_config_dict(randomization={"strategy": "not_a_strategy"})
    with pytest.raises(ConfigError, match="strategy"):
        config_from_dict(d)


def test_block_random_requires_block_size():
    d = base_config_dict(randomization={"strategy": "block_random", "block_size": None})
    with pytest.raises(ConfigError, match="block_size"):
        config_from_dict(d)


def test_wrong_type_rejected():
    d = base_config_dict()
    d["recording"]["sample_rate"] = "48000"
    with pytest.raises(ConfigError, match="sample_rate"):
        config_from_dict(d)


def test_spec_reference_config_with_integer_config_version_is_accepted():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load((Path(__file__).parents[3] / "configs" / "S01.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["config_version"] = 1  # exactly as written in spec §52
    cfg = config_from_dict(raw)
    assert cfg.experiment.config_version == "1"
