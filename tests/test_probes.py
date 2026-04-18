from latent_harness.core.config import LatentRuntimeConfig


def test_probe_config_defaults_off():
    cfg = LatentRuntimeConfig()
    assert cfg.probe_mode is False
    assert cfg.probe_output_dir is None


def test_probe_config_accepts_output_dir():
    cfg = LatentRuntimeConfig(probe_mode=True, probe_output_dir="artifacts/probe/out")
    assert cfg.probe_mode is True
    assert cfg.probe_output_dir == "artifacts/probe/out"


def test_probe_mode_requires_output_dir():
    import pytest
    with pytest.raises(ValueError, match="probe_output_dir"):
        LatentRuntimeConfig(probe_mode=True, probe_output_dir=None)
