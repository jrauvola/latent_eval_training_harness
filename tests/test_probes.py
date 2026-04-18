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


import torch
from pathlib import Path


def test_register_dgrad_probe_captures_per_layer_max(tmp_path):
    from latent_harness.core.probes import DgradProbe

    # Fake 4-layer model: stack of 3 linear layers with tanh between
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
        def forward(self, x):
            outs = []
            for layer in self.layers:
                x = torch.tanh(layer(x))
                outs.append(x)
            return outs

    model = Toy()
    probe = DgradProbe(output_path=tmp_path / "dgrad.csv")
    probe.attach(model.layers, layer_names=["l0", "l1", "l2"])
    x = torch.randn(2, 8, requires_grad=True)
    outs = model(x)
    loss = outs[-1].sum()
    loss.backward()
    probe.flush(step=0)
    probe.detach_all()

    csv_content = (tmp_path / "dgrad.csv").read_text()
    assert "step,layer,max_abs_dgrad" in csv_content
    assert "0,l0," in csv_content
    assert "0,l1," in csv_content
    assert "0,l2," in csv_content
