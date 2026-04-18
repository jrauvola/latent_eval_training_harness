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


def test_dgrad_probe_captures_known_analytical_value(tmp_path):
    """Loss = out.sum() → grad_output for last layer is all-ones → max should be 1.0."""
    from latent_harness.core.probes import DgradProbe

    layer = torch.nn.Linear(8, 8)
    probe = DgradProbe(output_path=tmp_path / "dgrad.csv")
    probe.attach([layer], layer_names=["l0"])
    x = torch.randn(2, 8, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    probe.flush(step=0)
    probe.detach_all()

    import csv as _csv
    with (tmp_path / "dgrad.csv").open() as f:
        rows = list(_csv.DictReader(f))
    # grad_output for the layer output is dloss/dout = all-ones tensor → max|.|=1.0
    assert abs(float(rows[0]["max_abs_dgrad"]) - 1.0) < 1e-5


def test_dgrad_probe_takes_running_max_across_multiple_backwards(tmp_path):
    """CODI performs multiple forward/backward passes per training step.
    Probe must take the max across all backwards within a single flush window.

    Uses scales [3.0, 7.0, 2.0] so:
      - running-max   → 7.0  (correct)
      - first-write   → 3.0  (wrong)
      - last-write    → 2.0  (wrong)
    Any implementation that doesn't actually take a max will fail.
    """
    from latent_harness.core.probes import DgradProbe

    layer = torch.nn.Linear(4, 4)
    probe = DgradProbe(output_path=tmp_path / "dgrad.csv")
    probe.attach([layer], layer_names=["l0"])

    for scale in (3.0, 7.0, 2.0):
        x = torch.randn(2, 4, requires_grad=True)
        (layer(x).sum() * scale).backward()

    probe.flush(step=0)
    probe.detach_all()

    import csv as _csv
    with (tmp_path / "dgrad.csv").open() as f:
        rows = list(_csv.DictReader(f))
    captured = float(rows[0]["max_abs_dgrad"])
    assert abs(captured - 7.0) < 1e-5, (
        f"expected running max 7.0, got {captured} "
        "(3.0 = first-write bug, 2.0 = last-write bug)"
    )


def test_dgrad_probe_detach_stops_hook_firing(tmp_path):
    """After detach_all, further forward/backward must not update state."""
    from latent_harness.core.probes import DgradProbe

    layer = torch.nn.Linear(4, 4)
    probe = DgradProbe(output_path=tmp_path / "dgrad.csv")
    probe.attach([layer], layer_names=["l0"])

    # First pass — probe active
    (layer(torch.randn(2, 4, requires_grad=True)).sum()).backward()
    probe.flush(step=0)
    probe.detach_all()

    # Second pass — probe detached, should NOT appear in the CSV
    (layer(torch.randn(2, 4, requires_grad=True) * 100).sum()).backward()
    probe.flush(step=1)  # will write NaN (no hook data collected)

    import csv as _csv
    with (tmp_path / "dgrad.csv").open() as f:
        rows = list(_csv.DictReader(f))
    # Row for step 1 should have NaN (no captured value after detach)
    step1_rows = [r for r in rows if r["step"] == "1"]
    assert len(step1_rows) == 1
    val = step1_rows[0]["max_abs_dgrad"]
    import math
    assert val == "nan" or math.isnan(float(val))


def test_dgrad_probe_attach_length_mismatch_raises(tmp_path):
    from latent_harness.core.probes import DgradProbe
    probe = DgradProbe(output_path=tmp_path / "dgrad.csv")
    layers = [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
    import pytest
    with pytest.raises(ValueError, match="layer count mismatch"):
        probe.attach(layers, layer_names=["only_one"])
