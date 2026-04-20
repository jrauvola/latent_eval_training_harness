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


def test_rmsnorm_denom_probe_captures_per_step(tmp_path):
    from latent_harness.core.probes import RMSNormDenomProbe

    # Minimal RMSNorm-like module
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    norm = RMSNorm(16)
    probe = RMSNormDenomProbe(output_path=tmp_path / "denom.csv")
    probe.attach({"q_norm_l0": norm})
    x = torch.randn(4, 16)
    _ = norm(x)
    probe.flush(step=0)
    probe.detach_all()
    content = (tmp_path / "denom.csv").read_text()
    assert "step,name,denom_min,denom_median,denom_max" in content
    assert "0,q_norm_l0," in content


def test_rmsnorm_denom_probe_captures_known_analytical_value(tmp_path):
    """For x = all-ones tensor, mean(x²) = 1.0, denom = 1/sqrt(1+eps) ≈ 1.0."""
    from latent_harness.core.probes import RMSNormDenomProbe

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    norm = RMSNorm(16, eps=1e-6)
    probe = RMSNormDenomProbe(output_path=tmp_path / "denom.csv")
    probe.attach({"l0": norm})
    _ = norm(torch.ones(4, 16))   # x² mean = 1.0 exactly → denom = rsqrt(1 + 1e-6) ≈ 0.9999995
    probe.flush(step=0)
    probe.detach_all()

    import csv as _csv
    with (tmp_path / "denom.csv").open() as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    # min == median == max since every element is identical
    for col in ("denom_min", "denom_median", "denom_max"):
        assert abs(float(row[col]) - 1.0) < 1e-4, f"{col}: {row[col]}"


def test_rmsnorm_denom_probe_last_forward_wins(tmp_path):
    """Unlike DgradProbe's running-max, RMSNormDenomProbe stores only the LAST forward's distribution.

    Three forwards with constant inputs at scales 3.0, 7.0, 2.0 (so denom is
    rsqrt(9), rsqrt(49), rsqrt(4) = 0.333, 0.143, 0.500).
    Last-forward-wins → emitted values reflect scale 2.0 → denom ≈ 0.5.
    Running-max would give 0.5 here (same answer, bad discriminator), so we
    additionally check the minimum is also 0.5 (not 0.143 from the middle forward).
    """
    from latent_harness.core.probes import RMSNormDenomProbe

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    norm = RMSNorm(16, eps=1e-6)
    probe = RMSNormDenomProbe(output_path=tmp_path / "denom.csv")
    probe.attach({"l0": norm})

    for scale in (3.0, 7.0, 2.0):
        _ = norm(torch.ones(4, 16) * scale)

    probe.flush(step=0)
    probe.detach_all()

    import csv as _csv
    with (tmp_path / "denom.csv").open() as f:
        rows = list(_csv.DictReader(f))
    # Expected denom for scale=2: rsqrt(4 + 1e-6) ≈ 0.5
    expected = 0.5
    # With constant input at scale 2.0, min == median == max ≈ 0.5.
    # Running-max bug would yield min ≈ 0.143 (scale=7 forward leaked in).
    # First-forward-wins bug would yield ≈ 0.333 (scale=3).
    for col in ("denom_min", "denom_median", "denom_max"):
        val = float(rows[0][col])
        assert abs(val - expected) < 1e-3, (
            f"{col}: expected {expected} (last-forward scale 2.0), got {val}. "
            f"running-max bug → ~0.143, first-write bug → ~0.333"
        )


def test_rmsnorm_denom_probe_detach_stops_hook_firing(tmp_path):
    """After detach_all, further forward must not update state."""
    from latent_harness.core.probes import RMSNormDenomProbe

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    norm = RMSNorm(16, eps=1e-6)
    probe = RMSNormDenomProbe(output_path=tmp_path / "denom.csv")
    probe.attach({"l0": norm})

    # First forward — hook fires
    _ = norm(torch.ones(4, 16))
    probe.flush(step=0)
    probe.detach_all()

    # Second forward with scale 100 — hook should NOT fire
    _ = norm(torch.ones(4, 16) * 100)
    probe.flush(step=1)

    import csv as _csv
    with (tmp_path / "denom.csv").open() as f:
        rows = list(_csv.DictReader(f))
    step1_rows = [r for r in rows if r["step"] == "1"]
    assert len(step1_rows) == 1
    # Step 1 should have empty values (no data captured)
    assert step1_rows[0]["denom_min"] in ("", "nan")
    assert step1_rows[0]["denom_median"] in ("", "nan")
    assert step1_rows[0]["denom_max"] in ("", "nan")


def test_probe_callback_attaches_and_flushes(tmp_path, monkeypatch):
    """Unit test for ProbeCallback: verifies on_train_begin attaches probes
    and on_step_end flushes them."""
    from latent_harness.training.trainer import ProbeCallback
    from latent_harness.core.config import LatentRuntimeConfig

    # Track calls via patched maybe_init_probes
    call_log = {"init": 0, "flush": 0, "detach": 0}

    class FakeProbe:
        def flush(self, step):
            call_log["flush"] += 1
        def detach_all(self):
            call_log["detach"] += 1

    def fake_init(model, runtime_config):
        call_log["init"] += 1
        return {"dgrad": FakeProbe(), "rmsnorm": None}

    monkeypatch.setattr("latent_harness.training.trainer.maybe_init_probes", fake_init)

    runtime_cfg = LatentRuntimeConfig(probe_mode=True, probe_output_dir=str(tmp_path))
    cb = ProbeCallback(runtime_config=runtime_cfg)

    # Simulate trainer lifecycle
    class _DummyState:
        global_step = 5
    cb.on_train_begin(args=None, state=_DummyState(), control=None, model=None)
    cb.on_step_end(args=None, state=_DummyState(), control=None)
    cb.on_train_end(args=None, state=_DummyState(), control=None)

    assert call_log == {"init": 1, "flush": 1, "detach": 1}


def test_probe_callback_skips_when_probe_mode_off(tmp_path):
    from latent_harness.training.trainer import ProbeCallback
    from latent_harness.core.config import LatentRuntimeConfig

    runtime_cfg = LatentRuntimeConfig()  # probe_mode=False by default
    cb = ProbeCallback(runtime_config=runtime_cfg)

    class _DummyState:
        global_step = 0
    cb.on_train_begin(args=None, state=_DummyState(), control=None, model=None)
    assert cb.probes is None
    # on_step_end and on_train_end should be no-ops (no exceptions)
    cb.on_step_end(args=None, state=_DummyState(), control=None)
    cb.on_train_end(args=None, state=_DummyState(), control=None)


# ---------- Wrapper-walker regression tests (feature/probe-walker-fix) ----------
#
# These tests exercise ``maybe_init_probes`` against toy modules that mimic the
# Qwen3 and Gemma-3 wrapper chains we observe at runtime. They are the
# regression harness for the 16-hop walker bug where Gemma-3 dgrad CSVs never
# got produced because the walker only followed ``.model``/``.base_model`` and
# missed ``Gemma3Model.language_model``.


def _make_toy_layers(n: int = 4) -> torch.nn.ModuleList:
    return torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(n)])


def test_maybe_init_probes_walks_qwen3_style_chain(tmp_path):
    """Simulates ``LatentReasoningRuntime → PeftModel → Qwen3ForCausalLM → Qwen3Model → layers``."""
    from latent_harness.core.config import LatentRuntimeConfig
    from latent_harness.core.probes import maybe_init_probes

    layers = _make_toy_layers()

    class FakeQwen3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers

    class FakeQwen3ForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeQwen3Model()

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # PeftModel delegates ``.model`` to base_model.model in practice;
            # for the walker we just need the attribute chain to resolve.
            self.model = FakeQwen3ForCausalLM()

    class FakeRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakePeftModel()

    cfg = LatentRuntimeConfig(probe_mode=True, probe_output_dir=str(tmp_path))
    probes = maybe_init_probes(FakeRuntime(), cfg)
    assert probes is not None, "walker should locate Qwen3 .layers via .model chain"
    assert probes.get("dgrad") is not None
    probes["dgrad"].detach_all()


def test_maybe_init_probes_walks_gemma3_conditional_generation_chain(tmp_path):
    """Regression: ``LatentReasoningRuntime → PeftModel → Gemma3ForConditionalGeneration
    → Gemma3Model → language_model (Gemma3TextModel) → layers``.

    Before the fix, the 16-hop walker followed ``.model``/``.base_model`` only
    and got stuck at ``Gemma3Model`` (which has ``.language_model`` but neither
    ``.layers`` nor ``.model`` nor ``.base_model``), producing no dgrad CSV.
    """
    from latent_harness.core.config import LatentRuntimeConfig
    from latent_harness.core.probes import maybe_init_probes

    layers = _make_toy_layers()

    class FakeGemma3TextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers

    class FakeGemma3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Note: no ``.layers`` / ``.model`` / ``.base_model`` — only
            # ``.language_model``. This is what trips the old walker.
            self.language_model = FakeGemma3TextModel()

    class FakeGemma3ForCondGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeGemma3Model()

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeGemma3ForCondGen()

    class FakeRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakePeftModel()

    cfg = LatentRuntimeConfig(probe_mode=True, probe_output_dir=str(tmp_path))
    probes = maybe_init_probes(FakeRuntime(), cfg)
    assert probes is not None, (
        "walker should locate Gemma-3 .layers via .language_model branch; "
        "failure here reproduces the 16-hop walker bug."
    )
    assert probes.get("dgrad") is not None
    probes["dgrad"].detach_all()


def test_maybe_init_probes_prefers_runtime_layers_property(tmp_path):
    """If the runtime exposes a ``.layers`` property, ``maybe_init_probes``
    should use it directly instead of walking wrappers. This guarantees the
    Gemma-3 fix works even if a future HF release renames the inner
    attribute (e.g. ``.text_model`` vs ``.language_model``)."""
    from latent_harness.core.config import LatentRuntimeConfig
    from latent_harness.core.probes import maybe_init_probes

    canonical_layers = _make_toy_layers()
    decoy_layers = _make_toy_layers()

    class FakeRuntime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Decoy ``.model.layers`` that the walker WOULD pick up if the
            # property weren't preferred.
            self._inner = torch.nn.Module()
            self._inner.layers = decoy_layers

        @property
        def model(self):  # walker would descend here
            return self._inner

        @property
        def layers(self):  # property takes precedence over the walker
            return canonical_layers

    probes = maybe_init_probes(
        FakeRuntime(),
        LatentRuntimeConfig(probe_mode=True, probe_output_dir=str(tmp_path)),
    )
    assert probes is not None
    # The probe's layer_names length must match the CANONICAL layers, not the decoy.
    # We expose this via _layer_names on DgradProbe.
    assert len(probes["dgrad"]._layer_names) == len(canonical_layers)
    probes["dgrad"].detach_all()


def test_maybe_init_probes_returns_none_when_chain_has_no_layers(tmp_path):
    """If the wrapper chain truly has no decoder stack, degrade gracefully."""
    from latent_harness.core.config import LatentRuntimeConfig
    from latent_harness.core.probes import maybe_init_probes

    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)  # no .layers, .model, .base_model

    probes = maybe_init_probes(
        Bare(),
        LatentRuntimeConfig(probe_mode=True, probe_output_dir=str(tmp_path)),
    )
    assert probes is None
