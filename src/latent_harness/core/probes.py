from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DgradProbe:
    """Capture per-layer max |dL/dhidden| via backward hooks.

    Usage:
        probe = DgradProbe(output_path=Path("dgrad.csv"))
        probe.attach(model.model.layers, layer_names=[f"layer_{i}" for i in range(len(...))])
        ... (training step: forward + backward) ...
        probe.flush(step=global_step)
        probe.detach_all()  # at end of run
    """

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current: dict[str, float] = {}
        self._wrote_header = False
        self._layer_names: list[str] = []

    def attach(self, layers: Iterable[nn.Module], layer_names: list[str]) -> None:
        layers_list = list(layers)
        if len(layers_list) != len(layer_names):
            raise ValueError(
                f"layer count mismatch: {len(layers_list)} modules vs "
                f"{len(layer_names)} names"
            )
        self._layer_names = list(layer_names)
        for name, layer in zip(self._layer_names, layers_list):
            handle = layer.register_full_backward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            # grad_output is a tuple; take the first element (output-side gradient)
            if grad_output and grad_output[0] is not None:
                val = grad_output[0].detach().abs().max().item()
                prev = self._current.get(name)
                if prev is None or val > prev:
                    self._current[name] = val
        return hook

    def flush(self, step: int) -> None:
        with self.output_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self._wrote_header:
                writer.writerow(["step", "layer", "max_abs_dgrad"])
                self._wrote_header = True
            for name in self._layer_names:
                writer.writerow([step, name, self._current.get(name, float("nan"))])
        self._current.clear()

    def detach_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def maybe_init_probes(model, runtime_config) -> dict | None:
    """If runtime_config.probe_mode is True, attach DgradProbe + optional RMSNormDenomProbe.

    Returns:
        None if probe_mode is False.
        Otherwise dict: {"dgrad": DgradProbe, "rmsnorm": RMSNormDenomProbe | None}.
        Caller is responsible for calling flush(step) per optimizer step and detach_all() at end.
    """
    if not getattr(runtime_config, "probe_mode", False):
        return None

    if model is None:
        logger.error(
            "maybe_init_probes called with model=None but probe_mode=True. "
            "Probes will NOT be attached. Check trainer callback wiring."
        )
        return None

    out_dir = Path(runtime_config.probe_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk the model wrappers (PEFT/LoRA/etc.) looking for the decoder layer stack.
    m = model
    layers = None
    # Safety guard against infinite loops from self-referential wrappers.
    for _ in range(16):
        if hasattr(m, "layers") and isinstance(m.layers, torch.nn.ModuleList):
            layers = m.layers
            break
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break

    if layers is None:
        logger.error(
            "maybe_init_probes: could not locate decoder .layers ModuleList by walking "
            "model wrapper chain (16-hop limit). Probes will NOT be attached. "
            "Model top-level type: %s. Check that the base model exposes .model.layers "
            "or set probe_mode=False.",
            type(model).__name__,
        )
        return None

    dgrad_probe = DgradProbe(output_path=out_dir / "dgrad_per_layer.csv")
    dgrad_probe.attach(layers, [f"layer_{i}" for i in range(len(layers))])

    rmsnorm_probe = None
    if (
        len(layers) > 0
        and hasattr(layers[0], "self_attn")
        and hasattr(layers[0].self_attn, "q_norm")
        and hasattr(layers[0].self_attn, "k_norm")
    ):
        rmsnorm_probe = RMSNormDenomProbe(output_path=out_dir / "qk_rmsnorm_denom.csv")
        rmsnorm_probe.attach(
            {
                "q_norm_l0": layers[0].self_attn.q_norm,
                "k_norm_l0": layers[0].self_attn.k_norm,
            }
        )

    return {"dgrad": dgrad_probe, "rmsnorm": rmsnorm_probe}


class RMSNormDenomProbe:
    """Capture distribution of 1/sqrt(mean(x^2) + eps) at selected RMSNorm modules.

    Logs per-step quantiles (min, median, max) of the denominator across the
    batch × sequence axes. Used to test whether Q/K RMSNorm denominator
    variability correlates with gradient instability in Gemma-3.
    """

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current: dict[str, torch.Tensor] = {}
        self._wrote_header = False
        self._names: list[str] = []

    def attach(self, named_modules: dict[str, nn.Module]) -> None:
        self._names = list(named_modules.keys())
        for name, mod in named_modules.items():
            handle = mod.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inp, output):
            # Recover denom from the input tensor. Works for standard RMSNorm.
            x = inp[0] if isinstance(inp, tuple) else inp
            eps = getattr(module, "eps", 1e-6)
            with torch.no_grad():
                var = x.detach().to(torch.float32).pow(2).mean(dim=-1)
                denom = torch.rsqrt(var + eps)
                self._current[name] = denom.flatten().cpu()
        return hook

    def flush(self, step: int) -> None:
        with self.output_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self._wrote_header:
                writer.writerow(["step", "name", "denom_min", "denom_median", "denom_max"])
                self._wrote_header = True
            for name in self._names:
                t = self._current.get(name)
                if t is None:
                    writer.writerow([step, name, "", "", ""])
                else:
                    writer.writerow([
                        step, name,
                        float(t.min()), float(t.median()), float(t.max()),
                    ])
        self._current.clear()

    def detach_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
