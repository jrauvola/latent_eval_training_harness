from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Default layer indices sampled by PerModuleGradProbe in addition to layer 0.
# Layer 0 receives FULL coverage (all q/k/v/o × lora_A/B). These layers receive
# spot coverage (same modules but only if present in the model).
DEFAULT_SPOT_LAYERS: tuple[int, ...] = (1, 5, 10, 20, 35)

# LoRA submodule kinds we log. Ordering matches the attn projection convention
# q → k → v → o, with lora_A before lora_B for each.
DEFAULT_SUBMODULES: tuple[str, ...] = (
    "q_proj.lora_A",
    "q_proj.lora_B",
    "k_proj.lora_A",
    "k_proj.lora_B",
    "v_proj.lora_A",
    "v_proj.lora_B",
    "o_proj.lora_A",
    "o_proj.lora_B",
)


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
    """If runtime_config.probe_mode is True, attach DgradProbe + optional probes.

    Returns:
        None if probe_mode is False.
        Otherwise dict:
          - "dgrad":      DgradProbe (always when probe_mode=True)
          - "rmsnorm":    RMSNormDenomProbe | None (if q_norm/k_norm present)
          - "per_module": PerModuleGradProbe | None (if enable_per_module_grad_probe=True)
        Caller is responsible for calling the appropriate flush/capture method per
        step and detach_all() at end.
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

    # Primary path: LatentReasoningRuntime exposes a ``.layers`` property that
    # knows how to unwrap the PEFT+Gemma3/Qwen3 wrapper chain. Preferred over
    # the generic walker because it handles Gemma-3's multimodal layout where
    # the decoder stack lives at ``.model.language_model.layers`` (the walker
    # follows ``.model``/``.base_model`` only and misses ``.language_model``).
    layers = None
    try:
        candidate = getattr(model, "layers", None)
    except AttributeError:
        # Property getter raised — fall through to the walker.
        candidate = None
    if isinstance(candidate, torch.nn.ModuleList):
        layers = candidate

    # Fallback: walk the model wrappers (PEFT/LoRA/etc.) looking for the
    # decoder layer stack. Kept for models that don't route through
    # ``LatentReasoningRuntime`` (e.g. raw HF CausalLM in a test harness).
    if layers is None:
        m = model
        # Safety guard against infinite loops from self-referential wrappers.
        for _ in range(16):
            if hasattr(m, "layers") and isinstance(m.layers, torch.nn.ModuleList):
                layers = m.layers
                break
            # Prefer ``.language_model`` (Gemma-3 ForConditionalGeneration's
            # text stack) before the generic ``.model`` / ``.base_model``
            # delegates, so the walker doesn't get stuck at Gemma3Model.
            if hasattr(m, "language_model"):
                m = m.language_model
            elif hasattr(m, "model"):
                m = m.model
            elif hasattr(m, "base_model"):
                m = m.base_model
            else:
                break

    if layers is None:
        logger.error(
            "maybe_init_probes: could not locate decoder .layers ModuleList via "
            "runtime.layers property or by walking model wrapper chain (16-hop limit). "
            "Probes will NOT be attached. Model top-level type: %s. Check that the "
            "runtime exposes .layers or that the base model has .model.layers; "
            "alternatively set probe_mode=False.",
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

    per_module_probe = None
    if getattr(runtime_config, "enable_per_module_grad_probe", False):
        # Default coverage: layer 0 (full) + spot layers {1, 5, 10, 20, 35}.
        # Filter spot layers to ones that actually exist in this model so we
        # don't silently skip — the probe will log a warning if nothing matches
        # (e.g. spec drift / wrong model family).
        spot = [i for i in DEFAULT_SPOT_LAYERS if i < len(layers)]
        layer_indices = sorted({0, *spot})
        per_module_probe = PerModuleGradProbe(
            output_path=out_dir / "per_module_grad.csv",
            layer_indices=layer_indices,
        )
        per_module_probe.attach(model, layer_indices=layer_indices)
        logger.info(
            "PerModuleGradProbe attached: %d targets across layers %s",
            len(per_module_probe._targets), layer_indices,
        )

    return {
        "dgrad": dgrad_probe,
        "rmsnorm": rmsnorm_probe,
        "per_module": per_module_probe,
    }


class PerModuleGradProbe:
    """Capture per-module parameter gradient norms for LoRA A/B weights.

    Complements DgradProbe (which logs layer-output / hidden-state gradient).
    DgradProbe sees the symptom of propagation (bounded ~1-4 even at crash step),
    whereas PerModuleGradProbe sees the underlying LoRA adapter weight gradient
    where the NaN actually originates (e.g. ``q_proj.lora_A.default.weight``
    at layer 0 blowing up before the hidden-state gradient does).

    Unlike the hook-based probes, this reads ``param.grad`` directly after
    backward. Must be called between ``backward()`` and ``optimizer.zero_grad()``
    — i.e. from the Trainer's ``on_pre_optimizer_step`` callback. Calling from
    ``on_step_end`` will read zeros because HF zeros grads after the step.

    Usage:
        probe = PerModuleGradProbe(output_path=Path("per_module_grad.csv"))
        probe.attach(model, layer_indices=[0, 1, 5, 10, 35])
        ... (training step: forward + backward; optimizer NOT stepped yet) ...
        probe.capture(step=global_step)
        probe.detach_all()  # at end of run
    """

    # Match  base_model.model.model.layers.<N>.self_attn.<PROJ>_proj.lora_<A|B>.default.weight
    # (PEFT standard layout). ``N`` is captured so we can filter by layer.
    _PARAM_RE = re.compile(
        r".*\.layers\.(?P<layer>\d+)\.self_attn\."
        r"(?P<proj>[qkvo])_proj\.lora_(?P<ab>[AB])\.default\.weight$"
    )

    def __init__(
        self,
        output_path: Path | str,
        layer_indices: Iterable[int] | None = None,
        submodules: Iterable[str] = DEFAULT_SUBMODULES,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()
        self._wrote_header = False
        # Ordered list of (param_full_name, layer_idx, submodule) we log per step.
        self._targets: list[tuple[str, int, str]] = []
        # Mapping param_full_name -> torch.nn.Parameter for fast lookup at capture time.
        self._params: dict[str, torch.nn.Parameter] = {}
        # Allowed layers (None = accept any layer that matches the regex).
        self._layer_set: set[int] | None = (
            set(int(i) for i in layer_indices) if layer_indices is not None else None
        )
        self._submodules = tuple(submodules)

    def attach(self, model: nn.Module, layer_indices: Iterable[int] | None = None) -> None:
        """Discover target LoRA params by walking ``model.named_parameters()``.

        If ``layer_indices`` is supplied here it overrides the constructor value.
        """
        if layer_indices is not None:
            self._layer_set = set(int(i) for i in layer_indices)

        # Compile a set of allowed submodule tokens like ``"q_proj.lora_A"``.
        allowed_submodules = set(self._submodules)

        discovered: list[tuple[str, int, str, torch.nn.Parameter]] = []
        for full_name, param in model.named_parameters():
            m = self._PARAM_RE.match(full_name)
            if m is None:
                continue
            layer_idx = int(m.group("layer"))
            if self._layer_set is not None and layer_idx not in self._layer_set:
                continue
            submodule = f"{m.group('proj')}_proj.lora_{m.group('ab')}"
            if submodule not in allowed_submodules:
                continue
            discovered.append((full_name, layer_idx, submodule, param))

        # Sort for stable CSV column order: layer first, then our canonical submodule order.
        submodule_rank = {s: i for i, s in enumerate(self._submodules)}
        discovered.sort(
            key=lambda t: (t[1], submodule_rank.get(t[2], len(submodule_rank)), t[0])
        )
        for full_name, layer_idx, submodule, param in discovered:
            self._targets.append((full_name, layer_idx, submodule))
            self._params[full_name] = param

        if not self._targets:
            logger.warning(
                "PerModuleGradProbe.attach: no LoRA A/B parameters matched the "
                "expected pattern ``.layers.<N>.self_attn.<proj>_proj.lora_<A|B>."
                "default.weight``. Dumping first 10 trainable parameter names as "
                "discovery aid: %s",
                [n for n, p in model.named_parameters() if p.requires_grad][:10],
            )

    def capture(self, step: int) -> None:
        """Read ``.grad`` off every target parameter and append a row per target.

        Must be called BEFORE the optimizer step (after backward, before
        ``optimizer.zero_grad()``). If a target's grad is ``None`` the row is
        still written with NaN so downstream heatmaps stay rectangular.
        """
        with self.output_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self._wrote_header:
                writer.writerow(
                    ["step", "module_name", "layer_idx", "submodule",
                     "max_abs_grad", "mean_abs_grad"]
                )
                self._wrote_header = True
            for full_name, layer_idx, submodule in self._targets:
                param = self._params.get(full_name)
                if param is None or param.grad is None:
                    writer.writerow([step, full_name, layer_idx, submodule,
                                     float("nan"), float("nan")])
                    continue
                g = param.grad.detach()
                # Cast to fp32 for stable reduction across bf16/fp16 grads.
                g_abs = g.to(torch.float32).abs()
                writer.writerow([
                    step, full_name, layer_idx, submodule,
                    float(g_abs.max().item()),
                    float(g_abs.mean().item()),
                ])

    def detach_all(self) -> None:
        # No hooks to unregister — ``.grad`` reads are passive — but clear
        # state for symmetry with the other probes.
        self._params.clear()
        self._targets.clear()


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
