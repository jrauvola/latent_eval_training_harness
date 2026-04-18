from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


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
