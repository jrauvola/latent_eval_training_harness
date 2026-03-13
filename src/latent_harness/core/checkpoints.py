from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


def load_checkpoint_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint)
    if checkpoint_path.endswith(".safetensors"):
        return load_safetensors(checkpoint_path)
    return torch.load(checkpoint_path, map_location="cpu")


def remap_runtime_state_dict_prefixes(
    state_dict: dict[str, torch.Tensor],
    *,
    target_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Translate legacy runtime prefixes after internal refactors.

    The shared runtime used to store backbone weights under ``codi.*`` and now
    stores them under ``model.*``. Official CODI checkpoints and older local
    runs still use the legacy prefix, so loading them without remapping silently
    drops most parameters when ``strict=False`` is used.
    """

    if not state_dict:
        return state_dict

    if any(key.startswith("model.") for key in state_dict):
        return state_dict

    if any(key.startswith("codi.") for key in state_dict) and any(key.startswith("model.") for key in target_keys):
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("codi."):
                remapped[f"model.{key[len('codi.') :]}"] = value
            else:
                remapped[key] = value
        return remapped

    return state_dict


def resolve_checkpoint_path(
    checkpoint_source: str | Path,
    checkpoint_type: str = "hf_repo",
    *,
    token: str | None = None,
) -> str:
    if checkpoint_type == "hf_repo":
        for filename in ("model.safetensors", "pytorch_model.bin"):
            try:
                return hf_hub_download(
                    repo_id=str(checkpoint_source),
                    filename=filename,
                    token=token,
                )
            except Exception:
                continue
        raise FileNotFoundError(f"Could not resolve checkpoint files for repo {checkpoint_source!r}")

    source = Path(checkpoint_source).expanduser().resolve()
    if source.is_file():
        return str(source)
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        candidate_path = source / candidate
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(f"Could not locate checkpoint under {source}")
