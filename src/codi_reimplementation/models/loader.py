from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from codi_reimplementation.training.codi_model import CODIRuntime, load_checkpoint_state
from codi_reimplementation.training.config import CodiModelConfig, CodiRuntimeConfig


@dataclass(slots=True)
class EvalModelSpec:
    name: str
    checkpoint_source: str
    base_model_name_or_path: str
    checkpoint_type: str = "hf_repo"
    hf_token: str | None = None
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_init: bool = True
    use_lora: bool = True
    full_precision: bool = True
    load_in_4bit: bool = False
    model_max_length: int = 512
    num_latent: int = 6
    use_prj: bool = True
    prj_dim: int = 2048
    prj_dropout: float = 0.0
    prj_no_ln: bool = False
    distill_loss_div_std: bool = True
    distill_loss_type: str = "smooth_l1"
    distill_loss_factor: float = 20.0
    ref_loss_factor: float = 1.0
    remove_eos: bool = True
    bf16: bool = True
    seed: int = 11


def _resolve_checkpoint_path(spec: EvalModelSpec) -> str:
    if spec.checkpoint_type == "hf_repo":
        filename = "model.safetensors"
        try:
            return hf_hub_download(
                repo_id=spec.checkpoint_source,
                filename=filename,
                token=spec.hf_token,
            )
        except Exception:
            return hf_hub_download(
                repo_id=spec.checkpoint_source,
                filename="pytorch_model.bin",
                token=spec.hf_token,
            )
    source = Path(spec.checkpoint_source).expanduser().resolve()
    if source.is_file():
        return str(source)
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        candidate_path = source / candidate
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(f"Could not locate checkpoint under {source}")


def load_eval_model(spec: EvalModelSpec, device: str | torch.device | None = None) -> CODIRuntime:
    runtime = CODIRuntime(
        model_config=CodiModelConfig(
            base_model_name_or_path=spec.base_model_name_or_path,
            hf_token=spec.hf_token,
            lora_r=spec.lora_r,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            lora_init=spec.lora_init,
            use_lora=spec.use_lora,
            full_precision=spec.full_precision,
            load_in_4bit=spec.load_in_4bit,
        ),
        runtime_config=CodiRuntimeConfig(
            model_max_length=spec.model_max_length,
            num_latent=spec.num_latent,
            use_prj=spec.use_prj,
            prj_dim=spec.prj_dim,
            prj_dropout=spec.prj_dropout,
            prj_no_ln=spec.prj_no_ln,
            distill_loss_div_std=spec.distill_loss_div_std,
            distill_loss_type=spec.distill_loss_type,
            distill_loss_factor=spec.distill_loss_factor,
            ref_loss_factor=spec.ref_loss_factor,
            remove_eos=spec.remove_eos,
            bf16=spec.bf16,
            seed=spec.seed,
        ),
        train_mode=False,
    )
    state_dict = load_checkpoint_state(_resolve_checkpoint_path(spec))
    runtime.load_state_dict(state_dict, strict=False)
    runtime.tie_weights_if_needed()
    if device is not None:
        runtime = runtime.to(device)
    runtime.eval()
    return runtime
