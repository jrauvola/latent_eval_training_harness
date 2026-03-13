from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    base_model_name_or_path: str
    hf_token: str | None = None
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_init: bool = True
    use_lora: bool = True
    full_precision: bool = True
    load_in_4bit: bool = False


@dataclass(slots=True)
class LatentRuntimeConfig:
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
