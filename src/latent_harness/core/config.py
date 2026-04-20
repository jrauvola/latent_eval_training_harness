from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def resolve_hf_hub_token(explicit: str | None = None) -> str | None:
    """Prefer YAML/model `hf_token`, then env vars, then the default CLI token file."""
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip() or None
    path = Path.home() / ".cache/huggingface/token"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


@dataclass(slots=True)
class ModelConfig:
    base_model_name_or_path: str
    hf_token: str | None = None
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None
    lora_init: bool = True
    use_lora: bool = True
    full_precision: bool = True
    load_in_4bit: bool = False
    freeze_base_embeddings: bool = False


@dataclass(slots=True)
class LatentRuntimeConfig:
    model_max_length: int = 512
    num_latent: int = 6
    #: Detach the latent hidden state between non-final latent rollout steps.
    detach_latent_between_steps: bool = False
    #: Detach the KV cache between non-final latent rollout steps (requires DynamicCache support).
    detach_cache_between_steps: bool = False
    #: Number of latent steps at the END of the loop to keep gradient-connected.
    #: None or 1 = current behavior (only the final latent step's gradient reaches the decoder).
    #: K > 1 = keep last K latent-step boundaries connected (requires num_latent >= K).
    detach_keep_last_k: int | None = None
    #: Position-based detach mode for the KV cache.
    #: "all" (default): detach all positions (current behavior).
    #: "reasoning_only": detach only question-origin positions [0, encoder_length);
    #: latent-origin positions keep grad_fn so gradient flows between latent steps.
    detach_position_mode: str = "all"
    use_prj: bool = True
    prj_dim: int = 2048
    prj_dropout: float = 0.0
    prj_no_ln: bool = False
    #: Run the projection MLP in float32 for numerical stability (activations cast back after).
    prj_fp32: bool = False
    #: Use ``x + alpha * F(x)`` with trainable ``alpha`` (ReZero-style) instead of ``F(x)`` alone.
    prj_residual_gated: bool = False
    #: Initial value for the residual gate ``alpha`` (typically 0.0 so the loop starts near identity).
    prj_gate_init: float = 0.0
    distill_loss_div_std: bool = True
    distill_loss_std_floor: float = 1e-6
    distill_loss_type: str = "smooth_l1"
    distill_loss_factor: float = 20.0
    ref_loss_factor: float = 1.0
    remove_eos: bool = True
    bf16: bool = True
    #: Force fp32 throughout. Overrides bf16. On CUDA this is the ONLY way to get true fp32 —
    #: setting bf16=False alone falls back to fp16 (not fp32) in runtime.py's dtype selection.
    #: Set fp32=True (and bf16=False) when you want real fp32 training as a precision control.
    fp32: bool = False
    seed: int = 11
    #: Wrap each benchmark prompt as a chat user turn (Qwen3, Llama-Instruct, etc.).
    use_chat_template: bool = False
    #: Extra kwargs for ``tokenizer.apply_chat_template`` (e.g. ``enable_thinking`` for Qwen3).
    chat_template_kwargs: dict[str, Any] | None = None
    #: When True, register per-layer dgrad + (optional) RMSNorm denom probes
    #: and write per-step CSV to probe_output_dir. Off by default.
    probe_mode: bool = False
    #: Directory for probe CSV output. Required when probe_mode is True.
    probe_output_dir: str | None = None
    # --- SIM-CoT auxiliary step-decoder (method="sim_cot") ----------------------
    #: Attach an auxiliary step decoder that predicts each explicit CoT step from
    #: the corresponding latent embedding. Train-only; dropped on checkpoint save.
    aux_decoder_enabled: bool = False
    #: When True, the auxiliary decoder is a full second LM copy of the base
    #: (SIM-CoT paper default — 2x model memory). When False, a shared-lm_head
    #: approximation is used (cheaper; single-layer transformer block + the base
    #: model's lm_head). Flip to False as a memory fallback. See spec §8.
    aux_decoder_full_lm: bool = True
    #: Weight on the explain-step CE averaged over effective steps.
    #: Spec: ``total = ce + 20*distill + ref_ce + explain_loss_factor * (explain / max(1, effective_steps))``.
    aux_decoder_explain_loss_factor: float = 1.0
    #: Hidden size of the fallback shared-lm_head decoder block. Only used when
    #: aux_decoder_full_lm=False. Defaults to a single-layer transformer.
    aux_decoder_shared_head_layers: int = 1

    def __post_init__(self) -> None:
        if self.detach_keep_last_k is not None:
            if self.detach_keep_last_k < 1:
                raise ValueError(
                    f"detach_keep_last_k must be >= 1 (got {self.detach_keep_last_k}); "
                    "use None for current behavior."
                )
            if self.detach_keep_last_k > self.num_latent:
                raise ValueError(
                    f"detach_keep_last_k ({self.detach_keep_last_k}) must not exceed "
                    f"num_latent ({self.num_latent})."
                )
        if self.detach_position_mode not in {"all", "reasoning_only"}:
            raise ValueError(
                f"detach_position_mode must be 'all' or 'reasoning_only' "
                f"(got {self.detach_position_mode!r})."
            )
        if self.fp32 and self.bf16:
            raise ValueError(
                "fp32 and bf16 are mutually exclusive; set at most one to True "
                "(fp32=True forces full 32-bit precision; bf16=True uses bfloat16 on CUDA)."
            )
        if self.probe_mode and not self.probe_output_dir:
            raise ValueError(
                "probe_mode=True requires probe_output_dir to be set "
                "(give a writable directory path)."
            )
