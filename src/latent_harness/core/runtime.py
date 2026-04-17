from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig, resolve_hf_hub_token


class NumericalInstabilityError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        tensor_name: str,
        summary: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.details = {
            "stage": stage,
            "tensor_name": tensor_name,
            "summary": summary,
            "extra": extra or {},
        }
        message = f"Non-finite tensor detected stage={stage} tensor={tensor_name} summary={summary}"
        if extra:
            message = f"{message} extra={extra}"
        super().__init__(message)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        summary.update(
            {
                "finite_count": 0,
                "nonfinite_count": 0,
                "absmax": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        )
        return summary

    float_view = detached.float()
    finite_mask = torch.isfinite(float_view)
    finite_count = int(finite_mask.sum().item())
    summary["finite_count"] = finite_count
    summary["nonfinite_count"] = int(detached.numel()) - finite_count
    if finite_count == 0:
        summary.update(
            {
                "absmax": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        )
        return summary

    finite_values = float_view[finite_mask]
    summary.update(
        {
            "absmax": float(finite_values.abs().max().cpu()),
            "min": float(finite_values.min().cpu()),
            "max": float(finite_values.max().cpu()),
        }
    )
    return summary


def get_lora_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if any(name in lowered for name in ("llama", "mistral", "falcon", "qwen", "gemma")):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    if "phi" in lowered:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    if "gpt2" in lowered:
        return ["c_attn", "c_proj", "c_fc"]
    raise ValueError(f"Unsupported model family for LoRA mapping: {model_name}")


def get_modules_to_save(model_name: str) -> list[str]:
    if "gpt2" in model_name.lower():
        return ["wte", "lm_head"]
    return ["embed_tokens", "lm_head"]


def maybe_prepare_gemma4_peft(model_name: str) -> None:
    lowered = model_name.lower()
    if "gemma-4" not in lowered and "gemma4" not in lowered:
        return
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError as exc:
        raise RuntimeError(
            "Gemma 4 support requires a transformers build that includes transformers.models.gemma4."
        ) from exc

    # PEFT expects Linear-like modules that expose weight metadata directly.
    if nn.Linear not in Gemma4ClippableLinear.__bases__:
        Gemma4ClippableLinear.__bases__ = (nn.Linear,)
    if not isinstance(getattr(Gemma4ClippableLinear, "weight", None), property):
        Gemma4ClippableLinear.weight = property(lambda self: self.linear.weight)
    if not isinstance(getattr(Gemma4ClippableLinear, "bias", None), property):
        Gemma4ClippableLinear.bias = property(lambda self: self.linear.bias)
    if not isinstance(getattr(Gemma4ClippableLinear, "in_features", None), property):
        Gemma4ClippableLinear.in_features = property(lambda self: self.linear.in_features)
    if not isinstance(getattr(Gemma4ClippableLinear, "out_features", None), property):
        Gemma4ClippableLinear.out_features = property(lambda self: self.linear.out_features)


def _detach_cache(
    past_key_values: Any,
    *,
    detach_up_to_pos: int | None = None,
) -> Any:
    """Detach KV cache tensors from the autograd graph.

    Handles both legacy tuple-of-tuples caches and the ``DynamicCache``
    objects used by modern transformers (5.x+).

    Args:
        past_key_values: KV cache to detach.
        detach_up_to_pos: If ``None`` (default), detach all positions in
            every tensor in every layer entry (legacy full-detach
            behavior). If an integer, detach only positions
            ``[0, detach_up_to_pos)`` along the sequence-length axis of
            the K and V tensors; positions ``[detach_up_to_pos, end]``
            retain their ``grad_fn`` so gradients continue to flow.
            Non-K/V tensor elements in a layer entry are detached
            wholesale in this mode.
    """
    if past_key_values is None:
        return None

    def _detach_tensor_full(t: torch.Tensor) -> torch.Tensor:
        return t.detach()

    def _slice_detach(t: torch.Tensor, cutoff: int) -> torch.Tensor:
        seq_len = t.size(-2)
        cutoff = min(cutoff, seq_len)
        if cutoff <= 0:
            return t
        if cutoff >= seq_len:
            return t.detach()
        return torch.cat([t[..., :cutoff, :].detach(), t[..., cutoff:, :]], dim=-2)

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx, layer_data in enumerate(past_key_values):
            k, v = layer_data[0], layer_data[1]
            if detach_up_to_pos is None:
                k_new, v_new = _detach_tensor_full(k), _detach_tensor_full(v)
            else:
                k_new = _slice_detach(k, detach_up_to_pos)
                v_new = _slice_detach(v, detach_up_to_pos)
            new_cache.update(k_new, v_new, layer_idx)
        return new_cache

    if isinstance(past_key_values, (tuple, list)):
        out = []
        for layer_kv in past_key_values:
            if not isinstance(layer_kv, (tuple, list)):
                out.append(layer_kv)
                continue
            if detach_up_to_pos is None:
                out.append(
                    tuple(
                        _detach_tensor_full(t) if isinstance(t, torch.Tensor) else t
                        for t in layer_kv
                    )
                )
                continue
            if len(layer_kv) < 2 or not isinstance(layer_kv[0], torch.Tensor):
                out.append(tuple(layer_kv) if isinstance(layer_kv, list) else layer_kv)
                continue
            k, v = layer_kv[0], layer_kv[1]
            k_new = _slice_detach(k, detach_up_to_pos)
            v_new = _slice_detach(v, detach_up_to_pos)
            rest = tuple(
                _detach_tensor_full(t) if isinstance(t, torch.Tensor) else t
                for t in layer_kv[2:]
            )
            out.append((k_new, v_new) + rest)
        return tuple(out)

    return past_key_values


def _resolve_should_detach(
    *,
    latent_index: int,
    num_latent: int,
    keep_last_k: int | None,
) -> bool:
    """Decide whether to detach at the boundary AFTER latent step ``latent_index``.

    Only called for non-final boundaries (``latent_index < num_latent - 1``).

    Rules:
    - ``keep_last_k`` in ``(None, 1)``: detach at every non-final boundary
      (current behavior).
    - ``keep_last_k == K`` with ``K > 1``: detach only at boundaries
      ``i < num_latent - K``. This keeps the last ``K`` latent steps'
      gradients connected.
    """
    if keep_last_k is None or keep_last_k <= 1:
        return True
    return latent_index < num_latent - keep_last_k


def _resolve_runtime_dtype(runtime_config: Any) -> torch.dtype:
    """Select the runtime dtype based on config flags and hardware.

    Precedence:
    - ``fp32=True``: ``torch.float32`` (overrides everything).
    - On CUDA: ``bf16=True`` → ``torch.bfloat16``, else ``torch.float16``.
    - On CPU: always ``torch.float32``.

    The common gotcha: on CUDA, ``bf16=False`` alone gives fp16, NOT fp32. Use
    ``fp32=True`` when you actually want 32-bit precision.
    """
    if getattr(runtime_config, "fp32", False):
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if runtime_config.bf16 else torch.float16
    return torch.float32


def _apply_boundary_detach(
    *,
    cache: Any,
    latent: torch.Tensor,
    encoder_length: int,
    runtime_config_detach_latent: bool,
    runtime_config_detach_cache: bool,
    detach_position_mode: str,
) -> tuple[Any, torch.Tensor]:
    """Apply latent-embedding and KV-cache detach at a non-final boundary.

    Returns the (possibly-detached) cache and latent. Called only when
    ``_resolve_should_detach`` decided this boundary should be detached.
    """
    new_latent = latent.detach() if runtime_config_detach_latent else latent
    new_cache = cache
    if runtime_config_detach_cache:
        if detach_position_mode == "reasoning_only":
            detach_up_to = encoder_length
        else:
            detach_up_to = None
        new_cache = _detach_cache(cache, detach_up_to_pos=detach_up_to)
    return new_cache, new_latent


def get_text_config_attr(config: Any, attr_name: str) -> Any:
    if hasattr(config, attr_name):
        return getattr(config, attr_name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, attr_name):
        return getattr(text_config, attr_name)
    raise AttributeError(f"Could not resolve config attribute {attr_name!r} on {type(config).__name__}")


class LatentReasoningRuntime(nn.Module):
    """Shared latent runtime used by both training and evaluation.

    The current implementation preserves the CODI-compatible latent interface and
    checkpoint format so the harness can split training/evaluation concerns
    without breaking existing runs.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: LatentRuntimeConfig,
        *,
        train_mode: bool,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.train_mode = train_mode

        self.runtime_dtype = _resolve_runtime_dtype(runtime_config)
        torch_dtype = self.runtime_dtype

        quantization_config = None
        if model_config.load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )

        _hf_token = resolve_hf_hub_token(model_config.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_name_or_path,
            token=_hf_token,
            torch_dtype=torch_dtype if model_config.full_precision else None,
            quantization_config=quantization_config,
        )

        hidden_size = get_text_config_attr(self.model.config, "hidden_size")
        original_vocab_size = get_text_config_attr(self.model.config, "vocab_size")
        self.pad_token_id = original_vocab_size
        self.bot_id = original_vocab_size + 1
        self.eot_id = original_vocab_size + 2
        self.model.resize_token_embeddings(original_vocab_size + 3)

        if model_config.use_lora:
            maybe_prepare_gemma4_peft(model_config.base_model_name_or_path)
            modules_to_save = (
                get_modules_to_save(model_config.base_model_name_or_path)
                if not model_config.freeze_base_embeddings
                else None
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=not train_mode,
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=(
                    model_config.lora_target_modules
                    or get_lora_target_modules(model_config.base_model_name_or_path)
                ),
                modules_to_save=modules_to_save,
                init_lora_weights=model_config.lora_init,
            )
            self.model = get_peft_model(self.model, lora_config)

        if runtime_config.use_prj:
            prj_dtype = torch.float32 if runtime_config.prj_fp32 else self.runtime_dtype
            blocks: list[nn.Module] = [
                nn.Dropout(runtime_config.prj_dropout),
                nn.Linear(hidden_size, runtime_config.prj_dim),
                nn.GELU(),
                nn.Linear(runtime_config.prj_dim, hidden_size),
            ]
            if not runtime_config.prj_no_ln:
                blocks.append(nn.LayerNorm(hidden_size))
            self.prj = nn.Sequential(*blocks).to(dtype=prj_dtype)
            if runtime_config.prj_residual_gated:
                self.prj_residual_alpha = nn.Parameter(
                    torch.tensor(float(runtime_config.prj_gate_init), dtype=torch.float32)
                )
            else:
                self.register_parameter("prj_residual_alpha", None)
        else:
            self.prj = nn.Identity()
            self.register_parameter("prj_residual_alpha", None)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if runtime_config.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif runtime_config.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported distillation loss {runtime_config.distill_loss_type}")

    @property
    def codi(self) -> nn.Module:
        """Compatibility alias for legacy eval code paths."""
        return self.model

    def build_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name_or_path,
            token=resolve_hf_hub_token(self.model_config.hf_token),
            model_max_length=self.runtime_config.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token_id = self.pad_token_id
        return tokenizer

    def get_input_embedding_layer(self) -> nn.Module:
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        model_name = self.model_config.base_model_name_or_path.lower()
        if "pythia" in model_name:
            return base.gpt_neox.embed_in
        if "gpt2" in model_name:
            return base.transformer.wte
        if hasattr(base, "model") and hasattr(base.model, "language_model"):
            language_model = base.model.language_model
            if hasattr(language_model, "embed_tokens"):
                return language_model.embed_tokens
            if hasattr(language_model, "model") and hasattr(language_model.model, "embed_tokens"):
                return language_model.model.embed_tokens
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens
        raise AttributeError("Could not locate input embedding layer")

    def maybe_project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if not self.runtime_config.use_prj:
            return hidden_state
        x = hidden_state
        if self.runtime_config.prj_fp32:
            x_work = x.float()
        else:
            x_work = x
        f_x = self.prj(x_work)
        if self.runtime_config.prj_residual_gated:
            assert self.prj_residual_alpha is not None
            alpha = self.prj_residual_alpha.to(device=f_x.device, dtype=f_x.dtype)
            out = x_work + alpha * f_x
        else:
            out = f_x
        return out.to(hidden_state.dtype)

    def _ensure_finite(
        self,
        tensor: torch.Tensor,
        *,
        stage: str,
        tensor_name: str,
        extra: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            raise NumericalInstabilityError(
                stage=stage,
                tensor_name=tensor_name,
                summary=tensor_summary(tensor),
                extra=extra,
            )
        return tensor

    def needs_token_type_ids(self) -> bool:
        model_name = self.model_config.base_model_name_or_path.lower()
        return "gemma-3" in model_name or "gemma3" in model_name

    def build_token_type_ids(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.LongTensor | None:
        if not self.needs_token_type_ids():
            return None
        if input_ids is not None:
            return torch.zeros_like(input_ids, dtype=torch.long)
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            return torch.zeros((batch_size, seq_len), dtype=torch.long, device=inputs_embeds.device)
        return None

    def encode_question(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[Any, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=self.build_token_type_ids(input_ids=input_ids),
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        latent = outputs.hidden_states[-1][:, -1:, :]
        return past_key_values, self.maybe_project(latent)

    def iterate_latent_steps(
        self,
        past_key_values: Any,
        latent: torch.Tensor,
        num_steps: int,
    ) -> tuple[Any, torch.Tensor]:
        for _ in range(num_steps):
            outputs = self.model(
                inputs_embeds=latent,
                token_type_ids=self.build_token_type_ids(inputs_embeds=latent),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent = outputs.hidden_states[-1][:, -1:, :]
            latent = self.maybe_project(latent)
        return past_key_values, latent

    def build_eot_embeds(
        self,
        batch_size: int,
        device: torch.device,
        *,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        token_ids = [self.eot_id]
        if not self.runtime_config.remove_eos and eos_token_id is not None:
            token_ids.append(eos_token_id)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        embeds = self.get_input_embedding_layer()(token_tensor).unsqueeze(0)
        return embeds.expand(batch_size, -1, -1)

    def generate_from_latent(
        self,
        *,
        tokenizer: AutoTokenizer,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        inf_latent_iterations: int,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[str]:
        device = input_ids.device
        batch_size = input_ids.size(0)
        past_key_values, latent = self.encode_question(input_ids=input_ids, attention_mask=attention_mask)
        past_key_values, _ = self.iterate_latent_steps(
            past_key_values=past_key_values,
            latent=latent,
            num_steps=inf_latent_iterations,
        )

        next_embeds = self.build_eot_embeds(
            batch_size=batch_size,
            device=device,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self.model(
                inputs_embeds=next_embeds,
                token_type_ids=self.build_token_type_ids(inputs_embeds=next_embeds),
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, : self.eot_id]
            token_ids = self._sample_tokens(
                logits=logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            for batch_index, token_id in enumerate(token_ids.tolist()):
                if finished[batch_index]:
                    continue
                predictions[batch_index].append(token_id)
                if token_id == tokenizer.eos_token_id:
                    finished[batch_index] = True
            if bool(finished.all()):
                break
            next_embeds = self.get_input_embedding_layer()(token_ids).unsqueeze(1)

        return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in predictions]

    def _sample_tokens(
        self,
        *,
        logits: torch.Tensor,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.LongTensor:
        if greedy:
            return torch.argmax(logits, dim=-1)
        working = logits / max(temperature, 1e-5)
        if top_k > 0:
            top_values, _ = torch.topk(working, min(top_k, working.shape[-1]), dim=-1)
            cutoff = top_values[:, -1].unsqueeze(-1)
            working = torch.where(working < cutoff, torch.full_like(working, float("-inf")), working)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(working, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove_mask = cumulative > top_p
            remove_mask = torch.roll(remove_mask, shifts=1, dims=-1)
            remove_mask[:, 0] = False
            filtered = working.clone()
            for row_index in range(filtered.size(0)):
                filtered[row_index, sorted_indices[row_index, remove_mask[row_index]]] = float("-inf")
            working = filtered
        probs = torch.softmax(working, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def tie_weights_if_needed(self) -> None:
        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
            except KeyError:
                return

    def forward(
        self,
        *,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        ref_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        ref_answer_position: torch.LongTensor,
        model_answer_position: torch.LongTensor,
        ref_attention_mask: torch.LongTensor,
        ref_labels: torch.LongTensor,
        step: int | None = None,
        step_ratio: float | None = None,
        collect_diagnostics: bool = False,
    ) -> dict[str, Any]:
        del step, step_ratio
        past_key_values, latent = self.encode_question(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                token_type_ids=self.build_token_type_ids(input_ids=ref_input_ids),
                output_hidden_states=True,
            )
        if self.runtime_config.ref_loss_factor > 0:
            teacher_outputs_with_grad = self.model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                token_type_ids=self.build_token_type_ids(input_ids=ref_input_ids),
                output_hidden_states=True,
            )
        else:
            teacher_outputs_with_grad = None

        student_logits = None
        distill_total = torch.tensor(0.0, device=encoder_input_ids.device)
        ce_total = torch.tensor(0.0, device=encoder_input_ids.device)
        teacher_selected_raw_stds: list[torch.Tensor] = []
        teacher_selected_effective_stds: list[torch.Tensor] = []
        teacher_selected_std_clamped_count = 0
        layer_loss_values: list[torch.Tensor] = []

        self._ensure_finite(
            latent,
            stage="encode_question",
            tensor_name="latent",
            extra={"num_latent": self.runtime_config.num_latent},
        )

        for latent_index in range(self.runtime_config.num_latent):
            latent_outputs = self.model(
                inputs_embeds=latent,
                token_type_ids=self.build_token_type_ids(inputs_embeds=latent),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_outputs.past_key_values
            latent = latent_outputs.hidden_states[-1][:, -1:, :]
            latent = self.maybe_project(latent)
            if latent_index < self.runtime_config.num_latent - 1:
                should_detach = _resolve_should_detach(
                    latent_index=latent_index,
                    num_latent=self.runtime_config.num_latent,
                    keep_last_k=self.runtime_config.detach_keep_last_k,
                )
                if should_detach:
                    past_key_values, latent = _apply_boundary_detach(
                        cache=past_key_values,
                        latent=latent,
                        encoder_length=encoder_input_ids.size(1),
                        runtime_config_detach_latent=self.runtime_config.detach_latent_between_steps,
                        runtime_config_detach_cache=self.runtime_config.detach_cache_between_steps,
                        detach_position_mode=self.runtime_config.detach_position_mode,
                    )
            self._ensure_finite(
                latent,
                stage="latent_rollout",
                tensor_name="latent",
                extra={"latent_index": latent_index},
            )

            if latent_index != self.runtime_config.num_latent - 1:
                continue

            decoder_embeds = self.get_input_embedding_layer()(decoder_input_ids)
            student_outputs = self.model(
                inputs_embeds=decoder_embeds,
                token_type_ids=self.build_token_type_ids(inputs_embeds=decoder_embeds),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            student_logits = student_outputs.logits
            self._ensure_finite(
                student_logits,
                stage="student_decode",
                tensor_name="student_logits",
                extra={"latent_index": latent_index},
            )

            layer_losses: list[torch.Tensor] = []
            for layer_index, (student_layer, teacher_layer) in enumerate(
                zip(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                )
            ):
                safe_ref_pos = ref_answer_position.clamp(max=teacher_layer.size(1) - 1)
                safe_model_pos = model_answer_position.clamp(max=student_layer.size(1) - 1)
                teacher_selected = teacher_layer.gather(
                    1,
                    safe_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, teacher_layer.size(-1)),
                )
                student_selected = student_layer.gather(
                    1,
                    safe_model_pos.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, student_layer.size(-1)
                    ),
                )
                self._ensure_finite(
                    teacher_selected,
                    stage="distill_alignment",
                    tensor_name="teacher_selected",
                    extra={"latent_index": latent_index, "layer_index": layer_index},
                )
                self._ensure_finite(
                    student_selected,
                    stage="distill_alignment",
                    tensor_name="student_selected",
                    extra={"latent_index": latent_index, "layer_index": layer_index},
                )
                loss_piece = self.distill_loss_fct(
                    student_selected.float(),
                    teacher_selected.detach().float(),
                )
                teacher_std = teacher_selected.detach().float().std(unbiased=False)
                effective_teacher_std = teacher_std.clamp_min(self.runtime_config.distill_loss_std_floor)
                if self.runtime_config.distill_loss_div_std:
                    loss_piece = loss_piece / effective_teacher_std.to(loss_piece.dtype)
                layer_losses.append(loss_piece)
                if collect_diagnostics:
                    teacher_selected_raw_stds.append(teacher_std.detach())
                    teacher_selected_effective_stds.append(effective_teacher_std.detach())
                    if teacher_std.item() < self.runtime_config.distill_loss_std_floor:
                        teacher_selected_std_clamped_count += 1
                    layer_loss_values.append(loss_piece.detach())
            stacked_layer_losses = torch.stack(layer_losses)
            self._ensure_finite(
                stacked_layer_losses,
                stage="distill_alignment",
                tensor_name="layer_losses",
                extra={"latent_index": latent_index},
            )
            distill_total = stacked_layer_losses.mean() * self.runtime_config.distill_loss_factor
            self._ensure_finite(
                distill_total,
                stage="distill_alignment",
                tensor_name="distill_total",
                extra={"latent_index": latent_index},
            )

            shifted_logits = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))
            shifted_labels = labels[:, 1:].reshape(-1)
            ce_total = self.loss_fct(shifted_logits.float(), shifted_labels)
            self._ensure_finite(
                ce_total,
                stage="student_decode",
                tensor_name="ce_total",
                extra={"latent_index": latent_index},
            )

        if teacher_outputs_with_grad is not None:
            ref_logits = teacher_outputs_with_grad.logits
            self._ensure_finite(
                ref_logits,
                stage="reference_decode",
                tensor_name="ref_logits",
            )
            shifted_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
            shifted_ref_labels = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = (
                self.loss_fct(shifted_ref_logits.float(), shifted_ref_labels) * self.runtime_config.ref_loss_factor
            )
            self._ensure_finite(
                ref_ce_loss,
                stage="reference_decode",
                tensor_name="ref_ce_loss",
            )
        else:
            ref_logits = None
            ref_ce_loss = torch.tensor(0.0, device=encoder_input_ids.device)

        total_loss = ce_total + distill_total + ref_ce_loss
        self._ensure_finite(
            total_loss,
            stage="loss_aggregation",
            tensor_name="total_loss",
        )
        diagnostics: dict[str, float] = {}
        if collect_diagnostics:
            if teacher_selected_raw_stds:
                stacked_raw_stds = torch.stack(teacher_selected_raw_stds)
                stacked_effective_stds = torch.stack(teacher_selected_effective_stds)
                diagnostics["teacher_selected_std_min_raw"] = float(stacked_raw_stds.min().cpu())
                diagnostics["teacher_selected_std_max_raw"] = float(stacked_raw_stds.max().cpu())
                diagnostics["teacher_selected_std_min_effective"] = float(stacked_effective_stds.min().cpu())
                diagnostics["teacher_selected_std_clamped_count"] = float(teacher_selected_std_clamped_count)
            if layer_loss_values:
                stacked_losses = torch.stack(layer_loss_values)
                diagnostics["distill_layer_loss_max"] = float(stacked_losses.max().cpu())
            if student_logits is not None:
                diagnostics["student_logits_absmax"] = float(student_logits.detach().abs().max().cpu())
            if ref_logits is not None:
                diagnostics["ref_logits_absmax"] = float(ref_logits.detach().abs().max().cpu())
            diagnostics["latent_absmax"] = float(latent.detach().abs().max().cpu())
            diagnostics["distill_to_ce_ratio"] = float(
                distill_total.detach().float().cpu() / max(float(abs(ce_total.detach().float().cpu())), 1e-12)
            )
        return {
            "loss": total_loss,
            "logits": student_logits,
            "ce_loss": float(ce_total.detach().cpu()),
            "distill_loss": float(distill_total.detach().cpu()),
            "ref_ce_loss": float(ref_ce_loss.detach().cpu()),
            "diagnostics": diagnostics,
            "config": {
                "model": asdict(self.model_config),
                "runtime": asdict(self.runtime_config),
            },
        }
