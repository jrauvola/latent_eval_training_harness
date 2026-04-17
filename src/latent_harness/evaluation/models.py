from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from latent_harness.core import (
    LatentReasoningRuntime,
    load_checkpoint_state,
    remap_runtime_state_dict_prefixes,
    resolve_checkpoint_path,
    resolve_hf_hub_token,
)
from latent_harness.evaluation.config import EvaluationModelSpec


@dataclass(slots=True)
class EvaluationModelHandle:
    name: str
    model_kind: str
    inference_strategy: str
    runtime_config: object
    model: object
    generation_model: object
    tokenizer: PreTrainedTokenizerBase
    bot_id: int | None = None
    remove_eos: bool = True


def _add_eval_special_tokens(tokenizer: PreTrainedTokenizerBase, spec: EvaluationModelSpec) -> None:
    if spec.hf_extra_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(spec.hf_extra_special_tokens)})


def _strip_hf_checkpoint_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    p = prefix if prefix.endswith(".") else f"{prefix}."
    return {k[len(p) :]: v for k, v in state_dict.items() if k.startswith(p)}


def _build_standard_tokenizer(spec: EvaluationModelSpec) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model.base_model_name_or_path,
        token=resolve_hf_hub_token(spec.model.hf_token),
        model_max_length=spec.runtime.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    _add_eval_special_tokens(tokenizer, spec)
    return tokenizer


def _resolve_vocab_size(model: object) -> int | None:
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    text_config = getattr(config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    embeddings = model.get_input_embeddings()
    num_embeddings = getattr(embeddings, "num_embeddings", None)
    if isinstance(num_embeddings, int):
        return num_embeddings
    return None


def _load_latent_runtime_model(
    spec: EvaluationModelSpec,
    *,
    device: str | torch.device | None = None,
) -> EvaluationModelHandle:
    runtime = LatentReasoningRuntime(
        model_config=spec.model,
        runtime_config=spec.runtime,
        train_mode=False,
    )
    if spec.checkpoint_type not in {"base_model", "none"}:
        state_dict = load_checkpoint_state(
            resolve_checkpoint_path(
                spec.checkpoint_source,
                spec.checkpoint_type,
                token=resolve_hf_hub_token(spec.model.hf_token),
                subfolder=spec.hf_subfolder,
                hf_hub_filename=spec.hf_hub_filename,
            )
        )
        state_dict = remap_runtime_state_dict_prefixes(
            state_dict,
            target_keys=set(runtime.state_dict().keys()),
        )
        incompatible = runtime.load_state_dict(state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.endswith(".weight") or not key.startswith("prj.")]
        if missing and len(missing) > 8:
            raise RuntimeError(
                f"Failed to load evaluation checkpoint {spec.name!r}: too many missing keys after compatibility "
                f"remapping (sample: {missing[:8]})"
            )
    runtime.tie_weights_if_needed()
    if device is not None:
        runtime = runtime.to(device)
    runtime.eval()
    return EvaluationModelHandle(
        name=spec.name,
        model_kind=spec.model_kind,
        inference_strategy=spec.inference_strategy,
        runtime_config=spec.runtime,
        model=runtime,
        generation_model=runtime.model,
        tokenizer=runtime.build_tokenizer(),
        bot_id=runtime.bot_id,
        remove_eos=runtime.runtime_config.remove_eos,
    )


def _load_standard_generation_model(
    spec: EvaluationModelSpec,
    *,
    device: str | torch.device | None = None,
) -> EvaluationModelHandle:
    if spec.checkpoint_type == "hf_pretrained":
        if not spec.checkpoint_source:
            raise ValueError("checkpoint_type 'hf_pretrained' requires checkpoint_source (HF repo id)")
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if spec.runtime.bf16 else torch.float16
        else:
            torch_dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            spec.checkpoint_source,
            subfolder=spec.hf_subfolder,
            token=resolve_hf_hub_token(spec.model.hf_token),
            model_max_length=spec.runtime.model_max_length,
            padding_side="left",
            use_fast=False,
            trust_remote_code=spec.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": ""})
        _add_eval_special_tokens(tokenizer, spec)
        load_kw: dict = {
            "token": resolve_hf_hub_token(spec.model.hf_token),
            "torch_dtype": torch_dtype if spec.model.full_precision else None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": spec.trust_remote_code,
        }
        if torch.cuda.is_available():
            load_kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            spec.checkpoint_source,
            subfolder=spec.hf_subfolder,
            **load_kw,
        )
        if device is not None and getattr(model, "hf_device_map", None) is None:
            model = model.to(device)
        model.eval()
        return EvaluationModelHandle(
            name=spec.name,
            model_kind=spec.model_kind,
            inference_strategy=spec.inference_strategy,
            runtime_config=spec.runtime,
            model=model,
            generation_model=model,
            tokenizer=tokenizer,
        )

    tokenizer = _build_standard_tokenizer(spec)
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if spec.runtime.bf16 else torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec.model.base_model_name_or_path,
        token=resolve_hf_hub_token(spec.model.hf_token),
        torch_dtype=torch_dtype if spec.model.full_precision else None,
    )
    vocab_size = _resolve_vocab_size(model)
    if vocab_size is not None and len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if spec.checkpoint_type not in {"base_model", "none"}:
        state_dict = load_checkpoint_state(
            resolve_checkpoint_path(
                spec.checkpoint_source,
                spec.checkpoint_type,
                token=resolve_hf_hub_token(spec.model.hf_token),
                subfolder=spec.hf_subfolder,
                hf_hub_filename=spec.hf_hub_filename,
            )
        )
        if spec.hf_checkpoint_state_dict_prefix:
            state_dict = _strip_hf_checkpoint_prefix(state_dict, spec.hf_checkpoint_state_dict_prefix)
        model.load_state_dict(state_dict, strict=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    return EvaluationModelHandle(
        name=spec.name,
        model_kind=spec.model_kind,
        inference_strategy=spec.inference_strategy,
        runtime_config=spec.runtime,
        model=model,
        generation_model=model,
        tokenizer=tokenizer,
    )


def load_evaluation_model(
    spec: EvaluationModelSpec,
    device: str | torch.device | None = None,
) -> EvaluationModelHandle:
    if spec.model_kind == "latent_runtime":
        return _load_latent_runtime_model(spec, device=device)
    if spec.model_kind == "causal_lm":
        return _load_standard_generation_model(spec, device=device)
    raise ValueError(f"Unsupported evaluation model kind: {spec.model_kind}")
