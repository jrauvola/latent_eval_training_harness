from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from latent_harness.core import LatentReasoningRuntime, load_checkpoint_state, remap_runtime_state_dict_prefixes, resolve_checkpoint_path
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


def _build_standard_tokenizer(spec: EvaluationModelSpec) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model.base_model_name_or_path,
        token=spec.model.hf_token,
        model_max_length=spec.runtime.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


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
                token=spec.model.hf_token,
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
    tokenizer = _build_standard_tokenizer(spec)
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if spec.runtime.bf16 else torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec.model.base_model_name_or_path,
        token=spec.model.hf_token,
        torch_dtype=torch_dtype if spec.model.full_precision else None,
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if spec.checkpoint_type not in {"base_model", "none"}:
        state_dict = load_checkpoint_state(
            resolve_checkpoint_path(
                spec.checkpoint_source,
                spec.checkpoint_type,
                token=spec.model.hf_token,
            )
        )
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
