from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig


@dataclass(slots=True)
class EvaluationRuntimeConfig:
    output_dir: str = "artifacts/eval/default"
    cache_dir: str = ".cache/huggingface"
    snapshot_dir: str = "artifacts/datasets"
    device: str = "auto"
    batch_size: int = 8
    progress_log_interval_batches: int = 10
    max_examples_per_benchmark: int | None = None
    max_new_tokens: int = 128
    greedy: bool = True
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.95
    num_passes: int = 1
    #: Phase 1 extensions. --------------------------------------------------
    #: When set, each model is evaluated once per value in this list by
    #: overriding ``runtime.num_latent``. Same outputs/trace dirs are reused,
    #: partitioned by ``numlatent_{n}`` subdir. Empty/None = use the value from
    #: the model spec as-is (legacy behavior).
    num_latent_sweep: list[int] | None = None
    #: When True, latent injection is bypassed entirely (the model runs as a
    #: plain instruction-finetuned causal LM with num_latent=0). Required when
    #: the runtime would otherwise error on num_latent=0. See spec §4 Phase 1.
    skip_latent_injection_at_zero: bool = True
    #: Write per-(example, latent_step) logit-lens traces to
    #: ``latent_traces_dir/{variant}/{benchmark}/numlatent_{n}/trace_{idx}.jsonl``.
    dump_latent_traces: bool = False
    latent_traces_dir: str = "research_findings/latent_traces"
    latent_trace_topk: int = 10
    #: Write per-example KV cache ``.npy`` dumps to
    #: ``kv_dump_dir/{variant}/{benchmark}/kv_example_{idx}.npy``.
    dump_kv_cache: bool = False
    kv_dump_dir: str = "research_findings/kv_pca"
    #: Cap on the number of examples (per variant/benchmark) that receive a
    #: KV dump, because the files are expensive on disk.
    kv_dump_max_examples: int = 200
    kv_dump_final_layer_only: bool = True
    #: Batch-level persistence threshold. After every ``persistence_every_examples``
    #: examples (or every batch, whichever is smaller), the eval runner flushes
    #: predictions.jsonl and running_summary.json. Non-negotiable per user memory
    #: feedback_batch_eval_persistence.md — crashes must lose <= this many examples.
    persistence_every_examples: int = 32
    seed: int = 11


InferenceStrategy = Literal["latent_cot", "standard_generation"]
ModelKind = Literal["latent_runtime", "causal_lm"]


@dataclass(slots=True)
class EvaluationModelSpec:
    name: str
    checkpoint_source: str | None
    checkpoint_type: str
    model: ModelConfig
    runtime: LatentRuntimeConfig
    inference_strategy: InferenceStrategy = "latent_cot"
    model_kind: ModelKind = "latent_runtime"
    #: Used with ``checkpoint_type: hf_pretrained`` — snapshot lives at ``repo_id/subfolder/``.
    hf_subfolder: str | None = None
    #: Exact repo-relative path for ``checkpoint_type: hf_repo`` when the weight file is not
    #: ``subfolder/{model.safetensors,pytorch_model.bin}`` (e.g. xet blob ``coconut/checkpoint_best``).
    hf_hub_filename: str | None = None
    #: Strip this prefix from checkpoint keys before ``load_state_dict`` (e.g. ``base_causallm.``).
    hf_checkpoint_state_dict_prefix: str | None = None
    #: Added via ``add_special_tokens`` before loading weights (e.g. COCONUT pause/latent tokens).
    hf_extra_special_tokens: list[str] | None = None
    #: Pass ``trust_remote_code=True`` to HF ``from_pretrained`` calls. Required for models
    #: with custom modeling code (e.g., Ouro's looped forward pass).
    trust_remote_code: bool = False
    #: Force ``use_cache=False`` during generation by overriding the model's ``generation_config``.
    #: Required for Ouro checkpoints whose custom ``UniversalTransformerCache`` class is
    #: incompatible with the ``Cache.key_cache`` property in transformers>=4.52.
    disable_kv_cache: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationModelSpec":
        if "model" in payload or "runtime" in payload:
            model_payload = payload.get("model", {})
            runtime_payload = payload.get("runtime", {})
        else:
            model_keys = {field.name for field in fields(ModelConfig)}
            runtime_keys = {field.name for field in fields(LatentRuntimeConfig)}
            model_payload = {key: value for key, value in payload.items() if key in model_keys}
            runtime_payload = {key: value for key, value in payload.items() if key in runtime_keys}
        return cls(
            name=payload["name"],
            checkpoint_source=payload.get("checkpoint_source"),
            checkpoint_type=payload.get("checkpoint_type", "hf_repo"),
            model=ModelConfig(**model_payload),
            runtime=LatentRuntimeConfig(**runtime_payload),
            inference_strategy=payload.get("inference_strategy", "latent_cot"),
            model_kind=payload.get("model_kind", "latent_runtime"),
            hf_subfolder=payload.get("hf_subfolder"),
            hf_hub_filename=payload.get("hf_hub_filename"),
            hf_checkpoint_state_dict_prefix=payload.get("hf_checkpoint_state_dict_prefix"),
            hf_extra_special_tokens=payload.get("hf_extra_special_tokens"),
            trust_remote_code=payload.get("trust_remote_code", False),
            disable_kv_cache=payload.get("disable_kv_cache", False),
        )


@dataclass(slots=True)
class EvaluationConfig:
    runtime: EvaluationRuntimeConfig
    benchmarks: list[str]
    models: list[EvaluationModelSpec]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        runtime = EvaluationRuntimeConfig(**payload.get("runtime", {}))
        benchmarks = payload.get("benchmarks", [])
        model_specs = [EvaluationModelSpec.from_dict(model_payload) for model_payload in payload.get("models", [])]
        return cls(runtime=runtime, benchmarks=benchmarks, models=model_specs)
