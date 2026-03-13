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
