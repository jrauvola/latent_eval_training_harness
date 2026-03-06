from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainingArguments


@dataclass(slots=True)
class CodiModelConfig:
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
class CodiDataConfig:
    dataset_names: list[str]
    cache_dir: str = ".cache/huggingface"
    max_samples: int | None = None
    snapshot_dir: str | None = None
    include_last_cot: bool = False
    answer_only: bool = False
    max_token_num: int = 1000


@dataclass(slots=True)
class CodiRuntimeConfig:
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


@dataclass(slots=True)
class TrainEntryConfig:
    model: CodiModelConfig
    data: CodiDataConfig
    runtime: CodiRuntimeConfig
    trainer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainEntryConfig":
        model_payload = payload.get("model", {})
        data_payload = payload.get("data", {})
        runtime_payload = payload.get("runtime", {})
        trainer_payload = payload.get("trainer", {})
        return cls(
            model=CodiModelConfig(**model_payload),
            data=CodiDataConfig(**data_payload),
            runtime=CodiRuntimeConfig(**runtime_payload),
            trainer=trainer_payload,
        )

    def to_hf_training_arguments(self) -> TrainingArguments:
        trainer_payload = dict(self.trainer)
        trainer_payload.setdefault("output_dir", "artifacts/train/default")
        trainer_payload.setdefault("logging_dir", "artifacts/train/default/logs")
        trainer_payload.setdefault("logging_steps", 10)
        trainer_payload.setdefault("logging_strategy", "steps")
        trainer_payload.setdefault("per_device_train_batch_size", 8)
        trainer_payload.setdefault("gradient_accumulation_steps", 1)
        trainer_payload.setdefault("num_train_epochs", 1)
        trainer_payload.setdefault("learning_rate", 8e-4)
        trainer_payload.setdefault("bf16", self.runtime.bf16)
        trainer_payload.setdefault("save_safetensors", True)
        trainer_payload.setdefault("report_to", [])
        trainer_payload.setdefault("seed", self.runtime.seed)
        trainer_payload.setdefault("remove_unused_columns", False)
        trainer_payload.setdefault("save_strategy", "epoch")
        trainer_payload.setdefault("lr_scheduler_type", "cosine")
        trainer_payload.setdefault("warmup_ratio", 0.03)
        trainer_payload.setdefault("weight_decay", 0.1)
        trainer_payload.setdefault("max_grad_norm", 2.0)
        trainer_payload.setdefault("dataloader_num_workers", 2)
        trainer_payload.setdefault("ddp_find_unused_parameters", False)
        trainer_payload["output_dir"] = str(Path(trainer_payload["output_dir"]).expanduser())
        trainer_payload["logging_dir"] = str(Path(trainer_payload["logging_dir"]).expanduser())
        return TrainingArguments(**trainer_payload)
