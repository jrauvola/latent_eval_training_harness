from __future__ import annotations

import inspect
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors
from transformers import Trainer

from latent_harness.core.io import ensure_dir, load_yaml_config
from latent_harness.training.config import TrainingConfig
from latent_harness.training.methods import get_method_recipe


class LatentTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        del num_items_in_batch
        step = self.state.global_step
        batch_size = self.args.per_device_train_batch_size
        accumulation = self.args.gradient_accumulation_steps
        world_size = max(getattr(self.args, "world_size", 1), 1)
        dataset_size = len(self.train_dataset)
        steps_per_epoch = ceil(dataset_size / max(batch_size * accumulation * world_size, 1))
        total_steps = max(steps_per_epoch * int(self.args.num_train_epochs), 1)

        outputs = model(
            **inputs,
            step=step,
            step_ratio=step / total_steps,
        )
        if step % max(self.args.logging_steps, 1) == 0:
            self.log(
                {
                    "loss": float(outputs["loss"].detach().cpu()),
                    "ce_loss": outputs["ce_loss"],
                    "distill_loss": outputs["distill_loss"],
                    "ref_ce_loss": outputs["ref_ce_loss"],
                }
            )

        if return_outputs:
            return outputs["loss"], outputs
        return outputs["loss"]

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        destination = output_dir or self.args.output_dir
        atomic_save_state_dict(self.model, destination)
        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)
        del _internal_call


class StandardSFTModel(nn.Module):
    def __init__(self, runtime_model: nn.Module) -> None:
        super().__init__()
        self.runtime_model = runtime_model

    def forward(self, *args, **kwargs):
        return self.runtime_model.model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.runtime_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.runtime_model.load_state_dict(state_dict, strict=strict)


class StandardSFTTrainer(Trainer):
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        destination = output_dir or self.args.output_dir
        atomic_save_state_dict(self.model, destination)
        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)
        del _internal_call


def atomic_save_state_dict(model: torch.nn.Module, output_dir: str | Path) -> Path:
    destination_dir = ensure_dir(output_dir)
    final_path = destination_dir / "model.safetensors"
    temp_path = destination_dir / "model.safetensors.tmp"
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    save_safetensors(state_dict, str(temp_path))
    temp_path.replace(final_path)
    return final_path


def _build_trainer_init_kwargs(
    trainer_cls: type[Trainer],
    *,
    model,
    training_args,
    tokenizer,
    data_module: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        **data_module,
    }
    supported = inspect.signature(trainer_cls.__init__).parameters
    if "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    return kwargs


def run_training_from_config(config_path: str) -> None:
    payload = load_yaml_config(config_path)
    config = TrainingConfig.from_dict(payload)
    recipe = get_method_recipe(config.method)
    recipe.assert_implemented()
    training_args = config.to_hf_training_arguments()

    runtime_builder = recipe.runtime_builder
    data_builder = recipe.data_module_builder
    assert runtime_builder is not None
    assert data_builder is not None

    runtime_model = runtime_builder(
        model_config=config.model,
        runtime_config=config.runtime,
        train_mode=True,
    )
    tokenizer = runtime_model.build_tokenizer()
    data_module = data_builder(
        tokenizer=tokenizer,
        data_config=config.data,
        runtime_config=config.runtime,
        bot_id=runtime_model.bot_id,
        eot_id=runtime_model.eot_id,
    )

    trainer_cls: type[Trainer]
    if recipe.training_style == "standard_sft":
        model = StandardSFTModel(runtime_model)
        trainer_cls = StandardSFTTrainer
    else:
        model = runtime_model
        trainer_cls = LatentTrainer

    trainer = trainer_cls(
        **_build_trainer_init_kwargs(
            trainer_cls,
            model=model,
            training_args=training_args,
            tokenizer=tokenizer,
            data_module=data_module,
        )
    )
    resume_checkpoint = config.trainer.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    atomic_save_state_dict(model, training_args.output_dir)
