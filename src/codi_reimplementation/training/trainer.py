from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as save_safetensors
from transformers import Trainer

from codi_reimplementation.config import ensure_dir, load_yaml_config
from codi_reimplementation.training.codi_model import CODIRuntime
from codi_reimplementation.training.config import TrainEntryConfig
from codi_reimplementation.training.datasets import make_supervised_data_module


class CodiTrainer(Trainer):
    def compute_loss(
        self,
        model: CODIRuntime,
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


def atomic_save_state_dict(model: torch.nn.Module, output_dir: str | Path) -> Path:
    destination_dir = ensure_dir(output_dir)
    final_path = destination_dir / "model.safetensors"
    temp_path = destination_dir / "model.safetensors.tmp"
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    save_safetensors(state_dict, str(temp_path))
    temp_path.replace(final_path)
    return final_path


def run_training_from_config(config_path: str) -> None:
    payload = load_yaml_config(config_path)
    config = TrainEntryConfig.from_dict(payload)
    training_args = config.to_hf_training_arguments()

    model = CODIRuntime(
        model_config=config.model,
        runtime_config=config.runtime,
        train_mode=True,
    )
    tokenizer = model.build_tokenizer()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_config=config.data,
        runtime_config=config.runtime,
        bot_id=model.bot_id,
        eot_id=model.eot_id,
    )

    trainer = CodiTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        **data_module,
    )
    resume_checkpoint = config.trainer.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    atomic_save_state_dict(model, training_args.output_dir)
