from __future__ import annotations

import copy
import inspect
import json
import logging
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file as save_safetensors
from transformers import Trainer, TrainerCallback

from latent_harness.core.io import ensure_dir, load_yaml_config
from latent_harness.core.probes import maybe_init_probes
from latent_harness.core.runtime import NumericalInstabilityError
from latent_harness.training.config import TrainingConfig
from latent_harness.training.lt_tuning import (
    CurriculumScheduler,
    LTTuningConfig,
    LTTuningRuntime,
)
from latent_harness.training.lt_tuning_data import (
    build_lt_tuning_dataset_for_stage,
    collect_lt_tuning_examples,
)
from latent_harness.training.methods import get_method_recipe

logger = logging.getLogger(__name__)

RISKY_STD_WARN_THRESHOLD = 1e-4
RISKY_MAGNITUDE_WARN_THRESHOLD = 1e2
RISKY_DISTILL_RATIO_WARN_THRESHOLD = 10.0
RISKY_GRAD_ABSMAX_WARN_THRESHOLD = 1e3
RISKY_PARAM_ABSMAX_WARN_THRESHOLD = 1e4


def _iter_trainable_parameters(model: nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            yield param


def summarize_gradients(model: nn.Module) -> dict[str, float]:
    total_norm_sq = 0.0
    grad_absmax = 0.0
    nonfinite_grad_count = 0
    tracked_grad_tensors = 0
    first_nonfinite_module: str | None = None
    module_grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            continue
        tracked_grad_tensors += 1
        grad_view = grad.detach().float()
        finite_mask = torch.isfinite(grad_view)
        n_nonfinite = int(grad_view.numel()) - int(finite_mask.sum().item())
        nonfinite_grad_count += n_nonfinite
        if n_nonfinite > 0 and first_nonfinite_module is None:
            first_nonfinite_module = name
        if finite_mask.any():
            finite_values = grad_view[finite_mask]
            param_norm_sq = float(torch.sum(finite_values * finite_values).cpu())
            total_norm_sq += param_norm_sq
            grad_absmax = max(grad_absmax, float(finite_values.abs().max().cpu()))
        else:
            param_norm_sq = 0.0
        bucket = _module_bucket(name)
        module_grad_norms[bucket] = module_grad_norms.get(bucket, 0.0) + param_norm_sq
    result = {
        "debug/grad_norm": total_norm_sq**0.5,
        "debug/grad_absmax": grad_absmax,
        "debug/nonfinite_grad_count": float(nonfinite_grad_count),
        "debug/tracked_grad_tensors": float(tracked_grad_tensors),
    }
    if first_nonfinite_module is not None:
        result["debug/first_nonfinite_grad_module"] = first_nonfinite_module
    top_modules = sorted(module_grad_norms.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for rank, (bucket, norm_sq) in enumerate(top_modules):
        result[f"debug/grad_norm_top{rank}_{bucket}"] = norm_sq**0.5
    return result


def _module_bucket(param_name: str) -> str:
    if "embed_tokens" in param_name:
        return "embed_tokens"
    if "lm_head" in param_name:
        return "lm_head"
    if "lora_" in param_name:
        return "lora"
    if ".prj" in param_name or param_name.startswith("prj"):
        return "prj"
    return "other"


def summarize_trainable_parameters(model: nn.Module) -> dict[str, float]:
    param_absmax = 0.0
    nonfinite_param_count = 0
    tracked_param_tensors = 0
    for param in _iter_trainable_parameters(model):
        tracked_param_tensors += 1
        param_view = param.detach().float()
        finite_mask = torch.isfinite(param_view)
        nonfinite_param_count += int(param_view.numel()) - int(finite_mask.sum().item())
        if finite_mask.any():
            param_absmax = max(param_absmax, float(param_view[finite_mask].abs().max().cpu()))
    return {
        "debug/param_absmax": param_absmax,
        "debug/nonfinite_param_count": float(nonfinite_param_count),
        "debug/tracked_param_tensors": float(tracked_param_tensors),
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if torch.isfinite(torch.tensor(value)):
            return value
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _summarize_batch_forensics(batch_forensics: Any) -> Any:
    if not isinstance(batch_forensics, list):
        return None
    summary: list[dict[str, Any]] = []
    for item in batch_forensics[:4]:
        if not isinstance(item, dict):
            summary.append({"value": str(item)})
            continue
        summary.append(
            {
                "dataset_key": item.get("dataset_key"),
                "example_index": item.get("example_index"),
                "source_row_index": item.get("source_row_index"),
                "example_hash": item.get("example_hash"),
                "encoder_length": item.get("encoder_length"),
                "decoder_length": item.get("decoder_length"),
                "ref_length": item.get("ref_length"),
                "ref_answer_position": item.get("ref_answer_position"),
                "model_answer_position": item.get("model_answer_position"),
            }
        )
    return {
        "batch_size": len(batch_forensics),
        "examples": summary,
    }


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


class TrainingTelemetryMixin:
    def _telemetry_root(self) -> Path:
        return ensure_dir(Path(self.args.output_dir) / "live")

    def _telemetry_paths(self) -> dict[str, Path]:
        root = self._telemetry_root()
        return {
            "metrics": root / "metrics.jsonl",
            "events": root / "events.jsonl",
            "status": root / "latest_status.json",
        }

    def _build_telemetry_row(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        sanitized = {key: _json_safe_value(value) for key, value in (payload or {}).items()}
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "global_step": int(getattr(self.state, "global_step", 0) or 0),
            "epoch": _json_safe_value(getattr(self.state, "epoch", None)),
            "model_training": bool(getattr(getattr(self, "model", None), "training", False)),
            **sanitized,
        }

    def _append_metric_row(self, payload: dict[str, Any]) -> None:
        paths = self._telemetry_paths()
        row = self._build_telemetry_row("metric", payload)
        _append_jsonl_row(paths["metrics"], row)
        _write_json(paths["status"], row)

    def _record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        row = self._build_telemetry_row(event_type, payload)
        paths = self._telemetry_paths()
        _append_jsonl_row(paths["events"], row)
        _write_json(paths["status"], row)

    def log(self, logs: dict[str, Any], *args, **kwargs) -> None:
        super().log(logs, *args, **kwargs)
        self._append_metric_row(logs)


class LatentTrainer(TrainingTelemetryMixin, Trainer):
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
        collect_diagnostics = bool(model.training) and step % max(self.args.logging_steps, 1) == 0
        batch_forensics = inputs.get("batch_forensics")
        model_inputs = dict(inputs)
        model_inputs.pop("batch_forensics", None)

        try:
            outputs = model(
                **model_inputs,
                step=step,
                step_ratio=step / total_steps,
                collect_diagnostics=collect_diagnostics,
            )
        except NumericalInstabilityError as exc:
            logger.error(
                "Numerical instability detected during forward step=%s epoch=%s details=%s",
                step,
                self.state.epoch,
                exc.details,
            )
            logger.error(
                "Forward failure batch_forensics=%s",
                _summarize_batch_forensics(batch_forensics),
            )
            raise RuntimeError(
                f"Numerical instability detected at step={step} epoch={self.state.epoch}; see logged diagnostics."
            ) from exc

        loss_value = outputs["loss"].detach()
        diagnostics = outputs.get("diagnostics", {})
        if not torch.isfinite(loss_value):
            logger.error(
                "Non-finite loss detected step=%s epoch=%s ce_loss=%s distill_loss=%s ref_ce_loss=%s diagnostics=%s batch_forensics=%s",
                step,
                self.state.epoch,
                outputs["ce_loss"],
                outputs["distill_loss"],
                outputs["ref_ce_loss"],
                diagnostics,
                _summarize_batch_forensics(batch_forensics),
            )
            raise RuntimeError(
                f"Non-finite loss detected at step={step} epoch={self.state.epoch}; see logged diagnostics."
            )
        if collect_diagnostics:
            self._warn_on_risky_forward_diagnostics(step=step, diagnostics=diagnostics)
            log_payload = {
                "train/loss": float(loss_value.cpu()),
                "train/ce_loss": outputs["ce_loss"],
                "train/distill_loss": outputs["distill_loss"],
                "train/ref_ce_loss": outputs["ref_ce_loss"],
            }
            for key, value in diagnostics.items():
                log_payload[f"debug/{key}"] = value
            self.log(log_payload)

        if return_outputs:
            return outputs["loss"], outputs
        return outputs["loss"]

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        batch_forensics = inputs.get("batch_forensics")
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not model.training or not getattr(self.accelerator, "sync_gradients", True):
            return loss

        step = self.state.global_step
        grad_metrics = summarize_gradients(model)
        param_metrics = summarize_trainable_parameters(model)
        if grad_metrics["debug/nonfinite_grad_count"] > 0 or param_metrics["debug/nonfinite_param_count"] > 0:
            logger.error(
                "Non-finite gradients or parameters detected before optimizer step=%s grad_metrics=%s param_metrics=%s batch_forensics=%s",
                step,
                grad_metrics,
                param_metrics,
                _summarize_batch_forensics(batch_forensics),
            )
            raise RuntimeError(
                f"Non-finite gradients or parameters detected at step={step}; aborting before optimizer step."
            )
        if step % max(self.args.logging_steps, 1) == 0:
            self._warn_on_risky_backward_diagnostics(
                step=step,
                grad_metrics=grad_metrics,
                param_metrics=param_metrics,
            )
            self.log({**grad_metrics, **param_metrics})
        return loss

    def evaluate(self, *args, **kwargs):
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        self._record_event("eval_start", {"metric_key_prefix": metric_key_prefix})
        logger.info(
            "Starting evaluation global_step=%s epoch=%s metric_key_prefix=%s",
            self.state.global_step,
            self.state.epoch,
            metric_key_prefix,
        )
        metrics = super().evaluate(*args, **kwargs)
        logger.info(
            "Finished evaluation global_step=%s epoch=%s metric_key_prefix=%s %s_loss=%s",
            self.state.global_step,
            self.state.epoch,
            metric_key_prefix,
            metric_key_prefix,
            metrics.get(f"{metric_key_prefix}_loss"),
        )
        self._record_event("eval_end", {"metric_key_prefix": metric_key_prefix, **metrics})
        return metrics

    def _warn_on_risky_forward_diagnostics(self, *, step: int, diagnostics: dict[str, Any]) -> None:
        std_min = diagnostics.get("teacher_selected_std_min_raw")
        if std_min is not None and std_min < RISKY_STD_WARN_THRESHOLD:
            logger.warning("Tiny teacher std detected step=%s teacher_selected_std_min_raw=%s", step, std_min)
        if diagnostics.get("teacher_selected_std_clamped_count", 0.0) > 0:
            logger.warning(
                "Teacher std floor applied step=%s clamped_count=%s effective_std_min=%s",
                step,
                diagnostics.get("teacher_selected_std_clamped_count"),
                diagnostics.get("teacher_selected_std_min_effective"),
            )
        for key in ("student_logits_absmax", "ref_logits_absmax", "latent_absmax"):
            value = diagnostics.get(key)
            if value is not None and value >= RISKY_MAGNITUDE_WARN_THRESHOLD:
                logger.warning("Large activation magnitude detected step=%s %s=%s", step, key, value)
        ratio = diagnostics.get("distill_to_ce_ratio")
        if ratio is not None and ratio >= RISKY_DISTILL_RATIO_WARN_THRESHOLD:
            logger.warning("Distillation is dominating CE step=%s distill_to_ce_ratio=%s", step, ratio)

    def _warn_on_risky_backward_diagnostics(
        self,
        *,
        step: int,
        grad_metrics: dict[str, float],
        param_metrics: dict[str, float],
    ) -> None:
        if grad_metrics["debug/grad_absmax"] >= RISKY_GRAD_ABSMAX_WARN_THRESHOLD:
            logger.warning(
                "Large gradient magnitude detected step=%s grad_absmax=%s grad_norm=%s",
                step,
                grad_metrics["debug/grad_absmax"],
                grad_metrics["debug/grad_norm"],
            )
        if param_metrics["debug/param_absmax"] >= RISKY_PARAM_ABSMAX_WARN_THRESHOLD:
            logger.warning(
                "Large parameter magnitude detected step=%s param_absmax=%s",
                step,
                param_metrics["debug/param_absmax"],
            )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        destination = output_dir or self.args.output_dir
        self._record_event("save_model", {"output_dir": str(destination)})
        atomic_save_state_dict(self.model, destination)
        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)
        del _internal_call


class ProbeCallback(TrainerCallback):
    """Attach dgrad / rmsnorm probes at train start, flush per step, detach at end."""

    def __init__(self, runtime_config):
        self.runtime_config = runtime_config
        self.probes: dict | None = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.probes = maybe_init_probes(model, self.runtime_config)

    def on_step_end(self, args, state, control, **kwargs):
        if self.probes is None:
            return
        step = state.global_step
        if self.probes.get("dgrad") is not None:
            self.probes["dgrad"].flush(step=step)
        if self.probes.get("rmsnorm") is not None:
            self.probes["rmsnorm"].flush(step=step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.probes is None:
            return
        if self.probes.get("dgrad") is not None:
            self.probes["dgrad"].detach_all()
        if self.probes.get("rmsnorm") is not None:
            self.probes["rmsnorm"].detach_all()


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


class StandardSFTTrainer(TrainingTelemetryMixin, Trainer):
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        destination = output_dir or self.args.output_dir
        self._record_event("save_model", {"output_dir": str(destination)})
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


def _configure_training_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger.setLevel(logging.INFO)



def _resize_embeddings_peft_safe(peft_or_base_model, new_num_tokens: int) -> None:
    """Resize token embeddings on a model that may be wrapped by PEFT.

    When ``modules_to_save`` includes ``embed_tokens`` / ``lm_head``, the PEFT
    LoRA setup wraps those modules in ``ModulesToSaveWrapper``. HF's
    ``resize_token_embeddings`` refuses anything that isn't a raw
    ``nn.Embedding`` / ``nn.Linear``. We temporarily unwrap, resize on the base
    HF model, then re-wrap so the PEFT adapter tracking is preserved.
    """
    # Local import so plain CODI / non-PEFT paths don't pay the import cost.
    from peft.utils.other import ModulesToSaveWrapper

    base = peft_or_base_model
    if hasattr(base, "get_base_model"):
        try:
            base = base.get_base_model()
        except Exception:
            base = peft_or_base_model

    in_embed = base.get_input_embeddings()
    out_embed = base.get_output_embeddings()

    in_adapters: list[str] = []
    out_adapters: list[str] = []
    in_active: str | None = None
    out_active: str | None = None

    if isinstance(in_embed, ModulesToSaveWrapper):
        in_adapters = list(in_embed.modules_to_save.keys())
        active = in_embed.active_adapter
        in_active = active[0] if isinstance(active, list) else active
        base.set_input_embeddings(in_embed.original_module)

    if isinstance(out_embed, ModulesToSaveWrapper):
        out_adapters = list(out_embed.modules_to_save.keys())
        active = out_embed.active_adapter
        out_active = active[0] if isinstance(active, list) else active
        base.set_output_embeddings(out_embed.original_module)

    base.resize_token_embeddings(new_num_tokens)

    if in_adapters:
        new_in = base.get_input_embeddings()
        anchor = in_active or in_adapters[0]
        wrapper = ModulesToSaveWrapper(new_in, anchor)
        for name in in_adapters:
            if name != anchor:
                wrapper.update(name)
        if in_active is not None:
            wrapper._active_adapter = [in_active]
        base.set_input_embeddings(wrapper)

    if out_adapters:
        new_out = base.get_output_embeddings()
        anchor = out_active or out_adapters[0]
        wrapper = ModulesToSaveWrapper(new_out, anchor)
        for name in out_adapters:
            if name != anchor:
                wrapper.update(name)
        if out_active is not None:
            wrapper._active_adapter = [out_active]
        base.set_output_embeddings(wrapper)


def run_lt_tuning_training(
    *,
    config: TrainingConfig,
    config_path: str,
    runtime_model: LTTuningRuntime,
    tokenizer: Any,
    data_module: dict[str, Any],
    training_args: Any,
    payload: dict[str, Any],
) -> None:
    """Stage-aware training runner for the LT-Tuning recipe.

    Runs the 3-stage curriculum sequentially. Between stages we:

    1. Regenerate the training dataset with the stage's thinking-token
       insertion strategy (stage 0 = no insertions; stage 2 optionally uses
       the confidence-triggered strategy that requires a forward pass through
       the current model).
    2. Flip the runtime's stage mode and fusion_alpha.
    3. Construct a fresh ``LatentTrainer`` for the stage with a stage-specific
       ``num_train_epochs`` and ``learning_rate`` (by override into
       ``training_args``). We deliberately recreate the trainer per stage so
       that HF Trainer's internal step counter, LR scheduler, and callbacks
       see a clean slate.

    The final model state is persisted once all stages complete. The
    registered thinking-token id is added to the tokenizer lazily once the
    first latent stage is about to start.
    """
    lt_config = LTTuningConfig.from_dict(payload.get("lt_tuning") if payload else None)
    # Rewire the runtime's lt_config to match the YAML (the generic
    # runtime_builder couldn't pass this keyword so LTTuningRuntime was
    # instantiated with defaults).
    runtime_model.lt_config = lt_config
    scheduler = CurriculumScheduler(lt_config)
    stages = scheduler.stages()
    logger.info(
        "LT-Tuning curriculum stages=%s epochs=%s",
        [s.name for s in stages],
        [s.epochs for s in stages],
    )

    train_examples = data_module.get("_lt_tuning_train_examples")
    eval_examples = data_module.get("_lt_tuning_eval_examples")
    if train_examples is None:
        # Happens if the data builder wasn't the lt_tuning-specific one.
        # Fall back to a fresh collection pass.
        train_examples, eval_examples = collect_lt_tuning_examples(
            tokenizer=tokenizer,
            data_config=config.data,
            runtime_config=config.runtime,
        )
    logger.info(
        "LT-Tuning formatted examples train=%d eval=%d",
        len(train_examples),
        len(eval_examples or []),
    )

    # Register the thinking token in the tokenizer and resize the model's
    # embedding table. This mirrors the clone's run.py setup.
    thinking_token = lt_config.thinking_token
    if thinking_token in tokenizer.get_vocab():
        thinking_token_id = tokenizer.convert_tokens_to_ids(thinking_token)
        logger.info("Thinking token %r already in tokenizer vocab id=%d", thinking_token, thinking_token_id)
    else:
        added = tokenizer.add_tokens([thinking_token])
        thinking_token_id = tokenizer.convert_tokens_to_ids(thinking_token)
        logger.info(
            "Added thinking token %r to tokenizer id=%d added_new=%d",
            thinking_token,
            thinking_token_id,
            added,
        )
        # Resize the underlying transformer's embeddings. When PEFT wraps
        # ``embed_tokens`` / ``lm_head`` via ``modules_to_save`` the HF
        # ``_get_resized_embeddings`` type-check fails, so we unwrap,
        # resize, and re-wrap via the helper below.
        _resize_embeddings_peft_safe(runtime_model.model, len(tokenizer))
        logger.info("Resized model embeddings to vocab size %d", len(tokenizer))
    runtime_model.set_thinking_token_id(thinking_token_id)

    output_root = Path(training_args.output_dir)
    ensure_dir(output_root)

    for stage_idx, stage in enumerate(stages):
        if stage.epochs <= 0:
            logger.info("Skipping stage=%s (epochs=0)", stage.name)
            continue
        logger.info(
            "=== LT-Tuning stage=%s mode=%s epochs=%d lr=%.3e fusion_alpha=%.3f ===",
            stage.name,
            stage.mode,
            stage.epochs,
            stage.learning_rate,
            stage.fusion_alpha,
        )

        # Build the stage-specific dataset. The confidence strategy needs the
        # *current* base model for probability estimation.
        base_model_for_confidence = runtime_model.model
        train_dataset = build_lt_tuning_dataset_for_stage(
            tokenizer=tokenizer,
            examples=train_examples,
            runtime_config=config.runtime,
            bot_id=runtime_model.bot_id,
            eot_id=runtime_model.eot_id,
            thinking_token_id=thinking_token_id,
            stage=stage,
            lt_config=lt_config,
            model_for_confidence=base_model_for_confidence,
            seed=config.runtime.seed,
            scheduled_stage_index=stage_idx,
        )
        eval_dataset = None
        if eval_examples:
            eval_dataset = build_lt_tuning_dataset_for_stage(
                tokenizer=tokenizer,
                examples=eval_examples,
                runtime_config=config.runtime,
                bot_id=runtime_model.bot_id,
                eot_id=runtime_model.eot_id,
                thinking_token_id=thinking_token_id,
                stage=stage,
                lt_config=lt_config,
                model_for_confidence=base_model_for_confidence,
                seed=config.runtime.seed,
                scheduled_stage_index=stage_idx,
            )

        # Update runtime stage mode BEFORE training begins for this stage.
        runtime_model.set_stage_mode(stage.mode, fusion_alpha=stage.fusion_alpha)

        # Clone training_args for this stage so the cumulative output dir
        # structure is preserved but per-stage overrides apply.
        stage_args = copy.deepcopy(training_args)
        stage_args.num_train_epochs = float(stage.epochs)
        stage_args.learning_rate = float(stage.learning_rate)
        stage_output_dir = str(output_root / f"stage_{stage_idx}_{stage.name}")
        stage_args.output_dir = stage_output_dir
        stage_args.logging_dir = stage_output_dir + "/logs"
        ensure_dir(stage_output_dir)

        from latent_harness.training.datasets import SupervisedLatentDataCollator

        stage_trainer = LatentTrainer(
            **_build_trainer_init_kwargs(
                LatentTrainer,
                model=runtime_model,
                training_args=stage_args,
                tokenizer=tokenizer,
                data_module={
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "data_collator": SupervisedLatentDataCollator(tokenizer=tokenizer),
                },
            )
        )
        stage_trainer._record_event(
            "lt_tuning_stage_start",
            {
                "config_path": config_path,
                "stage_index": stage_idx,
                "stage_name": stage.name,
                "stage_mode": stage.mode,
                "stage_epochs": stage.epochs,
                "stage_lr": stage.learning_rate,
                "fusion_alpha": stage.fusion_alpha,
                "num_train_examples": len(train_dataset),
            },
        )
        stage_trainer.train()
        stage_trainer._record_event(
            "lt_tuning_stage_end",
            {
                "stage_index": stage_idx,
                "stage_name": stage.name,
                "global_step": stage_trainer.state.global_step,
            },
        )
        # Persist checkpoint for this stage.
        atomic_save_state_dict(runtime_model, stage_output_dir)
        tokenizer.save_pretrained(stage_output_dir)

    # Persist final combined checkpoint at the top-level output_dir.
    atomic_save_state_dict(runtime_model, str(output_root))
    tokenizer.save_pretrained(str(output_root))
    logger.info("Finished LT-Tuning curriculum; saved final checkpoint to %s", output_root)


def run_training_from_config(config_path: str) -> None:
    _configure_training_logging()
    payload = load_yaml_config(config_path)
    config = TrainingConfig.from_dict(payload)
    recipe = get_method_recipe(config.method)
    recipe.assert_implemented()
    training_args = config.to_hf_training_arguments()
    logger.info(
        "Starting training config=%s method=%s model=%s output_dir=%s",
        config_path,
        config.method,
        config.model.base_model_name_or_path,
        training_args.output_dir,
    )
    logger.info(
        "Trainer args batch_size=%s grad_accum=%s epochs=%s logging_steps=%s eval_strategy=%s save_strategy=%s effective_batch_size=%s",
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
        training_args.num_train_epochs,
        training_args.logging_steps,
        getattr(training_args, "eval_strategy", getattr(training_args, "evaluation_strategy", None)),
        training_args.save_strategy,
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(getattr(training_args, "world_size", 1), 1),
    )
    logger.info(
        "Runtime controls bf16=%s num_latent=%s distill_type=%s distill_factor=%s distill_div_std=%s distill_std_floor=%s ref_loss_factor=%s max_grad_norm=%s learning_rate=%s",
        config.runtime.bf16,
        config.runtime.num_latent,
        config.runtime.distill_loss_type,
        config.runtime.distill_loss_factor,
        config.runtime.distill_loss_div_std,
        config.runtime.distill_loss_std_floor,
        config.runtime.ref_loss_factor,
        training_args.max_grad_norm,
        training_args.learning_rate,
    )
    logger.info(
        "Projection controls use_prj=%s prj_fp32=%s prj_residual_gated=%s prj_gate_init=%s prj_dim=%s prj_no_ln=%s",
        config.runtime.use_prj,
        config.runtime.prj_fp32,
        config.runtime.prj_residual_gated,
        config.runtime.prj_gate_init,
        config.runtime.prj_dim,
        config.runtime.prj_no_ln,
    )

    runtime_builder = recipe.runtime_builder
    data_builder = recipe.data_module_builder
    assert runtime_builder is not None
    assert data_builder is not None

    logger.info(
        "Model controls freeze_base_embeddings=%s use_lora=%s lora_r=%s lora_alpha=%s",
        config.model.freeze_base_embeddings,
        config.model.use_lora,
        config.model.lora_r,
        config.model.lora_alpha,
    )
    logger.info("Building runtime model")
    runtime_model = runtime_builder(
        model_config=config.model,
        runtime_config=config.runtime,
        train_mode=True,
    )
    alpha = getattr(runtime_model, "prj_residual_alpha", None)
    if alpha is not None:
        logger.info("prj_residual_alpha tensor value=%s", float(alpha.detach().cpu()))
    trainable_count = sum(p.numel() for p in runtime_model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in runtime_model.parameters() if not p.requires_grad)
    logger.info(
        "Parameter counts trainable=%s frozen=%s total=%s trainable_pct=%.2f%%",
        trainable_count,
        frozen_count,
        trainable_count + frozen_count,
        100.0 * trainable_count / max(trainable_count + frozen_count, 1),
    )
    logger.info("Building tokenizer")
    tokenizer = runtime_model.build_tokenizer()
    logger.info("Building data module")
    data_module = data_builder(
        tokenizer=tokenizer,
        data_config=config.data,
        runtime_config=config.runtime,
        bot_id=runtime_model.bot_id,
        eot_id=runtime_model.eot_id,
    )
    train_dataset = data_module.get("train_dataset")
    eval_dataset = data_module.get("eval_dataset")
    logger.info(
        "Finished data module build train_examples=%s eval_examples=%s",
        len(train_dataset) if train_dataset is not None else 0,
        len(eval_dataset) if eval_dataset is not None else 0,
    )

    trainer_cls: type[Trainer]
    if recipe.training_style == "standard_sft":
        model = StandardSFTModel(runtime_model)
        trainer_cls = StandardSFTTrainer
    elif recipe.training_style == "lt_tuning":
        # Stage-aware training runner lives in a dedicated helper below.
        run_lt_tuning_training(
            config=config,
            config_path=config_path,
            runtime_model=runtime_model,
            tokenizer=tokenizer,
            data_module=data_module,
            training_args=training_args,
            payload=payload,
        )
        return
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
    if getattr(config.runtime, "probe_mode", False):
        logger.info(
            "probe_mode enabled; attaching ProbeCallback output_dir=%s",
            config.runtime.probe_output_dir,
        )
        trainer.add_callback(ProbeCallback(config.runtime))
    resume_checkpoint = config.trainer.get("resume_from_checkpoint")
    trainer._record_event(
        "train_start",
        {
            "config_path": config_path,
            "output_dir": str(training_args.output_dir),
            "resume_from_checkpoint": resume_checkpoint,
        },
    )
    logger.info("Entering trainer.train resume_from_checkpoint=%s", resume_checkpoint)
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as exc:
        trainer._record_event("train_exception", {"error": str(exc)})
        raise
    logger.info("Finished trainer.train global_step=%s", trainer.state.global_step)
    trainer._record_event("train_end", {"global_step": trainer.state.global_step})
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    atomic_save_state_dict(model, training_args.output_dir)
    logger.info("Saved final training artifacts to %s", training_args.output_dir)
