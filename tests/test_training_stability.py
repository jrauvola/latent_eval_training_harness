from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Trainer

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig
from latent_harness.core.runtime import LatentReasoningRuntime, NumericalInstabilityError
from latent_harness.training.config import TrainingConfig
from latent_harness.training.trainer import LatentTrainer


class ScriptedSequenceModel(nn.Module):
    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        super().__init__()
        self.outputs = list(outputs)

    def forward(self, **kwargs):
        del kwargs
        return self.outputs.pop(0)


class DummyLossModel(nn.Module):
    def __init__(
        self,
        *,
        outputs: dict | None = None,
        error: Exception | None = None,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.outputs = outputs
        self.error = error
        self.forward_kwargs: dict | None = None
        if training:
            self.train()
        else:
            self.eval()

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        if self.error is not None:
            raise self.error
        assert self.outputs is not None
        return self.outputs


class TinyGradModel(nn.Module):
    def __init__(self, *, grad_value: float, param_value: float = 0.3) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([param_value], dtype=torch.float32))
        self.weight.grad = torch.tensor([grad_value], dtype=torch.float32)


def _runtime_inputs() -> dict[str, torch.Tensor | int | float | bool]:
    return {
        "encoder_input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "decoder_input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "ref_input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "encoder_attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "ref_answer_position": torch.tensor([1], dtype=torch.long),
        "model_answer_position": torch.tensor([1], dtype=torch.long),
        "ref_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "ref_labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
        "step": 0,
        "step_ratio": 0.0,
        "collect_diagnostics": True,
    }


def _finite_runtime_outputs() -> list[SimpleNamespace]:
    encoder_hidden = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    teacher_hidden = torch.tensor([[[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]], dtype=torch.float32)
    latent_hidden = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
    student_hidden = torch.tensor([[[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]], dtype=torch.float32)
    student_logits = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], [0.4, 0.3, 0.2, 0.1]]],
        dtype=torch.float32,
    )
    ref_logits = torch.tensor(
        [[[0.1, 0.3, 0.2, 0.4], [1.5, 2.5, 3.5, 4.5], [0.2, 0.4, 0.3, 0.1]]],
        dtype=torch.float32,
    )
    return [
        SimpleNamespace(past_key_values="pkv0", hidden_states=(encoder_hidden, encoder_hidden)),
        SimpleNamespace(hidden_states=(teacher_hidden, teacher_hidden)),
        SimpleNamespace(logits=ref_logits, hidden_states=(teacher_hidden, teacher_hidden)),
        SimpleNamespace(past_key_values="pkv1", hidden_states=(latent_hidden, latent_hidden)),
        SimpleNamespace(logits=student_logits, hidden_states=(student_hidden, student_hidden)),
    ]


def _build_runtime(outputs: list[SimpleNamespace], *, distill_loss_std_floor: float = 0.5) -> LatentReasoningRuntime:
    runtime = object.__new__(LatentReasoningRuntime)
    nn.Module.__init__(runtime)
    runtime.model_config = ModelConfig(base_model_name_or_path="gpt2", use_lora=False)
    runtime.runtime_config = LatentRuntimeConfig(
        num_latent=1,
        use_prj=False,
        remove_eos=True,
        bf16=False,
        distill_loss_div_std=True,
        distill_loss_std_floor=distill_loss_std_floor,
    )
    runtime.train_mode = True
    runtime.model = ScriptedSequenceModel(outputs)
    runtime.prj = nn.Identity()
    runtime.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    runtime.distill_loss_fct = nn.SmoothL1Loss()
    runtime.pad_token_id = 0
    runtime.bot_id = 10
    runtime.eot_id = 11
    embedding = nn.Embedding(32, 2)
    runtime.get_input_embedding_layer = lambda: embedding
    runtime.maybe_project = lambda hidden: hidden
    runtime.build_token_type_ids = lambda input_ids=None, inputs_embeds=None: None
    return runtime


def _build_trainer(step: int = 5) -> tuple[LatentTrainer, list[dict[str, float]]]:
    trainer = object.__new__(LatentTrainer)
    trainer.state = SimpleNamespace(global_step=step, epoch=0.25)
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        world_size=1,
        num_train_epochs=1,
        logging_steps=5,
    )
    trainer.train_dataset = [0] * 12
    trainer.accelerator = SimpleNamespace(sync_gradients=True)
    logged: list[dict[str, float]] = []
    trainer.log = logged.append
    return trainer, logged


def test_runtime_forward_reports_std_floor_diagnostics() -> None:
    runtime = _build_runtime(_finite_runtime_outputs(), distill_loss_std_floor=0.5)

    outputs = runtime.forward(**_runtime_inputs())

    assert torch.isfinite(outputs["loss"])
    assert outputs["diagnostics"]["teacher_selected_std_min_raw"] == pytest.approx(0.0)
    assert outputs["diagnostics"]["teacher_selected_std_min_effective"] == pytest.approx(0.5)
    assert outputs["diagnostics"]["teacher_selected_std_clamped_count"] == pytest.approx(2.0)


def test_runtime_forward_raises_on_nonfinite_student_logits() -> None:
    outputs = _finite_runtime_outputs()
    student_output = outputs[-1]
    outputs[-1] = SimpleNamespace(
        logits=student_output.logits.clone().index_fill_(2, torch.tensor([0]), float("nan")),
        hidden_states=student_output.hidden_states,
    )
    runtime = _build_runtime(outputs)

    with pytest.raises(NumericalInstabilityError) as exc_info:
        runtime.forward(**_runtime_inputs())

    assert exc_info.value.details["stage"] == "student_decode"
    assert exc_info.value.details["tensor_name"] == "student_logits"


def test_training_config_accepts_distill_loss_std_floor() -> None:
    config = TrainingConfig.from_dict(
        {
            "method": "codi",
            "model": {"base_model_name_or_path": "gpt2"},
            "data": {},
            "runtime": {"distill_loss_std_floor": 1e-3},
        }
    )

    assert config.runtime.distill_loss_std_floor == pytest.approx(1e-3)


def test_compute_loss_logs_train_metrics_only_for_training_steps() -> None:
    trainer, logged = _build_trainer(step=5)
    model = DummyLossModel(
        outputs={
            "loss": torch.tensor(1.5),
            "ce_loss": 0.4,
            "distill_loss": 0.8,
            "ref_ce_loss": 0.3,
            "diagnostics": {"teacher_selected_std_min_raw": 0.2},
        }
    )

    loss = LatentTrainer.compute_loss(trainer, model, inputs={})

    assert torch.equal(loss, torch.tensor(1.5))
    assert model.forward_kwargs is not None
    assert model.forward_kwargs["collect_diagnostics"] is True
    assert logged == [
        {
            "train/loss": 1.5,
            "train/ce_loss": 0.4,
            "train/distill_loss": 0.8,
            "train/ref_ce_loss": 0.3,
            "debug/teacher_selected_std_min_raw": 0.2,
        }
    ]


def test_compute_loss_skips_custom_logging_during_eval() -> None:
    trainer, logged = _build_trainer(step=5)
    model = DummyLossModel(
        outputs={
            "loss": torch.tensor(1.5),
            "ce_loss": 0.4,
            "distill_loss": 0.8,
            "ref_ce_loss": 0.3,
            "diagnostics": {"teacher_selected_std_min_raw": 0.2},
        },
        training=False,
    )

    LatentTrainer.compute_loss(trainer, model, inputs={})

    assert model.forward_kwargs is not None
    assert model.forward_kwargs["collect_diagnostics"] is False
    assert logged == []


def test_compute_loss_wraps_numerical_instability_errors() -> None:
    trainer, _ = _build_trainer(step=5)
    model = DummyLossModel(
        error=NumericalInstabilityError(
            stage="student_decode",
            tensor_name="student_logits",
            summary={"absmax": float("nan")},
        )
    )

    with pytest.raises(RuntimeError, match="Numerical instability detected"):
        LatentTrainer.compute_loss(trainer, model, inputs={})


def test_training_step_logs_gradient_and_parameter_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, logged = _build_trainer(step=5)
    model = TinyGradModel(grad_value=2.0)
    model.train()

    def _fake_training_step(self, model, inputs, num_items_in_batch=None):
        del self, model, inputs, num_items_in_batch
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "training_step", _fake_training_step)

    loss = LatentTrainer.training_step(trainer, model, inputs={})

    assert torch.equal(loss, torch.tensor(1.0))
    assert logged == [
        {
            "debug/grad_norm": pytest.approx(2.0),
            "debug/grad_absmax": pytest.approx(2.0),
            "debug/nonfinite_grad_count": 0.0,
            "debug/tracked_grad_tensors": 1.0,
            "debug/grad_norm_top0_other": pytest.approx(2.0),
            "debug/param_absmax": pytest.approx(0.3),
            "debug/nonfinite_param_count": 0.0,
            "debug/tracked_param_tensors": 1.0,
        }
    ]


def test_training_step_raises_on_nonfinite_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, _ = _build_trainer(step=5)
    model = TinyGradModel(grad_value=float("nan"))
    model.train()

    def _fake_training_step(self, model, inputs, num_items_in_batch=None):
        del self, model, inputs, num_items_in_batch
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "training_step", _fake_training_step)

    with pytest.raises(RuntimeError, match="Non-finite gradients or parameters detected"):
        LatentTrainer.training_step(trainer, model, inputs={})
