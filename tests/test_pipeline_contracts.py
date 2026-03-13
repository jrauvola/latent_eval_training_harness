from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from latent_harness.core import load_checkpoint_state, remap_runtime_state_dict_prefixes, resolve_checkpoint_path
from latent_harness.evaluation.config import EvaluationModelSpec
from latent_harness.training.config import _filter_supported_training_args
from latent_harness.training.trainer import _build_trainer_init_kwargs, atomic_save_state_dict, LatentTrainer


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))


def test_training_checkpoint_is_eval_resolvable(tmp_path) -> None:
    model = TinyModule()
    checkpoint_dir = tmp_path / "run"
    saved_path = atomic_save_state_dict(model, checkpoint_dir)

    spec = EvaluationModelSpec.from_dict(
        {
            "name": "tiny",
            "checkpoint_source": str(checkpoint_dir),
            "checkpoint_type": "local_path",
            "model": {"base_model_name_or_path": "gpt2"},
            "runtime": {},
        }
    )

    resolved = resolve_checkpoint_path(
        spec.checkpoint_source,
        spec.checkpoint_type,
        token=spec.model.hf_token,
    )
    state_dict = load_checkpoint_state(resolved)

    assert resolved == str(saved_path)
    assert torch.equal(state_dict["weight"], model.state_dict()["weight"])


def test_runtime_checkpoint_prefixes_remap_from_legacy_codi() -> None:
    legacy = {
        "codi.base_model.model.lm_head.weight": torch.tensor([1.0]),
        "prj.0.weight": torch.tensor([2.0]),
    }
    target_keys = {
        "model.base_model.model.lm_head.weight",
        "prj.0.weight",
    }

    remapped = remap_runtime_state_dict_prefixes(legacy, target_keys=target_keys)

    assert "model.base_model.model.lm_head.weight" in remapped
    assert "codi.base_model.model.lm_head.weight" not in remapped
    assert torch.equal(remapped["model.base_model.model.lm_head.weight"], torch.tensor([1.0]))
    assert torch.equal(remapped["prj.0.weight"], torch.tensor([2.0]))


def test_eval_model_spec_accepts_nested_runtime_and_model_blocks() -> None:
    spec = EvaluationModelSpec.from_dict(
        {
            "name": "codi_gpt2_official",
            "checkpoint_source": "zen-E/CODI-gpt2",
            "checkpoint_type": "hf_repo",
            "model": {
                "base_model_name_or_path": "gpt2",
                "lora_r": 64,
            },
            "runtime": {
                "num_latent": 4,
            },
        }
    )

    assert spec.model.base_model_name_or_path == "gpt2"
    assert spec.model.lora_r == 64
    assert spec.runtime.num_latent == 4
    assert spec.model_kind == "latent_runtime"
    assert spec.inference_strategy == "latent_cot"


def test_eval_model_spec_accepts_legacy_flat_fields() -> None:
    spec = EvaluationModelSpec.from_dict(
        {
            "name": "codi_gpt2_official",
            "checkpoint_source": "zen-E/CODI-gpt2",
            "checkpoint_type": "hf_repo",
            "base_model_name_or_path": "gpt2",
            "lora_r": 64,
            "num_latent": 4,
        }
    )

    assert spec.model.base_model_name_or_path == "gpt2"
    assert spec.model.lora_r == 64
    assert spec.runtime.num_latent == 4


def test_eval_model_spec_supports_base_model_standard_generation() -> None:
    spec = EvaluationModelSpec.from_dict(
        {
            "name": "base_gpt2",
            "checkpoint_type": "base_model",
            "checkpoint_source": None,
            "model_kind": "causal_lm",
            "inference_strategy": "standard_generation",
            "model": {
                "base_model_name_or_path": "gpt2",
                "use_lora": False,
            },
            "runtime": {
                "use_prj": False,
                "num_latent": 0,
            },
        }
    )

    assert spec.checkpoint_source is None
    assert spec.checkpoint_type == "base_model"
    assert spec.model_kind == "causal_lm"
    assert spec.inference_strategy == "standard_generation"


def test_training_args_filter_drops_unsupported_keys() -> None:
    payload = {
        "output_dir": "/tmp/out",
        "logging_dir": "/tmp/logs",
        "learning_rate": 1e-3,
        "save_safetensors": True,
        "made_up_flag": "ignore-me",
    }

    filtered = _filter_supported_training_args(payload)

    assert "output_dir" in filtered
    assert "logging_dir" in filtered
    assert "learning_rate" in filtered
    assert "made_up_flag" not in filtered


def test_trainer_init_uses_tokenizer_when_supported() -> None:
    class TokenizerTrainer:
        def __init__(self, model, args, tokenizer=None, train_dataset=None):
            del model, args, tokenizer, train_dataset

    kwargs = _build_trainer_init_kwargs(
        TokenizerTrainer,
        model="model",
        training_args="args",
        tokenizer="tok",
        data_module={"train_dataset": "data"},
    )

    assert kwargs["tokenizer"] == "tok"
    assert "processing_class" not in kwargs


def test_trainer_init_falls_back_to_processing_class() -> None:
    class ProcessingTrainer:
        def __init__(self, model, args, processing_class=None, train_dataset=None):
            del model, args, processing_class, train_dataset

    kwargs = _build_trainer_init_kwargs(
        ProcessingTrainer,
        model="model",
        training_args="args",
        tokenizer="tok",
        data_module={"train_dataset": "data"},
    )

    assert kwargs["processing_class"] == "tok"
    assert "tokenizer" not in kwargs


def test_latent_trainer_save_model_uses_atomic_contract(tmp_path) -> None:
    model = TinyModule()

    class DummyProcessor:
        def __init__(self) -> None:
            self.saved_to: str | None = None

        def save_pretrained(self, destination: str) -> None:
            self.saved_to = destination

    processor = DummyProcessor()
    dummy_trainer = SimpleNamespace(
        model=model,
        args=SimpleNamespace(output_dir=str(tmp_path / "default")),
        processing_class=processor,
        tokenizer=None,
    )

    LatentTrainer.save_model(dummy_trainer, output_dir=str(tmp_path / "checkpoint"))

    assert (tmp_path / "checkpoint" / "model.safetensors").exists()
    assert processor.saved_to == str(tmp_path / "checkpoint")
