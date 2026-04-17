from __future__ import annotations

from unittest.mock import MagicMock, patch

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig
from latent_harness.evaluation.config import EvaluationModelSpec
from latent_harness.evaluation.models import _load_standard_generation_model


def _build_spec(*, trust_remote_code: bool) -> EvaluationModelSpec:
    return EvaluationModelSpec(
        name="ouro-test",
        checkpoint_source="ByteDance/Ouro-2.6B",
        checkpoint_type="hf_pretrained",
        model=ModelConfig(base_model_name_or_path="ByteDance/Ouro-2.6B"),
        runtime=LatentRuntimeConfig(model_max_length=512, num_latent=0, use_prj=False, bf16=True),
        inference_strategy="standard_generation",
        model_kind="causal_lm",
        trust_remote_code=trust_remote_code,
    )


def _install_mocks() -> tuple[MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"

    model = MagicMock()
    model.config = MagicMock(vocab_size=50000)
    model.hf_device_map = None
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return tokenizer, model


@patch("latent_harness.evaluation.models.AutoModelForCausalLM")
@patch("latent_harness.evaluation.models.AutoTokenizer")
def test_trust_remote_code_true_flows_to_hf(mock_tokenizer_cls, mock_model_cls) -> None:
    tok, mdl = _install_mocks()
    mock_tokenizer_cls.from_pretrained.return_value = tok
    mock_model_cls.from_pretrained.return_value = mdl

    _load_standard_generation_model(_build_spec(trust_remote_code=True), device="cpu")

    _, tok_kwargs = mock_tokenizer_cls.from_pretrained.call_args
    assert tok_kwargs.get("trust_remote_code") is True

    _, mdl_kwargs = mock_model_cls.from_pretrained.call_args
    assert mdl_kwargs.get("trust_remote_code") is True


@patch("latent_harness.evaluation.models.AutoModelForCausalLM")
@patch("latent_harness.evaluation.models.AutoTokenizer")
def test_trust_remote_code_false_flows_to_hf(mock_tokenizer_cls, mock_model_cls) -> None:
    tok, mdl = _install_mocks()
    mock_tokenizer_cls.from_pretrained.return_value = tok
    mock_model_cls.from_pretrained.return_value = mdl

    _load_standard_generation_model(_build_spec(trust_remote_code=False), device="cpu")

    _, tok_kwargs = mock_tokenizer_cls.from_pretrained.call_args
    assert tok_kwargs.get("trust_remote_code") is False

    _, mdl_kwargs = mock_model_cls.from_pretrained.call_args
    assert mdl_kwargs.get("trust_remote_code") is False
