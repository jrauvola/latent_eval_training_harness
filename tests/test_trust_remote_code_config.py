from __future__ import annotations

from latent_harness.evaluation.config import EvaluationModelSpec


def test_trust_remote_code_defaults_to_false() -> None:
    spec = EvaluationModelSpec.from_dict(
        {
            "name": "test",
            "checkpoint_source": "some/repo",
            "checkpoint_type": "hf_pretrained",
            "model_kind": "causal_lm",
            "model": {"base_model_name_or_path": "some/repo"},
            "runtime": {},
        }
    )
    assert spec.trust_remote_code is False


def test_trust_remote_code_parsed_when_true() -> None:
    spec = EvaluationModelSpec.from_dict(
        {
            "name": "test",
            "checkpoint_source": "some/repo",
            "checkpoint_type": "hf_pretrained",
            "model_kind": "causal_lm",
            "trust_remote_code": True,
            "model": {"base_model_name_or_path": "some/repo"},
            "runtime": {},
        }
    )
    assert spec.trust_remote_code is True
