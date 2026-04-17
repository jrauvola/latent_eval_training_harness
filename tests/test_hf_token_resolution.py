from __future__ import annotations

import pytest

from latent_harness.core.config import resolve_hf_hub_token


def test_resolve_hf_hub_token_prefers_explicit() -> None:
    assert resolve_hf_hub_token("from-yaml") == "from-yaml"


def test_resolve_hf_hub_token_falls_back_to_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-hf")
    assert resolve_hf_hub_token(None) == "env-hf"


def test_resolve_hf_hub_token_falls_back_to_hub_alt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "env-alt")
    assert resolve_hf_hub_token(None) == "env-alt"


def test_explicit_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env")
    assert resolve_hf_hub_token("yaml") == "yaml"


def test_empty_string_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty YAML token is treated as unset so HF_TOKEN is used."""
    monkeypatch.setenv("HF_TOKEN", "env")
    assert resolve_hf_hub_token("") == "env"
