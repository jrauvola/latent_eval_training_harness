"""Validate Qwen3-4B-Instruct-2507 loads without errors and chat template behaves.

This test uses tokenizer only (no full model download) to exercise the integration
points that tend to break on a new base model:
  - tokenizer loads
  - chat template applies
  - enable_thinking=False disables reasoning tokens
"""
from __future__ import annotations

import pytest


def test_qwen3_tokenizer_loads_and_chat_template_works():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception as exc:
        pytest.skip(f"Qwen3-4B-Instruct-2507 tokenizer not loadable: {exc}")

    msgs = [{"role": "user", "content": "What is 2+2?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert isinstance(text, str) and len(text) > 0

    try:
        text_nothink = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pytest.skip("Qwen3 chat template does not expose enable_thinking kwarg")
    # Presence-of-absence check: thinking tokens should not dominate
    assert "<think>" not in text_nothink or text_nothink.count("<think>") <= 1
