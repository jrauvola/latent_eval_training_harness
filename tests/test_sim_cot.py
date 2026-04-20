"""Tests for the SIM-CoT auxiliary step-decoder integration.

These tests intentionally avoid downloading real model weights; the
runtime-level behaviors are exercised with a tiny in-process model built
via the ``object.__new__`` pattern used elsewhere in the suite (see
``test_training_stability.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig
from latent_harness.training.methods import get_method_recipe
from latent_harness.training.sim_cot import (
    AuxiliaryStepDecoder,
    SimCotLatentDataCollator,
    SimCotLatentRuntime,
    build_step_token_lists,
    dedup_trailing_pads,
    extract_step_texts,
    make_sim_cot_data_module,
    pad_step_batch,
)


# -----------------------------------------------------------------------------
# Step extraction + padding helpers
# -----------------------------------------------------------------------------


class TestExtractStepTexts:
    def test_extracts_multiple_chunks(self) -> None:
        cot = "<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>>"
        assert extract_step_texts(cot) == [
            "<<600*30/100=180>>",
            "<<600*10/100=60>>",
            "<<180+60=240>>",
        ]

    def test_single_chunk(self) -> None:
        assert extract_step_texts("<<150/0.2=750>>") == ["<<150/0.2=750>>"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert extract_step_texts("") == []

    def test_fallback_when_no_chunks(self) -> None:
        # Natural-language CoT: still return a single-chunk fallback so the
        # aux decoder has at least one supervision target.
        nl_cot = "Rebecca saves 20% of her salary."
        assert extract_step_texts(nl_cot) == [nl_cot]


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer.

    Tokenizes by splitting on whitespace and mapping each token to a
    deterministic id via ``hash``. Good enough for pad/truncate tests.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = 100 + len(self._vocab)
            ids.append(self._vocab[token])
        return ids


class TestBuildStepTokenLists:
    def test_tokenizes_and_appends_eot(self) -> None:
        tokenizer = _FakeTokenizer()
        steps = build_step_token_lists(
            ["alpha beta", "gamma"],
            tokenizer=tokenizer,
            eot_id=999,
            pad_id=0,
            latent_num=2,
        )
        assert len(steps) == 2
        assert steps[0][-1] == 999
        assert steps[1][-1] == 999

    def test_pads_when_fewer_steps_than_requested(self) -> None:
        tokenizer = _FakeTokenizer()
        steps = build_step_token_lists(
            ["alpha"],
            tokenizer=tokenizer,
            eot_id=999,
            pad_id=0,
            latent_num=3,
        )
        assert len(steps) == 3
        assert steps[-1] == [0]
        assert steps[-2] == [0]

    def test_merges_tail_when_more_steps_than_slots(self) -> None:
        tokenizer = _FakeTokenizer()
        # 4 steps, 2 slots: first slot keeps "alpha", second slot merges
        # the tail three steps into one long sequence ending with eot.
        steps = build_step_token_lists(
            ["alpha", "beta gamma", "delta", "epsilon"],
            tokenizer=tokenizer,
            eot_id=999,
            pad_id=0,
            latent_num=2,
        )
        assert len(steps) == 2
        assert steps[0][-1] == 999
        assert steps[1][-1] == 999
        assert len(steps[1]) > len(steps[0])


class TestPadStepBatch:
    def test_pads_outer_and_inner_dimensions(self) -> None:
        batch = [
            [[1, 2, 3], [4, 5]],
            [[6]],
        ]
        padded = pad_step_batch(batch, pad_id=0)
        assert len(padded) == 2
        assert all(len(steps) == 2 for steps in padded)
        assert all(len(step) == 3 for steps in padded for step in steps)
        assert padded[1][0] == [6, 0, 0]
        assert padded[1][1] == [0, 0, 0]

    def test_empty_batch(self) -> None:
        assert pad_step_batch([], pad_id=0) == []


class TestDedupTrailingPads:
    def test_drops_all_pad_column(self) -> None:
        rows = [[1, 2, 0, 0], [3, 4, 0, 0]]
        trimmed = dedup_trailing_pads(rows, pad_id=0)
        assert trimmed == [[1, 2, 0], [3, 4, 0]]

    def test_keeps_single_column(self) -> None:
        rows = [[0], [0]]
        trimmed = dedup_trailing_pads(rows, pad_id=0)
        assert trimmed == [[0], [0]]

    def test_noop_when_non_pad_in_tail(self) -> None:
        rows = [[1, 2, 3], [4, 5, 6]]
        assert dedup_trailing_pads(rows, pad_id=0) == rows


# -----------------------------------------------------------------------------
# Method registry wiring
# -----------------------------------------------------------------------------


def test_sim_cot_recipe_is_implemented() -> None:
    recipe = get_method_recipe("sim_cot")
    assert recipe.implemented is True
    assert recipe.runtime_builder is SimCotLatentRuntime
    assert recipe.data_module_builder is make_sim_cot_data_module
    # Explicitly should not raise.
    recipe.assert_implemented()


# -----------------------------------------------------------------------------
# Shared-head AuxiliaryStepDecoder (no real HF model downloaded)
# -----------------------------------------------------------------------------


def _build_shared_head_aux_decoder(hidden: int = 16, vocab: int = 32) -> tuple[
    AuxiliaryStepDecoder, nn.Module, nn.Module
]:
    lm_head = nn.Linear(hidden, vocab, bias=False)
    embedding = nn.Embedding(vocab, hidden)
    runtime_config = LatentRuntimeConfig(
        num_latent=2,
        aux_decoder_enabled=True,
        aux_decoder_full_lm=False,
        aux_decoder_shared_head_layers=1,
        use_prj=False,
        bf16=False,
        fp32=True,
    )
    model_config = ModelConfig(
        base_model_name_or_path="sshleifer/tiny-gpt2",  # not actually loaded
        use_lora=False,
        full_precision=True,
    )
    aux = AuxiliaryStepDecoder(
        model_config=model_config,
        runtime_config=runtime_config,
        base_hidden_size=hidden,
        pad_token_id=0,
        bot_id=vocab - 2,
        eot_id=vocab - 1,
        runtime_dtype=torch.float32,
        base_lm_head=lm_head,
        base_embedding=embedding,
    )
    return aux, lm_head, embedding


class TestAuxiliaryStepDecoderSharedHead:
    def test_shared_head_decoder_builds(self) -> None:
        aux, _, _ = _build_shared_head_aux_decoder()
        # Shared-head fallback should NOT allocate a full LM.
        assert aux.decoder is None
        assert aux.full_lm is False

    def test_forward_step_emits_logits_and_labels(self) -> None:
        aux, _, _ = _build_shared_head_aux_decoder(hidden=16, vocab=32)
        batch_size = 2
        step_len = 4
        latent_embd = torch.zeros(batch_size, 1, 16)
        step_ids = torch.tensor(
            [
                [5, 6, 7, 31],
                [8, 0, 0, 0],  # pad_id=0 in trailing positions
            ],
            dtype=torch.long,
        )
        logits, labels = aux.forward_step(
            latent_embd=latent_embd,
            step_token_ids=step_ids,
        )
        assert logits.shape == (batch_size, step_len + 1, 32)
        assert labels.shape == (batch_size, step_len + 1)
        # First column (the prefix sentinel) must always be IGNORE_INDEX.
        assert (labels[:, 0] == -100).all()
        # Pad positions in row 1 must map to IGNORE_INDEX.
        assert (labels[1, 2:] == -100).all()

    def test_forward_step_loss_is_finite_and_nonzero(self) -> None:
        aux, _, _ = _build_shared_head_aux_decoder(hidden=16, vocab=32)
        latent_embd = torch.randn(1, 1, 16)
        step_ids = torch.tensor([[5, 6, 7, 31]], dtype=torch.long)
        logits, labels = aux.forward_step(
            latent_embd=latent_embd,
            step_token_ids=step_ids,
        )
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss = loss_fct(shift_logits.float(), shift_labels)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0


# -----------------------------------------------------------------------------
# SimCotLatentRuntime: state_dict strip / load tolerance, drop_aux_decoder.
# -----------------------------------------------------------------------------


class _FakeBaseModel(nn.Module):
    def __init__(self, hidden: int = 16, vocab: int = 32) -> None:
        super().__init__()
        self.base_linear = nn.Linear(hidden, hidden, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden, vocab_size=vocab)


def _build_sim_cot_runtime_mock(hidden: int = 16, vocab: int = 32) -> SimCotLatentRuntime:
    """Build a SimCotLatentRuntime in-process without touching HF hub."""
    runtime = object.__new__(SimCotLatentRuntime)
    nn.Module.__init__(runtime)
    runtime.model_config = ModelConfig(
        base_model_name_or_path="sshleifer/tiny-gpt2",
        use_lora=False,
        full_precision=True,
    )
    runtime.runtime_config = LatentRuntimeConfig(
        num_latent=2,
        aux_decoder_enabled=True,
        aux_decoder_full_lm=False,
        use_prj=False,
        bf16=False,
        fp32=True,
    )
    runtime.train_mode = True
    runtime._sim_cot_train_mode = True
    runtime.runtime_dtype = torch.float32
    fake = _FakeBaseModel(hidden=hidden, vocab=vocab)
    runtime.model = fake
    runtime.prj = nn.Identity()
    runtime.pad_token_id = 0
    runtime.bot_id = vocab - 2
    runtime.eot_id = vocab - 1
    runtime.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    runtime.distill_loss_fct = nn.SmoothL1Loss()
    lm_head = nn.Linear(hidden, vocab, bias=False)
    embedding = nn.Embedding(vocab, hidden)
    runtime.aux_decoder = AuxiliaryStepDecoder(
        model_config=runtime.model_config,
        runtime_config=runtime.runtime_config,
        base_hidden_size=hidden,
        pad_token_id=runtime.pad_token_id,
        bot_id=runtime.bot_id,
        eot_id=runtime.eot_id,
        runtime_dtype=torch.float32,
        base_lm_head=lm_head,
        base_embedding=embedding,
    )
    return runtime


class TestSimCotRuntimeCheckpointContract:
    def test_state_dict_strips_aux_decoder_keys(self) -> None:
        runtime = _build_sim_cot_runtime_mock()
        state = runtime.state_dict()
        aux_keys = [k for k in state.keys() if k.startswith("aux_decoder.")]
        assert aux_keys == []

    def test_state_dict_retains_base_model_keys(self) -> None:
        runtime = _build_sim_cot_runtime_mock()
        state = runtime.state_dict()
        # The fake base model has a ``base_linear`` param; confirm it
        # survives under its namespaced key.
        assert any("base_linear" in key for key in state.keys())

    def test_load_tolerates_codi_checkpoint_without_aux_keys(self) -> None:
        """A checkpoint saved from the CODI path should load into a SIM-CoT
        runtime without raising. The aux decoder remains attached in memory
        afterwards — its absence in the on-disk state is expected and the
        loader uses ``strict=False`` so the missing keys are informational."""
        runtime = _build_sim_cot_runtime_mock()
        assert runtime.aux_decoder is not None
        codi_state = runtime.state_dict()  # already stripped of aux keys
        # No raise: we accept the stripped state dict. The runtime must
        # still have a functional aux decoder after the load (we're still
        # in train mode).
        runtime.load_state_dict(codi_state, strict=False)
        assert runtime.aux_decoder is not None

    def test_load_tolerates_state_with_aux_keys(self) -> None:
        """A legacy state dict that *does* contain aux_decoder.* entries
        must not leak into base-model parameter keys. The SIM-CoT loader
        filters them out."""
        runtime = _build_sim_cot_runtime_mock()
        state = dict(runtime.state_dict())
        state["aux_decoder.phantom.weight"] = torch.zeros(2, 2)
        # No raise: extraneous aux_decoder key is silently dropped.
        runtime.load_state_dict(state, strict=False)
        assert runtime.aux_decoder is not None

    def test_drop_aux_decoder_removes_module(self) -> None:
        runtime = _build_sim_cot_runtime_mock()
        assert runtime.aux_decoder is not None
        runtime.drop_aux_decoder()
        assert runtime.aux_decoder is None
        assert runtime._sim_cot_train_mode is False

    def test_drop_then_state_dict_still_clean(self) -> None:
        runtime = _build_sim_cot_runtime_mock()
        runtime.drop_aux_decoder()
        state = runtime.state_dict()
        assert all(not k.startswith("aux_decoder.") for k in state.keys())


class TestSimCotCollator:
    def test_collator_attaches_step_tokens(self) -> None:
        # Build fake per-example records with a ``step_tokens`` field, run
        # through the collator, and verify the batched field exists.
        tokenizer = _FakeTokenizer()
        collator = SimCotLatentDataCollator(tokenizer=tokenizer, pad_id=0)
        records = [
            {
                "encoder_input_ids": torch.tensor([1, 2], dtype=torch.long),
                "decoder_input_ids": torch.tensor([3, 4], dtype=torch.long),
                "ref_input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "labels": torch.tensor([3, 4], dtype=torch.long),
                "ref_labels": torch.tensor([-100, -100, 3, 4], dtype=torch.long),
                "ref_answer_position": torch.tensor(2, dtype=torch.long),
                "model_answer_position": torch.tensor(0, dtype=torch.long),
                "step_tokens": [[100, 101, 999], [102, 999]],
                "forensics": {},
            },
            {
                "encoder_input_ids": torch.tensor([5], dtype=torch.long),
                "decoder_input_ids": torch.tensor([6], dtype=torch.long),
                "ref_input_ids": torch.tensor([5, 6], dtype=torch.long),
                "labels": torch.tensor([6], dtype=torch.long),
                "ref_labels": torch.tensor([-100, 6], dtype=torch.long),
                "ref_answer_position": torch.tensor(1, dtype=torch.long),
                "model_answer_position": torch.tensor(0, dtype=torch.long),
                "step_tokens": [[103, 999]],
                "forensics": {},
            },
        ]
        batch = collator(records)
        assert "step_tokens" in batch
        padded = batch["step_tokens"]
        assert len(padded) == 2
        assert all(len(s) == 2 for s in padded)
        # Row 1 (shorter) should be padded with an all-pad slot.
        assert padded[1][1] == [0, 0, 0]


# -----------------------------------------------------------------------------
# Config plumbing
# -----------------------------------------------------------------------------


class TestSimCotConfig:
    def test_defaults_disabled(self) -> None:
        cfg = LatentRuntimeConfig()
        assert cfg.aux_decoder_enabled is False
        assert cfg.aux_decoder_full_lm is True
        assert cfg.aux_decoder_explain_loss_factor == pytest.approx(1.0)

    def test_enable_and_fallback_knobs(self) -> None:
        cfg = LatentRuntimeConfig(
            aux_decoder_enabled=True,
            aux_decoder_full_lm=False,
            aux_decoder_explain_loss_factor=2.5,
            aux_decoder_shared_head_layers=2,
        )
        assert cfg.aux_decoder_enabled is True
        assert cfg.aux_decoder_full_lm is False
        assert cfg.aux_decoder_explain_loss_factor == pytest.approx(2.5)
        assert cfg.aux_decoder_shared_head_layers == 2


# -----------------------------------------------------------------------------
# Forward integration: aux loss term is wired into the total loss.
# -----------------------------------------------------------------------------


class _ScriptedModelForSimCot(nn.Module):
    """Emits scripted outputs for the SIM-CoT forward call chain.

    The total number of required ``model(...)`` calls in the combined
    base-forward + aux-rollout pass is:

    * 1 question encode
    * 1 no-grad teacher forward
    * 1 with-grad teacher forward (when ``ref_loss_factor > 0``)
    * ``num_latent`` latent steps (base forward)
    * 1 student decode
    * 1 question re-encode for aux pass
    * up to ``num_latent`` aux-rollout latent steps
    """

    def __init__(self, hidden: int, vocab: int, num_latent: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, vocab_size=vocab)
        self._num_latent = num_latent
        # Scripted outputs shared across forward calls.
        self._hidden = torch.randn(1, 2, hidden)
        self._logits = torch.randn(1, 2, vocab)

    def forward(self, **kwargs):
        pkv = kwargs.get("past_key_values")
        has_input_ids = kwargs.get("input_ids") is not None
        has_embeds = kwargs.get("inputs_embeds") is not None
        if has_input_ids and pkv is None:
            # Question encode or teacher forward.
            return SimpleNamespace(
                past_key_values="pkv_q",
                hidden_states=(self._hidden, self._hidden),
                logits=self._logits,
            )
        if has_embeds:
            # Latent rollout step or student decode.
            return SimpleNamespace(
                past_key_values="pkv_l",
                hidden_states=(self._hidden, self._hidden),
                logits=self._logits,
            )
        return SimpleNamespace(
            past_key_values="pkv_x",
            hidden_states=(self._hidden, self._hidden),
            logits=self._logits,
        )


def _build_full_sim_cot_runtime(hidden: int = 16, vocab: int = 32, num_latent: int = 1) -> SimCotLatentRuntime:
    runtime = object.__new__(SimCotLatentRuntime)
    nn.Module.__init__(runtime)
    runtime.model_config = ModelConfig(
        base_model_name_or_path="sshleifer/tiny-gpt2",
        use_lora=False,
        full_precision=True,
    )
    runtime.runtime_config = LatentRuntimeConfig(
        num_latent=num_latent,
        aux_decoder_enabled=True,
        aux_decoder_full_lm=False,
        use_prj=False,
        bf16=False,
        fp32=True,
        distill_loss_div_std=True,
        distill_loss_std_floor=0.5,
        ref_loss_factor=1.0,
        aux_decoder_explain_loss_factor=1.0,
    )
    runtime.train_mode = True
    runtime._sim_cot_train_mode = True
    runtime.runtime_dtype = torch.float32
    runtime.model = _ScriptedModelForSimCot(hidden=hidden, vocab=vocab, num_latent=num_latent)
    runtime.prj = nn.Identity()
    runtime.pad_token_id = 0
    runtime.bot_id = vocab - 2
    runtime.eot_id = vocab - 1
    runtime.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    runtime.distill_loss_fct = nn.SmoothL1Loss()
    embedding = nn.Embedding(vocab, hidden)
    runtime.get_input_embedding_layer = lambda: embedding
    runtime.maybe_project = lambda hidden_state: hidden_state
    runtime.build_token_type_ids = lambda input_ids=None, inputs_embeds=None: None
    lm_head = nn.Linear(hidden, vocab, bias=False)
    runtime.aux_decoder = AuxiliaryStepDecoder(
        model_config=runtime.model_config,
        runtime_config=runtime.runtime_config,
        base_hidden_size=hidden,
        pad_token_id=runtime.pad_token_id,
        bot_id=runtime.bot_id,
        eot_id=runtime.eot_id,
        runtime_dtype=torch.float32,
        base_lm_head=lm_head,
        base_embedding=embedding,
    )
    return runtime


class TestSimCotComputeExplainLoss:
    def test_explain_loss_nonzero_with_supervision(self) -> None:
        """Call ``_compute_explain_loss`` directly with a scripted model.

        The base-forward chain is covered by the existing
        ``test_training_stability`` suite; here we isolate the aux-rollout
        loop to confirm it emits a finite, nonzero loss when supervision
        is present.
        """
        runtime = _build_full_sim_cot_runtime(hidden=16, vocab=32, num_latent=1)
        # Two step supervisions per batch element.
        step_tokens = [[[5, 6, 31], [7, 8, 31]]]
        total, effective = runtime._compute_explain_loss(
            encoder_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            encoder_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            step_tokens=step_tokens,
        )
        assert effective >= 1
        assert torch.isfinite(total)
        assert total.item() > 0.0

    def test_explain_loss_zero_when_no_step_tokens(self) -> None:
        runtime = _build_full_sim_cot_runtime(hidden=16, vocab=32, num_latent=1)
        total, effective = runtime._compute_explain_loss(
            encoder_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            encoder_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            step_tokens=None,
        )
        assert effective == 0
        assert total.item() == 0.0

    def test_explain_loss_zero_when_all_slots_padded(self) -> None:
        runtime = _build_full_sim_cot_runtime(hidden=16, vocab=32, num_latent=1)
        # All pad tokens -> labels are all IGNORE_INDEX -> loss is skipped.
        step_tokens = [[[0, 0, 0], [0, 0, 0]]]
        total, effective = runtime._compute_explain_loss(
            encoder_input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            encoder_attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            step_tokens=step_tokens,
        )
        assert effective == 0
        assert total.item() == 0.0
