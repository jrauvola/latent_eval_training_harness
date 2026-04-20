"""LT-Tuning correctness tests.

Three categories:

1. CPF numerical correctness on a fixed handcrafted input.
2. Curriculum scheduler fires stage transitions at the right boundaries.
3. Stage-2 confidence-triggered thinking-token placement matches the
   reference repo's ``ConfidenceThinkingStrategy`` on a shared sample.

These tests intentionally avoid loading a real transformer — they use tiny
synthetic ``nn.Embedding`` layers and stub models so they run fast on CI.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from latent_harness.training.lt_tuning import (
    ArithmeticThinkingStrategy,
    ConfidenceThinkingStrategy,
    CurriculumScheduler,
    LTTuningConfig,
    RandomThinkingStrategy,
    StageSpec,
    cpf_fuse,
)
from latent_harness.training.methods import get_method_recipe


# ---------------------------------------------------------------------------
# CPF numerical correctness
# ---------------------------------------------------------------------------


def test_cpf_fuse_handcomputed_alpha_zero():
    """When alpha=0 and top_p=1 and T=1, CPF = softmax(logits) @ E."""
    torch.manual_seed(0)
    vocab, hidden = 4, 3
    # Deterministic embedding
    emb = nn.Embedding(vocab, hidden)
    emb.weight.data = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    hidden_state = torch.tensor([[0.5, 0.5, 0.5]])
    # logits chosen so softmax is roughly uniform; we then mask token 3 (thinking).
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    fused = cpf_fuse(
        hidden_state=hidden_state,
        logits=logits,
        embedding=emb,
        fusion_alpha=0.0,
        fusion_top_p=1.0,
        fusion_temperature=1.0,
        thinking_token_id=3,
    )
    # After masking token 3 (thinking), probs over {0,1,2} are 1/3 each.
    # Expected embedding = mean of first 3 rows = (1/3, 1/3, 1/3).
    expected = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert torch.allclose(fused, expected, atol=1e-5)


def test_cpf_fuse_convex_blend():
    """alpha=0.6 blends exactly: z = 0.6 * h + 0.4 * expected_emb."""
    torch.manual_seed(0)
    vocab, hidden = 3, 2
    emb = nn.Embedding(vocab, hidden)
    emb.weight.data = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    hidden_state = torch.tensor([[1.0, 2.0]])
    # logits chosen so token 0 takes all mass; masked token 2 (thinking).
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    fused = cpf_fuse(
        hidden_state=hidden_state,
        logits=logits,
        embedding=emb,
        fusion_alpha=0.6,
        fusion_top_p=0.8,
        fusion_temperature=1.0,
        thinking_token_id=2,
    )
    # Expected embedding ~ E[0] = (1,0). Blend: 0.6*(1,2) + 0.4*(1,0) = (1.0, 1.2)
    expected = torch.tensor([[1.0, 1.2]])
    assert torch.allclose(fused, expected, atol=1e-4)


def test_cpf_fuse_masks_thinking_token():
    """The thinking token must never contribute mass to the expected embedding."""
    torch.manual_seed(0)
    vocab, hidden = 3, 1
    emb = nn.Embedding(vocab, hidden)
    emb.weight.data = torch.tensor([[1.0], [2.0], [100.0]], dtype=torch.float32)
    hidden_state = torch.tensor([[0.0]])
    # logits make thinking token (id=2) dominant; after masking, tokens 0/1
    # share mass equally.
    logits = torch.tensor([[0.0, 0.0, 1e6]])
    fused = cpf_fuse(
        hidden_state=hidden_state,
        logits=logits,
        embedding=emb,
        fusion_alpha=0.0,
        fusion_top_p=1.0,
        fusion_temperature=1.0,
        thinking_token_id=2,
    )
    # Expected = (1 + 2) / 2 = 1.5
    assert torch.allclose(fused, torch.tensor([[1.5]]), atol=1e-4)


def test_cpf_fuse_matches_clone_soft_fusion_embedding():
    """Reproduce the clone's ``_soft_fusion_embedding`` on a fixed input and
    confirm our ``cpf_fuse`` produces the same output.

    Clone reference (Latent-Thoughts-Tuning/model.py::_soft_fusion_embedding):

        scaled_logits = logits / T
        masked = scaled.clone(); masked[thinking_id] = -inf
        probs = softmax(masked)
        sorted_probs, sorted_idx = sort(probs, desc)
        cum = cumsum(sorted_probs)
        cutoff = cum <= top_p
        if cutoff.sum() == 0: cutoff[0] = True
        filtered = zeros_like(probs)
        filtered[sorted_idx[cutoff]] = sorted_probs[cutoff]
        filtered = filtered / filtered.sum()
        token_embed = filtered @ E.weight
        fused = alpha * hidden + (1 - alpha) * token_embed
    """
    torch.manual_seed(42)
    vocab, hidden = 8, 4
    emb = nn.Embedding(vocab, hidden)
    emb.weight.data = torch.randn(vocab, hidden)
    hidden_state = torch.randn(1, hidden)
    logits = torch.randn(vocab)  # 1D per clone's impl
    alpha, top_p, temp, thinking = 0.6, 0.8, 1.3, 4

    # --- Clone-faithful reproduction ---
    scaled = logits / temp
    masked = scaled.clone()
    masked[thinking] = float("-inf")
    probs = torch.softmax(masked, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumsum <= top_p
    if cutoff.sum() == 0:
        cutoff[0] = True
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[cutoff]] = sorted_probs[cutoff]
    filtered = filtered / filtered.sum()
    token_embed = filtered @ emb.weight
    clone_fused = alpha * hidden_state + (1 - alpha) * token_embed.unsqueeze(0)

    # --- Harness cpf_fuse ---
    fused = cpf_fuse(
        hidden_state=hidden_state,
        logits=logits.unsqueeze(0),
        embedding=emb,
        fusion_alpha=alpha,
        fusion_top_p=top_p,
        fusion_temperature=temp,
        thinking_token_id=thinking,
    )
    # Our ``top_p <= cum`` mask differs by 1 from the clone when exactly at the
    # boundary, because the clone tests ``cum <= top_p`` (exclusive of the
    # token that tips cum past top_p), whereas both must still include the
    # top-1 when cumsum of top-1 already exceeds top_p. We match in both
    # formulations when no probability is exactly at the boundary. Tolerate a
    # tight numerical equivalence.
    # Note: clone keeps cumsum <= top_p; we also keep cumsum <= top_p and
    # additionally force top-1 inclusion. Both match in the generic case.
    assert torch.allclose(fused, clone_fused, atol=1e-5)


def test_cpf_fuse_top_p_keeps_top_1_min():
    """Even when top_p is tiny, we always keep at least the top token."""
    torch.manual_seed(0)
    vocab, hidden = 4, 2
    emb = nn.Embedding(vocab, hidden)
    emb.weight.data = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [10.0, 10.0]], dtype=torch.float32
    )
    hidden_state = torch.tensor([[0.0, 0.0]])
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    fused = cpf_fuse(
        hidden_state=hidden_state,
        logits=logits,
        embedding=emb,
        fusion_alpha=0.0,
        fusion_top_p=1e-6,
        fusion_temperature=1.0,
        thinking_token_id=3,
    )
    # Even with top_p~0, keep top-1 which is token 0 -> embedding (1, 0).
    assert torch.allclose(fused, torch.tensor([[1.0, 0.0]]), atol=1e-4)


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------


def test_curriculum_default_stages_order_and_epochs():
    cfg = LTTuningConfig()
    scheduler = CurriculumScheduler(cfg)
    stages = scheduler.stages()
    assert [s.name for s in stages] == ["stage0-cot", "stage1-hidden", "stage2-fusion"]
    assert [s.mode for s in stages] == ["explicit", "hidden_state", "soft_fusion"]
    assert [s.epochs for s in stages] == [1, 1, 3]


def test_curriculum_boundaries_cumulative():
    cfg = LTTuningConfig()
    scheduler = CurriculumScheduler(cfg)
    # With 100 steps/epoch, stages run [0,100), [100,200), [200,500).
    boundaries = scheduler.stage_boundaries(steps_per_epoch=100)
    assert boundaries == [
        ("stage0-cot", 0, 100),
        ("stage1-hidden", 100, 200),
        ("stage2-fusion", 200, 500),
    ]


def test_curriculum_stage_at_step_fires_at_boundaries():
    cfg = LTTuningConfig()
    scheduler = CurriculumScheduler(cfg)
    spe = 100
    # Last step of stage 0 is 99; first of stage 1 is 100.
    assert scheduler.stage_at_step(99, spe).name == "stage0-cot"
    assert scheduler.stage_at_step(100, spe).name == "stage1-hidden"
    assert scheduler.stage_at_step(199, spe).name == "stage1-hidden"
    assert scheduler.stage_at_step(200, spe).name == "stage2-fusion"
    # After the curriculum ends -> None.
    assert scheduler.stage_at_step(500, spe) is None


def test_curriculum_rejects_misaligned_tuples():
    with pytest.raises(ValueError, match="Inconsistent"):
        LTTuningConfig(
            stage_modes=("explicit", "hidden_state", "soft_fusion"),
            stage_epochs=(1, 1),  # wrong length
        ).stages()


def test_lt_tuning_in_registry_and_implemented():
    recipe = get_method_recipe("lt_tuning")
    assert recipe.implemented is True
    assert recipe.training_style == "lt_tuning"
    assert recipe.paper_name == "LT-Tuning"


# ---------------------------------------------------------------------------
# Stage-2 confidence threshold matches clone's dataset.py semantics
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub for the non-model-dependent strategies."""

    def __init__(self, vocab_map: dict[int, str] | None = None, pad_token_id: int = 0):
        self._vocab = vocab_map or {}
        self.pad_token_id = pad_token_id

    def decode(self, ids, *args, **kwargs):
        if isinstance(ids, int):
            return self._vocab.get(ids, f"<id_{ids}>")
        return " ".join(self._vocab.get(int(i), f"<id_{int(i)}>") for i in ids)


class _StubCausalLM(nn.Module):
    """Tiny stub model that returns fixed logits.

    The ConfidenceThinkingStrategy only reads ``outputs.logits`` and indexes
    with ``input_ids[0, 1:]`` to gather gold-next-token probabilities. Any
    tensor of shape ``[1, seq_len, vocab_size]`` is sufficient here.
    """

    def __init__(self, logits_per_step: torch.Tensor):
        super().__init__()
        # A trivial parameter so ``next(model.parameters()).device`` works.
        self.dummy = nn.Parameter(torch.zeros(1))
        self._logits = logits_per_step  # [seq_len, vocab]

    def forward(self, input_ids=None, attention_mask=None, **_):
        seq_len = input_ids.shape[1]
        # Truncate/pad to the requested sequence length.
        if self._logits.shape[0] < seq_len:
            pad = torch.zeros(
                seq_len - self._logits.shape[0],
                self._logits.shape[1],
                dtype=self._logits.dtype,
            )
            logits = torch.cat([self._logits, pad], dim=0)
        else:
            logits = self._logits[:seq_len]
        return SimpleNamespace(logits=logits.unsqueeze(0))


def test_confidence_strategy_selects_low_prob_positions():
    """Ground-truth next token with probability < threshold -> candidate."""
    vocab_size = 5
    # Build a 6-token input: question (2 tokens) + trace (4 tokens).
    # Token ids: [10, 11, 20, 21, 22, 23] mapped to vocab modulo. We'll use
    # vocab 0-4 and pick ids in that range for simplicity.
    input_ids = [0, 1, 2, 3, 4, 2]
    question_length = 2
    seq_len = len(input_ids)

    # Build per-position logits so the model thinks:
    # - at pos 0 (predicting token at pos 1 == 1): HIGH confidence (pass)
    # - at pos 1 (predicting token at pos 2 == 2): LOW confidence
    # - at pos 2 (predicting token at pos 3 == 3): HIGH confidence (pass)
    # - at pos 3 (predicting token at pos 4 == 4): LOW confidence
    # - at pos 4 (predicting token at pos 5 == 2): HIGH confidence (pass)
    # Insertion candidates are positions >= question_length (2). The strategy
    # checks token_probs[pos - 1] < threshold for pos in [2, 6). So it tests
    # probs at positions 1, 2, 3, 4 (gold tokens 2, 3, 4, 2).
    logits = torch.full((seq_len, vocab_size), -1e9, dtype=torch.float32)
    # pos 0 predicts token 1: confident
    logits[0, 1] = 10.0
    # pos 1 predicts token 2: LOW confidence (competing tokens are close)
    logits[1, 2] = 0.0
    logits[1, 0] = 0.0
    logits[1, 1] = 0.0
    logits[1, 3] = 0.0
    logits[1, 4] = 0.0
    # pos 2 predicts token 3: confident
    logits[2, 3] = 10.0
    # pos 3 predicts token 4: LOW
    logits[3, 4] = 0.0
    logits[3, 0] = 0.0
    logits[3, 1] = 0.0
    logits[3, 2] = 0.0
    logits[3, 3] = 0.0
    # pos 4 predicts token 2: confident
    logits[4, 2] = 10.0

    model = _StubCausalLM(logits)
    tokenizer = _StubTokenizer()

    strat = ConfidenceThinkingStrategy(
        tokenizer,
        thinking_token_id=99,
        model=model,
        tokens_per_stage=10,
        insertion_prob=1.0,
        secondary_insertion_prob=0.0,
        seed=42,
        probability_threshold=0.3,
    )
    candidates = strat._candidate_indices(
        input_ids=input_ids,
        question_length=question_length,
    )
    # Expected: positions 2 and 4 (gold token probs are ~0.2 < 0.3).
    assert candidates == [2, 4]


def test_confidence_strategy_apply_inserts_thinking_tokens():
    """End-to-end apply: selected candidates get <thinking> ids inserted."""
    vocab_size = 5
    input_ids = [0, 1, 2, 3, 4, 2]
    question_length = 2
    seq_len = len(input_ids)

    logits = torch.full((seq_len, vocab_size), -1e9, dtype=torch.float32)
    logits[0, 1] = 10.0
    # low confidence at pos 1 predicting token 2
    for t in range(vocab_size):
        logits[1, t] = 0.0
    logits[2, 3] = 10.0
    logits[3, 4] = 10.0
    logits[4, 2] = 10.0

    model = _StubCausalLM(logits)
    tokenizer = _StubTokenizer()
    strat = ConfidenceThinkingStrategy(
        tokenizer,
        thinking_token_id=99,
        model=model,
        tokens_per_stage=10,
        insertion_prob=1.0,
        secondary_insertion_prob=0.0,
        seed=42,
        probability_threshold=0.3,
    )
    updated, inserted = strat.apply(
        input_ids=input_ids,
        question_length=question_length,
        sample_idx=0,
        scheduled_stage=1,
    )
    # One low-confidence position (2) -> one inserted <thinking> (id=99).
    assert 99 in updated
    assert len(inserted) == 1
    # The inserted <thinking> should appear at position 2 (after the 2 question tokens).
    assert inserted == [2]


def test_confidence_threshold_matches_clone_formula():
    """Reimplement the clone's gather-based probability computation and confirm
    the harness strategy identifies the same candidate indices.
    """
    import torch.nn.functional as F

    vocab_size = 7
    input_ids = [1, 2, 3, 4, 5, 6, 3, 2]  # question_len=3, trace len=5
    question_length = 3
    threshold = 0.25

    torch.manual_seed(11)
    logits = torch.randn(len(input_ids), vocab_size)

    # --- Clone-equivalent reference computation ---
    # Clone: log_probs over logits[idx, : len(ids)], gather next_tokens,
    # token_probs[pos - 1] < threshold for pos in [question_len, len).
    log_probs = F.log_softmax(logits[: len(input_ids)].float(), dim=-1)
    next_tokens = torch.tensor(input_ids[1:], dtype=torch.long)
    token_log_probs = log_probs[:-1].gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
    token_probs = token_log_probs.exp().tolist()
    expected: list[int] = []
    for pos in range(question_length, len(input_ids)):
        if pos - 1 < len(token_probs) and token_probs[pos - 1] < threshold:
            expected.append(pos)

    # --- Harness strategy ---
    model = _StubCausalLM(logits)
    tokenizer = _StubTokenizer()
    strat = ConfidenceThinkingStrategy(
        tokenizer,
        thinking_token_id=0,
        model=model,
        tokens_per_stage=100,
        insertion_prob=1.0,
        seed=0,
        probability_threshold=threshold,
    )
    candidates = strat._candidate_indices(
        input_ids=input_ids,
        question_length=question_length,
    )
    assert candidates == expected


# ---------------------------------------------------------------------------
# Strategy wiring
# ---------------------------------------------------------------------------


def test_build_thinking_strategy_returns_none_for_explicit_stage():
    from latent_harness.training.lt_tuning import build_thinking_strategy

    cfg = LTTuningConfig()
    stage = cfg.stages()[0]  # explicit
    strategy = build_thinking_strategy(
        stage=stage,
        lt_config=cfg,
        tokenizer=_StubTokenizer(),
        thinking_token_id=0,
        model=None,
    )
    assert strategy is None


def test_build_thinking_strategy_random_works_without_model():
    from latent_harness.training.lt_tuning import build_thinking_strategy

    cfg = LTTuningConfig(thinking_strategy="random")
    stage = cfg.stages()[1]  # hidden_state stage
    strategy = build_thinking_strategy(
        stage=stage,
        lt_config=cfg,
        tokenizer=_StubTokenizer(),
        thinking_token_id=5,
        model=None,
    )
    assert isinstance(strategy, RandomThinkingStrategy)
