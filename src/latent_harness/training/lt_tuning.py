"""LT-Tuning (Latent-Thoughts Tuning) method implementation.

Port of the Latent-Thoughts-Tuning paper / reference repo into the harness.

Three training stages over a curriculum:

- Stage 0 ("explicit" / common): explicit CoT warmup, no ``<thinking>`` tokens
  are inserted; the model learns standard CoT-SFT under our latent data
  contract (encoder/decoder + reference path).
- Stage 1 ("hidden_state"): a single ``<thinking>`` token is inserted into the
  reasoning trace and at that position the model replaces the token embedding
  with the previous position's hidden state (the CODI-style latent).
- Stage 2 ("soft_fusion" / CPF): the replacement embedding is a convex blend
  of the previous hidden state and a top-p weighted expected embedding derived
  from the previous-position logits:

      e_pred = sum_w top_p(softmax(l_{t-1} / T)) * E(w)
      z_t    = alpha * h_{t-1, I} + (1 - alpha) * e_pred

Stage 2 also uses a confidence-triggered thinking-insertion strategy that
inserts a ``<thinking>`` token before every trace position where the teacher
probability of the gold next token falls below a threshold (clone-of-reference
``ConfidenceThinkingStrategy``).

The CPF forward pass and the 3-stage curriculum are both first-class harness
objects here; ``run_lt_tuning_from_config`` is the entrypoint wired into the
method registry.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig
from latent_harness.core.runtime import LatentReasoningRuntime
from latent_harness.training.config import TrainingDataConfig

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Default curriculum values used if a config is silent. These match the
# reference repo's ``example_config.yaml`` except for per-stage epochs which
# come from paper Table 7 interpolated to 4B ({1, 1, 3} biased toward stage 2).
_DEFAULT_STAGE_NAMES = ("stage0-cot", "stage1-hidden", "stage2-fusion")
_DEFAULT_STAGE_MODES = ("explicit", "hidden_state", "soft_fusion")
_DEFAULT_STAGE_EPOCHS = (1, 1, 3)
_DEFAULT_STAGE_LRS = (2e-5, 2e-5, 2e-5)


# ---------------------------------------------------------------------------
# Curriculum configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageSpec:
    """Configuration for a single curriculum stage."""

    name: str
    mode: str  # one of {"explicit", "hidden_state", "soft_fusion"}
    epochs: int
    learning_rate: float
    fusion_alpha: float
    thinking_insertion_prob: float
    thinking_secondary_insertion_prob: float
    reinforce_prob_threshold: float
    tokens_per_stage: int

    def is_latent(self) -> bool:
        return self.mode in {"hidden_state", "soft_fusion"}


@dataclass(slots=True)
class LTTuningConfig:
    """Top-level config for LT-Tuning curriculum + CPF.

    Values sourced from ``Latent-Thoughts-Tuning/configs/example_config.yaml``
    except where explicitly noted.
    """

    thinking_token: str = "<thinking>"
    thinking_strategy: str = "confidence"  # "confidence" | "arithmetic" | "random"
    # Layer index to pull the "previous hidden state" from. -1 = last layer
    # (matches clone's ``thinking_hidden_state_layer: -1``). The spec allows
    # "penultimate unless clone specifies otherwise"; the clone specifies -1
    # so we use that.
    hidden_state_layer_index: int = -1
    # CPF hyperparameters
    fusion_top_p: float = 0.9
    fusion_temperature: float = 1.0
    # Stage specs
    stage_names: tuple[str, ...] = _DEFAULT_STAGE_NAMES
    stage_modes: tuple[str, ...] = _DEFAULT_STAGE_MODES
    stage_epochs: tuple[int, ...] = _DEFAULT_STAGE_EPOCHS
    stage_learning_rates: tuple[float, ...] = _DEFAULT_STAGE_LRS
    stage_fusion_alphas: tuple[float, ...] = (0.5, 0.5, 0.6)
    stage_insertion_probs: tuple[float, ...] = (0.0, 0.85, 0.95)
    stage_secondary_insertion_probs: tuple[float, ...] = (0.0, 0.15, 0.2)
    stage_reinforce_prob_thresholds: tuple[float, ...] = (0.0, 0.3, 0.2)
    stage_tokens_per_stage: tuple[int, ...] = (0, 1, 1)
    reinforce_max_eval_length: int = 2048
    # Each stage in the clone paper optionally re-tokenises its dataset with a
    # stage-specific insertion probability. When True we regenerate the dataset
    # between stages (costly). When False we fall back to the stage-2 dataset
    # for stages 1 and 2 (cheap; stage-0 always has no insertions).
    regenerate_dataset_per_stage: bool = True

    def stages(self) -> list[StageSpec]:
        n = len(self.stage_modes)
        if not (
            len(self.stage_names) == n
            and len(self.stage_epochs) == n
            and len(self.stage_learning_rates) == n
            and len(self.stage_fusion_alphas) == n
            and len(self.stage_insertion_probs) == n
            and len(self.stage_secondary_insertion_probs) == n
            and len(self.stage_reinforce_prob_thresholds) == n
            and len(self.stage_tokens_per_stage) == n
        ):
            raise ValueError("Inconsistent LT-Tuning stage tuple lengths in config")
        out: list[StageSpec] = []
        for i, mode in enumerate(self.stage_modes):
            out.append(
                StageSpec(
                    name=self.stage_names[i],
                    mode=mode,
                    epochs=int(self.stage_epochs[i]),
                    learning_rate=float(self.stage_learning_rates[i]),
                    fusion_alpha=float(self.stage_fusion_alphas[i]),
                    thinking_insertion_prob=float(self.stage_insertion_probs[i]),
                    thinking_secondary_insertion_prob=float(
                        self.stage_secondary_insertion_probs[i]
                    ),
                    reinforce_prob_threshold=float(self.stage_reinforce_prob_thresholds[i]),
                    tokens_per_stage=int(self.stage_tokens_per_stage[i]),
                )
            )
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LTTuningConfig":
        if not payload:
            return cls()
        kwargs: dict[str, Any] = {}
        for field_name in (
            "thinking_token",
            "thinking_strategy",
            "hidden_state_layer_index",
            "fusion_top_p",
            "fusion_temperature",
            "reinforce_max_eval_length",
            "regenerate_dataset_per_stage",
        ):
            if field_name in payload:
                kwargs[field_name] = payload[field_name]
        for field_name in (
            "stage_names",
            "stage_modes",
            "stage_epochs",
            "stage_learning_rates",
            "stage_fusion_alphas",
            "stage_insertion_probs",
            "stage_secondary_insertion_probs",
            "stage_reinforce_prob_thresholds",
            "stage_tokens_per_stage",
        ):
            if field_name in payload:
                kwargs[field_name] = tuple(payload[field_name])
        return cls(**kwargs)


class CurriculumScheduler:
    """Iterate stages of an LT-Tuning curriculum.

    The scheduler owns no training state; it is a pure iterator over
    :class:`StageSpec` instances plus helpers to tell callers which stage a
    given global-step index would fall in (useful for analyses).
    """

    def __init__(self, config: LTTuningConfig) -> None:
        self.config = config
        self.stages_list = config.stages()

    def __iter__(self):
        return iter(self.stages_list)

    def __len__(self) -> int:
        return len(self.stages_list)

    def stages(self) -> list[StageSpec]:
        return list(self.stages_list)

    def stage_boundaries(self, steps_per_epoch: int) -> list[tuple[str, int, int]]:
        """Return ``[(name, start_step, end_step_exclusive), ...]``.

        Useful for callers that want to inspect the cumulative step schedule
        (tests, loggers) without actually running training.
        """
        boundaries: list[tuple[str, int, int]] = []
        cursor = 0
        for stage in self.stages_list:
            span = max(stage.epochs, 0) * steps_per_epoch
            boundaries.append((stage.name, cursor, cursor + span))
            cursor += span
        return boundaries

    def stage_at_step(
        self, step: int, steps_per_epoch: int
    ) -> StageSpec | None:
        for name, start, end in self.stage_boundaries(steps_per_epoch):
            if start <= step < end:
                return next(s for s in self.stages_list if s.name == name)
        return None


# ---------------------------------------------------------------------------
# Thinking-token insertion strategies (stage-2 dataset processing)
# ---------------------------------------------------------------------------


class _BaseThinkingStrategy:
    """Base class for ``<thinking>`` insertion strategies.

    Matches the clone's abstract ``ThinkingTokenStrategy`` contract closely
    enough that its logic is portable; we consume raw ``input_ids`` +
    ``question_length`` tuples instead of the clone's dict-of-lists so the
    strategies can plug into our harness-style tokenization pipeline.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        *,
        tokens_per_stage: int | float = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.thinking_token_id = thinking_token_id
        self.tokens_per_stage = tokens_per_stage
        self.insertion_prob = insertion_prob
        self.secondary_insertion_prob = secondary_insertion_prob
        self._seed = seed
        self.requires_model_pass = False

    def _candidate_indices(
        self,
        *,
        input_ids: list[int],
        question_length: int,
    ) -> list[int]:
        raise NotImplementedError

    def _rng_for(self, sample_idx: int, scheduled_stage: int) -> random.Random:
        if self._seed is None:
            return random.Random()
        derived = (1000003 * sample_idx + 9176 * scheduled_stage + 23 + self._seed) & 0xFFFFFFFF
        return random.Random(derived)

    def _select_indices(
        self,
        candidates: list[int],
        scheduled_stage: int,
        rng: random.Random,
    ) -> list[int]:
        if scheduled_stage <= 0 or not candidates:
            return []
        if self.tokens_per_stage < 1:
            target = int(len(candidates) * self.tokens_per_stage)
        else:
            target = int(self.tokens_per_stage * scheduled_stage)
        if target <= 0:
            return []
        target = min(target, len(candidates))
        if target < len(candidates):
            return sorted(rng.sample(candidates, target))
        return sorted(candidates[:target])

    def apply(
        self,
        *,
        input_ids: list[int],
        question_length: int,
        sample_idx: int,
        scheduled_stage: int = 1,
        candidate_indices: list[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Return ``(updated_input_ids, inserted_positions)``.

        Mirrors the clone's ``ThinkingTokenStrategy.apply`` control flow.
        """
        if len(input_ids) == 0 or question_length >= len(input_ids):
            return list(input_ids), []
        rng = self._rng_for(sample_idx, scheduled_stage)
        if candidate_indices is None:
            candidate_indices = self._candidate_indices(
                input_ids=list(input_ids),
                question_length=question_length,
            )
        selected = self._select_indices(candidate_indices, scheduled_stage, rng)
        if not selected:
            return list(input_ids), []

        updated = list(input_ids)
        inserted: list[int] = []
        offset = 0
        for idx in selected:
            if rng.random() > self.insertion_prob:
                continue
            insert_at = max(idx + offset, 0)
            if insert_at < question_length:
                # Match clone: warn but continue. Clone prints a warning; we
                # skip to avoid ever inserting into the question prompt.
                continue
            updated.insert(insert_at, self.thinking_token_id)
            inserted.append(insert_at)
            offset += 1
            if self.secondary_insertion_prob > 0.0 and rng.random() < self.secondary_insertion_prob:
                insert_at += 1
                updated.insert(insert_at, self.thinking_token_id)
                inserted.append(insert_at)
                offset += 1
        return updated, inserted


class RandomThinkingStrategy(_BaseThinkingStrategy):
    def _candidate_indices(
        self, *, input_ids: list[int], question_length: int
    ) -> list[int]:
        return list(range(max(question_length, 1), len(input_ids)))


class ArithmeticThinkingStrategy(_BaseThinkingStrategy):
    _OPERATORS = set("+-=*/%()")

    def __init__(self, *args, operator_regex: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import re

        self._operator_regex = re.compile(operator_regex) if operator_regex else None

    def _is_numeric_or_operator(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if self._operator_regex and self._operator_regex.search(stripped):
            return True
        if any(char in self._OPERATORS for char in stripped):
            return True
        return stripped.replace(".", "", 1).isdigit()

    def _candidate_indices(
        self, *, input_ids: list[int], question_length: int
    ) -> list[int]:
        candidates: list[int] = []
        for idx in range(question_length, len(input_ids)):
            token_text = self.tokenizer.decode([input_ids[idx]])
            if self._is_numeric_or_operator(token_text):
                candidates.append(idx)
        return candidates


class ConfidenceThinkingStrategy(_BaseThinkingStrategy):
    """Insert ``<thinking>`` before low-confidence tokens per the teacher model.

    Direct port of the clone's ``ConfidenceThinkingStrategy``. Given the
    teacher's next-token probability at each trace position, return indices
    where ``p(gold_{t+1}) < threshold`` as candidate insertion points.

    Unlike the clone we run this as a single-sample op (the dataset builder
    calls it per example) so no batched padding is required. This sacrifices
    some throughput but keeps the data pipeline straightforward.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        *,
        model: nn.Module | None,
        tokens_per_stage: int | float = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: int | None = None,
        probability_threshold: float = 0.3,
        max_sequence_length: int = 2048,
    ) -> None:
        super().__init__(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_insertion_prob,
            seed=seed,
        )
        if model is None:
            raise ValueError(
                "ConfidenceThinkingStrategy requires a model for probability estimation."
            )
        self.model = model
        self.model_device = next(model.parameters()).device
        self.threshold = float(max(min(probability_threshold, 1.0), 0.0))
        self.max_sequence_length = max_sequence_length
        self.requires_model_pass = True

    def _candidate_indices(
        self, *, input_ids: list[int], question_length: int
    ) -> list[int]:
        # Truncate to avoid OOM on pathological inputs.
        input_ids = input_ids[: self.max_sequence_length]
        if len(input_ids) <= question_length + 1:
            return []
        device = self.model_device
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)
        with torch.no_grad():
            training_was = self.model.training
            self.model.eval()
            outputs = self.model(input_ids=ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq, vocab]
            if training_was:
                self.model.train()
        log_probs = F.log_softmax(logits[0, : len(input_ids) - 1, :].float(), dim=-1)
        next_tokens = ids[0, 1:]
        token_log_probs = log_probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
        token_probs = token_log_probs.exp().tolist()
        candidates: list[int] = []
        for pos in range(question_length, len(input_ids)):
            if pos - 1 < 0 or pos - 1 >= len(token_probs):
                continue
            if token_probs[pos - 1] < self.threshold:
                candidates.append(pos)
        return candidates


def build_thinking_strategy(
    *,
    stage: StageSpec,
    lt_config: LTTuningConfig,
    tokenizer: PreTrainedTokenizerBase,
    thinking_token_id: int,
    model: nn.Module | None = None,
    seed: int | None = None,
) -> _BaseThinkingStrategy | None:
    """Return a thinking-token strategy for ``stage``, or ``None`` for explicit."""
    if stage.mode == "explicit":
        return None
    strategy_name = lt_config.thinking_strategy.lower()
    if strategy_name == "random":
        return RandomThinkingStrategy(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=stage.tokens_per_stage,
            insertion_prob=stage.thinking_insertion_prob,
            secondary_insertion_prob=stage.thinking_secondary_insertion_prob,
            seed=seed,
        )
    if strategy_name == "arithmetic":
        return ArithmeticThinkingStrategy(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=stage.tokens_per_stage,
            insertion_prob=stage.thinking_insertion_prob,
            secondary_insertion_prob=stage.thinking_secondary_insertion_prob,
            seed=seed,
        )
    if strategy_name in {"confidence", "reinforce", "policy"}:
        return ConfidenceThinkingStrategy(
            tokenizer,
            thinking_token_id,
            model=model,
            tokens_per_stage=stage.tokens_per_stage,
            insertion_prob=stage.thinking_insertion_prob,
            secondary_insertion_prob=stage.thinking_secondary_insertion_prob,
            seed=seed,
            probability_threshold=stage.reinforce_prob_threshold,
            max_sequence_length=lt_config.reinforce_max_eval_length,
        )
    raise ValueError(f"Unsupported thinking_strategy: {strategy_name!r}")


# ---------------------------------------------------------------------------
# CPF forward pass
# ---------------------------------------------------------------------------


def cpf_fuse(
    *,
    hidden_state: torch.Tensor,
    logits: torch.Tensor,
    embedding: nn.Module,
    fusion_alpha: float,
    fusion_top_p: float,
    fusion_temperature: float,
    thinking_token_id: int,
) -> torch.Tensor:
    """Context-Prediction Fusion: blend hidden state with expected embedding.

    Pure-function version of the clone's ``_soft_fusion_embedding`` so we can
    unit-test numerical correctness independently of the runtime.

    Args:
        hidden_state: ``[..., hidden_size]``.
        logits: ``[..., vocab_size]``.
        embedding: ``nn.Embedding`` providing ``weight`` of shape
            ``[vocab_size, hidden_size]``.
        fusion_alpha: weight on ``hidden_state``; ``(1 - alpha)`` on the
            expected embedding.
        fusion_top_p: keep smallest set of tokens whose cumulative probability
            exceeds this threshold; renormalize within that set.
        fusion_temperature: softmax temperature applied to logits.
        thinking_token_id: mask this token id out of the distribution to
            prevent the model from soft-assigning its own thinking token.

    Returns:
        Fused embedding of shape ``[..., hidden_size]``.
    """
    scaled = logits / max(fusion_temperature, 1e-6)
    masked = scaled.clone()
    masked[..., thinking_token_id] = float("-inf")
    probs = torch.softmax(masked, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # Keep tokens whose cumulative probability is <= top_p; always keep the
    # top-1 even if it already exceeds top_p (mirrors clone's fallback).
    keep = cum <= fusion_top_p
    keep[..., 0] = True
    filtered_sorted = torch.where(
        keep, sorted_probs, torch.zeros_like(sorted_probs)
    )
    denom = filtered_sorted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    filtered_sorted = filtered_sorted / denom
    # Scatter back to vocab order
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, filtered_sorted)
    # Expected embedding: filtered_probs @ E
    # Support arbitrary leading dims
    expected = torch.matmul(filtered.to(embedding.weight.dtype), embedding.weight)
    alpha = float(fusion_alpha)
    return alpha * hidden_state + (1.0 - alpha) * expected


class LTTuningRuntime(LatentReasoningRuntime):
    """LatentReasoningRuntime extended with CPF + stage-mode forward.

    The base ``LatentReasoningRuntime`` handles LoRA, V2 step-boundary detach,
    projection head, teacher/student distillation, etc. This subclass adds:

    - ``stage_mode`` state (``explicit`` / ``hidden_state`` / ``soft_fusion``)
      that the trainer flips at stage boundaries.
    - ``fusion_alpha`` / ``fusion_top_p`` / ``fusion_temperature`` / layer
      index / thinking_token_id buffers for CPF.
    - A stage-aware ``iterate_latent_steps`` that applies CPF fusion when
      ``stage_mode == 'soft_fusion'``.

    Invariants:
    - Stage 0 (explicit) trains under our existing latent-distillation contract
      with ``num_latent=0`` — i.e., no latent rollout happens and the CE path
      dominates. This is the CoT warmup phase.
    - Stages 1 and 2 run ``num_latent`` latent rollout steps; at each step
      stage 1 uses the raw hidden state (base behaviour), stage 2 blends it
      with the CPF expected embedding.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: LatentRuntimeConfig,
        *,
        train_mode: bool,
        lt_config: LTTuningConfig | None = None,
        thinking_token_id: int | None = None,
    ) -> None:
        super().__init__(model_config, runtime_config, train_mode=train_mode)
        self.lt_config = lt_config or LTTuningConfig()
        self.stage_mode: str = "explicit"
        self._current_fusion_alpha: float = 0.5
        if thinking_token_id is not None:
            self.thinking_token_id = int(thinking_token_id)
        else:
            # Until the dataset builder registers a dedicated thinking token we
            # alias to the harness's ``bot_id`` just to have a valid index. The
            # caller is expected to overwrite this via ``set_thinking_token_id``
            # before the first stage-1 forward pass.
            self.thinking_token_id = int(self.bot_id)

    # ----- Stage control ----------------------------------------------------

    def set_stage_mode(self, mode: str, *, fusion_alpha: float | None = None) -> None:
        if mode not in {"explicit", "hidden_state", "soft_fusion"}:
            raise ValueError(f"Unknown LT-Tuning stage mode {mode!r}")
        self.stage_mode = mode
        if fusion_alpha is not None:
            self._current_fusion_alpha = float(fusion_alpha)
        logger.info(
            "LT-Tuning stage_mode=%s fusion_alpha=%.3f",
            self.stage_mode,
            self._current_fusion_alpha,
        )

    def set_thinking_token_id(self, token_id: int) -> None:
        self.thinking_token_id = int(token_id)

    # ----- CPF application --------------------------------------------------

    def _cpf_apply(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply CPF fusion to a latent hidden state, using the latest logits."""
        return cpf_fuse(
            hidden_state=hidden,
            logits=logits,
            embedding=self.get_input_embedding_layer(),
            fusion_alpha=self._current_fusion_alpha,
            fusion_top_p=self.lt_config.fusion_top_p,
            fusion_temperature=self.lt_config.fusion_temperature,
            thinking_token_id=self.thinking_token_id,
        )

    def _select_hidden_state(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pick ``hidden_states[hidden_state_layer_index]`` safely.

        Indexing mirrors the clone: negative indices count from the end. Per
        spec, layer index == -1 = last layer (clone default). The spec also
        mentions "penultimate unless clone specifies otherwise"; the clone
        specifies -1 so we honour that.
        """
        idx = int(self.lt_config.hidden_state_layer_index)
        n = len(hidden_states)
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise IndexError(
                f"hidden_state_layer_index {self.lt_config.hidden_state_layer_index} "
                f"out of range for {n} hidden_states"
            )
        return hidden_states[idx]

    # ----- Forward path overrides -------------------------------------------

    def iterate_latent_steps(
        self,
        past_key_values: Any,
        latent: torch.Tensor,
        num_steps: int,
    ) -> tuple[Any, torch.Tensor]:
        """Override inference-time latent iteration to apply CPF per step."""
        if self.stage_mode != "soft_fusion":
            return super().iterate_latent_steps(past_key_values, latent, num_steps)
        for _ in range(num_steps):
            outputs = self.model(
                inputs_embeds=latent,
                token_type_ids=self.build_token_type_ids(inputs_embeds=latent),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden = self._select_hidden_state(outputs.hidden_states)[:, -1:, :]
            logits = outputs.logits[:, -1, :]
            # maybe_project operates on the hidden state; CPF operates on the
            # post-projection latent so the interfaces between stages stay
            # consistent.
            hidden = self.maybe_project(hidden)
            # Fuse (fusion_alpha is applied per position; broadcast across the
            # single-position latent dim).
            fused = self._cpf_apply(hidden, logits.unsqueeze(1))
            latent = fused
        return past_key_values, latent

    # We intentionally do NOT override ``forward``. The base ``forward``
    # already handles stage-0 (explicit; ``num_latent=0`` short-circuits the
    # latent loop), stage-1 (hidden_state; current behaviour), and stage-2
    # via our ``iterate_latent_steps`` override — except that the training
    # forward inlines the loop rather than calling ``iterate_latent_steps``.
    # To propagate CPF into training we override the training-time loop too.

    def forward(  # type: ignore[override]
        self,
        *,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        ref_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        ref_answer_position: torch.LongTensor,
        model_answer_position: torch.LongTensor,
        ref_attention_mask: torch.LongTensor,
        ref_labels: torch.LongTensor,
        step: int | None = None,
        step_ratio: float | None = None,
        collect_diagnostics: bool = False,
    ) -> dict[str, Any]:
        if self.stage_mode != "soft_fusion":
            return super().forward(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_attention_mask,
                ref_answer_position=ref_answer_position,
                model_answer_position=model_answer_position,
                ref_attention_mask=ref_attention_mask,
                ref_labels=ref_labels,
                step=step,
                step_ratio=step_ratio,
                collect_diagnostics=collect_diagnostics,
            )

        # Fused soft-fusion forward: replicate the base training loop but apply
        # CPF at every latent boundary.
        from dataclasses import asdict

        from latent_harness.core.runtime import (
            _apply_boundary_detach,
            _resolve_should_detach,
        )

        del step, step_ratio

        past_key_values, latent = self.encode_question(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                token_type_ids=self.build_token_type_ids(input_ids=ref_input_ids),
                output_hidden_states=True,
            )
        if self.runtime_config.ref_loss_factor > 0:
            teacher_outputs_with_grad = self.model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                token_type_ids=self.build_token_type_ids(input_ids=ref_input_ids),
                output_hidden_states=True,
            )
        else:
            teacher_outputs_with_grad = None

        student_logits = None
        distill_total = torch.tensor(0.0, device=encoder_input_ids.device)
        ce_total = torch.tensor(0.0, device=encoder_input_ids.device)

        self._ensure_finite(
            latent,
            stage="encode_question",
            tensor_name="latent",
            extra={"num_latent": self.runtime_config.num_latent},
        )

        for latent_index in range(self.runtime_config.num_latent):
            latent_outputs = self.model(
                inputs_embeds=latent,
                token_type_ids=self.build_token_type_ids(inputs_embeds=latent),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_outputs.past_key_values
            hidden = self._select_hidden_state(latent_outputs.hidden_states)[:, -1:, :]
            hidden = self.maybe_project(hidden)
            # CPF fusion uses the latest-position logits
            logits_at_step = latent_outputs.logits[:, -1, :].unsqueeze(1)
            latent = self._cpf_apply(hidden, logits_at_step)

            if latent_index < self.runtime_config.num_latent - 1:
                should_detach = _resolve_should_detach(
                    latent_index=latent_index,
                    num_latent=self.runtime_config.num_latent,
                    keep_last_k=self.runtime_config.detach_keep_last_k,
                )
                if should_detach:
                    past_key_values, latent = _apply_boundary_detach(
                        cache=past_key_values,
                        latent=latent,
                        encoder_length=encoder_input_ids.size(1),
                        runtime_config_detach_latent=self.runtime_config.detach_latent_between_steps,
                        runtime_config_detach_cache=self.runtime_config.detach_cache_between_steps,
                        detach_position_mode=self.runtime_config.detach_position_mode,
                    )
            self._ensure_finite(
                latent,
                stage="latent_rollout_cpf",
                tensor_name="latent",
                extra={"latent_index": latent_index},
            )

            if latent_index != self.runtime_config.num_latent - 1:
                continue

            decoder_embeds = self.get_input_embedding_layer()(decoder_input_ids)
            student_outputs = self.model(
                inputs_embeds=decoder_embeds,
                token_type_ids=self.build_token_type_ids(inputs_embeds=decoder_embeds),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            student_logits = student_outputs.logits
            self._ensure_finite(
                student_logits,
                stage="student_decode",
                tensor_name="student_logits",
                extra={"latent_index": latent_index},
            )

            layer_losses: list[torch.Tensor] = []
            for layer_index, (student_layer, teacher_layer) in enumerate(
                zip(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                )
            ):
                safe_ref_pos = ref_answer_position.clamp(max=teacher_layer.size(1) - 1)
                safe_model_pos = model_answer_position.clamp(max=student_layer.size(1) - 1)
                teacher_selected = teacher_layer.gather(
                    1,
                    safe_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, teacher_layer.size(-1)
                    ),
                )
                student_selected = student_layer.gather(
                    1,
                    safe_model_pos.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, student_layer.size(-1)
                    ),
                )
                loss_piece = self.distill_loss_fct(
                    student_selected.float(),
                    teacher_selected.detach().float(),
                )
                teacher_std = teacher_selected.detach().float().std(unbiased=False)
                effective_std = teacher_std.clamp_min(self.runtime_config.distill_loss_std_floor)
                if self.runtime_config.distill_loss_div_std:
                    loss_piece = loss_piece / effective_std.to(loss_piece.dtype)
                layer_losses.append(loss_piece)
            distill_total = torch.stack(layer_losses).mean() * self.runtime_config.distill_loss_factor

            shifted_logits = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))
            shifted_labels = labels[:, 1:].reshape(-1)
            ce_total = self.loss_fct(shifted_logits.float(), shifted_labels)

        if teacher_outputs_with_grad is not None:
            ref_logits = teacher_outputs_with_grad.logits
            shifted_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
            shifted_ref_labels = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = (
                self.loss_fct(shifted_ref_logits.float(), shifted_ref_labels)
                * self.runtime_config.ref_loss_factor
            )
        else:
            ref_ce_loss = torch.tensor(0.0, device=encoder_input_ids.device)

        total_loss = ce_total + distill_total + ref_ce_loss
        self._ensure_finite(
            total_loss,
            stage="loss_aggregation",
            tensor_name="total_loss",
        )
        return {
            "loss": total_loss,
            "logits": student_logits,
            "ce_loss": float(ce_total.detach().cpu()),
            "distill_loss": float(distill_total.detach().cpu()),
            "ref_ce_loss": float(ref_ce_loss.detach().cpu()),
            "diagnostics": {
                "stage_mode": 2.0,  # soft_fusion encoded as numeric for log pipes
                "fusion_alpha": float(self._current_fusion_alpha),
            },
            "config": {
                "model": asdict(self.model_config),
                "runtime": asdict(self.runtime_config),
            },
        }
