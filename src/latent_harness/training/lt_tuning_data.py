"""Stage-aware data module for LT-Tuning.

Builds the latent training records from our standard dataset pipeline and then
optionally decorates them with ``<thinking>`` tokens per the current
curriculum stage. For stage 0 ("explicit") no tokens are inserted. For stages
1 and 2 the selected thinking strategy (``arithmetic`` or ``confidence``) is
used to pick insertion indices within the reasoning-trace span of each
example.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from latent_harness.core.config import LatentRuntimeConfig
from latent_harness.training.config import TrainingDataConfig
from latent_harness.training.datasets import (
    IGNORE_INDEX,
    SupervisedLatentDataCollator,
    SupervisedLatentDataset,
    _build_formatted_splits,
    _get_answer_token_position,
    _prepend_tokens,
    _tokenize_texts,
    _trim_bos,
    _append_eos,
)
from latent_harness.training.lt_tuning import (
    LTTuningConfig,
    StageSpec,
    build_thinking_strategy,
    _BaseThinkingStrategy,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LTTuningFormattedExample:
    """Per-example data carried across stages so we don't re-format from disk."""

    question: str
    cot: str
    answer: str
    dataset_key: str
    example_index: int
    source_row_index: int
    example_hash: str


def _format_bundle(formatted: list[dict[str, Any]]) -> list[LTTuningFormattedExample]:
    return [
        LTTuningFormattedExample(
            question=str(f["question"]),
            cot=str(f["cot"]),
            answer=str(f["answer"]),
            dataset_key=str(f.get("dataset_key") or "unknown"),
            example_index=int(f.get("example_index", -1)),
            source_row_index=int(f.get("source_row_index", -1)),
            example_hash=str(f.get("example_hash") or "unknown"),
        )
        for f in formatted
    ]


class LTTuningLatentDataset(Dataset):
    """Stage-aware latent dataset.

    Produces records compatible with the harness's ``SupervisedLatentDataset``
    contract (same keys: ``encoder_input_ids``, ``decoder_input_ids``,
    ``ref_input_ids``, ``labels``, ``ref_labels``, ``ref_answer_position``,
    ``model_answer_position``, ``forensics``) but optionally inserts one or
    more ``<thinking>`` tokens into the decoder/reference trace regions based
    on the stage's thinking strategy.

    The injected token ids must already exist in the tokenizer/model embedding
    table (the caller is responsible for calling
    ``tokenizer.add_tokens([thinking_token])`` + ``model.resize_token_embeddings``
    at setup time).
    """

    def __init__(
        self,
        *,
        examples: Sequence[LTTuningFormattedExample],
        tokenizer: PreTrainedTokenizerBase,
        runtime_config: LatentRuntimeConfig,
        bot_id: int,
        eot_id: int,
        thinking_token_id: int,
        thinking_strategy: _BaseThinkingStrategy | None,
        scheduled_stage: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.thinking_token_id = thinking_token_id
        self.thinking_strategy = thinking_strategy
        self.scheduled_stage = scheduled_stage
        self.records = self._build(examples)

    def _build(self, examples: Sequence[LTTuningFormattedExample]) -> list[dict[str, Any]]:
        logger.info(
            "Building LT-Tuning latent dataset count=%d stage=%d strategy=%s",
            len(examples),
            self.scheduled_stage,
            type(self.thinking_strategy).__name__ if self.thinking_strategy else "None",
        )
        max_length = self.runtime_config.model_max_length
        tokenizer = self.tokenizer

        questions = [e.question for e in examples]
        cots = [e.cot for e in examples]
        answers = [e.answer for e in examples]
        source_ids = _tokenize_texts(questions, tokenizer, max_length=max_length)
        cot_ids = _tokenize_texts(cots, tokenizer, max_length=max_length)
        answer_ids = _tokenize_texts(answers, tokenizer, max_length=max_length)

        if not self.runtime_config.remove_eos:
            source_ids = [_append_eos(ids, tokenizer) for ids in source_ids]
            cot_ids = [_append_eos(ids, tokenizer) for ids in cot_ids]
            answer_ids = [_append_eos(ids, tokenizer) for ids in answer_ids]

        cot_ids = [_trim_bos(ids, tokenizer) for ids in cot_ids]
        answer_ids = [_trim_bos(ids, tokenizer) for ids in answer_ids]

        answer_prompt = "The answer is:"
        answer_prompt_ids = torch.tensor(
            tokenizer.encode(answer_prompt, add_special_tokens=False),
            dtype=torch.long,
        )

        records: list[dict[str, Any]] = []
        for i, (example, src, cot, ans) in enumerate(
            zip(examples, source_ids, cot_ids, answer_ids)
        ):
            teacher = torch.cat([src, cot, ans]).long()
            src_list = src.tolist()
            cot_list = cot.tolist()
            ans_list = ans.tolist()
            question_length = len(src_list)
            teacher_list = list(teacher.tolist())

            # Apply thinking strategy over the trace (cot + answer) region only.
            inserted_positions: list[int] = []
            if self.thinking_strategy is not None:
                updated_teacher, inserted_positions = self.thinking_strategy.apply(
                    input_ids=teacher_list,
                    question_length=question_length,
                    sample_idx=example.example_index if example.example_index >= 0 else i,
                    scheduled_stage=self.scheduled_stage,
                )
                teacher_list = updated_teacher

            # Reconstruct trace + answer ids from the possibly-updated teacher
            # sequence. Insertion positions are absolute within teacher_list;
            # we split after question_length.
            teacher = torch.tensor(teacher_list, dtype=torch.long)
            trace_and_answer = teacher_list[question_length:]

            # Separate answer vs cot region. We rely on the original answer
            # length to locate where the answer starts in the (possibly
            # augmented) trace. Inserted tokens land within the cot region in
            # practice; we conservatively treat everything before the first
            # occurrence of the answer-prompt tokens (if any) as cot.
            cot_augmented_ids = torch.tensor(
                trace_and_answer[: max(len(trace_and_answer) - len(ans_list), 0)],
                dtype=torch.long,
            )
            ans_augmented_ids = torch.tensor(
                trace_and_answer[max(len(trace_and_answer) - len(ans_list), 0):],
                dtype=torch.long,
            )

            teacher_label = teacher.clone()
            teacher_label[: src.numel()] = IGNORE_INDEX
            # Labels at thinking-token positions stay as the thinking id so the
            # model still sees them in the loss (matches clone behavior).

            encoder = _prepend_tokens(src, [self.bot_id])
            if self.runtime_config.remove_eos:
                decoder = _prepend_tokens(ans_augmented_ids, [self.eot_id])
            else:
                decoder = _prepend_tokens(
                    ans_augmented_ids, [self.eot_id, tokenizer.eos_token_id]
                )

            answer_text = tokenizer.decode(ans_augmented_ids, skip_special_tokens=False)
            if answer_prompt_ids.numel() > 0 and not answer_text.startswith(answer_prompt):
                ref_pos = max(teacher.numel() - ans_augmented_ids.numel(), 0)
                model_pos = 1
            else:
                ref_pos = _get_answer_token_position(teacher, answer_prompt_ids)
                model_pos = _get_answer_token_position(decoder, answer_prompt_ids)

            records.append(
                {
                    "encoder_input_ids": encoder,
                    "decoder_input_ids": decoder,
                    "ref_input_ids": teacher,
                    "labels": decoder.clone(),
                    "ref_labels": teacher_label,
                    "ref_answer_position": torch.tensor(ref_pos, dtype=torch.long),
                    "model_answer_position": torch.tensor(model_pos, dtype=torch.long),
                    "forensics": {
                        "dataset_key": example.dataset_key,
                        "example_index": example.example_index,
                        "source_row_index": example.source_row_index,
                        "example_hash": example.example_hash,
                        "encoder_length": int(encoder.numel()),
                        "decoder_length": int(decoder.numel()),
                        "ref_length": int(teacher.numel()),
                        "ref_answer_position": int(ref_pos),
                        "model_answer_position": int(model_pos),
                        "num_thinking_tokens": int(len(inserted_positions)),
                        "thinking_positions": inserted_positions,
                    },
                }
            )
        logger.info("Built %d LT-Tuning latent records", len(records))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collect_lt_tuning_examples(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
) -> tuple[list[LTTuningFormattedExample], list[LTTuningFormattedExample]]:
    """Collect formatted (question, cot, answer) examples for all stages."""
    train_raw, eval_raw = _build_formatted_splits(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    return _format_bundle(train_raw), _format_bundle(eval_raw)


def build_lt_tuning_dataset_for_stage(
    *,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[LTTuningFormattedExample],
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
    thinking_token_id: int,
    stage: StageSpec,
    lt_config: LTTuningConfig,
    model_for_confidence: Any = None,
    seed: int | None = None,
    scheduled_stage_index: int = 0,
) -> LTTuningLatentDataset:
    """Build a stage-specific latent dataset.

    Stages are indexed starting at 0 for explicit CoT warmup; for the
    thinking-token insertion logic ``scheduled_stage = stage_index`` (matches
    the clone's convention where stage 0 inserts zero tokens, stage k inserts
    ``tokens_per_stage * k``).
    """
    strategy = build_thinking_strategy(
        stage=stage,
        lt_config=lt_config,
        tokenizer=tokenizer,
        thinking_token_id=thinking_token_id,
        model=model_for_confidence,
        seed=seed,
    )
    return LTTuningLatentDataset(
        examples=examples,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        bot_id=bot_id,
        eot_id=eot_id,
        thinking_token_id=thinking_token_id,
        thinking_strategy=strategy,
        scheduled_stage=max(scheduled_stage_index, 0),
    )


def make_lt_tuning_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    """Method registry-compatible data module builder.

    Returns the stage-0 (explicit-CoT warmup) dataset — the stage orchestrator
    swaps it out for later stages via :func:`build_lt_tuning_dataset_for_stage`.
    This keeps the method-registry contract intact: any caller that just wants
    a CODI-shape dataset + collator gets sensible stage-0 defaults.
    """
    train_examples, eval_examples = collect_lt_tuning_examples(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    # For the default module, use no thinking tokens yet — stage-0 explicit CoT.
    train_dataset = SupervisedLatentDataset(
        formatted_examples=[
            {
                "question": e.question,
                "cot": e.cot,
                "answer": e.answer,
                "dataset_key": e.dataset_key,
                "example_index": e.example_index,
                "source_row_index": e.source_row_index,
                "example_hash": e.example_hash,
            }
            for e in train_examples
        ],
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        bot_id=bot_id,
        eot_id=eot_id,
    )
    eval_dataset = None
    if eval_examples:
        eval_dataset = SupervisedLatentDataset(
            formatted_examples=[
                {
                    "question": e.question,
                    "cot": e.cot,
                    "answer": e.answer,
                    "dataset_key": e.dataset_key,
                    "example_index": e.example_index,
                    "source_row_index": e.source_row_index,
                    "example_hash": e.example_hash,
                }
                for e in eval_examples
            ],
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            bot_id=bot_id,
            eot_id=eot_id,
        )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": SupervisedLatentDataCollator(tokenizer=tokenizer),
        # Extras consumed by the orchestrator; trainers ignore unknown keys.
        "_lt_tuning_train_examples": train_examples,
        "_lt_tuning_eval_examples": eval_examples,
    }
