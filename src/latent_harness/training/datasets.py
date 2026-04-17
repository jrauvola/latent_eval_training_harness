from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from latent_harness.core.io import ensure_dir
from latent_harness.core.config import LatentRuntimeConfig
from latent_harness.data.dataset_registry import load_dataset_registry
from latent_harness.training.config import TrainingDataConfig

IGNORE_INDEX = -100
DATASET_PROGRESS_LOG_EVERY = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSource:
    key: str
    formatter: str
    path: str
    subset: str | None = None
    split: str = "train"
    streaming: bool = False
    fallbacks: tuple[tuple[str, str | None, str], ...] = ()


LEGACY_DATASET_SOURCES: dict[str, DatasetSource] = {
    "gsm8k_aug": DatasetSource(
        key="gsm8k_aug",
        formatter="gsm8k_aug",
        path="zen-E/GSM8k-Aug",
    ),
    "gsm8k_aug_nl": DatasetSource(
        key="gsm8k_aug_nl",
        formatter="gsm8k_aug_nl",
        path="zen-E/GSM8k-Aug-NL",
    ),
    "commonsense_cot": DatasetSource(
        key="commonsense_cot",
        formatter="commonsense_cot",
        path="zen-E/CommonsenseQA-GPT4omini",
    ),
    "strategyqa_cot": DatasetSource(
        key="strategyqa_cot",
        formatter="strategyqa_cot",
        path="zen-E/StrategyQA_CoT_GPT4o",
    ),
    "prontoqa": DatasetSource(
        key="prontoqa",
        formatter="prontoqa",
        path="longface/prontoqa-train",
        streaming=True,
        fallbacks=(("tasksource/prontoqa", None, "train"),),
    ),
}

REGISTRY_FORMATTERS: dict[str, str] = {
    "numinamath_15": "numinamath_15",
    "openthoughts3_12m": "openthoughts3_12m",
    "natural_reasoning": "natural_reasoning",
    "proofwriter": "proofwriter",
}

REGISTRY_SOURCE_OVERRIDES: dict[str, DatasetSource] = {
    "gsm8k_aug_nl": LEGACY_DATASET_SOURCES["gsm8k_aug_nl"],
    "commonsense_cot_qingyi": LEGACY_DATASET_SOURCES["commonsense_cot"],
    "strategyqa_cot": LEGACY_DATASET_SOURCES["strategyqa_cot"],
    "prontoqa": LEGACY_DATASET_SOURCES["prontoqa"],
}


def _tokenize_texts(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> list[torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    return [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]


def _trim_bos(tokens: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    if tokens.numel() > 0 and tokenizer.bos_token_id is not None and tokens[0].item() == tokenizer.bos_token_id:
        return tokens[1:]
    return tokens


def _append_eos(tokens: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    if tokenizer.eos_token_id is None:
        return tokens
    return torch.tensor([*tokens.tolist(), tokenizer.eos_token_id], dtype=torch.long)


def _prepend_tokens(tokens: torch.Tensor, prefix: list[int]) -> torch.Tensor:
    return torch.tensor([*prefix, *tokens.tolist()], dtype=torch.long)


def _get_answer_token_position(tokens: torch.Tensor, answer_prompt_tokens: torch.Tensor) -> int:
    window = answer_prompt_tokens.numel()
    if window == 0 or tokens.numel() < window:
        return 0
    matches = (tokens.unfold(0, window, 1) == answer_prompt_tokens).all(dim=1).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return 0
    pos = int(matches[0].item() + window)
    return min(pos, tokens.numel() - 1)


def _extract_numeric_answer(raw: str) -> str | None:
    cleaned = raw.replace("####", "").strip()
    parts = cleaned.split()
    if not parts:
        return None
    terminal = parts[-1].replace(",", "")
    if terminal and (terminal[0].isdigit() or terminal[0] == "-"):
        return terminal
    return None


def _normalize_answer(raw: Any) -> str:
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "entailment"}:
        return "True"
    if lowered in {"false", "no", "contradiction"}:
        return "False"
    return text


def _load_registry_entry_map(path: str | None) -> dict[str, dict[str, Any]]:
    registry = load_dataset_registry(path)
    return {entry["id"]: entry for entry in registry.get("entries", []) if entry.get("id")}


def _resolve_registry_source(dataset_id: str, data_config: TrainingDataConfig) -> DatasetSource:
    if dataset_id in REGISTRY_SOURCE_OVERRIDES:
        return REGISTRY_SOURCE_OVERRIDES[dataset_id]
    formatter = REGISTRY_FORMATTERS.get(dataset_id)
    if formatter is None:
        raise ValueError(
            f"Unsupported registry dataset {dataset_id!r} for the current training contract. "
            "Use a near-drop-in latent dataset or add a formatter/adapter first."
        )
    entry = _load_registry_entry_map(data_config.registry_path).get(dataset_id)
    if entry is None:
        raise ValueError(f"Registry dataset {dataset_id!r} was not found in the configured dataset registry.")
    fallbacks = tuple(
        (
            fallback["path"],
            fallback.get("subset"),
            fallback.get("smoke_split", "train").split("[", 1)[0],
        )
        for fallback in entry.get("fallbacks", [])
        if fallback.get("path")
    )
    return DatasetSource(
        key=dataset_id,
        formatter=formatter,
        path=entry["path"],
        subset=entry.get("subset"),
        split="train",
        streaming=dataset_id in {"numinamath_15", "openthoughts3_12m", "natural_reasoning"},
        fallbacks=fallbacks,
    )


def _resolve_dataset_sources(data_config: TrainingDataConfig) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for dataset_name in data_config.dataset_names:
        try:
            sources.append(LEGACY_DATASET_SOURCES[dataset_name])
        except KeyError as exc:
            known = ", ".join(sorted(LEGACY_DATASET_SOURCES))
            raise ValueError(f"Unsupported training dataset {dataset_name!r}. Known legacy datasets: {known}") from exc
    for dataset_id in data_config.registry_ids:
        sources.append(_resolve_registry_source(dataset_id, data_config))
    if not sources:
        raise ValueError("Training config must specify at least one dataset via data.dataset_names or data.registry_ids.")
    return sources


def _snapshot_file_path(
    *,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    sources: Sequence[DatasetSource],
) -> Path | None:
    if not data_config.snapshot_dir:
        return None
    snapshot_root = ensure_dir(data_config.snapshot_dir)
    signature_payload = {
        "dataset_names": data_config.dataset_names,
        "registry_ids": data_config.registry_ids,
        "registry_path": data_config.registry_path,
        "max_samples": data_config.max_samples,
        "max_samples_per_dataset": data_config.max_samples_per_dataset,
        "include_last_cot": data_config.include_last_cot,
        "answer_only": data_config.answer_only,
        "max_token_num": data_config.max_token_num,
        "validation_split_ratio": data_config.validation_split_ratio,
        "validation_max_samples": data_config.validation_max_samples,
        "sources": [
            {
                "key": source.key,
                "path": source.path,
                "subset": source.subset,
                "split": source.split,
                "streaming": source.streaming,
                "fallbacks": list(source.fallbacks),
            }
            for source in sources
        ],
    }
    digest = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return snapshot_root / f"formatted_examples_{digest}.jsonl"


def _load_formatted_examples_snapshot(snapshot_path: Path) -> list[dict[str, Any]]:
    logger.info("Loading local training snapshot from %s", snapshot_path)
    examples: list[dict[str, Any]] = []
    with snapshot_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            record = json.loads(line)
            question = str(record["question"])
            cot = str(record["cot"])
            answer = str(record["answer"])
            example_hash = str(
                record.get("example_hash")
                or hashlib.sha256(
                    json.dumps(
                        {
                            "question": question,
                            "cot": cot,
                            "answer": answer,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:16]
            )
            examples.append(
                {
                    "question": question,
                    "cot": cot,
                    "answer": answer,
                    "dataset_key": str(record.get("dataset_key") or "snapshot"),
                    "example_index": int(record.get("example_index", line_index)),
                    "source_row_index": int(record.get("source_row_index", line_index)),
                    "example_hash": example_hash,
                }
            )
    logger.info("Loaded %d formatted examples from local snapshot", len(examples))
    return examples


def _write_formatted_examples_snapshot(
    snapshot_path: Path,
    formatted_examples: Sequence[dict[str, Any]],
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d formatted examples to local snapshot %s", len(formatted_examples), snapshot_path)
    with snapshot_path.open("w", encoding="utf-8") as handle:
        for record in formatted_examples:
            handle.write(
                json.dumps(
                    {
                        "question": record["question"],
                        "cot": record["cot"],
                        "answer": record["answer"],
                        "dataset_key": record["dataset_key"],
                        "example_index": record["example_index"],
                        "source_row_index": record["source_row_index"],
                        "example_hash": record["example_hash"],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _iter_dataset_rows(source: DatasetSource, cache_dir: str, *, prefer_local_cache: bool) -> Iterator[dict[str, Any]]:
    attempts: list[tuple[str, str | None, str]] = [(source.path, source.subset, source.split), *list(source.fallbacks)]
    last_error: Exception | None = None
    use_streaming = source.streaming and not prefer_local_cache
    for path, subset, split in attempts:
        logger.info(
            "Loading dataset source key=%s path=%s subset=%s split=%s streaming=%s",
            source.key,
            path,
            subset,
            split,
            use_streaming,
        )
        kwargs: dict[str, Any] = {
            "path": path,
            "split": split,
            "cache_dir": cache_dir,
        }
        if subset is not None:
            kwargs["name"] = subset
        if use_streaming:
            kwargs["streaming"] = True
        try:
            dataset = load_dataset(**kwargs)
            for row in dataset:
                yield row
            return
        except Exception as exc:  # noqa: BLE001 - preserve fallback behavior for HF mirrors
            logger.warning(
                "Dataset load attempt failed for key=%s path=%s subset=%s split=%s: %s",
                source.key,
                path,
                subset,
                split,
                exc,
            )
            last_error = exc
    raise RuntimeError(f"Failed to load dataset {source.key!r}: {last_error}")


def _extract_openthoughts_trace_and_answer(conversations: Any) -> tuple[str, str] | None:
    if not isinstance(conversations, list):
        return None
    question = ""
    assistant = ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from") or message.get("role") or "").lower()
        value = str(message.get("value") or message.get("content") or "").strip()
        if not value:
            continue
        if not question and role in {"human", "user"}:
            question = value
        elif not assistant and role in {"gpt", "assistant"}:
            assistant = value
    if not question or not assistant or "</think>" not in assistant:
        return None
    cot, answer = assistant.split("</think>", 1)
    cot = cot.replace("<think>", "", 1).strip()
    answer = answer.strip()
    if not cot or not answer:
        return None
    return question, cot, answer


def _format_training_example(
    formatter: str,
    row: dict[str, Any],
    *,
    include_last_cot: bool,
    answer_only: bool,
) -> tuple[str, str, str] | None:
    question = str(row.get("question") or row.get("input") or "").strip()
    if formatter in {"gsm8k_aug", "gsm8k_aug_nl"}:
        if not question:
            return None
        cot = str(row.get("cot") or "").strip()
        answer = _extract_numeric_answer(str(row.get("answer") or ""))
        if answer is None:
            return None
        if not include_last_cot and cot:
            if formatter == "gsm8k_aug_nl":
                cot_parts = cot.split(". ")
                cot = ". ".join(cot_parts[:-1]).strip()
                if cot:
                    cot = cot + "."
            else:
                cot_tokens = cot.split()
                cot = " ".join(cot_tokens[:-1]).strip()
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if formatter in {"commonsense_cot", "strategyqa_cot"}:
        if not question:
            return None
        cot = str(row.get("cot") or "").strip()
        answer = _normalize_answer(row.get("answer") or "")
        if not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if formatter == "prontoqa":
        prompt = str(row.get("prompt") or "").strip()
        if prompt:
            question_text = prompt
            cot = ""
            answer = ""
            if "###Response:" in prompt:
                question_text, response = prompt.split("###Response:", 1)
                question_text = question_text.replace("###Context:", "", 1).strip()
                response = response.strip()
                if "The answer is:" in response:
                    cot, answer = response.rsplit("The answer is:", 1)
                    cot = cot.replace("Let's think step by step.", "", 1).strip()
                    answer = _normalize_answer(answer)
            if question_text and answer:
                answer_text = answer if answer_only else f"The answer is: {answer}"
                return question_text, cot, answer_text
        cot_steps = row.get("steps") or row.get("chain_of_thought") or []
        cot = "\n".join(str(step).strip() for step in cot_steps[:-1] if str(step).strip())
        answer = _normalize_answer(row.get("answer") or row.get("target") or "")
        question_text = str(row.get("question") or row.get("query") or row.get("input") or "").strip()
        if not question_text or not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question_text, cot, answer_text

    if formatter == "numinamath_15":
        question = str(row.get("problem") or "").strip()
        cot = str(row.get("solution") or "").strip()
        answer = _normalize_answer(row.get("answer") or "")
        if not question or not cot or not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if formatter == "openthoughts3_12m":
        parsed = _extract_openthoughts_trace_and_answer(row.get("conversations"))
        if parsed is None:
            return None
        question, cot, answer = parsed
        answer_text = answer if answer_only else answer
        return question, cot, answer_text

    if formatter == "natural_reasoning":
        question = str(row.get("question") or "").strip()
        responses = row.get("responses") or []
        cot = ""
        if isinstance(responses, list) and responses:
            cot = str((responses[0] or {}).get("response") or "").strip()
        answer = _normalize_answer(row.get("reference_answer") or "")
        if not question or not cot or not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if formatter == "proofwriter":
        theory = str(row.get("theory") or "").strip()
        question = str(row.get("question") or "").strip()
        cot = str(row.get("allProofs") or "").strip()
        answer = _normalize_answer(row.get("answer") or "")
        if not theory or not question or not cot or not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return f"{theory}\n\nQuestion: {question}", cot, answer_text

    return None


def _build_standard_completion(*, cot: str, answer: str, include_cot: bool) -> str:
    if include_cot and cot:
        return f"{cot}\n{answer}".strip()
    return answer.strip()


def _collect_formatted_examples(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
) -> list[dict[str, Any]]:
    sources = _resolve_dataset_sources(data_config)
    snapshot_path = _snapshot_file_path(
        data_config=data_config,
        runtime_config=runtime_config,
        sources=sources,
    )
    if snapshot_path is not None and snapshot_path.exists():
        formatted_examples = _load_formatted_examples_snapshot(snapshot_path)
        rng = random.Random(runtime_config.seed)
        rng.shuffle(formatted_examples)
        logger.info(
            "Loaded %d formatted examples from snapshot and reshuffled with seed=%d",
            len(formatted_examples),
            runtime_config.seed,
        )
        return formatted_examples

    formatted_examples: list[dict[str, Any]] = []
    total_examples = 0
    prefer_local_cache = snapshot_path is not None
    if prefer_local_cache:
        logger.info("Local-first dataset mode enabled; streaming sources will be materialized into cache first")
    logger.info("Collecting formatted examples from %d dataset sources", len(sources))
    for source in sources:
        per_dataset_examples = 0
        rows_seen = 0
        format_skips = 0
        token_skips = 0
        logger.info("Starting dataset collection for key=%s formatter=%s", source.key, source.formatter)
        for row in _iter_dataset_rows(source, data_config.cache_dir, prefer_local_cache=prefer_local_cache):
            rows_seen += 1
            formatted = _format_training_example(
                source.formatter,
                row,
                include_last_cot=data_config.include_last_cot,
                answer_only=data_config.answer_only,
            )
            if formatted is None:
                format_skips += 1
                continue
            question, cot, answer = formatted
            token_estimate = len(tokenizer.encode(f"{question} {cot} {answer}"))
            if token_estimate > data_config.max_token_num:
                token_skips += 1
                continue
            example_hash = hashlib.sha256(
                json.dumps(
                    {
                        "dataset_key": source.key,
                        "question": question,
                        "cot": cot,
                        "answer": answer,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            formatted_examples.append(
                {
                    "question": question,
                    "cot": cot,
                    "answer": answer,
                    "dataset_key": source.key,
                    "example_index": per_dataset_examples,
                    "source_row_index": rows_seen - 1,
                    "example_hash": example_hash,
                }
            )
            per_dataset_examples += 1
            total_examples += 1
            if per_dataset_examples % DATASET_PROGRESS_LOG_EVERY == 0:
                logger.info(
                    "Dataset progress key=%s accepted=%d seen=%d format_skips=%d token_skips=%d total_accepted=%d",
                    source.key,
                    per_dataset_examples,
                    rows_seen,
                    format_skips,
                    token_skips,
                    total_examples,
                )
            if data_config.max_samples_per_dataset is not None and per_dataset_examples >= data_config.max_samples_per_dataset:
                logger.info(
                    "Reached per-dataset cap for key=%s accepted=%d cap=%d",
                    source.key,
                    per_dataset_examples,
                    data_config.max_samples_per_dataset,
                )
                break
            if data_config.max_samples is not None and total_examples >= data_config.max_samples:
                logger.info(
                    "Reached global cap while processing key=%s total_accepted=%d cap=%d",
                    source.key,
                    total_examples,
                    data_config.max_samples,
                )
                break
        logger.info(
            "Finished dataset key=%s accepted=%d seen=%d format_skips=%d token_skips=%d",
            source.key,
            per_dataset_examples,
            rows_seen,
            format_skips,
            token_skips,
        )
        if data_config.max_samples is not None and total_examples >= data_config.max_samples:
            break
    if not formatted_examples:
        raise ValueError("No training examples were collected from the configured datasets.")
    if snapshot_path is not None:
        _write_formatted_examples_snapshot(snapshot_path, formatted_examples)
    rng = random.Random(runtime_config.seed)
    rng.shuffle(formatted_examples)
    logger.info("Shuffled %d formatted examples with seed=%d", len(formatted_examples), runtime_config.seed)
    return formatted_examples


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    validation_split_ratio: float,
    validation_max_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_split_ratio <= 0 or len(examples) < 2:
        logger.info("Validation split disabled or too small; using all %d examples for training", len(examples))
        return examples, []
    eval_count = max(1, int(round(len(examples) * validation_split_ratio)))
    eval_count = min(eval_count, len(examples) - 1)
    eval_examples = examples[:eval_count]
    if validation_max_samples is not None:
        eval_examples = eval_examples[:validation_max_samples]
    train_examples = examples[eval_count:]
    logger.info(
        "Split formatted examples into train=%d eval=%d using ratio=%.4f max_eval=%s",
        len(train_examples),
        len(eval_examples),
        validation_split_ratio,
        validation_max_samples,
    )
    return train_examples, eval_examples


class SupervisedLatentDataset(Dataset):
    def __init__(
        self,
        *,
        formatted_examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        runtime_config: LatentRuntimeConfig,
        bot_id: int,
        eot_id: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.examples = self._preprocess(formatted_examples)

    def _preprocess(self, formatted_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        logger.info("Tokenizing latent training dataset with %d formatted examples", len(formatted_examples))
        max_length = self.runtime_config.model_max_length
        questions = [str(example["question"]) for example in formatted_examples]
        cots = [str(example["cot"]) for example in formatted_examples]
        answers = [str(example["answer"]) for example in formatted_examples]
        source_ids = _tokenize_texts(questions, self.tokenizer, max_length=max_length)
        cot_ids = _tokenize_texts(cots, self.tokenizer, max_length=max_length)
        answer_ids = _tokenize_texts(answers, self.tokenizer, max_length=max_length)

        if not self.runtime_config.remove_eos:
            source_ids = [_append_eos(ids, self.tokenizer) for ids in source_ids]
            cot_ids = [_append_eos(ids, self.tokenizer) for ids in cot_ids]
            answer_ids = [_append_eos(ids, self.tokenizer) for ids in answer_ids]

        cot_ids = [_trim_bos(ids, self.tokenizer) for ids in cot_ids]
        answer_ids = [_trim_bos(ids, self.tokenizer) for ids in answer_ids]

        teacher_ids = [torch.cat([src, cot, ans]).long() for src, cot, ans in zip(source_ids, cot_ids, answer_ids)]
        teacher_labels: list[torch.Tensor] = []
        encoder_ids: list[torch.Tensor] = []
        decoder_ids: list[torch.Tensor] = []
        ref_answer_positions: list[int] = []
        model_answer_positions: list[int] = []

        answer_prompt = "The answer is:"
        answer_prompt_ids = torch.tensor(
            self.tokenizer.encode(answer_prompt, add_special_tokens=False),
            dtype=torch.long,
        )

        for src, ans, teacher in zip(source_ids, answer_ids, teacher_ids):
            teacher_label = teacher.clone()
            teacher_label[: src.numel()] = IGNORE_INDEX
            teacher_labels.append(teacher_label)

            encoder_ids.append(_prepend_tokens(src, [self.bot_id]))
            if self.runtime_config.remove_eos:
                decoder = _prepend_tokens(ans, [self.eot_id])
            else:
                decoder = _prepend_tokens(ans, [self.eot_id, self.tokenizer.eos_token_id])
            decoder_ids.append(decoder)

            answer_text = self.tokenizer.decode(ans, skip_special_tokens=False)
            if answer_prompt_ids.numel() > 0 and not answer_text.startswith(answer_prompt):
                ref_answer_positions.append(max(teacher.numel() - ans.numel(), 0))
                model_answer_positions.append(1)
            else:
                ref_answer_positions.append(_get_answer_token_position(teacher, answer_prompt_ids))
                model_answer_positions.append(_get_answer_token_position(decoder, answer_prompt_ids))

        records: list[dict[str, Any]] = []
        for example, src, dec, teacher, labels, ref_pos, model_pos in zip(
            formatted_examples,
            encoder_ids,
            decoder_ids,
            teacher_ids,
            teacher_labels,
            ref_answer_positions,
            model_answer_positions,
        ):
            records.append(
                {
                    "encoder_input_ids": src,
                    "decoder_input_ids": dec,
                    "ref_input_ids": teacher,
                    "labels": dec.clone(),
                    "ref_labels": labels,
                    "ref_answer_position": torch.tensor(ref_pos, dtype=torch.long),
                    "model_answer_position": torch.tensor(model_pos, dtype=torch.long),
                    "forensics": {
                        "dataset_key": str(example.get("dataset_key") or "unknown"),
                        "example_index": int(example.get("example_index", -1)),
                        "source_row_index": int(example.get("source_row_index", -1)),
                        "example_hash": str(example.get("example_hash") or "unknown"),
                        "encoder_length": int(src.numel()),
                        "decoder_length": int(dec.numel()),
                        "ref_length": int(teacher.numel()),
                        "ref_answer_position": int(ref_pos),
                        "model_answer_position": int(model_pos),
                    },
                }
            )
        logger.info("Built latent training records: %d", len(records))
        return records

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass(slots=True)
class SupervisedLatentDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        encoder_input_ids = [item["encoder_input_ids"] for item in instances]
        decoder_input_ids = [item["decoder_input_ids"] for item in instances]
        ref_input_ids = [item["ref_input_ids"] for item in instances]
        labels = [item["labels"] for item in instances]
        ref_labels = [item["ref_labels"] for item in instances]
        ref_answer_position = [item["ref_answer_position"] for item in instances]
        model_answer_position = [item["model_answer_position"] for item in instances]
        batch_forensics = [dict(item.get("forensics", {})) for item in instances]

        reversed_encoder = [ids.flip(0) for ids in encoder_input_ids]
        padded_encoder = torch.nn.utils.rnn.pad_sequence(
            reversed_encoder,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(1)
        padded_decoder = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_ref = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        padded_ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return {
            "encoder_input_ids": padded_encoder,
            "decoder_input_ids": padded_decoder,
            "ref_input_ids": padded_ref,
            "labels": padded_labels,
            "encoder_attention_mask": padded_encoder.ne(self.tokenizer.pad_token_id),
            "ref_attention_mask": padded_ref.ne(self.tokenizer.pad_token_id),
            "ref_labels": padded_ref_labels,
            "ref_answer_position": torch.stack(ref_answer_position),
            "model_answer_position": torch.stack(model_answer_position),
            "batch_forensics": batch_forensics,
        }


class StandardSupervisedDataset(Dataset):
    def __init__(
        self,
        *,
        formatted_examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        runtime_config: LatentRuntimeConfig,
        include_cot: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.examples: list[dict[str, torch.Tensor]] = []
        logger.info(
            "Tokenizing standard supervised dataset with %d formatted examples include_cot=%s",
            len(formatted_examples),
            include_cot,
        )

        for example in formatted_examples:
            question = str(example["question"])
            cot = str(example["cot"])
            answer = str(example["answer"])
            completion = _build_standard_completion(cot=cot, answer=answer, include_cot=include_cot)
            full_text = f"{question}\n{completion}".strip()
            prompt_text = question.strip()

            full_ids = torch.tensor(
                tokenizer(
                    full_text,
                    truncation=True,
                    max_length=runtime_config.model_max_length,
                    padding=False,
                    return_attention_mask=False,
                )["input_ids"],
                dtype=torch.long,
            )
            prompt_ids = torch.tensor(
                tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=runtime_config.model_max_length,
                    padding=False,
                    return_attention_mask=False,
                )["input_ids"],
                dtype=torch.long,
            )
            if full_ids.numel() == 0:
                continue
            if tokenizer.eos_token_id is not None and full_ids[-1].item() != tokenizer.eos_token_id:
                full_ids = torch.cat([full_ids, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])
            labels = full_ids.clone()
            labels[: min(prompt_ids.numel(), labels.numel())] = IGNORE_INDEX
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "labels": labels,
                }
            )
        logger.info("Built standard supervised records: %d", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass(slots=True)
class StandardDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in instances]
        labels = [item["labels"] for item in instances]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_input_ids.ne(self.tokenizer.pad_token_id),
            "labels": padded_labels,
        }


def _build_formatted_splits(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    formatted_examples = _collect_formatted_examples(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    return _split_examples(
        formatted_examples,
        validation_split_ratio=data_config.validation_split_ratio,
        validation_max_samples=data_config.validation_max_samples,
    )


def make_supervised_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    train_examples, eval_examples = _build_formatted_splits(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    train_dataset = SupervisedLatentDataset(
        formatted_examples=train_examples,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        bot_id=bot_id,
        eot_id=eot_id,
    )
    eval_dataset = (
        SupervisedLatentDataset(
            formatted_examples=eval_examples,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            bot_id=bot_id,
            eot_id=eot_id,
        )
        if eval_examples
        else None
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": SupervisedLatentDataCollator(tokenizer=tokenizer),
    }


def make_standard_answer_only_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    del bot_id, eot_id
    train_examples, eval_examples = _build_formatted_splits(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    train_dataset = StandardSupervisedDataset(
        formatted_examples=train_examples,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        include_cot=False,
    )
    eval_dataset = (
        StandardSupervisedDataset(
            formatted_examples=eval_examples,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            include_cot=False,
        )
        if eval_examples
        else None
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": StandardDataCollator(tokenizer=tokenizer),
    }


def make_standard_cot_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    del bot_id, eot_id
    train_examples, eval_examples = _build_formatted_splits(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    train_dataset = StandardSupervisedDataset(
        formatted_examples=train_examples,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        include_cot=True,
    )
    eval_dataset = (
        StandardSupervisedDataset(
            formatted_examples=eval_examples,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            include_cot=True,
        )
        if eval_examples
        else None
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": StandardDataCollator(tokenizer=tokenizer),
    }
