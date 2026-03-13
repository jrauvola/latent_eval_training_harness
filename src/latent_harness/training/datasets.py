from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from latent_harness.core.config import LatentRuntimeConfig
from latent_harness.training.config import TrainingDataConfig

IGNORE_INDEX = -100


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
    return int(matches[0].item() + window)


def _extract_numeric_answer(raw: str) -> str | None:
    cleaned = raw.replace("####", "").strip()
    parts = cleaned.split()
    if not parts:
        return None
    terminal = parts[-1].replace(",", "")
    if terminal and (terminal[0].isdigit() or terminal[0] == "-"):
        return terminal
    return None


def _load_training_split(dataset_name: str, cache_dir: str) -> list[dict[str, Any]]:
    if dataset_name == "gsm8k_aug":
        return list(load_dataset("zen-E/GSM8k-Aug", cache_dir=cache_dir)["train"])
    if dataset_name == "gsm8k_aug_nl":
        return list(load_dataset("zen-E/GSM8k-Aug-NL", cache_dir=cache_dir)["train"])
    if dataset_name == "commonsense_cot":
        return list(load_dataset("zen-E/CommonsenseQA-GPT4omini", cache_dir=cache_dir)["train"])
    if dataset_name == "strategyqa_cot":
        return list(load_dataset("zen-E/StrategyQA_CoT_GPT4o", cache_dir=cache_dir)["train"])
    if dataset_name == "prontoqa":
        dataset = load_dataset("tasksource/prontoqa", cache_dir=cache_dir)
        split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
        return list(dataset[split_name])
    raise ValueError(f"Unsupported training dataset {dataset_name!r}")


def _format_training_example(
    dataset_name: str,
    row: dict[str, Any],
    *,
    include_last_cot: bool,
    answer_only: bool,
) -> tuple[str, str, str] | None:
    question = str(row.get("question") or row.get("input") or "").strip()
    if not question:
        return None

    if dataset_name in {"gsm8k_aug", "gsm8k_aug_nl"}:
        cot = str(row.get("cot") or "").strip()
        answer = _extract_numeric_answer(str(row.get("answer") or ""))
        if answer is None:
            return None
        if not include_last_cot and cot:
            if dataset_name == "gsm8k_aug_nl":
                cot_parts = cot.split(". ")
                cot = ". ".join(cot_parts[:-1]).strip()
                if cot:
                    cot = cot + "."
            else:
                cot_tokens = cot.split()
                cot = " ".join(cot_tokens[:-1]).strip()
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if dataset_name in {"commonsense_cot", "strategyqa_cot"}:
        cot = str(row.get("cot") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    if dataset_name == "prontoqa":
        cot_steps = row.get("steps") or []
        cot = "\n".join(str(step).strip() for step in cot_steps[:-1] if str(step).strip())
        answer = str(row.get("answer") or row.get("target") or "").strip()
        if not answer:
            return None
        answer_text = answer if answer_only else f"The answer is: {answer}"
        return question, cot, answer_text

    return None


def _build_standard_completion(*, cot: str, answer: str, include_cot: bool) -> str:
    if include_cot and cot:
        return f"{cot}\n{answer}".strip()
    return answer.strip()


class SupervisedLatentDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_names: list[str],
        tokenizer: PreTrainedTokenizerBase,
        data_config: TrainingDataConfig,
        runtime_config: LatentRuntimeConfig,
        bot_id: int,
        eot_id: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.bot_id = bot_id
        self.eot_id = eot_id

        questions: list[str] = []
        cots: list[str] = []
        answers: list[str] = []

        for dataset_name in dataset_names:
            for row in _load_training_split(dataset_name, data_config.cache_dir):
                formatted = _format_training_example(
                    dataset_name,
                    row,
                    include_last_cot=data_config.include_last_cot,
                    answer_only=data_config.answer_only,
                )
                if formatted is None:
                    continue
                question, cot, answer = formatted
                token_estimate = len(tokenizer.encode(f"{question} {cot} {answer}"))
                if token_estimate > data_config.max_token_num:
                    continue
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
                if data_config.max_samples is not None and len(questions) >= data_config.max_samples:
                    break
            if data_config.max_samples is not None and len(questions) >= data_config.max_samples:
                break

        self.examples = self._preprocess(questions=questions, cots=cots, answers=answers)

    def _preprocess(self, *, questions: list[str], cots: list[str], answers: list[str]) -> list[dict[str, torch.Tensor]]:
        max_length = self.runtime_config.model_max_length
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

        for src, cot, ans, teacher in zip(source_ids, cot_ids, answer_ids, teacher_ids):
            del cot
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

        records: list[dict[str, torch.Tensor]] = []
        for src, dec, teacher, labels, ref_pos, model_pos in zip(
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
                }
            )
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
        }


class StandardSupervisedDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_names: list[str],
        tokenizer: PreTrainedTokenizerBase,
        data_config: TrainingDataConfig,
        runtime_config: LatentRuntimeConfig,
        include_cot: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config
        self.examples: list[dict[str, torch.Tensor]] = []

        for dataset_name in dataset_names:
            for row in _load_training_split(dataset_name, data_config.cache_dir):
                formatted = _format_training_example(
                    dataset_name,
                    row,
                    include_last_cot=data_config.include_last_cot,
                    answer_only=data_config.answer_only,
                )
                if formatted is None:
                    continue
                question, cot, answer = formatted
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
                if data_config.max_samples is not None and len(self.examples) >= data_config.max_samples:
                    break
            if data_config.max_samples is not None and len(self.examples) >= data_config.max_samples:
                break

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


def make_supervised_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    train_dataset = SupervisedLatentDataset(
        dataset_names=data_config.dataset_names,
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
        bot_id=bot_id,
        eot_id=eot_id,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
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
    train_dataset = StandardSupervisedDataset(
        dataset_names=data_config.dataset_names,
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
        include_cot=False,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
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
    train_dataset = StandardSupervisedDataset(
        dataset_names=data_config.dataset_names,
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
        include_cot=True,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": StandardDataCollator(tokenizer=tokenizer),
    }
