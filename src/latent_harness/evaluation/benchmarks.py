from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from datasets import DatasetDict, load_dataset

from latent_harness.core.io import ensure_dir

TaskType = Literal["numeric", "multiple_choice", "boolean", "answer_choice_text"]


@dataclass(slots=True)
class BenchmarkExample:
    benchmark_name: str
    example_id: str
    prompt: str
    target: str
    task_type: TaskType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSpec:
    name: str
    hf_path: str
    split: str
    revision: str = "main"
    hf_subset: str | None = None
    task_type: TaskType = "numeric"


@dataclass(slots=True)
class LoadedBenchmark:
    spec: BenchmarkSpec
    examples: list[BenchmarkExample]


@dataclass(slots=True)
class ParsedAnswer:
    value: str
    is_valid: bool
    parser: Literal["numeric", "multiple_choice", "boolean", "answer_choice_text", "raw"]


DEFAULT_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        hf_path="gsm8k",
        hf_subset="main",
        split="test",
        revision="main",
        task_type="numeric",
    ),
    "svamp": BenchmarkSpec(
        name="svamp",
        hf_path="ChilleD/SVAMP",
        split="test",
        revision="main",
        task_type="numeric",
    ),
    "gsm-hard": BenchmarkSpec(
        name="gsm-hard",
        hf_path="reasoning-machines/gsm-hard",
        split="train",
        revision="main",
        task_type="numeric",
    ),
    "multi-arith": BenchmarkSpec(
        name="multi-arith",
        hf_path="ChilleD/MultiArith",
        split="test",
        revision="main",
        task_type="numeric",
    ),
    "commonsense_qa": BenchmarkSpec(
        name="commonsense_qa",
        hf_path="tau/commonsense_qa",
        split="validation",
        revision="main",
        task_type="multiple_choice",
    ),
    "strategyqa": BenchmarkSpec(
        name="strategyqa",
        hf_path="ChilleD/StrategyQA",
        split="test",
        revision="main",
        task_type="boolean",
    ),
    "prontoqa": BenchmarkSpec(
        name="prontoqa",
        hf_path="longface/prontoqa-train",
        split="train",
        revision="main",
        task_type="boolean",
    ),
    "p3_arc_challenge_letter": BenchmarkSpec(
        name="p3_arc_challenge_letter",
        hf_path="bigscience/P3",
        hf_subset="ai2_arc_ARC_Challenge_pick_the_most_correct_option",
        split="validation",
        revision="main",
        task_type="multiple_choice",
    ),
    "p3_arc_challenge_text": BenchmarkSpec(
        name="p3_arc_challenge_text",
        hf_path="bigscience/P3",
        hf_subset="ai2_arc_ARC_Challenge_multiple_choice",
        split="validation",
        revision="main",
        task_type="answer_choice_text",
    ),
    "p3_arc_challenge_qa_options": BenchmarkSpec(
        name="p3_arc_challenge_qa_options",
        hf_path="bigscience/P3",
        hf_subset="ai2_arc_ARC_Challenge_qa_options",
        split="validation",
        revision="main",
        task_type="answer_choice_text",
    ),
}

NUMERIC_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
BOOLEAN_PATTERN = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
CHOICE_PATTERN = re.compile(r"\b([A-E])\b")
PRONTOQA_ANSWER_RE = re.compile(r"(?:###The answer is:|The answer is:)\s*(True|False)", re.IGNORECASE)


def list_benchmark_names() -> list[str]:
    return sorted(DEFAULT_BENCHMARKS)


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return DEFAULT_BENCHMARKS[name]
    except KeyError as exc:
        known = ", ".join(list_benchmark_names())
        raise KeyError(f"Unknown benchmark {name!r}. Known benchmarks: {known}") from exc


def normalize_numeric(text: str) -> ParsedAnswer:
    matches = NUMERIC_PATTERN.findall(text)
    if not matches:
        return ParsedAnswer(value="nan", is_valid=False, parser="numeric")
    candidate = matches[-1].replace(",", "")
    try:
        number = float(candidate)
    except ValueError:
        return ParsedAnswer(value="nan", is_valid=False, parser="numeric")
    if math.isinf(number) or math.isnan(number):
        return ParsedAnswer(value="nan", is_valid=False, parser="numeric")
    if number.is_integer():
        return ParsedAnswer(value=str(int(number)), is_valid=True, parser="numeric")
    return ParsedAnswer(value=f"{number:.6f}".rstrip("0").rstrip("."), is_valid=True, parser="numeric")


def normalize_multiple_choice(text: str) -> ParsedAnswer:
    anchor = text.split("The answer is:")[-1]
    match = CHOICE_PATTERN.search(anchor.upper())
    if not match:
        return ParsedAnswer(value="", is_valid=False, parser="multiple_choice")
    return ParsedAnswer(value=match.group(1), is_valid=True, parser="multiple_choice")


def normalize_boolean(text: str) -> ParsedAnswer:
    match = BOOLEAN_PATTERN.search(text)
    if not match:
        return ParsedAnswer(value="", is_valid=False, parser="boolean")
    token = match.group(1).lower()
    value = "true" if token in {"true", "yes"} else "false"
    return ParsedAnswer(value=value, is_valid=True, parser="boolean")


def normalize_answer_choice_text(value: object) -> str:
    text = str(value).strip()
    return " ".join(text.split()).casefold()


def parse_answer_choice_text(
    text: str,
    answer_choices: list[str] | None = None,
) -> ParsedAnswer:
    normalized_text = normalize_answer_choice_text(text)
    if not normalized_text:
        return ParsedAnswer(value="", is_valid=False, parser="answer_choice_text")

    if not answer_choices:
        return ParsedAnswer(value=normalized_text, is_valid=True, parser="answer_choice_text")

    normalized_choices = [normalize_answer_choice_text(choice) for choice in answer_choices if str(choice).strip()]
    if normalized_text in normalized_choices:
        return ParsedAnswer(value=normalized_text, is_valid=True, parser="answer_choice_text")

    anchors = [text.split("The answer is:")[-1], text]
    for candidate in anchors:
        normalized_candidate = normalize_answer_choice_text(candidate)
        for choice in normalized_choices:
            if choice and choice in normalized_candidate:
                return ParsedAnswer(value=choice, is_valid=True, parser="answer_choice_text")

    return ParsedAnswer(value="", is_valid=False, parser="answer_choice_text")


def normalize_reference(
    value: object,
    task_type: TaskType,
    *,
    answer_choices: list[str] | None = None,
) -> str:
    if task_type == "numeric":
        return normalize_numeric(str(value)).value
    if task_type == "multiple_choice":
        return normalize_multiple_choice(str(value)).value
    if task_type == "boolean":
        return normalize_boolean(str(value)).value
    if task_type == "answer_choice_text":
        return parse_answer_choice_text(str(value), answer_choices).value
    raise ValueError(f"Unsupported task type: {task_type}")


def parse_prediction(
    text: str,
    task_type: TaskType,
    *,
    answer_choices: list[str] | None = None,
) -> ParsedAnswer:
    if task_type == "numeric":
        return normalize_numeric(text)
    if task_type == "multiple_choice":
        return normalize_multiple_choice(text)
    if task_type == "boolean":
        return normalize_boolean(text)
    if task_type == "answer_choice_text":
        return parse_answer_choice_text(text, answer_choices)
    return ParsedAnswer(value=text.strip(), is_valid=True, parser="raw")


def _resolve_split(dataset: DatasetDict, preferred: str) -> str:
    if preferred in dataset:
        return preferred
    for candidate in ("validation", "test", "train"):
        if candidate in dataset:
            return candidate
    raise KeyError(f"Could not resolve split from dataset keys: {list(dataset.keys())}")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise KeyError(f"Missing expected keys {keys!r} in example with keys {sorted(row.keys())}")


def _format_commonsense_prompt(row: dict[str, Any]) -> str:
    choices = row.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    option_lines = [f"{label}. {text}" for label, text in zip(labels, texts)]
    options_block = "\n".join(option_lines)
    return f"{_stringify(row['question'])}\n{options_block}\nAnswer with one letter."


def _format_strategyqa_prompt(row: dict[str, Any]) -> str:
    return f"{_stringify(row['question'])}\nAnswer with yes or no."


def _format_prontoqa_prompt(row: dict[str, Any]) -> str:
    if "prompt" in row:
        prompt = _stringify(row["prompt"])
        if "###Response:" in prompt:
            prompt = prompt.split("###Response:")[0].strip()
        if prompt.startswith("###Context:"):
            prompt = prompt.replace("###Context:", "", 1).strip()
        return prompt
    question = _stringify(row.get("question") or row.get("query") or row.get("input"))
    facts = row.get("facts") or row.get("context") or row.get("premises")
    if isinstance(facts, list):
        facts_block = "\n".join(_stringify(item) for item in facts)
        return f"{facts_block}\n\nQuestion: {question}\nAnswer with true or false."
    if facts:
        return f"{_stringify(facts)}\n\nQuestion: {question}\nAnswer with true or false."
    return f"{question}\nAnswer with true or false."


def _format_svamp_prompt(row: dict[str, Any]) -> str:
    if "question_concat" in row:
        return _stringify(row["question_concat"])
    body = _stringify(row.get("Body"))
    question = _stringify(row.get("Question"))
    return f"{body} {question}".strip()


def _format_prompt(spec: BenchmarkSpec, row: dict[str, Any]) -> str:
    if spec.hf_path == "bigscience/P3":
        return _stringify(row.get("inputs_pretokenized") or row.get("inputs"))
    if spec.name == "commonsense_qa":
        return _format_commonsense_prompt(row)
    if spec.name == "strategyqa":
        return _format_strategyqa_prompt(row)
    if spec.name == "prontoqa":
        return _format_prontoqa_prompt(row)
    if spec.name == "svamp":
        return _format_svamp_prompt(row)
    if spec.name == "gsm-hard":
        return _stringify(row.get("instruction") or row.get("input"))
    return _stringify(row.get("question") or row.get("prompt") or row.get("input"))


def _extract_target(spec: BenchmarkSpec, row: dict[str, Any]) -> str:
    if spec.hf_path == "bigscience/P3":
        return normalize_reference(
            _extract(row, "targets_pretokenized", "targets"),
            spec.task_type,
            answer_choices=row.get("answer_choices"),
        )
    if spec.name == "gsm8k":
        return normalize_reference(_extract(row, "answer"), spec.task_type)
    if spec.name == "svamp":
        return normalize_reference(_extract(row, "Answer", "answer"), spec.task_type)
    if spec.name == "gsm-hard":
        return normalize_reference(_extract(row, "target", "response"), spec.task_type)
    if spec.name == "multi-arith":
        return normalize_reference(_extract(row, "final_ans", "answer"), spec.task_type)
    if spec.name == "commonsense_qa":
        return normalize_reference(_extract(row, "answerKey", "answer"), spec.task_type)
    if spec.name == "strategyqa":
        return normalize_reference(_extract(row, "answer"), spec.task_type)
    if spec.name == "prontoqa":
        if "prompt" in row:
            match = PRONTOQA_ANSWER_RE.search(_stringify(row["prompt"]))
            if match:
                return normalize_reference(match.group(1), spec.task_type)
        return normalize_reference(_extract(row, "answer", "target", "label"), spec.task_type)
    return normalize_reference(_extract(row, "answer"), spec.task_type)


def _extract_example_id(spec: BenchmarkSpec, row: dict[str, Any], index: int) -> str:
    for key in ("id", "idx", "example_id", "qid"):
        if key in row:
            return f"{spec.name}-{row[key]}"
    return f"{spec.name}-{index}"


def export_snapshot(loaded: LoadedBenchmark, snapshot_path: str | Path) -> Path:
    destination = Path(snapshot_path).expanduser().resolve()
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for example in loaded.examples:
            handle.write(
                json.dumps(
                    {
                        "benchmark_name": example.benchmark_name,
                        "example_id": example.example_id,
                        "prompt": example.prompt,
                        "target": example.target,
                        "task_type": example.task_type,
                        "metadata": example.metadata,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    return destination


def load_benchmark(
    spec: BenchmarkSpec,
    *,
    cache_dir: str | Path,
    snapshot_dir: str | Path | None = None,
    max_examples: int | None = None,
) -> LoadedBenchmark:
    dataset = load_dataset(
        path=spec.hf_path,
        name=spec.hf_subset,
        revision=spec.revision,
        cache_dir=str(Path(cache_dir).expanduser().resolve()),
    )
    split_name = _resolve_split(dataset, spec.split)
    split = dataset[split_name]

    examples: list[BenchmarkExample] = []
    for index, row in enumerate(split):
        prompt = _format_prompt(spec, row)
        target = _extract_target(spec, row)
        example_id = _extract_example_id(spec, row, index)
        examples.append(
            BenchmarkExample(
                benchmark_name=spec.name,
                example_id=example_id,
                prompt=prompt,
                target=target,
                task_type=spec.task_type,
                metadata={
                    "split": split_name,
                    "source_revision": spec.revision,
                    "hf_subset": spec.hf_subset,
                    "answer_choices": row.get("answer_choices"),
                },
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    loaded = LoadedBenchmark(spec=spec, examples=examples)
    if snapshot_dir is not None:
        snapshot_root = ensure_dir(snapshot_dir)
        export_snapshot(loaded, snapshot_root / f"{spec.name}.jsonl")
    return loaded
