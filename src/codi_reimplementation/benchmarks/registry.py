from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from codi_reimplementation.benchmarks.answer_parsing import normalize_reference
from codi_reimplementation.benchmarks.types import BenchmarkExample, BenchmarkSpec, LoadedBenchmark
from codi_reimplementation.config import ensure_dir


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
}


PRONTOQA_ANSWER_RE = re.compile(r"(?:###The answer is:|The answer is:)\s*(True|False)", re.IGNORECASE)


def list_benchmark_names() -> list[str]:
    return sorted(DEFAULT_BENCHMARKS)


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return DEFAULT_BENCHMARKS[name]
    except KeyError as exc:
        known = ", ".join(list_benchmark_names())
        raise KeyError(f"Unknown benchmark {name!r}. Known benchmarks: {known}") from exc


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
                metadata={"split": split_name, "source_revision": spec.revision},
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    loaded = LoadedBenchmark(spec=spec, examples=examples)
    if snapshot_dir is not None:
        snapshot_root = ensure_dir(snapshot_dir)
        export_snapshot(loaded, snapshot_root / f"{spec.name}.jsonl")
    return loaded
