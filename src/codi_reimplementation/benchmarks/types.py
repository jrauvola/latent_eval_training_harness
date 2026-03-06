from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["numeric", "multiple_choice", "boolean"]


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
    prompt_field: str | None = None
    answer_field: str | None = None
    prompt_template: str = "{question}"
    fewshot_template: str | None = None


@dataclass(slots=True)
class LoadedBenchmark:
    spec: BenchmarkSpec
    examples: list[BenchmarkExample]
