from __future__ import annotations

from dataclasses import dataclass

from codi_reimplementation.benchmarks.answer_parsing import ParsedAnswer, parse_prediction
from codi_reimplementation.benchmarks.types import BenchmarkExample


@dataclass(slots=True)
class ExampleScore:
    parsed_prediction: ParsedAnswer
    correct: bool


def score_prediction(example: BenchmarkExample, prediction_text: str) -> ExampleScore:
    parsed = parse_prediction(prediction_text, example.task_type)
    correct = parsed.is_valid and parsed.value == example.target
    return ExampleScore(parsed_prediction=parsed, correct=correct)
