from __future__ import annotations

from dataclasses import dataclass

from latent_harness.evaluation.benchmarks import BenchmarkExample, ParsedAnswer, parse_prediction


@dataclass(slots=True)
class ExampleScore:
    parsed_prediction: ParsedAnswer
    correct: bool


def score_prediction(example: BenchmarkExample, prediction_text: str) -> ExampleScore:
    parsed = parse_prediction(
        prediction_text,
        example.task_type,
        answer_choices=example.metadata.get("answer_choices"),
    )
    correct = parsed.is_valid and parsed.value == example.target
    return ExampleScore(parsed_prediction=parsed, correct=correct)
