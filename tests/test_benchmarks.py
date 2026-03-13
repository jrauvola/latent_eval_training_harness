from __future__ import annotations

from latent_harness.evaluation.benchmarks import get_benchmark_spec, parse_prediction


def test_p3_benchmark_specs_are_registered() -> None:
    spec = get_benchmark_spec("p3_arc_challenge_text")
    assert spec.hf_path == "bigscience/P3"
    assert spec.hf_subset == "ai2_arc_ARC_Challenge_multiple_choice"
    assert spec.task_type == "answer_choice_text"


def test_answer_choice_text_parser_matches_exact_choice_text() -> None:
    parsed = parse_prediction(
        "Record the details of the investigation.",
        "answer_choice_text",
        answer_choices=[
            "Put the objects in groups.",
            "Change the height of the ramp.",
            "Choose different objects to roll.",
            "Record the details of the investigation.",
        ],
    )
    assert parsed.is_valid is True
    assert parsed.value == "record the details of the investigation."


def test_answer_choice_text_parser_matches_embedded_choice_text() -> None:
    parsed = parse_prediction(
        "The answer is: Record the details of the investigation.",
        "answer_choice_text",
        answer_choices=[
            "Put the objects in groups.",
            "Change the height of the ramp.",
            "Choose different objects to roll.",
            "Record the details of the investigation.",
        ],
    )
    assert parsed.is_valid is True
    assert parsed.value == "record the details of the investigation."
