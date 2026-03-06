from codi_reimplementation.benchmarks.answer_parsing import parse_prediction


def test_numeric_parser_uses_last_number() -> None:
    parsed = parse_prediction("reasoning... The answer is 42", "numeric")
    assert parsed.is_valid
    assert parsed.value == "42"


def test_multiple_choice_parser_finds_letter() -> None:
    parsed = parse_prediction("The answer is: C", "multiple_choice")
    assert parsed.is_valid
    assert parsed.value == "C"


def test_boolean_parser_normalizes_yes() -> None:
    parsed = parse_prediction("yes, that is correct", "boolean")
    assert parsed.is_valid
    assert parsed.value == "true"
