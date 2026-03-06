from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from codi_reimplementation.benchmarks.types import TaskType


@dataclass(slots=True)
class ParsedAnswer:
    value: str
    is_valid: bool
    parser: Literal["numeric", "multiple_choice", "boolean", "raw"]


NUMERIC_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
BOOLEAN_PATTERN = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
CHOICE_PATTERN = re.compile(r"\b([A-E])\b")


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


def normalize_reference(value: object, task_type: TaskType) -> str:
    if task_type == "numeric":
        return normalize_numeric(str(value)).value
    if task_type == "multiple_choice":
        return normalize_multiple_choice(str(value)).value
    if task_type == "boolean":
        return normalize_boolean(str(value)).value
    raise ValueError(f"Unsupported task type: {task_type}")


def parse_prediction(text: str, task_type: TaskType) -> ParsedAnswer:
    if task_type == "numeric":
        return normalize_numeric(text)
    if task_type == "multiple_choice":
        return normalize_multiple_choice(text)
    if task_type == "boolean":
        return normalize_boolean(text)
    return ParsedAnswer(value=text.strip(), is_valid=True, parser="raw")
