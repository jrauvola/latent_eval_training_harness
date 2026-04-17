#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import torch

from latent_harness.core.io import ensure_dir, load_yaml_config
from latent_harness.evaluation.benchmarks import normalize_answer_choice_text, parse_prediction
from latent_harness.evaluation.config import EvaluationConfig
from latent_harness.evaluation.models import load_evaluation_model
from latent_harness.evaluation.runner import _generate_predictions


DEFAULT_INPUT_CSV = (
    "/Users/jrauvola/Desktop/Latent_Reasoning_Project/latent_eval_training_harness/"
    "artifacts/datasets/p3_llama32_vs_codi_llama32_text_5_examples/p3_llama32_qa_comparison.csv"
)


def extract_answer_choices(prompt: str) -> list[str]:
    return [line[2:].strip() for line in prompt.splitlines() if line.strip().startswith("- ")]


def build_variants(prompt: str) -> list[tuple[str, str]]:
    variants = [("original", prompt)]

    no_choice_count = re.sub(
        r"Only one answer is correct among these \d+ choices:",
        "Choose the correct answer from the options below:",
        prompt,
        flags=re.IGNORECASE,
    )
    variants.append(("no_choice_count", no_choice_count))

    text_only = (
        no_choice_count
        + "\n\nRespond with only the exact answer text from the options. No explanation."
    )
    variants.append(("text_only", text_only))

    strict_text_only = (
        no_choice_count
        + "\n\nRespond with only the exact answer text from the options."
        + " Do not say 'The answer is'. Do not explain."
    )
    variants.append(("strict_text_only", strict_text_only))

    question_first = re.sub(
        r"^I gave my students this multiple choice question:\s*",
        "Question: ",
        prompt,
        flags=re.IGNORECASE,
    )
    question_first = re.sub(
        r"Only one answer is correct among these \d+ choices:",
        "Options:",
        question_first,
        flags=re.IGNORECASE,
    )
    question_first = (
        question_first
        + "\n\nReply with only the exact option text."
    )
    variants.append(("question_first_text_only", question_first))

    no_preamble = re.sub(
        r"^I gave my students this multiple choice question:\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    variants.append(("no_preamble", no_preamble))

    question_and_options = re.sub(
        r"^I gave my students this multiple choice question:\s*",
        "Question: ",
        prompt,
        flags=re.IGNORECASE,
    )
    question_and_options = re.sub(
        r"Only one answer is correct among these \d+ choices:",
        "Options:",
        question_and_options,
        flags=re.IGNORECASE,
    )
    variants.append(("question_and_options", question_and_options))

    inline_options = prompt
    choices = extract_answer_choices(prompt)
    if choices:
        question_lines = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                continue
            if "Only one answer is correct among these" in stripped:
                continue
            if stripped:
                question_lines.append(stripped)
        inline_prompt = " ".join(question_lines)
        inline_prompt += f" Options: {', '.join(choices)}."
        inline_prompt += " Reply with only the exact option text."
        inline_options = inline_prompt
    variants.append(("inline_options_text_only", inline_options))

    short_answer = re.sub(
        r"\n\nOnly one answer is correct among these \d+ choices:\n(?:- .+\n)+\nCould you tell me which one is correct\?",
        "\n\nAnswer with a short phrase only.",
        prompt,
        flags=re.IGNORECASE,
    )
    variants.append(("short_answer_no_options", short_answer))

    letters = ["A", "B", "C", "D", "E", "F"]
    if choices:
        stem = re.sub(
            r"\n\nOnly one answer is correct among these \d+ choices:\n(?:- .+\n)+\nCould you tell me which one is correct\?",
            "",
            prompt,
            flags=re.IGNORECASE,
        ).strip()
        letter_lines = [f"{letters[idx]}. {choice}" for idx, choice in enumerate(choices)]
        letter_prompt = (
            f"{stem}\n\nOptions:\n" + "\n".join(letter_lines) + "\n\nAnswer with one letter only."
        )
        variants.append(("letter_options", letter_prompt))

        first = choices[0]
        shuffled = choices[1:] + [first]
        shuffled_lines = "\n".join(f"- {choice}" for choice in shuffled)
        shuffled_prompt = re.sub(
            r"Only one answer is correct among these \d+ choices:\n(?:- .+\n)+",
            f"Choose the correct answer from the options below:\n{shuffled_lines}\n",
            prompt,
            flags=re.IGNORECASE,
        )
        shuffled_prompt += "\nRespond with only the exact answer text."
        variants.append(("shuffled_options_text_only", shuffled_prompt))

    return variants


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt variants against a harness model for diagnostics.")
    parser.add_argument(
        "--config",
        default="/Users/jrauvola/Desktop/Latent_Reasoning_Project/latent_eval_training_harness/configs/evaluation/broader_suite_plus_p3_gemma3_gh200.yaml",
        help="Evaluation config containing the named model.",
    )
    parser.add_argument(
        "--model-name",
        default="codi_llama32_1b_official",
        help="Model spec name to test.",
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="CSV containing prompt/gold rows from the focused P3 comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/codi_prompt_diagnostics",
        help="Directory for diagnostic CSV output.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of examples to probe.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Generation length override for the diagnostic run.",
    )
    parser.add_argument(
        "--model-names",
        default=None,
        help="Optional comma-separated list of model names. Overrides --model-name when set.",
    )
    args = parser.parse_args()

    payload = load_yaml_config(args.config)
    config = EvaluationConfig.from_dict(payload)
    config.runtime.max_new_tokens = args.max_new_tokens
    config.runtime.greedy = True
    config.runtime.temperature = 0.0
    config.runtime.top_k = 0
    config.runtime.top_p = 1.0

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    selected_model_names = (
        [name.strip() for name in args.model_names.split(",") if name.strip()]
        if args.model_names
        else [args.model_name]
    )
    available_model_names = {model.name for model in config.models}
    missing_models = [name for name in selected_model_names if name not in available_model_names]
    if missing_models:
        available = ", ".join(sorted(available_model_names))
        raise ValueError(f"Unknown models {missing_models!r}. Available: {available}")

    examples: list[dict[str, str]] = []
    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index >= args.max_examples:
                break
            examples.append(row)

    rows_out: list[dict[str, object]] = []
    for model_name in selected_model_names:
        spec = next(model for model in config.models if model.name == model_name)
        model = load_evaluation_model(spec, device=device)

        for row in examples:
            prompt = row["prompt"]
            gold_answer = row["gold_answer"]
            answer_choices = extract_answer_choices(prompt)
            for variant_name, variant_prompt in build_variants(prompt):
                tokenized = model.tokenizer(
                    [variant_prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=model.runtime_config.model_max_length,
                )
                prepared_batch = {
                    "input_ids": tokenized["input_ids"].to(device),
                    "attention_mask": tokenized["attention_mask"].to(device),
                }
                with torch.inference_mode():
                    prediction_text = _generate_predictions(
                        loaded_model=model,
                        prepared_batch=prepared_batch,
                        config=config.runtime,
                    )[0]
                parsed = parse_prediction(
                    prediction_text,
                    "answer_choice_text",
                    answer_choices=answer_choices,
                )
                rows_out.append(
                    {
                        "model_name": model_name,
                        "example_id": row["example_id"],
                        "variant": variant_name,
                        "gold_answer": gold_answer,
                        "normalized_gold_answer": normalize_answer_choice_text(gold_answer),
                        "prediction_text": prediction_text,
                        "parsed_prediction": parsed.value,
                        "prediction_valid": parsed.is_valid,
                        "correct": parsed.value == normalize_answer_choice_text(gold_answer) and parsed.is_valid,
                        "prediction_hash": hashlib.sha1(prediction_text.encode("utf-8")).hexdigest()[:10],
                    }
                )

    output_path = output_dir / "prompt_diagnostics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "example_id",
                "variant",
                "gold_answer",
                "normalized_gold_answer",
                "prediction_text",
                "parsed_prediction",
                "prediction_valid",
                "correct",
                "prediction_hash",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    summary_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows_out:
        grouped.setdefault((str(row["model_name"]), str(row["variant"])), []).append(row)

    for (model_name, variant), group_rows in sorted(grouped.items()):
        num_correct = sum(bool(row["correct"]) for row in group_rows)
        unique_outputs = len({str(row["prediction_hash"]) for row in group_rows})
        invalid = sum(not bool(row["prediction_valid"]) for row in group_rows)
        sample_output = str(group_rows[0]["prediction_text"])[:120].replace("\n", " ")
        summary_rows.append(
            {
                "model_name": model_name,
                "variant": variant,
                "num_examples": len(group_rows),
                "num_correct": num_correct,
                "accuracy": num_correct / max(len(group_rows), 1),
                "num_invalid": invalid,
                "unique_output_count": unique_outputs,
                "sample_output": sample_output,
            }
        )

    summary_path = output_dir / "prompt_diagnostics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "variant",
                "num_examples",
                "num_correct",
                "accuracy",
                "num_invalid",
                "unique_output_count",
                "sample_output",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_lines = [
        "# CODI Prompt Diagnostics",
        "",
        f"Examples tested: {len(examples)}",
        f"Models tested: {', '.join(selected_model_names)}",
        "",
        "| Model | Variant | Accuracy | Invalid | Unique outputs | Sample output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    str(row["variant"]),
                    f"{float(row['accuracy']):.2f}",
                    str(row["num_invalid"]),
                    str(row["unique_output_count"]),
                    str(row["sample_output"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    markdown_path = output_dir / "prompt_diagnostics_summary.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"Wrote diagnostics to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
