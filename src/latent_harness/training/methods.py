from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from latent_harness.core.runtime import LatentReasoningRuntime
from latent_harness.training.datasets import (
    make_standard_answer_only_data_module,
    make_standard_cot_data_module,
    make_supervised_data_module,
)

RuntimeBuilder = Callable[..., LatentReasoningRuntime]
DataModuleBuilder = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class MethodRecipe:
    key: str
    paper_name: str
    summary: str
    training_signal: str
    inference_path: str
    implemented: bool
    training_style: str = "latent_distillation"
    runtime_builder: RuntimeBuilder | None = None
    data_module_builder: DataModuleBuilder | None = None
    validation_focus: list[str] = field(default_factory=list)

    def assert_implemented(self) -> None:
        if self.runtime_builder is None or self.data_module_builder is None:
            raise NotImplementedError(
                f"Method {self.key!r} is tracked in the methodology but not wired into the "
                "trainer yet."
            )


METHOD_RECIPES: dict[str, MethodRecipe] = {
    "codi": MethodRecipe(
        key="codi",
        paper_name="CODI",
        summary="Self-distill explicit CoT into a latent student with hidden-state alignment.",
        training_signal="teacher CE + student CE + answer-position hidden-state distillation",
        inference_path="latent loop followed by answer decoding",
        implemented=True,
        runtime_builder=LatentReasoningRuntime,
        data_module_builder=make_supervised_data_module,
        validation_focus=[
            "answer-position alignment",
            "checkpoint compatibility",
            "latent generation parity",
        ],
    ),
    "cot_sft": MethodRecipe(
        key="cot_sft",
        paper_name="CoT-SFT",
        summary="Supervise explicit reasoning traces followed by the final answer.",
        training_signal="standard LM loss on question -> CoT -> answer",
        inference_path="standard autoregressive generation",
        implemented=True,
        training_style="standard_sft",
        runtime_builder=LatentReasoningRuntime,
        data_module_builder=make_standard_cot_data_module,
        validation_focus=[
            "final answer extraction",
            "prompt-to-trace continuation",
            "checkpoint compatibility",
        ],
    ),
    "coconut": MethodRecipe(
        key="coconut",
        paper_name="COCONUT",
        summary="Progressively replace explicit reasoning spans with continuous latent thoughts.",
        training_signal="curriculum over latent replacement schedule",
        inference_path="latent token rollout with curriculum-matched interface",
        implemented=False,
        validation_focus=["curriculum staging", "latent replacement schedule"],
    ),
    "sim_cot": MethodRecipe(
        key="sim_cot",
        paper_name="SIM-CoT",
        summary="Supervise each latent reasoning step with an auxiliary decoder during training.",
        training_signal="step-level latent supervision via auxiliary decoder",
        inference_path="drop auxiliary decoder and keep latent inference path",
        implemented=False,
        validation_focus=["latent diversity", "auxiliary supervision stability"],
    ),
    "colar": MethodRecipe(
        key="colar",
        paper_name="CoLaR",
        summary="Compress reasoning chains into controllable latent chunks with dynamic compression.",
        training_signal="compressed-embedding prediction and compression-factor conditioning",
        inference_path="compression-conditioned latent decoding",
        implemented=False,
        validation_focus=["compression ratio control", "reasoning quality under compression"],
    ),
    "no_cot_sft": MethodRecipe(
        key="no_cot_sft",
        paper_name="No-CoT-SFT",
        summary="Supervise the model only on the final answer without explicit reasoning text.",
        training_signal="standard LM loss on question -> answer",
        inference_path="standard autoregressive generation",
        implemented=True,
        training_style="standard_sft",
        runtime_builder=LatentReasoningRuntime,
        data_module_builder=make_standard_answer_only_data_module,
        validation_focus=[
            "answer-only formatting",
            "checkpoint compatibility",
            "generation cleanliness",
        ],
    ),
}


def get_method_recipe(name: str) -> MethodRecipe:
    try:
        return METHOD_RECIPES[name]
    except KeyError as exc:
        known = ", ".join(sorted(METHOD_RECIPES))
        raise KeyError(f"Unknown training method {name!r}. Known methods: {known}") from exc


def list_method_recipes() -> list[MethodRecipe]:
    return [METHOD_RECIPES[key] for key in sorted(METHOD_RECIPES)]
