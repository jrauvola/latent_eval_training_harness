from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from latent_harness.core.io import load_yaml_config
from latent_harness.evaluation.config import EvaluationConfig, EvaluationRuntimeConfig
from latent_harness.evaluation.models import EvaluationModelHandle, load_evaluation_model
from latent_harness.evaluation.runner import _generate_predictions


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=128, alias="max_completion_tokens")
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    stream: bool = False
    stop: str | list[str] | None = None

    model_config = {"populate_by_name": True}


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model_spec(config_path: str, model_name: str):
    payload = load_yaml_config(config_path)
    config = EvaluationConfig.from_dict(payload)
    for spec in config.models:
        if spec.name == model_name:
            return spec
    available = ", ".join(spec.name for spec in config.models)
    raise ValueError(f"Model {model_name!r} not found in {config_path}. Available: {available}")


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    normalized_messages = [
        {"role": message.role, "content": flatten_content(message.content)} for message in messages
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    prompt_parts: list[str] = []
    for message in normalized_messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        if not content:
            continue
        label = role.capitalize() if role else "User"
        prompt_parts.append(f"{label}: {content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def apply_stop_sequences(text: str, stop: str | list[str] | None) -> str:
    if stop is None:
        return text
    candidates = [stop] if isinstance(stop, str) else [item for item in stop if item]
    cutoff: int | None = None
    for candidate in candidates:
        index = text.find(candidate)
        if index != -1:
            cutoff = index if cutoff is None else min(cutoff, index)
    return text if cutoff is None else text[:cutoff]


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    tokenized = tokenizer(text, add_special_tokens=False)
    return len(tokenized.get("input_ids", []))


class LocalOpenAIServer:
    def __init__(
        self,
        *,
        config_path: str,
        model_name: str,
        device_name: str,
        external_model_id: str | None = None,
    ) -> None:
        self.spec = load_model_spec(config_path, model_name)
        self.device = resolve_device(device_name)
        self.loaded_model = load_evaluation_model(self.spec, device=self.device)
        self.model_id = external_model_id or f"openai/{self.spec.name}"

    def generate(self, request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this local adapter")

        prompt = build_prompt(self.loaded_model.tokenizer, request.messages)
        tokenized = self.loaded_model.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.loaded_model.runtime_config.model_max_length,
        )
        prepared_batch = {
            "input_ids": tokenized["input_ids"].to(self.device),
            "attention_mask": tokenized["attention_mask"].to(self.device),
        }

        runtime_config = EvaluationRuntimeConfig(
            max_new_tokens=request.max_tokens or 128,
            greedy=(request.temperature or 0.0) <= 0.0,
            temperature=max(request.temperature or 0.0, 1e-5),
            top_p=request.top_p or 1.0,
        )
        with torch.inference_mode():
            text = _generate_predictions(
                loaded_model=self.loaded_model,
                prepared_batch=prepared_batch,
                config=runtime_config,
            )[0]
        text = apply_stop_sequences(text, request.stop).strip()

        prompt_tokens = count_tokens(self.loaded_model.tokenizer, prompt)
        completion_tokens = count_tokens(self.loaded_model.tokenizer, text)
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model or self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def list_models(self) -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "latent-harness",
                    "metadata": {
                        "spec_name": self.spec.name,
                        "model_kind": self.spec.model_kind,
                        "inference_strategy": self.spec.inference_strategy,
                        "base_model_name_or_path": self.spec.model.base_model_name_or_path,
                    },
                }
            ],
        }


def create_app(
    *,
    config_path: str,
    model_name: str,
    device_name: str,
    external_model_id: str | None = None,
) -> FastAPI:
    server = LocalOpenAIServer(
        config_path=config_path,
        model_name=model_name,
        device_name=device_name,
        external_model_id=external_model_id,
    )
    app = FastAPI(title="Latent Harness OpenAI Adapter", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": server.spec.name}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return server.list_models()

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        return server.generate(request)

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "latent-harness-openai-adapter",
            "model": server.spec.name,
            "config": asdict(server.spec),
        }

    return app
