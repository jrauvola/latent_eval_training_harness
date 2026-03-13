from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from latent_harness.core.config import LatentRuntimeConfig, ModelConfig


def get_lora_target_modules(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if any(name in lowered for name in ("llama", "mistral", "falcon", "qwen")):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    if "phi" in lowered:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    if "gpt2" in lowered:
        return ["c_attn", "c_proj", "c_fc"]
    raise ValueError(f"Unsupported model family for LoRA mapping: {model_name}")


def get_modules_to_save(model_name: str) -> list[str]:
    if "gpt2" in model_name.lower():
        return ["wte", "lm_head"]
    return ["embed_tokens", "lm_head"]


class LatentReasoningRuntime(nn.Module):
    """Shared latent runtime used by both training and evaluation.

    The current implementation preserves the CODI-compatible latent interface and
    checkpoint format so the harness can split training/evaluation concerns
    without breaking existing runs.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: LatentRuntimeConfig,
        *,
        train_mode: bool,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.train_mode = train_mode

        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if runtime_config.bf16 else torch.float16
        else:
            torch_dtype = torch.float32
        self.runtime_dtype = torch_dtype

        quantization_config = None
        if model_config.load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_name_or_path,
            token=model_config.hf_token,
            torch_dtype=torch_dtype if model_config.full_precision else None,
            quantization_config=quantization_config,
        )

        original_vocab_size = self.model.config.vocab_size
        self.pad_token_id = original_vocab_size
        self.bot_id = original_vocab_size + 1
        self.eot_id = original_vocab_size + 2
        self.model.resize_token_embeddings(original_vocab_size + 3)

        if model_config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=not train_mode,
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=get_lora_target_modules(model_config.base_model_name_or_path),
                modules_to_save=get_modules_to_save(model_config.base_model_name_or_path),
                init_lora_weights=model_config.lora_init,
            )
            self.model = get_peft_model(self.model, lora_config)

        if runtime_config.use_prj:
            blocks: list[nn.Module] = [
                nn.Dropout(runtime_config.prj_dropout),
                nn.Linear(self.model.config.hidden_size, runtime_config.prj_dim),
                nn.GELU(),
                nn.Linear(runtime_config.prj_dim, self.model.config.hidden_size),
            ]
            if not runtime_config.prj_no_ln:
                blocks.append(nn.LayerNorm(self.model.config.hidden_size))
            self.prj = nn.Sequential(*blocks).to(dtype=self.runtime_dtype)
        else:
            self.prj = nn.Identity()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if runtime_config.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif runtime_config.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported distillation loss {runtime_config.distill_loss_type}")

    @property
    def codi(self) -> nn.Module:
        """Compatibility alias for legacy eval code paths."""
        return self.model

    def build_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name_or_path,
            token=self.model_config.hf_token,
            model_max_length=self.runtime_config.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token_id = self.pad_token_id
        return tokenizer

    def get_input_embedding_layer(self) -> nn.Module:
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        model_name = self.model_config.base_model_name_or_path.lower()
        if "pythia" in model_name:
            return base.gpt_neox.embed_in
        if "gpt2" in model_name:
            return base.transformer.wte
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens
        if hasattr(base, "embed_tokens"):
            return base.embed_tokens
        raise AttributeError("Could not locate input embedding layer")

    def maybe_project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.prj(hidden_state) if self.runtime_config.use_prj else hidden_state

    def encode_question(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[Any, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        latent = outputs.hidden_states[-1][:, -1:, :]
        return past_key_values, self.maybe_project(latent)

    def iterate_latent_steps(
        self,
        past_key_values: Any,
        latent: torch.Tensor,
        num_steps: int,
    ) -> tuple[Any, torch.Tensor]:
        for _ in range(num_steps):
            outputs = self.model(
                inputs_embeds=latent,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent = outputs.hidden_states[-1][:, -1:, :]
            latent = self.maybe_project(latent)
        return past_key_values, latent

    def build_eot_embeds(
        self,
        batch_size: int,
        device: torch.device,
        *,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        token_ids = [self.eot_id]
        if not self.runtime_config.remove_eos and eos_token_id is not None:
            token_ids.append(eos_token_id)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        embeds = self.get_input_embedding_layer()(token_tensor).unsqueeze(0)
        return embeds.expand(batch_size, -1, -1)

    def generate_from_latent(
        self,
        *,
        tokenizer: AutoTokenizer,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        inf_latent_iterations: int,
        max_new_tokens: int,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[str]:
        device = input_ids.device
        batch_size = input_ids.size(0)
        past_key_values, latent = self.encode_question(input_ids=input_ids, attention_mask=attention_mask)
        past_key_values, _ = self.iterate_latent_steps(
            past_key_values=past_key_values,
            latent=latent,
            num_steps=inf_latent_iterations,
        )

        next_embeds = self.build_eot_embeds(
            batch_size=batch_size,
            device=device,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self.model(
                inputs_embeds=next_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, : self.model.config.vocab_size - 1]
            token_ids = self._sample_tokens(
                logits=logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            for batch_index, token_id in enumerate(token_ids.tolist()):
                if finished[batch_index]:
                    continue
                predictions[batch_index].append(token_id)
                if token_id == tokenizer.eos_token_id:
                    finished[batch_index] = True
            if bool(finished.all()):
                break
            next_embeds = self.get_input_embedding_layer()(token_ids).unsqueeze(1)

        return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in predictions]

    def _sample_tokens(
        self,
        *,
        logits: torch.Tensor,
        greedy: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.LongTensor:
        if greedy:
            return torch.argmax(logits, dim=-1)
        working = logits / max(temperature, 1e-5)
        if top_k > 0:
            top_values, _ = torch.topk(working, min(top_k, working.shape[-1]), dim=-1)
            cutoff = top_values[:, -1].unsqueeze(-1)
            working = torch.where(working < cutoff, torch.full_like(working, float("-inf")), working)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(working, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove_mask = cumulative > top_p
            remove_mask = torch.roll(remove_mask, shifts=1, dims=-1)
            remove_mask[:, 0] = False
            filtered = working.clone()
            for row_index in range(filtered.size(0)):
                filtered[row_index, sorted_indices[row_index, remove_mask[row_index]]] = float("-inf")
            working = filtered
        probs = torch.softmax(working, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def tie_weights_if_needed(self) -> None:
        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
            except KeyError:
                return

    def forward(
        self,
        *,
        encoder_input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        ref_input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        ref_answer_position: torch.LongTensor,
        model_answer_position: torch.LongTensor,
        ref_attention_mask: torch.LongTensor,
        ref_labels: torch.LongTensor,
        step: int | None = None,
        step_ratio: float | None = None,
    ) -> dict[str, Any]:
        del step, step_ratio
        past_key_values, latent = self.encode_question(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        with torch.no_grad():
            teacher_outputs = self.model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                output_hidden_states=True,
            )
        teacher_outputs_with_grad = self.model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            output_hidden_states=True,
        )

        student_logits = None
        distill_total = torch.tensor(0.0, device=encoder_input_ids.device)
        ce_total = torch.tensor(0.0, device=encoder_input_ids.device)

        for latent_index in range(self.runtime_config.num_latent):
            latent_outputs = self.model(
                inputs_embeds=latent,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_outputs.past_key_values
            latent = latent_outputs.hidden_states[-1][:, -1:, :]
            latent = self.maybe_project(latent)

            if latent_index != self.runtime_config.num_latent - 1:
                continue

            decoder_embeds = self.get_input_embedding_layer()(decoder_input_ids)
            student_outputs = self.model(
                inputs_embeds=decoder_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            student_logits = student_outputs.logits

            layer_losses: list[torch.Tensor] = []
            for student_layer, teacher_layer in zip(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states,
            ):
                teacher_selected = teacher_layer.gather(
                    1,
                    ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, teacher_layer.size(-1)),
                )
                student_selected = student_layer.gather(
                    1,
                    model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, student_layer.size(-1)
                    ),
                )
                loss_piece = self.distill_loss_fct(student_selected, teacher_selected.detach())
                if self.runtime_config.distill_loss_div_std:
                    loss_piece = loss_piece / teacher_selected.std().clamp_min(1e-6)
                layer_losses.append(loss_piece)
            distill_total = torch.stack(layer_losses).mean() * self.runtime_config.distill_loss_factor

            shifted_logits = student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))
            shifted_labels = labels[:, 1:].reshape(-1)
            ce_total = self.loss_fct(shifted_logits, shifted_labels)

        ref_logits = teacher_outputs_with_grad.logits
        shifted_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
        shifted_ref_labels = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(shifted_ref_logits, shifted_ref_labels) * self.runtime_config.ref_loss_factor

        total_loss = ce_total + distill_total + ref_ce_loss
        return {
            "loss": total_loss,
            "logits": student_logits,
            "ce_loss": float(ce_total.detach().cpu()),
            "distill_loss": float(distill_total.detach().cpu()),
            "ref_ce_loss": float(ref_ce_loss.detach().cpu()),
            "config": {
                "model": asdict(self.model_config),
                "runtime": asdict(self.runtime_config),
            },
        }
