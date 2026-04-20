"""SIM-CoT: auxiliary step-decoder supervision for latent reasoning.

Implements the InternLM/SIM-CoT recipe on top of ``LatentReasoningRuntime``:

- An ``AuxiliaryStepDecoder`` (full second LM copy, per paper; with a
  shared-lm_head fallback path for memory-tight regimes) predicts the
  explicit CoT step text that corresponds to each latent rollout step.
- ``SimCotLatentRuntime`` wraps the base latent runtime, attaches the aux
  decoder during training, and discards it on checkpoint save so the
  saved artifact matches the inference-time CODI contract.
- The loss composition is (per spec §4.2a):
  ``total = ce + 20 * distill + ref_ce + explain_loss_factor * (explain / max(1, effective_steps))``.
- Data preparation parses ``<< expr=value >>`` chunks out of the
  gsm8k_aug CoT string to produce per-step token sequences — a
  tokenizer-agnostic re-derivation of SIM-CoT's Llama-specific
  ``get_steps`` helper.

Refs:
- Paper: arXiv 2509.20317
- Repo: https://github.com/InternLM/SIM-CoT (CODI/src/model.py lines 305-748)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase

from latent_harness.core.config import (
    LatentRuntimeConfig,
    ModelConfig,
    resolve_hf_hub_token,
)
from latent_harness.core.runtime import (
    LatentReasoningRuntime,
    _apply_boundary_detach,
    _resolve_runtime_dtype,
    _resolve_should_detach,
)
from latent_harness.training.config import TrainingDataConfig
from latent_harness.training.datasets import (
    IGNORE_INDEX,
    SupervisedLatentDataCollator,
    SupervisedLatentDataset,
    _build_formatted_splits,
)

logger = logging.getLogger(__name__)

# Token ID we reserve for the "prefix" column that sits in front of each
# per-step explain-target sequence (see SIM-CoT ``model.py`` line 499 —
# they use -570 because it's guaranteed to never appear in vocab).
EXPLAIN_PREFIX_SENTINEL = -570


# -----------------------------------------------------------------------------
# Step-trace extraction (tokenizer-agnostic; see ``get_steps`` in SIM-CoT repo)
# -----------------------------------------------------------------------------

# gsm8k_aug CoT format: sequence of ``<<expr=value>>`` chunks separated by
# whitespace. We re-extract per step at the raw-text level so the extraction
# does not depend on a specific tokenizer's BPE choice for ``<<`` / ``>>``.
_STEP_CHUNK_RE = re.compile(r"<<[^<>]*?>>")


def extract_step_texts(cot: str) -> list[str]:
    """Return the ``<<...>>`` step chunks found in a gsm8k_aug CoT string.

    Example::

        >>> extract_step_texts("<<600*30/100=180>> <<600-240=360>>")
        ['<<600*30/100=180>>', '<<600-240=360>>']

    Returns an empty list when no step chunks are present (e.g., NL CoT).
    Falls back to a single-chunk of the stripped input in that case so the
    aux decoder still gets *some* supervision.
    """
    matches = _STEP_CHUNK_RE.findall(cot or "")
    if matches:
        return [match.strip() for match in matches]
    text = (cot or "").strip()
    if not text:
        return []
    return [text]


def build_step_token_lists(
    step_texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    eot_id: int,
    pad_id: int,
    *,
    latent_num: int,
) -> list[list[int]]:
    """Tokenize per-step explanation texts and pad/truncate to ``latent_num+1`` slots.

    Mirrors the ``len(steps) > max_steps`` / ``< max_steps`` logic of
    SIM-CoT's ``get_steps``: keep the first ``max_steps - 1`` steps as-is,
    merge the tail into the last slot, and pad slots with ``[pad_id]``
    when we are short. Every kept slot ends with ``eot_id``.
    """
    max_steps = max(1, latent_num)
    tokenized: list[list[int]] = []
    for text in step_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        tokenized.append(list(ids) + [eot_id])

    if len(tokenized) > max_steps:
        kept = tokenized[: max_steps - 1]
        tail = tokenized[max_steps - 1 :]
        merged: list[int] = []
        for seq in tail:
            if seq and seq[-1] == eot_id:
                merged.extend(seq[:-1])
            else:
                merged.extend(seq)
        merged.append(eot_id)
        kept.append(merged)
        tokenized = kept
    while len(tokenized) < max_steps:
        tokenized.append([pad_id])
    return tokenized


def _max_step_length(padded_steps_batch: Sequence[Sequence[Sequence[int]]]) -> int:
    return max(
        (len(step) for steps in padded_steps_batch for step in steps),
        default=1,
    )


def pad_step_batch(
    step_lists: list[list[list[int]]],
    pad_id: int,
) -> list[list[list[int]]]:
    """Pad a ragged ``[batch, n_steps, step_len]`` tensor on both inner axes.

    Matches the behavior of SIM-CoT's ``pad_steps`` helper.
    """
    if not step_lists:
        return []
    s_max = max(len(steps) for steps in step_lists)
    l_max = _max_step_length(step_lists)
    out: list[list[list[int]]] = []
    for steps in step_lists:
        padded_steps: list[list[int]] = []
        for step in steps:
            cur = list(step)
            pad_len = l_max - len(cur)
            if pad_len > 0:
                cur = cur + [pad_id] * pad_len
            padded_steps.append(cur)
        while len(padded_steps) < s_max:
            padded_steps.append([pad_id] * l_max)
        out.append(padded_steps)
    return out


def dedup_trailing_pads(rows: list[list[int]], pad_id: int) -> list[list[int]]:
    """Trim all-padding columns from the right side of a ``[batch, len]`` block.

    Drops a column only if every row has the pad token in it (ignoring the
    last column so we keep at least one genuine step boundary token).
    Matches the behavior of SIM-CoT's ``dedup_trailing_pads``.
    """
    if not rows:
        return rows
    max_len = len(rows[0])
    while max_len > 1:
        if all(row[max_len - 2] == pad_id for row in rows):
            max_len -= 1
        else:
            break
    return [row[:max_len] for row in rows]


# -----------------------------------------------------------------------------
# Auxiliary step decoder
# -----------------------------------------------------------------------------


class AuxiliaryStepDecoder(nn.Module):
    """Per-latent-step decoder that predicts the gold explicit step text.

    Two implementations are supported:

    1. ``full_lm=True`` (paper default): load a second copy of the base
       language model, resize its embedding table to match the latent
       runtime, and run autoregressive decoding over
       ``concat([latent_embd, explain_embds])``. Memory: ~2x base model.
    2. ``full_lm=False`` (shared-head fallback): reuse the base runtime's
       ``lm_head`` plus a small stack of fresh transformer layers. Memory
       overhead is small but the fallback does NOT match the paper
       numerically — document the deviation in the writeup.

    The decoder is intentionally *not* wrapped with LoRA in either path;
    SIM-CoT's reference config trains the aux decoder at full precision
    within the LoRA training loop.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        runtime_config: LatentRuntimeConfig,
        base_hidden_size: int,
        pad_token_id: int,
        bot_id: int,
        eot_id: int,
        runtime_dtype: torch.dtype,
        base_lm_head: nn.Module | None = None,
        base_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.full_lm = runtime_config.aux_decoder_full_lm
        self.pad_token_id = pad_token_id
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.runtime_dtype = runtime_dtype

        if self.full_lm:
            quantization_config = None
            if model_config.load_in_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                )
            token = resolve_hf_hub_token(model_config.hf_token)
            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_config.base_model_name_or_path,
                token=token,
                torch_dtype=runtime_dtype if model_config.full_precision else None,
                quantization_config=quantization_config,
            )
            # Resize embeddings to match the runtime vocab (base + {pad, bot, eot}).
            new_vocab_size = eot_id + 1
            self.decoder.resize_token_embeddings(new_vocab_size)
            self.projector_in = nn.Identity()
            self.projector_out = nn.Identity()
        else:
            # Shared-head fallback: a tiny transformer block tower wired to the
            # base model's lm_head. Runs the aux CE through the same lm_head as
            # the latent runtime uses, so the approximation costs only
            # attention blocks.
            if base_lm_head is None or base_embedding is None:
                raise ValueError(
                    "Shared-head fallback requires base_lm_head and base_embedding."
                )
            num_layers = max(1, runtime_config.aux_decoder_shared_head_layers)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=base_hidden_size,
                nhead=max(4, base_hidden_size // 128),
                dim_feedforward=base_hidden_size * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.shared_head_tower = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.shared_head_tower.to(dtype=runtime_dtype)
            # We deliberately do NOT register these as parameters of
            # AuxiliaryStepDecoder to avoid double-counting weights; the
            # runtime owns them and we simply reference them.
            self._shared_lm_head = base_lm_head
            self._shared_embedding = base_embedding
            self.decoder = None
            self.projector_in = nn.Identity()
            self.projector_out = nn.Identity()

    @property
    def parameters_owned(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_input_embedding_layer(self) -> nn.Module:
        if self.full_lm:
            base = self.decoder
            if hasattr(base, "get_input_embeddings"):
                return base.get_input_embeddings()
            return base.model.embed_tokens
        return self._shared_embedding

    def forward_step(
        self,
        *,
        latent_embd: torch.Tensor,
        step_token_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one aux-decoder step and return ``(logits, labels)``.

        ``latent_embd`` has shape ``[batch, 1, hidden]`` and is prepended to
        the embedded ``step_token_ids``. Labels correspond to predicting
        ``step_token_ids`` (the prefix column is masked via the ``-100``
        label) shifted left by one in the main loss call site.
        """
        batch_size = step_token_ids.size(0)
        device = step_token_ids.device
        prefix_sentinel = torch.full(
            (batch_size, 1),
            EXPLAIN_PREFIX_SENTINEL,
            dtype=step_token_ids.dtype,
            device=device,
        )
        indices_with_prefix = torch.cat([prefix_sentinel, step_token_ids], dim=1)
        attention_mask = (indices_with_prefix != self.pad_token_id).long()

        embed_layer = self.get_input_embedding_layer()
        step_embeds = embed_layer(step_token_ids)
        if latent_embd.dtype != step_embeds.dtype:
            latent_embd = latent_embd.to(dtype=step_embeds.dtype)
        explain_embeds = torch.cat([latent_embd, step_embeds], dim=1)

        if self.full_lm:
            outputs = self.decoder(
                inputs_embeds=explain_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            logits = outputs.logits
        else:
            # Build a causal key_padding_mask for TransformerEncoder; it
            # expects ``True`` for positions that should be masked.
            key_padding_mask = attention_mask == 0
            seq_len = explain_embeds.size(1)
            attn_causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            tower_out = self.shared_head_tower(
                explain_embeds,
                mask=attn_causal_mask,
                src_key_padding_mask=key_padding_mask,
            )
            logits = self._shared_lm_head(tower_out)

        labels = indices_with_prefix.clone()
        labels = torch.where(
            (labels == EXPLAIN_PREFIX_SENTINEL) | (labels == self.pad_token_id),
            torch.full_like(labels, IGNORE_INDEX),
            labels,
        )
        return logits, labels


# -----------------------------------------------------------------------------
# Dataset augmentation: attach per-example step_tokens (padded ``[n_steps, L]``)
# -----------------------------------------------------------------------------


class SimCotLatentDataset(SupervisedLatentDataset):
    """``SupervisedLatentDataset`` extended with per-example step token lists.

    Step texts are re-extracted from the stored CoT string at dataset
    construction time (they are discarded from ``SupervisedLatentDataset``
    after tokenization of the teacher sequence). This keeps the
    SIM-CoT-specific logic co-located with the SIM-CoT dataset class and
    avoids touching the base dataset pipeline.
    """

    def __init__(
        self,
        *,
        formatted_examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        runtime_config: LatentRuntimeConfig,
        bot_id: int,
        eot_id: int,
        pad_id: int,
    ) -> None:
        self.pad_id = pad_id
        # Re-extract step texts before we forward to the base init (which
        # discards the raw CoT).
        # We store per-example step token id lists keyed by example hash so
        # indexing order is preserved.
        step_tokens_by_hash: dict[str, list[list[int]]] = {}
        for example in formatted_examples:
            step_texts = extract_step_texts(str(example.get("cot") or ""))
            tokens = build_step_token_lists(
                step_texts,
                tokenizer=tokenizer,
                eot_id=eot_id,
                pad_id=pad_id,
                latent_num=runtime_config.num_latent + 1,
            )
            step_tokens_by_hash[str(example.get("example_hash"))] = tokens
        super().__init__(
            formatted_examples=formatted_examples,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            bot_id=bot_id,
            eot_id=eot_id,
        )
        # Attach step_tokens to each preprocessed record by example hash.
        for example, record in zip(formatted_examples, self.examples):
            record["step_tokens"] = step_tokens_by_hash[
                str(example.get("example_hash"))
            ]


@dataclass(slots=True)
class SimCotLatentDataCollator:
    """Wraps ``SupervisedLatentDataCollator`` and adds a batched ``step_tokens`` field.

    Composition-over-inheritance sidesteps the ``super()`` breakage that
    ``@dataclass(slots=True)`` causes with multi-level inheritance.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_id: int = 0

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        inner = SupervisedLatentDataCollator(tokenizer=self.tokenizer)
        batch = inner(instances)
        step_tokens_per_example = [
            list(item.get("step_tokens") or []) for item in instances
        ]
        padded = pad_step_batch(step_tokens_per_example, pad_id=self.pad_id)
        batch["step_tokens"] = padded
        return batch


# -----------------------------------------------------------------------------
# SIM-CoT runtime: wraps LatentReasoningRuntime, adds aux decoder.
# -----------------------------------------------------------------------------


class SimCotLatentRuntime(LatentReasoningRuntime):
    """``LatentReasoningRuntime`` + SIM-CoT auxiliary step decoder.

    The aux decoder is attached only when ``train_mode=True`` AND
    ``runtime_config.aux_decoder_enabled=True``. On save we delete
    ``self.aux_decoder`` so downstream eval / checkpoint loaders see the
    plain CODI interface (per the SIM-CoT paper §3.2: "drop the auxiliary
    decoder at inference time").
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: LatentRuntimeConfig,
        *,
        train_mode: bool,
    ) -> None:
        super().__init__(model_config, runtime_config, train_mode=train_mode)
        self._sim_cot_train_mode = bool(
            train_mode and runtime_config.aux_decoder_enabled
        )
        self.aux_decoder: AuxiliaryStepDecoder | None = None
        if self._sim_cot_train_mode:
            hidden_size = int(self.model.config.hidden_size) if hasattr(
                self.model, "config"
            ) else self.get_input_embedding_layer().embedding_dim
            self.aux_decoder = AuxiliaryStepDecoder(
                model_config=model_config,
                runtime_config=runtime_config,
                base_hidden_size=hidden_size,
                pad_token_id=self.pad_token_id,
                bot_id=self.bot_id,
                eot_id=self.eot_id,
                runtime_dtype=self.runtime_dtype,
                base_lm_head=self._get_base_lm_head(),
                base_embedding=self.get_input_embedding_layer(),
            )
            logger.info(
                "SIM-CoT aux decoder attached full_lm=%s explain_loss_factor=%.4f",
                runtime_config.aux_decoder_full_lm,
                runtime_config.aux_decoder_explain_loss_factor,
            )

    # ---- aux-decoder lifecycle helpers --------------------------------------

    def _get_base_lm_head(self) -> nn.Module:
        base = self.model
        if hasattr(base, "get_base_model"):
            try:
                base = base.get_base_model()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(base, "lm_head"):
            return base.lm_head
        if hasattr(base, "model") and hasattr(base.model, "lm_head"):
            return base.model.lm_head
        raise AttributeError("Could not locate base lm_head for shared-head aux decoder")

    def drop_aux_decoder(self) -> None:
        """Detach and discard the auxiliary decoder for checkpoint save."""
        if self.aux_decoder is not None:
            logger.info("Dropping SIM-CoT aux decoder before checkpoint save")
        self.aux_decoder = None
        self._sim_cot_train_mode = False

    def state_dict(self, *args, **kwargs):  # noqa: D401 - match nn.Module
        state = super().state_dict(*args, **kwargs)
        # Strip aux_decoder.* keys from the saved state dict so the
        # on-disk artifact matches the CODI inference contract.
        stripped = type(state)((k, v) for k, v in state.items() if not k.startswith("aux_decoder."))
        return stripped

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("aux_decoder.")}
        # With strict=False we tolerate a saved CODI state_dict with no aux
        # decoder loading into a runtime that currently has one attached.
        return super().load_state_dict(filtered, strict=False)

    # ---- core forward -------------------------------------------------------

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
        collect_diagnostics: bool = False,
        step_tokens: Any = None,
    ) -> dict[str, Any]:
        """Extended CODI forward with per-step auxiliary decoder supervision.

        When ``aux_decoder`` is attached and ``step_tokens`` are provided,
        we compute an additional ``explain_loss`` term from per-step
        autoregressive CE against the aux decoder, add it (weighted and
        normalized by effective step count) to the total loss, and return
        it under the ``explain_loss`` key.
        """
        base_outputs = super().forward(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=labels,
            encoder_attention_mask=encoder_attention_mask,
            ref_answer_position=ref_answer_position,
            model_answer_position=model_answer_position,
            ref_attention_mask=ref_attention_mask,
            ref_labels=ref_labels,
            step=step,
            step_ratio=step_ratio,
            collect_diagnostics=collect_diagnostics,
        )
        if self.aux_decoder is None or step_tokens is None:
            base_outputs.setdefault("explain_loss", 0.0)
            return base_outputs

        # We need a *second* latent rollout (the base forward discards the
        # per-step latents for the aux-decoder step loss). This is the
        # straight SIM-CoT pattern — they actually run one combined loop
        # that emits both. We factor it as two passes to keep the base
        # runtime unchanged.
        explain_loss_total, effective_steps_cnt = self._compute_explain_loss(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            step_tokens=step_tokens,
        )
        if effective_steps_cnt == 0:
            weighted = torch.tensor(0.0, device=encoder_input_ids.device)
        else:
            weighted = (
                explain_loss_total
                * self.runtime_config.aux_decoder_explain_loss_factor
                / float(max(1, effective_steps_cnt))
            )
        base_outputs["loss"] = base_outputs["loss"] + weighted
        base_outputs["explain_loss"] = float(weighted.detach().cpu())
        base_outputs["explain_loss_raw"] = float(
            explain_loss_total.detach().cpu()
            if isinstance(explain_loss_total, torch.Tensor)
            else float(explain_loss_total)
        )
        base_outputs["explain_effective_steps"] = effective_steps_cnt
        return base_outputs

    def _compute_explain_loss(
        self,
        *,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        step_tokens: Any,
    ) -> tuple[torch.Tensor, int]:
        """Run a dedicated latent rollout that emits per-step explain CE.

        ``step_tokens`` comes from the collator as a Python nested list
        ``[batch, n_steps, step_len]`` (already padded by
        ``pad_step_batch``). We re-pad per step before tensorizing so the
        batch is rectangular.
        """
        if not isinstance(step_tokens, list) or not step_tokens:
            return torch.tensor(0.0, device=encoder_input_ids.device), 0

        batch_size = len(step_tokens)
        n_steps = max(len(s) for s in step_tokens)
        device = encoder_input_ids.device
        pad_id = self.pad_token_id

        past_key_values, latent = self.encode_question(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )
        total = torch.tensor(0.0, device=device)
        effective_cnt = 0
        num_latent = self.runtime_config.num_latent

        def _run_aux_step(forward_idx: int, current_latent: torch.Tensor) -> None:
            nonlocal total, effective_cnt
            rows = []
            for b in range(batch_size):
                row = list(step_tokens[b][forward_idx]) if forward_idx < len(
                    step_tokens[b]
                ) else [pad_id]
                rows.append(row or [pad_id])
            rows = dedup_trailing_pads(rows, pad_id=pad_id)
            indices = torch.tensor(rows, dtype=torch.long, device=device)
            logits, labels_tensor = self.aux_decoder.forward_step(
                latent_embd=current_latent,
                step_token_ids=indices,
            )
            if (labels_tensor != IGNORE_INDEX).sum().item() == 0:
                return
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels_tensor[..., 1:].contiguous().view(-1)
            if (shift_labels != IGNORE_INDEX).sum().item() == 0:
                return
            step_loss = self.loss_fct(shift_logits.float(), shift_labels)
            total = total + step_loss
            effective_cnt += 1

        # Step 0: before any latent rollout — use the question-tail latent.
        if n_steps > 0:
            _run_aux_step(0, latent)

        for latent_index in range(num_latent):
            forward_idx = latent_index + 1
            if forward_idx >= n_steps:
                # We've already consumed all supplied step targets; no more
                # aux forward passes are needed, so we can exit early.
                break
            latent_outputs = self.model(
                inputs_embeds=latent,
                token_type_ids=self.build_token_type_ids(inputs_embeds=latent),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = latent_outputs.past_key_values
            latent = latent_outputs.hidden_states[-1][:, -1:, :]
            latent = self.maybe_project(latent)
            if latent_index < num_latent - 1:
                should_detach = _resolve_should_detach(
                    latent_index=latent_index,
                    num_latent=num_latent,
                    keep_last_k=self.runtime_config.detach_keep_last_k,
                )
                if should_detach:
                    past_key_values, latent = _apply_boundary_detach(
                        cache=past_key_values,
                        latent=latent,
                        encoder_length=encoder_input_ids.size(1),
                        runtime_config_detach_latent=self.runtime_config.detach_latent_between_steps,
                        runtime_config_detach_cache=self.runtime_config.detach_cache_between_steps,
                        detach_position_mode=self.runtime_config.detach_position_mode,
                    )
            _run_aux_step(forward_idx, latent)

        return total, effective_cnt


__all__ = [
    "AuxiliaryStepDecoder",
    "SimCotLatentDataCollator",
    "SimCotLatentDataset",
    "SimCotLatentRuntime",
    "build_step_token_lists",
    "dedup_trailing_pads",
    "extract_step_texts",
    "make_sim_cot_data_module",
    "pad_step_batch",
]


def make_sim_cot_data_module(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_config: TrainingDataConfig,
    runtime_config: LatentRuntimeConfig,
    bot_id: int,
    eot_id: int,
) -> dict[str, Any]:
    """Build train/eval datasets + collator that expose per-example ``step_tokens``.

    Reuses ``_build_formatted_splits`` (shared with the CODI path) so data
    filtering and deterministic shuffling match the other methods.
    """
    train_examples, eval_examples = _build_formatted_splits(
        tokenizer=tokenizer,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_dataset = SimCotLatentDataset(
        formatted_examples=train_examples,
        tokenizer=tokenizer,
        runtime_config=runtime_config,
        bot_id=bot_id,
        eot_id=eot_id,
        pad_id=pad_id,
    )
    eval_dataset = (
        SimCotLatentDataset(
            formatted_examples=eval_examples,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            bot_id=bot_id,
            eot_id=eot_id,
            pad_id=pad_id,
        )
        if eval_examples
        else None
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": SimCotLatentDataCollator(tokenizer=tokenizer, pad_id=pad_id),
    }
