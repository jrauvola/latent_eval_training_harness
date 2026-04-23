"""Instrumentation helpers for Phase 1 latent-reasoning eval.

This module implements two interpretability hooks that piggyback on the
LatentReasoningRuntime forward pass used during evaluation:

* :func:`generate_from_latent_with_taps` — a drop-in variant of
  ``LatentReasoningRuntime.generate_from_latent`` that additionally returns the
  per-latent-step hidden states and KV-cache snapshots. It is used by the eval
  runner to produce decoded-latent traces and KV dumps without a second forward
  pass.

* :func:`project_hidden_to_topk` — logit-lens projection utility. Projects a
  latent-step hidden state through the model's ``lm_head`` and returns the
  top-K token ids / probabilities / strings.

* :func:`extract_kv_at_latent_positions` — extracts the (k, v) tensors at the
  final-latent positions of a cache returned from the latent rollout, returning
  per-example tensors suitable for ``.npy`` dump.

None of these helpers mutate runtime state: they operate on tensors returned
from stock runtime forward passes.

See `docs/superpowers/specs/2026-04-19-qwen3-eval-and-comparison-pipeline-design.md`
§4 Phase 1 for the deliverable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class LatentStepTrace:
    """Per-latent-step snapshot captured during a tapped forward pass.

    ``hidden_state`` is the final-layer hidden state at the newly-emitted
    latent position, shape ``[batch, hidden_size]`` (the ``seq_len=1`` axis
    has been squeezed).
    """

    latent_step: int
    hidden_state: torch.Tensor  # [batch, hidden_size]


@dataclass(slots=True)
class GenerationWithTaps:
    """Return value of :func:`generate_from_latent_with_taps`."""

    predictions: list[str]
    latent_traces: list[LatentStepTrace]
    #: Final cache after the last latent step (so downstream code can slice
    #: the KV-tensors at latent positions).
    final_cache: Any
    #: Length of the encoder (question+BOT) prefix, so KV slicing downstream
    #: can find where latent positions begin in the cache's sequence axis.
    encoder_prefix_length: int
    #: Number of latent steps actually iterated through (may be zero when
    #: ``skip_latent_injection`` is True or the configured value is 0).
    num_latent_iterated: int


def _get_lm_head(runtime: Any) -> torch.nn.Module:
    """Locate the language-model head on a (possibly-PEFT-wrapped) CausalLM."""

    model = runtime.model
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
    else:
        base = model
    for attr in ("lm_head", "embed_out", "output_projection"):
        candidate = getattr(base, attr, None)
        if candidate is not None:
            return candidate
    # Fall back to the HF get_output_embeddings protocol.
    if hasattr(base, "get_output_embeddings"):
        out = base.get_output_embeddings()
        if out is not None:
            return out
    raise AttributeError("Could not locate lm_head on model for logit-lens projection")


def project_hidden_to_topk(
    runtime: Any,
    hidden_state: torch.Tensor,
    tokenizer,
    *,
    top_k: int = 10,
) -> list[list[dict[str, Any]]]:
    """Project a final-layer hidden state through ``lm_head`` and return top-K.

    ``hidden_state`` is ``[batch, hidden_size]``. Returns a list of length
    ``batch``, each element a list of ``top_k`` dicts with keys
    ``token_id`` / ``prob`` / ``token_str``.
    """

    lm_head = _get_lm_head(runtime)
    lm_head_dtype = next(lm_head.parameters()).dtype
    with torch.no_grad():
        logits = lm_head(hidden_state.to(dtype=lm_head_dtype, device=next(lm_head.parameters()).device))
    # Strip CODI special tokens (pad/bot/eot); those are the last 3 appended entries.
    valid_vocab = getattr(runtime, "eot_id", logits.size(-1))
    logits = logits[..., :valid_vocab]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs.topk(k=top_k, dim=-1)
    batch_rows: list[list[dict[str, Any]]] = []
    for row_probs, row_ids in zip(top_probs.tolist(), top_ids.tolist()):
        row: list[dict[str, Any]] = []
        for token_id, prob in zip(row_ids, row_probs):
            try:
                token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            except Exception:
                token_str = ""
            row.append({"token_id": int(token_id), "prob": float(prob), "token_str": token_str})
        batch_rows.append(row)
    return batch_rows


def perturb_latent_kv_inplace(
    cache: Any,
    *,
    encoder_prefix_length: int,
    num_latent: int,
    noise_sigma: float,
    seed: int,
) -> None:
    """F6: in-place Gaussian perturbation of K/V at latent positions.

    For each layer's K and V tensors (shape ``[batch, num_heads, seq_len, head_dim]``),
    slice the latent positions ``[encoder_prefix_length, encoder_prefix_length + num_latent)``
    and add ``noise_sigma * std(slice) * randn_like(slice)`` in-place. ``std`` is
    computed per tensor over all elements of the slice (a single scalar per K- or
    V-slice per layer) so the perturbation scale tracks each tensor's own scale.

    No-op when ``num_latent <= 0`` or ``noise_sigma == 0.0`` — the ``sigma=0`` call
    must introduce **zero** drift versus the un-perturbed control.

    Handles both ``DynamicCache`` and legacy tuple-of-tuples caches. Uses a
    dedicated torch ``Generator`` seeded with ``seed`` so noise is reproducible
    and independent of any other RNG stream.
    """
    if noise_sigma == 0.0 or num_latent <= 0:
        return

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:  # pragma: no cover
        DynamicCache = None

    # Normalize to a list of (k, v) layer tensors we can mutate in place.
    layer_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if DynamicCache is not None and isinstance(cache, DynamicCache):
        for layer_idx in range(len(cache)):
            layer_data = cache[layer_idx]
            layer_kvs.append((layer_data[0], layer_data[1]))
    elif isinstance(cache, (tuple, list)):
        for layer_kv in cache:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                layer_kvs.append((layer_kv[0], layer_kv[1]))
    else:
        raise TypeError(f"Unsupported cache type for KV perturbation: {type(cache).__name__}")

    if not layer_kvs:
        return

    k_probe = layer_kvs[0][0]
    device = k_probe.device
    seq_len = k_probe.size(-2)
    # Latent positions: from encoder_prefix_length to encoder_prefix_length + num_latent.
    # Guard defensively (should always match because the latent rollout appended
    # exactly num_latent tokens after the encoder prefix).
    latent_start = encoder_prefix_length
    latent_stop = min(encoder_prefix_length + num_latent, seq_len)
    if latent_stop <= latent_start:
        return

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    for k, v in layer_kvs:
        for tensor in (k, v):
            slice_ = tensor[:, :, latent_start:latent_stop, :]
            # Per-tensor scalar std over the latent slice. Cast to float32 to avoid
            # catastrophic precision loss in bf16 when computing std of small values.
            std_val = slice_.detach().float().std().item()
            if std_val == 0.0:
                continue
            # Sample noise in float32 for precision, then cast to the slice's dtype.
            noise = torch.empty(slice_.shape, dtype=torch.float32, device=device).normal_(
                mean=0.0, std=1.0, generator=generator
            )
            additive = (noise_sigma * std_val) * noise.to(dtype=slice_.dtype)
            slice_.add_(additive)


def generate_from_latent_with_taps(
    runtime: Any,
    *,
    tokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    inf_latent_iterations: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    skip_latent_injection: bool = False,
    capture_latent_hidden: bool = True,
    perturb_latent_noise_sigma: float = 0.0,
    perturb_latent_noise_seed: int = 11,
) -> GenerationWithTaps:
    """Run latent generation and capture per-step hidden states + final cache.

    This mirrors ``LatentReasoningRuntime.generate_from_latent`` but keeps the
    latent-step hidden states for downstream logit-lens projection, and returns
    the final KV cache so the caller can dump it.

    When ``skip_latent_injection`` is True or ``inf_latent_iterations == 0``,
    the latent rollout is skipped entirely and generation proceeds directly
    from the encoder output — this is the "zero-latent ablation" path.

    When ``perturb_latent_noise_sigma > 0`` and latent iterations > 0, Gaussian
    noise is added to the K/V tensors at the latent positions after the latent
    rollout loop and before the answer generation loop (F6 ablation).
    """

    device = input_ids.device
    batch_size = input_ids.size(0)

    past_key_values, latent = runtime.encode_question(input_ids=input_ids, attention_mask=attention_mask)
    encoder_prefix_length = input_ids.size(1)

    latent_traces: list[LatentStepTrace] = []
    effective_iterations = 0 if skip_latent_injection else inf_latent_iterations
    for step_index in range(effective_iterations):
        outputs = runtime.model(
            inputs_embeds=latent,
            token_type_ids=runtime.build_token_type_ids(inputs_embeds=latent),
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        final_hidden = outputs.hidden_states[-1][:, -1:, :]
        if capture_latent_hidden:
            latent_traces.append(
                LatentStepTrace(
                    latent_step=step_index,
                    hidden_state=final_hidden.squeeze(1).detach().to("cpu"),
                )
            )
        latent = runtime.maybe_project(final_hidden)

    # F6: apply Gaussian-noise perturbation on the latent-position K/V entries
    # BEFORE the answer loop. This preserves the latent rollout (so latent
    # hidden states are unchanged for trace dumping) but perturbs the cached
    # representations the answer loop will attend over.
    if perturb_latent_noise_sigma > 0.0 and effective_iterations > 0:
        perturb_latent_kv_inplace(
            past_key_values,
            encoder_prefix_length=encoder_prefix_length,
            num_latent=effective_iterations,
            noise_sigma=perturb_latent_noise_sigma,
            seed=perturb_latent_noise_seed,
        )

    # From this point forward: reproduce the stock generation loop.
    next_embeds = runtime.build_eot_embeds(
        batch_size=batch_size,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
    )
    predictions_tokens: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        outputs = runtime.model(
            inputs_embeds=next_embeds,
            token_type_ids=runtime.build_token_type_ids(inputs_embeds=next_embeds),
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, : runtime.eot_id]
        token_ids = runtime._sample_tokens(  # noqa: SLF001 — internal helper; stable API in-repo.
            logits=logits,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        for batch_index, token_id in enumerate(token_ids.tolist()):
            if finished[batch_index]:
                continue
            predictions_tokens[batch_index].append(token_id)
            if token_id == tokenizer.eos_token_id:
                finished[batch_index] = True
        if bool(finished.all()):
            break
        next_embeds = runtime.get_input_embedding_layer()(token_ids).unsqueeze(1)

    predictions = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in predictions_tokens]
    return GenerationWithTaps(
        predictions=predictions,
        latent_traces=latent_traces,
        final_cache=past_key_values,
        encoder_prefix_length=encoder_prefix_length,
        num_latent_iterated=effective_iterations,
    )


def extract_kv_at_latent_positions(
    cache: Any,
    *,
    encoder_prefix_length: int,
    num_latent: int,
    final_layer_only: bool = True,
) -> np.ndarray:
    """Extract (k, v) tensors at the latent positions from a KV cache.

    The returned array has shape ``[batch, num_latent, 2, num_heads, head_dim]``
    when ``final_layer_only=True`` (``[batch, num_layers, num_latent, 2, num_heads, head_dim]``
    otherwise). The K and V are stacked along axis 2 so a caller can easily
    concatenate them at analysis time (PCA inputs are typically ``k ⊕ v``).

    Handles both ``DynamicCache`` and legacy tuple-of-tuples caches.

    When ``num_latent == 0`` (zero-latent ablation) the function returns an
    empty array of shape ``[batch, 0, 2, num_heads, head_dim]`` — callers
    should check for this and skip the ``.npy`` dump.
    """

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    # Normalize to a list of (k, v) layer tensors.
    layer_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if DynamicCache is not None and isinstance(cache, DynamicCache):
        for layer_idx in range(len(cache)):
            layer_data = cache[layer_idx]
            layer_kvs.append((layer_data[0], layer_data[1]))
    elif isinstance(cache, (tuple, list)):
        for layer_kv in cache:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                layer_kvs.append((layer_kv[0], layer_kv[1]))
    else:
        raise TypeError(f"Unsupported cache type for KV extraction: {type(cache).__name__}")

    if not layer_kvs:
        raise RuntimeError("Empty KV cache passed to extract_kv_at_latent_positions")

    if final_layer_only:
        layer_kvs = [layer_kvs[-1]]

    k_probe = layer_kvs[0][0]
    # k_probe shape is typically [batch, num_heads, seq_len, head_dim].
    batch_size, num_heads, seq_len, head_dim = k_probe.shape
    # Latent positions: the last `num_latent` entries of the seq axis.
    # (The encoder prefix begins at 0 and extends to `encoder_prefix_length`.)
    latent_start = seq_len - num_latent
    if latent_start < encoder_prefix_length and num_latent > 0:
        # Sanity — shouldn't happen, but guard defensively.
        latent_start = max(encoder_prefix_length, latent_start)

    per_layer: list[np.ndarray] = []
    for k, v in layer_kvs:
        if num_latent == 0:
            slice_k = k.new_zeros((batch_size, num_heads, 0, head_dim))
            slice_v = v.new_zeros((batch_size, num_heads, 0, head_dim))
        else:
            slice_k = k[:, :, latent_start : latent_start + num_latent, :]
            slice_v = v[:, :, latent_start : latent_start + num_latent, :]
        # Re-arrange to [batch, num_latent, 2, num_heads, head_dim]
        kv_stack = torch.stack([slice_k, slice_v], dim=2)  # [b, heads, 2, n_lat, hd]
        kv_stack = kv_stack.permute(0, 3, 2, 1, 4).contiguous()  # [b, n_lat, 2, heads, hd]
        per_layer.append(kv_stack.detach().float().cpu().numpy())

    if final_layer_only:
        return per_layer[0]
    return np.stack(per_layer, axis=1)  # [b, layers, n_lat, 2, heads, hd]
