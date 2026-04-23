"""Tests for F6 latent-position KV perturbation.

Contract:
- ``sigma=0.0`` is a true no-op (not a tiny-mutation no-op).
- ``num_latent=0`` is a true no-op.
- Only the latent-position slice is mutated; encoder-prefix and post-latent
  positions remain byte-identical.
- Same seed + same sigma reproduces the noise exactly across two separate calls.
- Different seed yields different noise (with high probability given tensor size).
- Works on both tuple-of-tuples caches and ``DynamicCache`` (when available).
"""

from __future__ import annotations

import torch

from latent_harness.evaluation.latent_tap import perturb_latent_kv_inplace


def _fabricate_tuple_cache(*, layers: int, batch: int, heads: int, seq_len: int, head_dim: int):
    torch.manual_seed(0)
    return tuple(
        (
            torch.randn(batch, heads, seq_len, head_dim),
            torch.randn(batch, heads, seq_len, head_dim),
        )
        for _ in range(layers)
    )


def test_sigma_zero_is_noop():
    cache = _fabricate_tuple_cache(layers=2, batch=2, heads=4, seq_len=10, head_dim=8)
    before = [(k.clone(), v.clone()) for k, v in cache]
    perturb_latent_kv_inplace(cache, encoder_prefix_length=6, num_latent=3, noise_sigma=0.0, seed=42)
    for (k, v), (k0, v0) in zip(cache, before):
        assert torch.equal(k, k0)
        assert torch.equal(v, v0)


def test_num_latent_zero_is_noop():
    cache = _fabricate_tuple_cache(layers=2, batch=2, heads=4, seq_len=10, head_dim=8)
    before = [(k.clone(), v.clone()) for k, v in cache]
    perturb_latent_kv_inplace(cache, encoder_prefix_length=6, num_latent=0, noise_sigma=1.0, seed=42)
    for (k, v), (k0, v0) in zip(cache, before):
        assert torch.equal(k, k0)
        assert torch.equal(v, v0)


def test_only_latent_positions_mutated():
    cache = _fabricate_tuple_cache(layers=3, batch=2, heads=4, seq_len=12, head_dim=8)
    before = [(k.clone(), v.clone()) for k, v in cache]
    enc_len, num_lat = 6, 3
    perturb_latent_kv_inplace(cache, encoder_prefix_length=enc_len, num_latent=num_lat, noise_sigma=1.0, seed=42)
    for (k, v), (k0, v0) in zip(cache, before):
        # Encoder prefix unchanged.
        assert torch.equal(k[:, :, :enc_len, :], k0[:, :, :enc_len, :])
        assert torch.equal(v[:, :, :enc_len, :], v0[:, :, :enc_len, :])
        # Post-latent positions unchanged.
        assert torch.equal(k[:, :, enc_len + num_lat :, :], k0[:, :, enc_len + num_lat :, :])
        assert torch.equal(v[:, :, enc_len + num_lat :, :], v0[:, :, enc_len + num_lat :, :])
        # Latent slice changed.
        assert not torch.equal(
            k[:, :, enc_len : enc_len + num_lat, :], k0[:, :, enc_len : enc_len + num_lat, :]
        )
        assert not torch.equal(
            v[:, :, enc_len : enc_len + num_lat, :], v0[:, :, enc_len : enc_len + num_lat, :]
        )


def test_reproducible_same_seed():
    fresh = _fabricate_tuple_cache(layers=2, batch=2, heads=4, seq_len=10, head_dim=8)
    baseline = tuple((k.clone(), v.clone()) for k, v in fresh)
    alt = tuple((k.clone(), v.clone()) for k, v in fresh)
    perturb_latent_kv_inplace(baseline, encoder_prefix_length=6, num_latent=3, noise_sigma=1.0, seed=42)
    perturb_latent_kv_inplace(alt, encoder_prefix_length=6, num_latent=3, noise_sigma=1.0, seed=42)
    for (k1, v1), (k2, v2) in zip(baseline, alt):
        assert torch.equal(k1, k2)
        assert torch.equal(v1, v2)


def test_different_seed_changes_noise():
    fresh = _fabricate_tuple_cache(layers=2, batch=2, heads=4, seq_len=10, head_dim=8)
    a = tuple((k.clone(), v.clone()) for k, v in fresh)
    b = tuple((k.clone(), v.clone()) for k, v in fresh)
    perturb_latent_kv_inplace(a, encoder_prefix_length=6, num_latent=3, noise_sigma=1.0, seed=42)
    perturb_latent_kv_inplace(b, encoder_prefix_length=6, num_latent=3, noise_sigma=1.0, seed=1234)
    any_diff = any(not torch.equal(k1, k2) for (k1, _), (k2, _) in zip(a, b))
    assert any_diff


def test_noise_scale_tracks_sigma_times_std():
    # Construct a cache with known std = ~1.0; then sigma=2.0 should produce a
    # perturbation with standard deviation approximately 2.0 * std(slice).
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 2, 8, 20, 16
    enc_len, num_lat = 10, 8
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    cache = ((k, v),)
    k0 = k.clone()
    base_std = k0[:, :, enc_len : enc_len + num_lat, :].float().std().item()
    perturb_latent_kv_inplace(cache, encoder_prefix_length=enc_len, num_latent=num_lat, noise_sigma=2.0, seed=42)
    delta = (k - k0)[:, :, enc_len : enc_len + num_lat, :].float()
    empirical = delta.std().item()
    expected = 2.0 * base_std
    # Empirical std is itself noisy; allow ~15% tolerance.
    assert abs(empirical - expected) < 0.15 * expected, (empirical, expected)


def test_handles_bf16_cache():
    cache = tuple(
        (torch.randn(2, 4, 10, 8).to(torch.bfloat16), torch.randn(2, 4, 10, 8).to(torch.bfloat16))
        for _ in range(2)
    )
    before = [(k.clone(), v.clone()) for k, v in cache]
    perturb_latent_kv_inplace(cache, encoder_prefix_length=6, num_latent=3, noise_sigma=1.0, seed=42)
    for (k, v), (k0, v0) in zip(cache, before):
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16
        # Mutated at least somewhere in latent slice.
        assert not torch.equal(k[:, :, 6:9, :], k0[:, :, 6:9, :])
