from __future__ import annotations

import pytest
import torch

from latent_harness.core.config import LatentRuntimeConfig
from latent_harness.core.runtime import _detach_cache


class TestDetachConfigValidation:
    def test_default_keeps_current_behavior(self):
        cfg = LatentRuntimeConfig()
        assert cfg.detach_keep_last_k is None
        assert cfg.detach_position_mode == "all"

    def test_keep_last_k_positive(self):
        cfg = LatentRuntimeConfig(num_latent=4, detach_keep_last_k=2)
        assert cfg.detach_keep_last_k == 2

    def test_keep_last_k_rejects_zero(self):
        with pytest.raises(ValueError, match="detach_keep_last_k"):
            LatentRuntimeConfig(num_latent=4, detach_keep_last_k=0)

    def test_keep_last_k_rejects_exceeding_num_latent(self):
        with pytest.raises(ValueError, match="detach_keep_last_k"):
            LatentRuntimeConfig(num_latent=2, detach_keep_last_k=3)

    def test_position_mode_valid_values(self):
        LatentRuntimeConfig(detach_position_mode="all")
        LatentRuntimeConfig(detach_position_mode="reasoning_only")

    def test_position_mode_rejects_invalid(self):
        with pytest.raises(ValueError, match="detach_position_mode"):
            LatentRuntimeConfig(detach_position_mode="garbage")


def _graph_pair(seq_len: int):
    """Return (src_k, src_v, graph_k, graph_v) where graph_* depend on src_*.

    Using ``src * 2`` makes d(graph)/d(src) trivial: backward of
    ``graph.sum()`` puts 2.0 into every src.grad element.
    """
    src_k = torch.randn(1, 4, seq_len, 8, requires_grad=True)
    src_v = torch.randn(1, 4, seq_len, 8, requires_grad=True)
    return src_k, src_v, src_k * 2, src_v * 2


class TestDetachCacheTupleFull:
    """Full-detach path must preserve legacy behavior bit-for-bit."""

    def test_none_input_returns_none(self):
        assert _detach_cache(None) is None

    def test_full_detach_severs_all_positions(self):
        src_k, src_v, k, v = _graph_pair(6)
        cache = ((k, v),)
        result = _detach_cache(cache, detach_up_to_pos=None)
        k_out, v_out = result[0]
        # grad_fn is None proves the entire autograd graph is severed; there is
        # no grad-capable tensor to call .backward() on after a full detach.
        assert k_out.grad_fn is None
        assert v_out.grad_fn is None
        assert src_k.grad is None
        assert src_v.grad is None

    def test_full_detach_handles_list_layer_entry(self):
        """Regression guard: legacy behavior accepted both tuple and list layer_kv."""
        src_k, src_v, k, v = _graph_pair(6)
        cache = ([k, v],)
        result = _detach_cache(cache, detach_up_to_pos=None)
        k_out, v_out = result[0]
        assert k_out.grad_fn is None
        assert v_out.grad_fn is None

    def test_full_detach_handles_extra_layer_elements(self):
        """Regression guard: legacy iterated ALL tensors in each layer_kv."""
        src_k, src_v, k, v = _graph_pair(4)
        extra = src_k * 3  # an additional tensor some cache formats include
        cache = ((k, v, extra),)
        result = _detach_cache(cache, detach_up_to_pos=None)
        for element in result[0]:
            if isinstance(element, torch.Tensor):
                assert element.grad_fn is None, (
                    f"tensor in layer entry not detached: shape={element.shape}"
                )


class TestDetachCacheTuplePartial:
    """Position-based detach must actually sever prefix and preserve suffix gradient."""

    def test_prefix_positions_are_gradient_severed(self):
        src_k, src_v, k, v = _graph_pair(6)
        cache = ((k, v),)
        result = _detach_cache(cache, detach_up_to_pos=4)
        k_out, v_out = result[0]
        prefix_loss = k_out[:, :, :4, :].sum() + v_out[:, :, :4, :].sum()
        prefix_loss.backward()
        # PyTorch's CatBackward may push a zero-gradient tensor to src via the
        # suffix branch even when only the prefix is summed.  The semantic
        # requirement is that *no meaningful gradient* reaches src — i.e. grad
        # is either None (never touched) or an all-zeros tensor (zero signal).
        def _no_signal(grad):
            return grad is None or torch.allclose(grad, torch.zeros_like(grad))
        assert _no_signal(src_k.grad)
        assert _no_signal(src_v.grad)

    def test_suffix_positions_route_gradient_to_source(self):
        src_k, src_v, k, v = _graph_pair(6)
        cache = ((k, v),)
        result = _detach_cache(cache, detach_up_to_pos=4)
        k_out, v_out = result[0]
        suffix_loss = k_out[:, :, 4:, :].sum() + v_out[:, :, 4:, :].sum()
        suffix_loss.backward()
        expected = torch.zeros_like(src_k)
        expected[:, :, 4:, :] = 2.0
        assert torch.allclose(src_k.grad, expected)
        assert torch.allclose(src_v.grad, expected)

    def test_cutoff_zero_keeps_everything_connected(self):
        src_k, src_v, k, v = _graph_pair(4)
        cache = ((k, v),)
        result = _detach_cache(cache, detach_up_to_pos=0)
        k_out, v_out = result[0]
        (k_out.sum() + v_out.sum()).backward()
        assert torch.allclose(src_k.grad, torch.full_like(src_k, 2.0))
        assert torch.allclose(src_v.grad, torch.full_like(src_v, 2.0))

    def test_cutoff_equal_seq_len_equals_full_detach(self):
        src_k, src_v, k, v = _graph_pair(4)
        cache = ((k, v),)
        result = _detach_cache(cache, detach_up_to_pos=4)
        k_out, v_out = result[0]
        # When cutoff == seq_len the whole tensor is detached; grad_fn is None
        # and no backward call is possible on the resulting leaf.
        assert k_out.grad_fn is None
        assert v_out.grad_fn is None
        assert src_k.grad is None
        assert src_v.grad is None


class TestDetachCacheDynamic:
    """DynamicCache path must mirror tuple path semantics."""

    def _require_dynamic_cache(self):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            pytest.skip("DynamicCache not available in this transformers build")
        return DynamicCache

    def test_full_detach_dynamic(self):
        DynamicCache = self._require_dynamic_cache()
        src_k, src_v, k, v = _graph_pair(6)
        cache = DynamicCache()
        cache.update(k, v, 0)
        result = _detach_cache(cache, detach_up_to_pos=None)
        assert isinstance(result, DynamicCache)
        layer_data = next(iter(result))
        k_out, v_out = layer_data[0], layer_data[1]
        # Full detach severs grad_fn; no backward possible on a detached leaf.
        assert k_out.grad_fn is None
        assert v_out.grad_fn is None
        assert src_k.grad is None
        assert src_v.grad is None

    def test_partial_detach_dynamic_routes_suffix_gradient(self):
        DynamicCache = self._require_dynamic_cache()
        src_k, src_v, k, v = _graph_pair(6)
        cache = DynamicCache()
        cache.update(k, v, 0)
        result = _detach_cache(cache, detach_up_to_pos=4)
        layer_data = next(iter(result))
        k_out, v_out = layer_data[0], layer_data[1]
        (k_out[:, :, 4:, :].sum() + v_out[:, :, 4:, :].sum()).backward()
        expected = torch.zeros_like(src_k)
        expected[:, :, 4:, :] = 2.0
        assert torch.allclose(src_k.grad, expected)
        assert torch.allclose(src_v.grad, expected)
