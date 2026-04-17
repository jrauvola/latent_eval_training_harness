from __future__ import annotations

import pytest

from latent_harness.core.config import LatentRuntimeConfig


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
