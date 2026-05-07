"""Unit tests for differential privacy mechanisms."""

from __future__ import annotations

import numpy as np
import pytest

from src.federated.privacy import add_dp_noise, clip_update, protect_update


class TestClipUpdate:
    """Tests for the clip_update() function."""

    def test_update_at_threshold_unchanged(self):
        """Update with L2 norm exactly at C should not be clipped."""
        update = np.array([3.0, 4.0])  # L2 norm = 5.0
        C = 5.0
        clipped = clip_update(update, C)
        np.testing.assert_array_almost_equal(clipped, update)

    def test_update_below_threshold_unchanged(self):
        """Update with L2 norm below C should be returned unchanged."""
        update = np.array([1.0, 1.0])  # L2 norm ≈ 1.414
        C = 5.0
        clipped = clip_update(update, C)
        np.testing.assert_array_almost_equal(clipped, update)

    def test_update_above_threshold_clipped_to_C(self):
        """Update above threshold should be clipped to exactly norm C."""
        update = np.array([6.0, 8.0])  # L2 norm = 10.0
        C = 5.0
        clipped = clip_update(update, C)
        assert abs(np.linalg.norm(clipped) - C) < 1e-5

    def test_clip_preserves_direction(self):
        """Clipped update should be in the same direction as original."""
        update = np.array([6.0, 8.0])
        C = 5.0
        clipped = clip_update(update, C)
        # Cosine similarity should be 1.0
        cos_sim = np.dot(update, clipped) / (np.linalg.norm(update) * np.linalg.norm(clipped))
        assert abs(cos_sim - 1.0) < 1e-5

    def test_invalid_C_raises_error(self):
        """Non-positive clipping threshold should raise ValueError."""
        update = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="must be positive"):
            clip_update(update, C=0.0)

    def test_negative_C_raises_error(self):
        """Negative clipping threshold should raise ValueError."""
        with pytest.raises(ValueError):
            clip_update(np.ones(10), C=-1.0)

    def test_returns_copy(self):
        """clip_update should return a new array, not modify the original."""
        update = np.array([1.0, 2.0])
        original = update.copy()
        _ = clip_update(update, C=10.0)
        np.testing.assert_array_equal(update, original)

    def test_multidimensional_array(self):
        """Should work on flattened higher-dimensional arrays."""
        update = np.ones((10, 10)).flatten()  # norm = 10.0
        C = 5.0
        clipped = clip_update(update, C)
        assert abs(np.linalg.norm(clipped) - C) < 1e-5


class TestAddDPNoise:
    """Tests for the add_dp_noise() function."""

    def test_shape_preserved(self):
        """Noise addition should preserve the shape of the input."""
        update = np.random.randn(100)
        noisy = add_dp_noise(update, sigma=0.1, C=1.0, seed=42)
        assert noisy.shape == update.shape

    def test_nonzero_noise_added(self):
        """Gaussian noise should change the update values."""
        update = np.zeros(100)
        noisy = add_dp_noise(update, sigma=1.0, C=1.0, seed=42)
        assert not np.allclose(update, noisy)

    def test_seeded_noise_is_reproducible(self):
        """Same seed should produce identical noise."""
        update = np.ones(50)
        noisy1 = add_dp_noise(update, sigma=0.5, C=1.0, seed=99)
        noisy2 = add_dp_noise(update, sigma=0.5, C=1.0, seed=99)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_give_different_noise(self):
        """Different seeds should produce different noise realizations."""
        update = np.ones(50)
        noisy1 = add_dp_noise(update, sigma=0.5, C=1.0, seed=1)
        noisy2 = add_dp_noise(update, sigma=0.5, C=1.0, seed=2)
        assert not np.allclose(noisy1, noisy2)

    def test_larger_sigma_gives_more_noise(self):
        """Higher sigma should result in larger average perturbation."""
        update = np.zeros(1000)
        noisy_small = add_dp_noise(update, sigma=0.01, C=1.0, seed=42)
        noisy_large = add_dp_noise(update, sigma=10.0, C=1.0, seed=42)
        assert np.std(noisy_large) > np.std(noisy_small)

    def test_invalid_sigma_raises_error(self):
        """Non-positive sigma should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            add_dp_noise(np.ones(10), sigma=0.0, C=1.0)

    def test_invalid_C_raises_error(self):
        """Non-positive C should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            add_dp_noise(np.ones(10), sigma=1.0, C=-1.0)


class TestProtectUpdate:
    """Tests for the combined protect_update() function."""

    def test_protect_update_clips_then_noises(self):
        """Protected update should have norm generally bounded near C + noise."""
        update = np.ones(100) * 10.0  # large norm
        C = 1.0
        sigma = 0.01
        protected = protect_update(update, C=C, sigma=sigma, seed=42)
        # After clipping, norm ≈ C; after small noise, should remain near C
        assert abs(np.linalg.norm(protected) - C) < 0.5  # within 0.5 of C

    def test_protect_update_changes_values(self):
        """Protected update should differ from original."""
        update = np.ones(50)
        protected = protect_update(update, C=0.5, sigma=0.1, seed=42)
        assert not np.allclose(update, protected)
