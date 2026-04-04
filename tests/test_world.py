"""Tests for World determinism and correctness."""

import numpy as np
from brobot.sim.world import World


def test_world_determinism():
    w1 = World("open", sigma=0.05, r=0.1, seed=42)
    w2 = World("open", sigma=0.05, r=0.1, seed=42)
    np.testing.assert_array_equal(w1.gt_poses, w2.gt_poses)
    np.testing.assert_array_equal(w1.observations, w2.observations)
    np.testing.assert_array_equal(w1.outlier_mask, w2.outlier_mask)


def test_world_shapes():
    w = World("open", sigma=0.05, r=0.0, T=100, seed=0)
    assert w.gt_poses.shape == (101, 3)
    assert w.controls.shape == (100, 2)
    assert w.observations.shape == (100, 36)
    assert w.outlier_mask.shape == (100, 36)


def test_world_no_outliers():
    w = World("open", sigma=0.05, r=0.0, seed=0)
    assert w.outlier_mask.sum() == 0


def test_world_outlier_rate():
    w = World("open", sigma=0.05, r=0.2, seed=42)
    actual_rate = w.outlier_mask.mean()
    assert 0.1 < actual_rate < 0.3  # within reasonable bounds
