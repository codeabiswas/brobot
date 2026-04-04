"""Tests for Numba-accelerated raycasting."""

import numpy as np
from brobot.sim.maps import open_map
from brobot.sim.raycast import raycast_single, batch_raycast


def test_raycast_single_hits_wall():
    m = open_map()
    # From center facing right: ~4.9m to wall
    d = raycast_single(5.0, 5.0, 0.0, m.grid, m.resolution, 5.6)
    assert 4.5 < d < 5.1


def test_raycast_single_max_range():
    m = open_map()
    # From near center facing into open space with short max range
    d = raycast_single(5.0, 5.0, 0.0, m.grid, m.resolution, 1.0)
    assert d == 1.0


def test_batch_raycast_shape():
    m = open_map()
    poses = np.array([[5.0, 5.0, 0.0], [3.0, 3.0, 1.0]])
    angles = np.linspace(-1.0, 1.0, 36)
    ranges = batch_raycast(poses, angles, m.grid, m.resolution, 5.6)
    assert ranges.shape == (2, 36)


def test_batch_raycast_positive():
    m = open_map()
    rng = np.random.default_rng(42)
    poses = np.column_stack([
        rng.uniform(2, 8, 50),
        rng.uniform(2, 8, 50),
        rng.uniform(-np.pi, np.pi, 50),
    ])
    angles = np.linspace(-2.09, 2.09, 36)
    ranges = batch_raycast(poses, angles, m.grid, m.resolution, 5.6)
    assert np.all(ranges >= 0)
    assert np.all(ranges <= 5.6)
