"""Numba-accelerated raycasting for LiDAR simulation."""

import numpy as np
from numba import njit, prange


@njit
def raycast_single(
    px: float,
    py: float,
    angle: float,
    occ_map: np.ndarray,
    resolution: float,
    d_max: float,
) -> float:
    """Cast a single ray and return distance to first occupied cell.

    Marches in steps of `resolution` along the ray direction,
    checking occupancy at each step.
    """
    dx = np.cos(angle) * resolution
    dy = np.sin(angle) * resolution
    height, width = occ_map.shape

    x = px
    y = py
    dist = 0.0

    while dist < d_max:
        col = int(x / resolution)
        row = int(y / resolution)

        if row < 0 or row >= height or col < 0 or col >= width:
            return dist
        if occ_map[row, col] == 1:
            return dist

        x += dx
        y += dy
        dist += resolution

    return d_max


@njit(parallel=True)
def batch_raycast(
    poses: np.ndarray,
    beam_angles: np.ndarray,
    occ_map: np.ndarray,
    resolution: float,
    d_max: float,
) -> np.ndarray:
    """Raycast for multiple poses and beam angles.

    Parameters
    ----------
    poses : ndarray (N, 3)
        Robot poses [px, py, theta].
    beam_angles : ndarray (K,)
        Beam angle offsets relative to robot heading.
    occ_map : ndarray (H, W)
        Binary occupancy grid.
    resolution : float
        Grid resolution in m/cell.
    d_max : float
        Maximum range in meters.

    Returns
    -------
    ranges : ndarray (N, K)
        Expected range for each pose and beam.
    """
    n_poses = poses.shape[0]
    n_beams = beam_angles.shape[0]
    ranges = np.empty((n_poses, n_beams))

    for i in prange(n_poses):
        px = poses[i, 0]
        py = poses[i, 1]
        theta = poses[i, 2]
        for k in range(n_beams):
            abs_angle = theta + beam_angles[k]
            ranges[i, k] = raycast_single(px, py, abs_angle, occ_map, resolution, d_max)

    return ranges
