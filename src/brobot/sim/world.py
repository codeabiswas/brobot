"""World: pre-generates deterministic ground truth and sensor data for a trial."""

import numpy as np

from brobot.sim.maps import get_map, OccupancyGrid
from brobot.sim.sensor import beam_angles, simulate_lidar, inject_outliers, N_BEAMS, D_MAX
from brobot.sim.trajectory import generate_trajectory


class World:
    """Pre-generates all ground truth and sensor data for one trial.

    All methods within a trial receive the same ground truth trajectory
    and the same injected outliers by sharing a World instance.

    Attributes
    ----------
    occ_map : OccupancyGrid
    gt_poses : ndarray (T+1, 3)
    controls : ndarray (T, 2)
    observations : ndarray (T, n_beams)
        Noisy (and possibly outlier-corrupted) LiDAR ranges.
    clean_observations : ndarray (T, n_beams)
        Noisy but outlier-free LiDAR ranges.
    expected_ranges : ndarray (T, n_beams)
        Noiseless expected ranges from ground truth poses.
    outlier_mask : ndarray (T, n_beams) bool
    beam_angles : ndarray (n_beams,)
    """

    def __init__(
        self,
        map_name: str,
        sigma: float,
        r: float,
        T: int = 200,
        seed: int = 0,
        n_beams: int = N_BEAMS,
        d_max: float = D_MAX,
    ):
        self.map_name = map_name
        self.sigma = sigma
        self.r = r
        self.T = T
        self.seed = seed
        self.n_beams = n_beams
        self.d_max = d_max

        # Build map
        self.occ_map = get_map(map_name)

        # Generate trajectory
        self.gt_poses, self.controls = generate_trajectory(
            self.occ_map, map_name, T
        )

        # Beam angles (fixed)
        self.beam_angles_arr = beam_angles(n_beams)

        # Generate sensor data
        rng = np.random.default_rng(seed)

        self.clean_observations = np.zeros((T, n_beams))
        self.observations = np.zeros((T, n_beams))
        self.expected_ranges = np.zeros((T, n_beams))
        self.outlier_mask = np.zeros((T, n_beams), dtype=bool)

        for t in range(T):
            noisy, expected = simulate_lidar(
                self.gt_poses[t + 1],  # pose after applying control t
                self.occ_map.grid,
                self.occ_map.resolution,
                sigma,
                rng,
                d_max,
                n_beams,
            )
            self.clean_observations[t] = noisy
            self.expected_ranges[t] = expected

            corrupted, mask = inject_outliers(noisy, r, rng, d_max)
            self.observations[t] = corrupted
            self.outlier_mask[t] = mask
