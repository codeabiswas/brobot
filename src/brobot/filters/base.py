"""Base filter class and result dataclass for all particle filter variants."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import numpy as np

from brobot.sim.world import World
from brobot.sim.raycast import batch_raycast
from brobot.sim.sensor import log_likelihood, beam_angles, N_BEAMS, D_MAX, FOV_DEG


@dataclass
class FilterResult:
    """Result from running a filter on a World."""

    rmse_per_step: np.ndarray   # (T,)
    mean_rmse: float
    std_rmse: float
    diverged: bool
    vpior_recall: float         # NaN if not applicable
    runtime_sec: float


class BaseFilter(ABC):
    """Abstract base class for all particle filter variants.

    Subclasses implement `step()` for one predict-update-resample cycle.
    The concrete `run()` method handles the loop, timing, and metrics.
    """

    def __init__(
        self,
        N: int,
        occ_map_grid: np.ndarray,
        resolution: float,
        sigma: float,
        beam_angles_arr: np.ndarray | None = None,
        d_max: float = D_MAX,
    ):
        self.N = N
        self.occ_map = occ_map_grid
        self.resolution = resolution
        self.sigma = sigma
        self.d_max = d_max
        self.beam_angles = (
            beam_angles_arr if beam_angles_arr is not None
            else beam_angles(N_BEAMS, FOV_DEG)
        )
        self.n_beams = len(self.beam_angles)

        # Particle state
        self.particles = np.zeros((N, 3))
        self.weights = np.ones(N) / N
        # Pre-normalization weights (for mutation)
        self.weights_prenorm = np.zeros(N)

    def initialize_particles(self, init_pose: np.ndarray, spread: float = 0.5):
        """Initialize particles around an initial pose."""
        rng = np.random.default_rng(0)
        self.particles[:, 0] = init_pose[0] + rng.normal(0, spread, self.N)
        self.particles[:, 1] = init_pose[1] + rng.normal(0, spread, self.N)
        self.particles[:, 2] = init_pose[2] + rng.normal(0, 0.2, self.N)
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi
        self.weights[:] = 1.0 / self.N

    def estimate(self) -> np.ndarray:
        """Weighted mean pose estimate (circular mean for theta)."""
        px = np.sum(self.weights * self.particles[:, 0])
        py = np.sum(self.weights * self.particles[:, 1])
        theta = np.arctan2(
            np.sum(self.weights * np.sin(self.particles[:, 2])),
            np.sum(self.weights * np.cos(self.particles[:, 2])),
        )
        return np.array([px, py, theta])

    def compute_expected_ranges(self) -> np.ndarray:
        """Raycast for all particles. Returns (N, K) array."""
        return batch_raycast(
            self.particles, self.beam_angles, self.occ_map,
            self.resolution, self.d_max,
        )

    def compute_log_weights(
        self, observation: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        """Log-likelihood weights for all particles."""
        return log_likelihood(observation, expected, self.sigma)

    @abstractmethod
    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        """One predict-update-resample cycle.

        Parameters
        ----------
        control : ndarray (2,) -- [v, omega]
        observation : ndarray (K,) -- observed beam ranges
        timestep : int -- current step (1-indexed)
        rng : Generator

        Returns
        -------
        info : dict
            Optional info (e.g., vpior_removed_mask for recall computation).
        """

    def run(self, world: World, seed: int = 0) -> FilterResult:
        """Run filter on a World instance and collect metrics."""
        rng = np.random.default_rng(seed)

        # Initialize particles around first GT pose
        self.initialize_particles(world.gt_poses[0])

        T = world.T
        rmse_per_step = np.zeros(T)
        total_outlier_beams = 0
        correctly_removed = 0
        has_vpior = False

        t_start = time.perf_counter()

        for t in range(T):
            info = self.step(
                world.controls[t],
                world.observations[t],
                t + 1,  # 1-indexed
                rng,
            )

            # RMSE
            est = self.estimate()
            gt = world.gt_poses[t + 1]
            rmse_per_step[t] = np.sqrt((est[0] - gt[0])**2 + (est[1] - gt[1])**2)

            # VPIOR recall tracking
            if "vpior_removed_mask" in info:
                has_vpior = True
                removed = info["vpior_removed_mask"]
                outliers = world.outlier_mask[t]
                total_outlier_beams += outliers.sum()
                correctly_removed += (removed & outliers).sum()

        runtime = time.perf_counter() - t_start

        # Divergence: RMSE > 1.0m for >20 consecutive steps
        diverged = False
        consec = 0
        for r in rmse_per_step:
            if r > 1.0:
                consec += 1
                if consec > 20:
                    diverged = True
                    break
            else:
                consec = 0

        vpior_recall = float("nan")
        if has_vpior and total_outlier_beams > 0:
            vpior_recall = correctly_removed / total_outlier_beams

        return FilterResult(
            rmse_per_step=rmse_per_step,
            mean_rmse=float(np.mean(rmse_per_step)),
            std_rmse=float(np.std(rmse_per_step)),
            diverged=diverged,
            vpior_recall=vpior_recall,
            runtime_sec=runtime,
        )
