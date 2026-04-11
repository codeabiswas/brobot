"""RAPF: Regularized Auxiliary Particle Filter (Liu et al. 2011).

Three-stage algorithm:
  Stage 1 — Auxiliary weighting with predictive likelihood lookahead.
  Stage 2 — Epanechnikov kernel regularization (Cholesky + h_opt).
  Stage 3 — Second-stage proposal with rejection/resampling (Eq. 17).
"""

import numpy as np

from brobot.filters.base import BaseFilter
from brobot.filters.components.resampling import systematic_resample
from brobot.filters.components.smoothing import regularize
from brobot.sim.motion import sample_motion_batch, predict_mean_batch
from brobot.sim.raycast import batch_raycast
from brobot.sim.sensor import log_likelihood


class RAPF(BaseFilter):
    """Regularized Auxiliary Particle Filter.

    Parameters
    ----------
    W : float
        Acceptance threshold for rejection/resampling (paper Section 2.2.2).
        Proposals are accepted if 1/W <= pi_j <= W. Default 2.0.
    max_reject : int
        Maximum rejection iterations before force-accepting. Default 30.
    """

    def __init__(self, *args, W: float = 2.0, max_reject: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = W
        self.max_reject = max_reject

    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> dict:
        v, omega = control

        # ==================================================================
        # Stage 1: First-stage auxiliary weighting
        # ==================================================================
        # Compute predicted means mu_i = E(alpha_t | alpha_{i-1})
        mu = predict_mean_batch(self.particles, v, omega)  # (N, 3)

        # Raycast from predicted means — RAYCAST PASS 1
        expected_mu = batch_raycast(
            mu, self.beam_angles, self.occ_map, self.resolution, self.d_max,
        )  # (N, K)

        # Auxiliary weights: lambda_i = pi_{i-1} * f(y_t | mu_i)
        log_lik_mu = log_likelihood(observation, expected_mu, self.sigma)  # (N,)
        log_lambda = np.log(self.weights + 1e-300) + log_lik_mu

        # Normalize (log-sum-exp for stability)
        log_lambda -= np.max(log_lambda)
        lambda_weights = np.exp(log_lambda)
        lam_sum = lambda_weights.sum()
        if lam_sum > 0:
            lambda_weights /= lam_sum
        else:
            lambda_weights[:] = 1.0 / self.N

        # ==================================================================
        # Stage 2: Regularization
        # ==================================================================
        # Save pre-resampling state for covariance computation
        original_particles = self.particles.copy()

        # Resample using auxiliary weights
        indices = systematic_resample(lambda_weights, rng)
        self.particles = self.particles[indices]

        # Regularize: Epanechnikov kernel + Cholesky + optimal bandwidth
        self.particles = regularize(
            self.particles, lambda_weights, original_particles, rng,
        )

        # ==================================================================
        # Stage 3: Second-stage proposal + rejection/resampling (Eq. 17)
        # ==================================================================
        # Predicted means from regularized particles
        mu_star = predict_mean_batch(self.particles, v, omega)  # (N, 3)

        # Raycast from predicted means — RAYCAST PASS 2
        expected_mu_star = batch_raycast(
            mu_star, self.beam_angles, self.occ_map, self.resolution, self.d_max,
        )  # (N, K)

        # Log-likelihood denominators: f(y_t | mu_j)
        log_lik_denom = log_likelihood(
            observation, expected_mu_star, self.sigma,
        )  # (N,)

        # Batch rejection loop
        accepted = np.zeros(self.N, dtype=bool)
        new_particles = np.empty_like(self.particles)

        for _attempt in range(self.max_reject):
            pending = np.where(~accepted)[0]
            if len(pending) == 0:
                break

            # Draw alpha_j ~ f(alpha_t | alpha*_{j,t-1})
            candidates = sample_motion_batch(
                self.particles[pending], v, omega, rng,
            )  # (n_pending, 3)

            # RAYCAST PASS 3: batch raycast for all pending candidates
            expected_cand = batch_raycast(
                candidates, self.beam_angles, self.occ_map,
                self.resolution, self.d_max,
            )  # (n_pending, K)

            # Second-stage weight: pi_j = f(y_t | alpha_j) / f(y_t | mu_j)
            log_lik_cand = log_likelihood(
                observation, expected_cand, self.sigma,
            )  # (n_pending,)
            log_pi = log_lik_cand - log_lik_denom[pending]

            # Acceptance test: 1/W <= pi_j <= W
            pi_vals = np.exp(np.clip(log_pi, -20, 20))
            accept_mask = (pi_vals >= 1.0 / self.W) & (pi_vals <= self.W)

            # Store accepted particles
            accepted_idx = pending[accept_mask]
            new_particles[accepted_idx] = candidates[accept_mask]
            accepted[accepted_idx] = True

            # Store last candidates for force-accept fallback
            last_candidates_pending = pending
            last_candidates_values = candidates

        # Force-accept remaining particles with their last drawn candidates
        if not np.all(accepted):
            remaining = np.where(~accepted)[0]
            # Use last candidates if available for these indices
            if 'last_candidates_pending' in dir():
                for idx in remaining:
                    pos = np.where(last_candidates_pending == idx)[0]
                    if len(pos) > 0:
                        new_particles[idx] = last_candidates_values[pos[0]]
                    else:
                        new_particles[idx] = sample_motion_batch(
                            self.particles[idx:idx+1], v, omega, rng,
                        )[0]
            else:
                forced = sample_motion_batch(
                    self.particles[remaining], v, omega, rng,
                )
                new_particles[remaining] = forced

        self.particles = new_particles
        self.weights[:] = 1.0 / self.N

        return {}
