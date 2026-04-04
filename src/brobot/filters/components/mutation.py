"""TST mutation operator (standard and entropy-based variant)."""

import numpy as np


def tst_mutation(
    particles: np.ndarray,
    weights_prenorm: np.ndarray,
    best_particle: np.ndarray,
    timestep: int,
) -> np.ndarray:
    """Standard TST mutation from the paper (Eq. 26).

    x_tilde_i = x_i + exp(-omega_i / t) * (x_hat - x_i)

    Parameters
    ----------
    particles : ndarray (N, 3)
        Current particle poses.
    weights_prenorm : ndarray (N,)
        Pre-normalization weights (used in exponent).
    best_particle : ndarray (3,)
        Particle with highest weight.
    timestep : int
        Current timestep (t >= 1).

    Returns
    -------
    mutated : ndarray (N, 3)
    """
    t = max(timestep, 1)
    phi = np.exp(-weights_prenorm / t)  # (N,)
    diff = best_particle[np.newaxis, :] - particles  # (N, 3)
    mutated = particles + phi[:, np.newaxis] * diff

    # Wrap theta
    mutated[:, 2] = (mutated[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return mutated


def tst_mutation_entropy(
    particles: np.ndarray,
    weights_prenorm: np.ndarray,
    best_particle: np.ndarray,
    entropy: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """MPF-E mutation: uses Shannon entropy instead of timestep.

    x_tilde_i = x_i + exp(-omega_i / (H_t + eps)) * (x_hat - x_i)

    Parameters
    ----------
    particles : ndarray (N, 3)
    weights_prenorm : ndarray (N,)
    best_particle : ndarray (3,)
    entropy : float
        Shannon entropy of current normalized weight distribution.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    mutated : ndarray (N, 3)
    """
    h = entropy + eps
    phi = np.exp(-weights_prenorm / h)  # (N,)
    diff = best_particle[np.newaxis, :] - particles  # (N, 3)
    mutated = particles + phi[:, np.newaxis] * diff

    mutated[:, 2] = (mutated[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return mutated


def shannon_entropy(weights: np.ndarray) -> float:
    """Compute Shannon entropy of normalized weights.

    H = -sum(w_i * log(w_i)), H in [0, log(N)]
    """
    w = weights[weights > 0]
    return -np.sum(w * np.log(w))
