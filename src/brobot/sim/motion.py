"""Differential-drive motion model with additive Gaussian noise."""

import numpy as np

# Fixed process noise standard deviations
SIGMA_V = 0.02   # m
SIGMA_W = 0.01   # rad
DT = 1.0         # timestep duration (seconds)


def motion_update(
    pose: np.ndarray,
    v: float,
    omega: float,
    rng: np.random.Generator,
    sigma_v: float = SIGMA_V,
    sigma_w: float = SIGMA_W,
    dt: float = DT,
) -> np.ndarray:
    """Propagate a single pose through differential-drive kinematics.

    x_t = x_{t-1} + [ds*cos(θ + dθ/2), ds*sin(θ + dθ/2), dθ]^T + ε
    """
    ds = v * dt
    dtheta = omega * dt
    theta = pose[2]

    new_pose = np.array([
        pose[0] + ds * np.cos(theta + dtheta / 2),
        pose[1] + ds * np.sin(theta + dtheta / 2),
        theta + dtheta,
    ])

    # Additive Gaussian noise
    new_pose[0] += rng.normal(0, sigma_v)
    new_pose[1] += rng.normal(0, sigma_v)
    new_pose[2] += rng.normal(0, sigma_w)

    # Wrap angle to [-pi, pi]
    new_pose[2] = (new_pose[2] + np.pi) % (2 * np.pi) - np.pi

    return new_pose


def sample_motion_batch(
    poses: np.ndarray,
    v: float,
    omega: float,
    rng: np.random.Generator,
    sigma_v: float = SIGMA_V,
    sigma_w: float = SIGMA_W,
    dt: float = DT,
) -> np.ndarray:
    """Propagate N particles through the motion model.

    Parameters
    ----------
    poses : ndarray (N, 3)
        Current particle poses [px, py, theta].

    Returns
    -------
    new_poses : ndarray (N, 3)
    """
    n = poses.shape[0]
    ds = v * dt
    dtheta = omega * dt
    theta = poses[:, 2]

    new_poses = np.empty_like(poses)
    new_poses[:, 0] = poses[:, 0] + ds * np.cos(theta + dtheta / 2)
    new_poses[:, 1] = poses[:, 1] + ds * np.sin(theta + dtheta / 2)
    new_poses[:, 2] = theta + dtheta

    # Additive Gaussian noise
    new_poses[:, 0] += rng.normal(0, sigma_v, n)
    new_poses[:, 1] += rng.normal(0, sigma_v, n)
    new_poses[:, 2] += rng.normal(0, sigma_w, n)

    # Wrap angle
    new_poses[:, 2] = (new_poses[:, 2] + np.pi) % (2 * np.pi) - np.pi

    return new_poses


def predict_mean_batch(
    poses: np.ndarray,
    v: float,
    omega: float,
    dt: float = DT,
) -> np.ndarray:
    """Noiseless batch motion prediction (deterministic).

    Computes E(x_t | x_{t-1}, u_t) — the expected next state without
    process noise. Used by the RAPF for auxiliary weight computation.

    Parameters
    ----------
    poses : ndarray (N, 3)
        Current particle poses [px, py, theta].

    Returns
    -------
    predicted : ndarray (N, 3)
    """
    ds = v * dt
    dtheta = omega * dt
    theta = poses[:, 2]

    predicted = np.empty_like(poses)
    predicted[:, 0] = poses[:, 0] + ds * np.cos(theta + dtheta / 2)
    predicted[:, 1] = poses[:, 1] + ds * np.sin(theta + dtheta / 2)
    predicted[:, 2] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi

    return predicted
