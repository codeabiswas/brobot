"""Metrics computation for experiment results."""

import numpy as np

from brobot.filters.base import FilterResult


def is_diverged(rmse_per_step: np.ndarray, threshold: float = 1.0, consec: int = 20) -> bool:
    """Check if filter diverged: RMSE > threshold for > consec consecutive steps."""
    count = 0
    for r in rmse_per_step:
        if r > threshold:
            count += 1
            if count > consec:
                return True
        else:
            count = 0
    return False


def aggregate_results(results: list[FilterResult]) -> dict:
    """Aggregate metrics over multiple trial results."""
    rmses = [r.mean_rmse for r in results]
    div_rate = sum(1 for r in results if r.diverged) / len(results)

    vpior_recalls = [r.vpior_recall for r in results if not np.isnan(r.vpior_recall)]
    mean_vpior = np.mean(vpior_recalls) if vpior_recalls else float("nan")

    return {
        "mean_rmse": np.mean(rmses),
        "std_rmse": np.std(rmses),
        "divergence_rate": div_rate,
        "mean_vpior_recall": mean_vpior,
        "mean_runtime": np.mean([r.runtime_sec for r in results]),
    }
