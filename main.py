"""Run the full MCL parameter sweep and generate figures."""

import argparse
import multiprocessing as mp
import os
import sys

from brobot.experiments.config import (
    generate_sweep_configs,
    SIGMAS,
    OUTLIER_RATES,
    MAPS,
    METHODS,
)
from brobot.experiments.runner import run_sweep
from brobot.analysis.figures import generate_all_figures
from brobot.analysis.tables import reproduction_check


def main():
    parser = argparse.ArgumentParser(
        description="Brobot: Monte Carlo Localization parameter sweep"
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Trials per configuration (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default="results/sweep.csv",
        help="Results CSV path (default: results/sweep.csv)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Skip sweep, generate figures from existing CSV",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 trial, reduced grid",
    )
    parser.add_argument(
        "--repro-check", action="store_true",
        help="Run reproduction check only (SIR/RA/MPF on clean data)",
    )
    args = parser.parse_args()

    if args.repro_check:
        reproduction_check(n_trials=args.trials)
        return

    if args.figures_only:
        if not os.path.exists(args.output):
            print(f"Error: {args.output} not found. Run the sweep first.")
            sys.exit(1)
        generate_all_figures(args.output)
        return

    # Configure sweep
    if args.smoke:
        sigmas = [0.05, 0.2]
        rates = [0.0, 0.15]
        maps = ["open"]
        methods = ["SIR", "MPF", "MPFE"]
        n_trials = 1
    else:
        sigmas = SIGMAS
        rates = OUTLIER_RATES
        maps = MAPS
        methods = METHODS
        n_trials = args.trials

    configs = generate_sweep_configs(
        sigmas=sigmas,
        rates=rates,
        maps=maps,
        methods=methods,
        n_trials=n_trials,
    )

    total = len(configs)
    print(f"Sweep: {total} runs ({len(sigmas)} σ × {len(rates)} r × "
          f"{len(maps)} maps × {len(methods)} methods × {n_trials} trials)")
    print(f"Workers: {args.workers}")
    print()

    # Run reproduction check first (quick sanity gate)
    if not args.smoke:
        print("--- Reproduction check ---")
        reproduction_check(n_trials=min(10, n_trials))
        print()

    # Run sweep
    print("--- Parameter sweep ---")
    df = run_sweep(configs, n_workers=args.workers, output_path=args.output)
    print(f"\nSweep complete: {len(df)} results")

    # Generate figures
    print("\n--- Generating figures ---")
    generate_all_figures(args.output)


if __name__ == "__main__":
    main()
