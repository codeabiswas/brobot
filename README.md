# Brobot

Monte Carlo Localization variant benchmarking for EECE 5550 — Mobile Robotics (Spring 2026).

Implements and compares 7 particle filter variants on simulated 2D occupancy grid environments with noisy LiDAR, based on [Mohseni et al. (2024)](https://doi.org/10.1016/j.engappai.2024.107897).

## Methods

| Key | Method | Description |
|-----|--------|-------------|
| `SIR` | SIR-PF | Standard Sequential Importance Resampling |
| `RA` | RA-PF | SIR + kernel smoothing (Silverman bandwidth) |
| `KLD` | KLD-AMCL | Adaptive particle count via KLD-sampling [50–500] |
| `MPF` | MPF | Rényi divergence outlier detection + VPIOR beam removal + TST mutation |
| `KLD_mut` | KLD + Mutation | KLD-AMCL + TST mutation only |
| `KLD_vpior` | KLD + VPIOR | KLD-AMCL + VPIOR only |
| `MPFE` | MPF-E | MPF with entropy-based mutation (proposed extension) |

## Setup

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

## Usage

### Full parameter sweep (12,600 runs)

```bash
uv run python main.py
```

Options:

```
--workers N       Number of parallel workers (default: CPU count)
--trials N        Trials per configuration (default: 30)
--output PATH     Results CSV path (default: results/sweep.csv)
--figures-only    Skip sweep, generate figures from existing CSV
--smoke           Smoke test: 1 trial, reduced grid
```

### Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/brobot/
├── sim/            # Environment: maps, Numba raycasting, motion, sensor, trajectory, World
├── filters/        # 7 filter implementations + shared components
├── experiments/    # Sweep config, runner (multiprocessing), metrics
└── analysis/       # Figure generation and reproduction check tables
```

## Parameter Sweep

- Sensor noise σ: 0.01, 0.05, 0.1, 0.2, 0.5 m
- Outlier rate r: 0%, 5%, 10%, 15%, 20%, 30%
- Maps: open room, corridor
- 500 particles, 200 timesteps, 30 trials per configuration

## Research Questions

1. **Q1:** At what (σ, r) does each method's mean RMSE exceed 0.5 m?
2. **Q2:** Does map geometry change the relative ranking of methods?
3. **Q3:** What fraction of MPF's gain over SIR-PF is attributable to KLD, mutation, and VPIOR individually?
4. **Q4:** Does MPF-E behave differently from MPF under high-stress, late-trajectory conditions?

## Output

- `results/sweep.csv` — full sweep results
- `figures/fig1_heatmaps.png` — RMSE heatmaps over (σ, r) for all methods (Q1)
- `figures/fig2_map_geometry.png` — RMSE by map geometry (Q2)
- `figures/fig3_ablation.png` — Ablation decomposition with marginal deltas (Q3)
- `figures/fig4_mpf_vs_mpfe.png` — RMSE trajectory, MPF vs MPF-E (Q4)

## Authors

- Andrei Biswas
- Rishabh Kumar
- Rishi Srikaanth
