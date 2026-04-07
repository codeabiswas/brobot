"""Generate the 4 paper figures from sweep results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def figure1_rmse_heatmaps(
    df: pd.DataFrame,
    map_name: str = "corridor",
    output_path: str = "figures/fig1_heatmaps.png",
):
    """Fig 1: RMSE heatmap over (σ, r) for all 7 methods.

    Answers Q1: At what (σ, r) does each method's mean RMSE exceed 0.5m?
    """
    _ensure_dir(output_path)
    methods = ["SIR", "RA", "KLD", "MPF", "KLD_mut", "KLD_vpior", "MPFE"]
    sub = df[df["map_name"] == map_name]

    sigmas = sorted(sub["sigma"].unique())
    rates = sorted(sub["r"].unique())

    n_methods = len(methods)
    # Use gridspec for a dedicated colorbar axis (avoids tight_layout conflict)
    fig = plt.figure(figsize=(3.2 * n_methods + 0.8, 4.5))
    gs = fig.add_gridspec(
        1, n_methods + 1,
        width_ratios=[1] * n_methods + [0.05],
        wspace=0.25,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_methods)]
    cbar_ax = fig.add_subplot(gs[0, n_methods])

    for idx, (ax, method) in enumerate(zip(axes, methods)):
        msub = sub[sub["method"] == method]
        pivot = msub.groupby(["sigma", "r"])["mean_rmse"].mean().unstack(level="r")
        pivot = pivot.reindex(index=sigmas, columns=rates)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1.0,
        )
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.0%}" for r in rates], fontsize=7)
        ax.set_xlabel("Outlier rate r", fontsize=8)
        ax.set_title(method, fontsize=10)

        if idx == 0:
            ax.set_yticks(range(len(sigmas)))
            ax.set_yticklabels([f"{s}" for s in sigmas], fontsize=7)
            ax.set_ylabel("Sensor noise σ (m)", fontsize=8)
        else:
            ax.set_yticks([])

        # Annotate cells
        for i in range(len(sigmas)):
            for j in range(len(rates)):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, cax=cbar_ax, label="Mean RMSE (m)")
    fig.suptitle(f"Mean RMSE by (σ, r) — {map_name} map", fontsize=12, y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def figure2_map_geometry(
    df: pd.DataFrame,
    sigma: float = 0.1,
    r: float = 0.2,
    output_path: str = "figures/fig2_map_geometry.png",
):
    """Fig 2: Mean RMSE by map geometry.

    Answers Q2: Does map geometry change the relative ranking of methods?
    """
    _ensure_dir(output_path)
    methods = ["SIR", "RA", "KLD", "MPF", "KLD_mut", "KLD_vpior", "MPFE"]
    sub = df[(df["sigma"] == sigma) & (df["r"] == r)]

    map_names = sorted(df["map_name"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods))
    n_maps = len(map_names)
    width = 0.7 / n_maps

    for i, map_name in enumerate(map_names):
        msub = sub[sub["map_name"] == map_name]
        means = [msub[msub["method"] == m]["mean_rmse"].mean() for m in methods]
        stds = [msub[msub["method"] == m]["mean_rmse"].std() for m in methods]
        offset = (i - (n_maps - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=map_name, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean RMSE (m)")
    ax.set_title(f"Mean RMSE by map geometry (σ={sigma}, r={r})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def figure3_ablation(
    df: pd.DataFrame,
    map_name: str = "corridor",
    output_path: str = "figures/fig3_ablation.png",
):
    """Fig 3: Ablation decomposition with marginal deltas.

    Answers Q3: What fraction of MPF's gain over SIR-PF is attributable to
    KLD, mutation, and VPIOR individually?
    """
    _ensure_dir(output_path)
    methods = ["SIR", "KLD", "KLD_mut", "KLD_vpior", "MPF"]
    labels = ["SIR-PF", "KLD-AMCL", "KLD+Mut", "KLD+VPIOR", "MPF"]
    sub = df[df["map_name"] == map_name]

    # Aggregate across all (σ, r) conditions
    means = []
    stds = []
    for m in methods:
        msub = sub[sub["method"] == m]
        means.append(msub["mean_rmse"].mean())
        stds.append(msub["mean_rmse"].std())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))

    # Annotate marginal deltas (relative to KLD-AMCL baseline)
    kld_rmse = means[1]  # KLD-AMCL is baseline
    for i in range(2, len(methods)):
        delta = means[i] - kld_rmse
        sign = "+" if delta > 0 else ""
        ax.annotate(
            f"Δ={sign}{delta:.4f}",
            xy=(x[i], means[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="red" if delta > 0 else "green",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean RMSE (m)")
    ax.set_title(f"Ablation: marginal contribution over KLD-AMCL — {map_name} map")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def figure4_mpf_vs_mpfe(
    df: pd.DataFrame,
    sigma: float = 0.2,
    r: float = 0.3,
    map_name: str = "four_rooms",
    output_path: str = "figures/fig4_mpf_vs_mpfe.png",
):
    """Fig 4: RMSE trajectory over 200 timesteps, MPF vs MPF-E.

    Answers Q4: Does MPF-E behave differently from MPF under high-stress conditions?

    Note: This requires per-step RMSE data which the sweep CSV does not
    store, so we re-run trials using the same seeds as the sweep to reproduce
    the exact same runs.
    """
    _ensure_dir(output_path)

    from brobot.sim.world import World
    from brobot.experiments.config import METHOD_REGISTRY, METHODS, MAPS, N_TRIALS

    fig, ax = plt.subplots(figsize=(12, 5))
    n_trials = N_TRIALS  # match the sweep (30)

    for method_name, color, label in [("MPF", "blue", "MPF"), ("MPFE", "red", "MPF-E")]:
        all_rmse = []
        for trial in range(n_trials):
            # Same deterministic seed formula as generate_sweep_configs (base_seed=0)
            map_idx = MAPS.index(map_name)
            seed = (0 * 100_000 + map_idx * 10_000 + trial) % (2**31)
            world = World(map_name=map_name, sigma=sigma, r=r, T=200, seed=seed)
            filter_cls = METHOD_REGISTRY[method_name]
            filt = filter_cls(
                N=500,
                occ_map_grid=world.occ_map.grid,
                resolution=world.occ_map.resolution,
                sigma=sigma,
                beam_angles_arr=world.beam_angles_arr,
            )
            method_idx = METHODS.index(method_name)
            filter_seed = (seed * 100 + method_idx) % (2**31)
            result = filt.run(world, seed=filter_seed)
            all_rmse.append(result.rmse_per_step)

        all_rmse = np.array(all_rmse)  # (n_trials, T)
        mean_rmse = all_rmse.mean(axis=0)
        std_rmse = all_rmse.std(axis=0)

        ax.plot(mean_rmse, color=color, label=label, linewidth=1.5)
        ax.fill_between(
            range(len(mean_rmse)),
            mean_rmse - std_rmse,
            mean_rmse + std_rmse,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"MPF vs MPF-E trajectory (σ={sigma}, r={r}, {map_name} map)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def figure_maps(
    output_path: str = "figures/maps.png",
):
    """Visualize the two occupancy grid maps with a sample trajectory overlay."""
    _ensure_dir(output_path)

    from brobot.sim.maps import open_map, corridor_map, four_rooms_map
    from brobot.sim.trajectory import generate_trajectory

    map_defs = [
        ("Open room",   open_map,       "open"),
        ("Corridor",    corridor_map,   "corridor"),
        ("Four rooms",  four_rooms_map, "four_rooms"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, build_fn, map_name) in zip(axes, map_defs):
        occ = build_fn()
        gt_poses, _ = generate_trajectory(occ, map_name)

        # Draw occupancy grid: walls dark, free light
        ax.imshow(
            occ.grid,
            cmap="Greys",
            origin="lower",
            extent=[0, occ.world_width, 0, occ.world_height],
        )

        # Overlay trajectory
        ax.plot(
            gt_poses[:, 0], gt_poses[:, 1],
            color="#e74c3c", linewidth=1.5, label="Trajectory",
        )
        ax.plot(
            gt_poses[0, 0], gt_poses[0, 1],
            "go", markersize=7, label="Start", zorder=5,
        )
        ax.plot(
            gt_poses[-1, 0], gt_poses[-1, 1],
            "rs", markersize=7, label="End", zorder=5,
        )

        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def generate_all_figures(csv_path: str = "results/sweep.csv"):
    """Generate all 4 paper figures plus map visualization from sweep results."""
    df = pd.read_csv(csv_path)
    os.makedirs("figures", exist_ok=True)

    figure_maps()
    figure1_rmse_heatmaps(df)
    figure2_map_geometry(df)
    figure3_ablation(df)
    figure4_mpf_vs_mpfe(df)
    print("All figures generated.")
