from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import simulate_and_compare as sim


OUT_DIR = sim.OUT_DIR


def _paths(tag: str) -> tuple[Path, Path, Path, Path, Path]:
    suffix = "" if not tag else f"_{tag}"
    runs_root = OUT_DIR / f"center_separation_linked_sweep_runs{suffix}"
    per_run_csv = OUT_DIR / f"center_separation_linked_sweep_per_run_metrics{suffix}.csv"
    summary_csv = OUT_DIR / f"center_separation_linked_sweep_summary_mean_sd{suffix}.csv"
    fig_path = OUT_DIR / f"center_separation_linked_core_comparison{suffix}.png"
    full_fig_path = OUT_DIR / f"center_separation_linked_sweep_all_methods{suffix}.png"
    return runs_root, per_run_csv, summary_csv, fig_path, full_fig_path


def run(
    dropout_prob: float = 0.045,
    tag: str = "",
    n_seeds: int = 20,
    separation_values: list[float] | None = None,
    z_range_um: tuple[float, float] = sim.DEFAULT_SIM_Z_RANGE_UM,
    force_rerun: bool = False,
) -> None:
    templates = sim.load_psf_templates()
    rows = []
    if separation_values is None:
        separation_values = [8.0, 18.0, 32.0, 55.0, 120.0]
    n_frames = 120
    n_particles = 4
    clutter_points_per_frame = 0
    runs_root, per_run_csv, summary_csv, fig_path, full_fig_path = _paths(tag)

    for sep in separation_values:
        condition_dir = runs_root / sim.safe_condition_name("center_separation_px", sep)
        for seed in range(n_seeds):
            run_dir = condition_dir / f"seed_{seed:03d}"
            required = [
                run_dir / "simulated_stack.tif",
                run_dir / "metrics_summary.csv",
                run_dir / "groups" / "geometry_previous_frame_linked.csv",
                run_dir / "groups" / "trajectory_lap_cost.csv",
            ]
            if not force_rerun and all(path.exists() for path in required):
                summary = pd.read_csv(run_dir / "metrics_summary.csv")
                print(f"reused center_separation={sep:g} seed={seed + 1}/{n_seeds}", flush=True)
            else:
                result = sim.run_experiment(
                    templates,
                    seed=seed,
                    n_frames=n_frames,
                    dropout_prob=dropout_prob,
                    center_separation_px=sep,
                    clutter_points_per_frame=clutter_points_per_frame,
                    n_particles=n_particles,
                    z_range_um=z_range_um,
                )
                sim.save_sweep_run_details(result, run_dir)
                summary = result["summary"].copy()
                print(f"saved center_separation={sep:g} seed={seed + 1}/{n_seeds}", flush=True)
            summary.insert(0, "sweep_name", "center_separation_px")
            summary.insert(1, "sweep_value", sep)
            rows.append(summary)

    per_run = pd.concat(rows, ignore_index=True)
    per_run.to_csv(per_run_csv, index=False)
    summary = sim.summarize_sweep(per_run)
    summary.to_csv(summary_csv, index=False)
    sim.plot_sweep(summary, full_fig_path)
    plot_core(summary, fig_path)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved outputs to {OUT_DIR}", flush=True)
    print(f"Per-run CSV: {per_run_csv}", flush=True)
    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Core figure: {fig_path}", flush=True)


def plot_core(summary: pd.DataFrame, out: Path) -> None:
    methods = [
        "framewise_geometry",
        "geometry_previous_frame",
        "geometry_previous_frame_linked",
        "trajectory_lap_cost",
        "oracle_grouping",
    ]
    labels = {
        "framewise_geometry": "Frame-wise geometry",
        "geometry_previous_frame": "Geometry + previous frame",
        "geometry_previous_frame_linked": "Geometry + linked trajectory",
        "trajectory_lap_cost": "Trajectory-level",
        "oracle_grouping": "Oracle",
    }
    colors = {
        "framewise_geometry": "#d62728",
        "geometry_previous_frame": "#ff7f0e",
        "geometry_previous_frame_linked": "#a65628",
        "trajectory_lap_cost": "#0072b2",
        "oracle_grouping": "#4d4d4d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("sweep_value")
        if sub.empty:
            continue
        x = sub["sweep_value"].astype(float).to_numpy()
        for ax, metric, ylabel in [
            (axes[0], "association_accuracy", "Association accuracy"),
            (axes[1], "source_recall", "Source recall"),
        ]:
            y = sub[f"{metric}_mean"].astype(float).to_numpy()
            e = sub[f"{metric}_sd"].fillna(0).astype(float).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=e,
                marker="o",
                lw=1.8,
                capsize=3,
                color=colors[method],
                label=labels[method],
                alpha=0.65 if method == "oracle_grouping" else 1.0,
                linestyle="--" if method == "oracle_grouping" else "-",
            )
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Center separation (px)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout-prob", type=float, default=0.045)
    parser.add_argument("--tag", default="")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--separations", nargs="*", type=float, default=None)
    parser.add_argument("--z-min-um", type=float, default=sim.DEFAULT_SIM_Z_RANGE_UM[0])
    parser.add_argument("--z-max-um", type=float, default=sim.DEFAULT_SIM_Z_RANGE_UM[1])
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    run(
        dropout_prob=args.dropout_prob,
        tag=args.tag,
        n_seeds=args.n_seeds,
        separation_values=args.separations,
        z_range_um=(args.z_min_um, args.z_max_um),
        force_rerun=args.force_rerun,
    )
