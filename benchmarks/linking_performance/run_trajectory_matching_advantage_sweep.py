from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_and_compare as sim


OUT_DIR = sim.OUT_DIR
_WORKER_TEMPLATES: list[sim.PsfTemplate] | None = None


def _condition_name(condition: dict[str, object]) -> str:
    name = str(condition["condition"])
    return name.replace(".", "p").replace("-", "m").replace(" ", "_")


def _parse_sources(text: object) -> list[str]:
    return [part for part in str(text).split(";") if part and part != "nan"]


def _group_has_real_source(row: pd.Series) -> bool:
    sources = _parse_sources(row.get("member_source_ids", ""))
    if any(source.startswith("P") for source in sources):
        return True
    pids = [part for part in str(row.get("member_particle_ids", "")).split(";") if part and part != "nan"]
    return any(int(float(pid)) > 0 for pid in pids)


def _hard_real_group_correct(row: pd.Series) -> bool:
    real_sources = [source for source in _parse_sources(row.get("member_source_ids", "")) if source.startswith("P")]
    if not real_sources:
        return False
    return len(set(real_sources)) == 1


def _fair_metrics_for_method(
    method: str,
    groups: pd.DataFrame,
    recovered: pd.DataFrame,
    expected_source_frames: pd.DataFrame,
    n_frames: int,
) -> dict[str, object]:
    if groups.empty:
        return {
            "method": method,
            "real_group_association_accuracy": 0.0,
            "hard_real_group_association_accuracy": 0.0,
            "real_group_count": 0,
            "pure_clutter_group_count": 0,
            "pure_clutter_groups_per_frame": 0.0,
            "real_source_recall": 0.0,
            "real_source_recall_recovered": 0,
            "real_source_recall_expected": int(len(expected_source_frames)),
        }

    has_real = groups.apply(_group_has_real_source, axis=1)
    real_groups = groups[has_real]
    pure_clutter_count = int((~has_real).sum())
    if real_groups.empty:
        real_assoc = 0.0
        hard_real_assoc = 0.0
    else:
        real_assoc = float(real_groups["is_correct"].astype(bool).mean())
        hard_real_assoc = float(real_groups.apply(_hard_real_group_correct, axis=1).mean())

    if recovered.empty or expected_source_frames.empty:
        recovered_count = 0
        real_source_recall = 0.0
    else:
        valid = recovered[recovered.get("is_correct", False).astype(bool)].copy()
        valid = valid[valid["assigned_source_id"].astype(str).str.startswith("P")]
        got = (
            valid[["FRAME", "assigned_source_id"]]
            .drop_duplicates()
            .rename(columns={"assigned_source_id": "source_id"})
        )
        recovered_count = int(
            len(expected_source_frames[["FRAME", "source_id"]].merge(got, on=["FRAME", "source_id"]))
        )
        real_source_recall = float(recovered_count / len(expected_source_frames))

    return {
        "method": method,
        "real_group_association_accuracy": real_assoc,
        "hard_real_group_association_accuracy": hard_real_assoc,
        "real_group_count": int(len(real_groups)),
        "pure_clutter_group_count": pure_clutter_count,
        "pure_clutter_groups_per_frame": float(pure_clutter_count / max(n_frames, 1)),
        "real_source_recall": real_source_recall,
        "real_source_recall_recovered": recovered_count,
        "real_source_recall_expected": int(len(expected_source_frames)),
    }


def add_fair_metrics(result: dict[str, object], n_frames: int) -> pd.DataFrame:
    rows = []
    expected = result["expected_source_frames"]
    for method, groups in result["groups"].items():
        rows.append(
            _fair_metrics_for_method(
                method,
                groups,
                result["recovered"][method],
                expected,
                n_frames,
            )
        )
    return pd.DataFrame(rows)


def _init_worker() -> None:
    global _WORKER_TEMPLATES
    _WORKER_TEMPLATES = sim.load_psf_templates()


def _run_one(task: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global _WORKER_TEMPLATES
    templates = _WORKER_TEMPLATES
    if templates is None:
        templates = sim.load_psf_templates()
        _WORKER_TEMPLATES = templates

    condition = task["condition"]
    assert isinstance(condition, dict)
    seed = int(task["seed"])
    n_frames = int(task["n_frames"])
    n_particles = int(task["n_particles"])
    run_dir = Path(str(task["run_dir"]))
    force_rerun = bool(task["force_rerun"])

    required_outputs = [
        run_dir / "metrics_summary.csv",
        run_dir / "fair_metrics_summary.csv",
        run_dir / "groups" / "trajectory_lap_cost.csv",
        run_dir / "lap_cost_raw_image_spot_tracks.csv",
    ]
    if not force_rerun and all(path.exists() for path in required_outputs):
        summary = pd.read_csv(run_dir / "metrics_summary.csv")
        fair = pd.read_csv(run_dir / "fair_metrics_summary.csv")
    else:
        result = sim.run_experiment(
            templates,
            seed=seed,
            n_frames=n_frames,
            dropout_prob=float(condition["dropout_prob"]),
            center_separation_px=float(condition["center_separation_px"]),
            clutter_points_per_frame=int(condition["clutter_points_per_frame"]),
            n_particles=n_particles,
            z_range_um=(4.0, 16.0),
        )
        sim.save_sweep_run_details(result, run_dir)
        fair = add_fair_metrics(result, n_frames=n_frames)
        fair.to_csv(run_dir / "fair_metrics_summary.csv", index=False)
        summary = result["summary"].copy()

    for frame in [summary, fair]:
        frame["condition"] = str(condition["condition"])
        frame["seed"] = seed
        frame["dropout_prob"] = float(condition["dropout_prob"])
        frame["center_separation_px"] = float(condition["center_separation_px"])
        frame["clutter_points_per_frame"] = int(condition["clutter_points_per_frame"])
        front_cols = ["condition", "seed", "dropout_prob", "center_separation_px", "clutter_points_per_frame"]
        other_cols = [col for col in frame.columns if col not in front_cols]
        frame = frame[front_cols + other_cols]
    return summary, fair


def summarize_advantage(per_run: pd.DataFrame, fair_per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        per_run.groupby(["condition", "dropout_prob", "center_separation_px", "clutter_points_per_frame", "method"])
        .agg(
            n=("seed", "nunique"),
            association_accuracy_mean=("association_accuracy", "mean"),
            association_accuracy_sd=("association_accuracy", "std"),
            source_recall_mean=("source_recall", "mean"),
            source_recall_sd=("source_recall", "std"),
        )
        .reset_index()
    )
    fair_summary = (
        fair_per_run.groupby(["condition", "dropout_prob", "center_separation_px", "clutter_points_per_frame", "method"])
        .agg(
            n=("seed", "nunique"),
            real_group_association_accuracy_mean=("real_group_association_accuracy", "mean"),
            real_group_association_accuracy_sd=("real_group_association_accuracy", "std"),
            hard_real_group_association_accuracy_mean=("hard_real_group_association_accuracy", "mean"),
            hard_real_group_association_accuracy_sd=("hard_real_group_association_accuracy", "std"),
            real_source_recall_mean=("real_source_recall", "mean"),
            real_source_recall_sd=("real_source_recall", "std"),
            pure_clutter_groups_per_frame_mean=("pure_clutter_groups_per_frame", "mean"),
            pure_clutter_groups_per_frame_sd=("pure_clutter_groups_per_frame", "std"),
        )
        .reset_index()
    )
    return summary, fair_summary


def plot_advantage(fair_summary: pd.DataFrame, out: Path) -> None:
    methods = [
        "framewise_geometry",
        "geometry_previous_frame",
        "geometry_previous_frame_linked",
        "trajectory_lap_cost",
        "oracle_grouping",
    ]
    labels = {
        "framewise_geometry": "Single-frame geometry",
        "geometry_previous_frame": "Frame-wise + previous frame",
        "geometry_previous_frame_linked": "Frame-wise + linked output",
        "trajectory_lap_cost": "Trajectory-level matching",
        "oracle_grouping": "Oracle",
    }
    colors = {
        "framewise_geometry": "#d62728",
        "geometry_previous_frame": "#ff7f0e",
        "geometry_previous_frame_linked": "#a65628",
        "trajectory_lap_cost": "#0072b2",
        "oracle_grouping": "#4d4d4d",
    }
    main = fair_summary[fair_summary["condition"].astype(str).str.startswith("dropout")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharex=True)
    for method in methods:
        sub = main[main["method"] == method].sort_values("dropout_prob")
        if sub.empty:
            continue
        x = sub["dropout_prob"].to_numpy(float)
        for ax, metric, ylabel in [
            (axes[0], "real_group_association_accuracy", "Real-particle group accuracy"),
            (axes[1], "real_source_recall", "Real source-frame recall"),
        ]:
            y = sub[f"{metric}_mean"].to_numpy(float)
            e = sub[f"{metric}_sd"].fillna(0).to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=e,
                marker="o",
                capsize=3,
                lw=1.8,
                color=colors[method],
                label=labels[method],
                linestyle="--" if method == "oracle_grouping" else "-",
                alpha=0.65 if method == "oracle_grouping" else 1.0,
            )
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Sub-spot dropout probability")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def run(tag: str, n_seeds: int = 20, n_frames: int = 40, workers: int = 4, force_rerun: bool = False) -> None:
    separations = [8.0, 18.0, 55.0, 120.0]
    dropouts = [0.0, 0.10, 0.20]
    conditions = [
        {
            "condition": f"dropout{int(round(dropout * 100))}_clutter8_sep{int(separation)}",
            "dropout_prob": dropout,
            "center_separation_px": separation,
            "clutter_points_per_frame": 8,
        }
        for separation in separations
        for dropout in dropouts
    ]
    runs_root = OUT_DIR / f"trajectory_matching_advantage_runs_{tag}"
    rows = []
    fair_rows = []
    tasks = []
    for condition in conditions:
        for seed in range(n_seeds):
            tasks.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "n_frames": n_frames,
                    "n_particles": 4,
                    "run_dir": runs_root / _condition_name(condition) / f"seed_{seed:03d}",
                    "force_rerun": force_rerun,
                }
            )

    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            summary, fair = _run_one(task)
            print(f"finished {summary['condition'].iloc[0]} seed={int(summary['seed'].iloc[0]):03d} ({i}/{len(tasks)})", flush=True)
            rows.append(summary)
            fair_rows.append(fair)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
            futures = [executor.submit(_run_one, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                summary, fair = future.result()
                print(f"finished {summary['condition'].iloc[0]} seed={int(summary['seed'].iloc[0]):03d} ({i}/{len(tasks)})", flush=True)
                rows.append(summary)
                fair_rows.append(fair)

    per_run = pd.concat(rows, ignore_index=True)
    fair_per_run = pd.concat(fair_rows, ignore_index=True)
    summary, fair_summary = summarize_advantage(per_run, fair_per_run)

    per_run_path = OUT_DIR / f"trajectory_matching_advantage_per_run_{tag}.csv"
    fair_per_run_path = OUT_DIR / f"trajectory_matching_advantage_fair_per_run_{tag}.csv"
    summary_path = OUT_DIR / f"trajectory_matching_advantage_summary_{tag}.csv"
    fair_summary_path = OUT_DIR / f"trajectory_matching_advantage_fair_summary_{tag}.csv"
    fig_path = OUT_DIR / f"trajectory_matching_advantage_fair_metrics_{tag}.png"
    per_run.to_csv(per_run_path, index=False)
    fair_per_run.to_csv(fair_per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    fair_summary.to_csv(fair_summary_path, index=False)
    plot_advantage(fair_summary, fig_path)

    core_methods = [
        "framewise_geometry",
        "geometry_previous_frame",
        "geometry_previous_frame_linked",
        "trajectory_lap_cost",
        "oracle_grouping",
    ]
    print(
        fair_summary[fair_summary["method"].isin(core_methods)]
        .sort_values(["condition", "method"])
        .to_string(index=False),
        flush=True,
    )
    print(f"Runs: {runs_root}", flush=True)
    print(f"Fair summary: {fair_summary_path}", flush=True)
    print(f"Figure: {fig_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="dropout_clutter_close_crossing")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    run(tag=args.tag, n_seeds=args.n_seeds, n_frames=args.frames, workers=args.workers, force_rerun=args.force_rerun)
