from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import simulate_and_compare as sim


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
TAG = "dropout_clutter_close_crossing_20seed_40f"
RUNS_ROOT = OUT_DIR / f"trajectory_matching_advantage_runs_{TAG}"

METHODS = [
    "framewise_geometry",
    "geometry_previous_frame",
    "geometry_previous_frame_linked",
    "trajectory_lap_cost",
    "oracle_grouping",
]

TRACK_FILE_BY_METHOD = {
    "trajectory_level": "repaired_interpolated_image_spot_tracks.csv",
    "trajectory_lap_cost": "lap_cost_repaired_interpolated_image_spot_tracks.csv",
}


def _split_sources(text: object, sep: str = ";") -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return [part.strip() for part in str(text).split(sep) if part.strip() and part.strip() != "nan"]


def _source_set(possible: object, fallback: object = "") -> set[str]:
    sources = set(_split_sources(possible, sep=";"))
    if not sources:
        sources = set(_split_sources(fallback, sep=";"))
    return sources


def _is_real_source(source: object) -> bool:
    return str(source).startswith("P")


def _track_source_candidates(tracks: pd.DataFrame) -> dict[int, set[str]]:
    candidates: dict[int, set[str]] = {}
    if tracks.empty or "TRACK_ID" not in tracks.columns:
        return candidates
    for tid, group in tracks.groupby("TRACK_ID"):
        source_set: set[str] = set()
        for _, row in group.iterrows():
            source_set.update(_source_set(row.get("possible_source_ids", ""), row.get("source_id", "")))
        candidates[int(tid)] = source_set
    return candidates


def _sets_from_frame_rows(rows: pd.DataFrame, track_candidates: dict[int, set[str]] | None = None) -> list[set[str]]:
    source_sets = []
    track_candidates = track_candidates or {}
    for _, row in rows.iterrows():
        source_set = _source_set(row.get("possible_source_ids", ""), row.get("source_id", ""))
        if not source_set and "TRACK_ID" in row:
            source_set = set(track_candidates.get(int(row["TRACK_ID"]), set()))
        source_sets.append(source_set)
    return source_sets


def _sets_from_group_row(row: pd.Series) -> list[set[str]]:
    hard_sources = _split_sources(row.get("member_source_ids", ""), sep=";")
    possible_parts = _split_sources(row.get("member_possible_source_ids", ""), sep="|")
    n_parts = max(len(hard_sources), len(possible_parts))
    source_sets = []
    for idx in range(n_parts):
        possible = possible_parts[idx] if idx < len(possible_parts) else ""
        fallback = hard_sources[idx] if idx < len(hard_sources) else ""
        source_sets.append(_source_set(possible, fallback))
    return source_sets


def _score_source_sets(source_sets: list[set[str]], assigned_source_id: object) -> dict[str, object]:
    assigned = str(assigned_source_id)
    informative = [set(s) for s in source_sets if s]
    if not informative:
        return {
            "has_real_candidate": False,
            "strict_overlap_is_correct": False,
            "strict_overlap_status": "STRICT_CLUTTER_OR_UNLABELED",
            "strict_overlap_assigned_source_id": "",
        }

    has_real_candidate = any(any(_is_real_source(source) for source in source_set) for source_set in informative)
    common = set.intersection(*informative)
    common_real = {source for source in common if _is_real_source(source)}
    assigned_is_real = _is_real_source(assigned)
    is_correct = bool(assigned_is_real and assigned in common_real)

    if not has_real_candidate:
        status = "STRICT_CLUTTER"
    elif is_correct and any(len(source_set) > 1 for source_set in informative):
        status = "STRICT_OVERLAP"
    elif is_correct:
        status = "STRICT_OK"
    elif common_real:
        status = "STRICT_ID_SWITCH"
    else:
        status = "STRICT_MIXED"

    return {
        "has_real_candidate": bool(has_real_candidate),
        "strict_overlap_is_correct": bool(is_correct),
        "strict_overlap_status": status,
        "strict_overlap_assigned_source_id": assigned if assigned_is_real else "",
    }


def _best_four_if_needed(frame_df: pd.DataFrame) -> pd.DataFrame:
    if len(frame_df) <= 4:
        return frame_df
    pts = frame_df[["X", "Y"]].to_numpy(float)
    best_score = np.inf
    best_combo: tuple[int, ...] | None = None
    for combo in combinations(range(len(frame_df)), 4):
        combo_pts = pts[list(combo)]
        metrics = sim.four_spot_shape_metrics(combo_pts)
        score = float(metrics["score"])
        if 45.0 <= float(metrics["mean_diag"]) <= 130.0 and score < best_score:
            best_score = score
            best_combo = combo
    if best_combo is None:
        return frame_df
    return frame_df.iloc[list(best_combo)]


def _framewise_units(groups: pd.DataFrame) -> list[dict[str, object]]:
    units = []
    for _, row in groups.iterrows():
        score = _score_source_sets(_sets_from_group_row(row), row.get("assigned_source_id", ""))
        units.append(
            {
                "FRAME": int(row["FRAME"]) if "FRAME" in row and pd.notna(row["FRAME"]) else -1,
                "assigned_source_id": score["strict_overlap_assigned_source_id"],
                "has_real_candidate": score["has_real_candidate"],
                "strict_overlap_is_correct": score["strict_overlap_is_correct"],
                "strict_overlap_status": score["strict_overlap_status"],
            }
        )
    return units


def _trajectory_units(groups: pd.DataFrame, tracks: pd.DataFrame) -> list[dict[str, object]]:
    units = []
    if groups.empty or tracks.empty:
        return units
    track_candidates = _track_source_candidates(tracks)
    for _, group in groups.iterrows():
        tids = [int(tid) for tid in _split_sources(group.get("member_track_ids", ""), sep=";")]
        if not tids:
            continue
        track_rows = tracks[tracks["TRACK_ID"].astype(int).isin(tids)]
        start_frame = group.get("segment_start_frame", np.nan)
        end_frame = group.get("segment_end_frame", np.nan)
        if pd.notna(start_frame):
            track_rows = track_rows[track_rows["FRAME"].astype(int) >= int(start_frame)]
        if pd.notna(end_frame):
            track_rows = track_rows[track_rows["FRAME"].astype(int) <= int(end_frame)]
        for frame, frame_df in track_rows.groupby("FRAME"):
            if len(frame_df) < 3:
                continue
            selected = _best_four_if_needed(frame_df)
            score = _score_source_sets(_sets_from_frame_rows(selected, track_candidates), group.get("assigned_source_id", ""))
            units.append(
                {
                    "FRAME": int(frame),
                    "assigned_source_id": score["strict_overlap_assigned_source_id"],
                    "has_real_candidate": score["has_real_candidate"],
                    "strict_overlap_is_correct": score["strict_overlap_is_correct"],
                    "strict_overlap_status": score["strict_overlap_status"],
                }
            )
    return units


def _strict_metrics_for_method(
    run_dir: Path,
    method: str,
    expected: pd.DataFrame,
    regroup_trajectory_lap_cost: bool = False,
) -> dict[str, object]:
    group_path = run_dir / "groups" / f"{method}.csv"
    groups = pd.DataFrame()
    if method == "trajectory_lap_cost" and regroup_trajectory_lap_cost:
        track_path = run_dir / TRACK_FILE_BY_METHOD[method]
        if track_path.exists():
            groups = sim.trajectory_lap_cost_group(pd.read_csv(track_path))
    elif group_path.exists():
        groups = pd.read_csv(group_path)

    if groups.empty:
        return {
            "method": method,
            "strict_overlap_association_accuracy": 0.0,
            "strict_overlap_real_source_recall": 0.0,
            "strict_overlap_real_group_count": 0,
            "strict_overlap_correct_group_count": 0,
            "strict_overlap_pure_clutter_group_count": 0,
            "strict_overlap_recovered_source_frames": 0,
            "strict_overlap_expected_source_frames": int(len(expected)),
        }

    if method in TRACK_FILE_BY_METHOD:
        track_path = run_dir / TRACK_FILE_BY_METHOD[method]
        tracks = pd.read_csv(track_path) if track_path.exists() else pd.DataFrame()
        units = _trajectory_units(groups, tracks)
    else:
        units = _framewise_units(groups)

    if not units:
        real_units = pd.DataFrame()
    else:
        real_units = pd.DataFrame(units)
        real_units = real_units[real_units["has_real_candidate"].astype(bool)].copy()

    pure_clutter_count = int(len(units) - len(real_units))
    correct_count = int(real_units["strict_overlap_is_correct"].astype(bool).sum()) if not real_units.empty else 0
    real_count = int(len(real_units))
    accuracy = float(correct_count / real_count) if real_count else 0.0

    if expected.empty or real_units.empty:
        recovered_count = 0
        recall = 0.0
    else:
        got = (
            real_units[real_units["strict_overlap_is_correct"].astype(bool)][["FRAME", "assigned_source_id"]]
            .drop_duplicates()
            .rename(columns={"assigned_source_id": "source_id"})
        )
        recovered_count = int(len(expected[["FRAME", "source_id"]].merge(got, on=["FRAME", "source_id"])))
        recall = float(recovered_count / len(expected)) if len(expected) else 0.0

    return {
        "method": method,
        "strict_overlap_association_accuracy": accuracy,
        "strict_overlap_real_source_recall": recall,
        "strict_overlap_real_group_count": real_count,
        "strict_overlap_correct_group_count": correct_count,
        "strict_overlap_pure_clutter_group_count": pure_clutter_count,
        "strict_overlap_recovered_source_frames": recovered_count,
        "strict_overlap_expected_source_frames": int(len(expected)),
    }


def _metadata_from_run(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        first = metrics.iloc[0]
        return {
            "condition": str(first["condition"]) if "condition" in metrics.columns else run_dir.parent.name,
            "seed": int(first["seed"]) if "seed" in metrics.columns else int(run_dir.name.split("_")[-1]),
            "dropout_prob": float(first["dropout_prob"]),
            "center_separation_px": float(first["center_separation_px"]),
            "clutter_points_per_frame": int(first["clutter_points_per_frame"]),
        }
    condition = run_dir.parent.name
    seed = int(run_dir.name.split("_")[-1])
    parts = condition.split("_")
    dropout = float(parts[0].replace("dropout", "")) / 100.0
    sep = float(parts[-1].replace("sep", ""))
    return {
        "condition": condition,
        "seed": seed,
        "dropout_prob": dropout,
        "center_separation_px": sep,
        "clutter_points_per_frame": 8,
    }


def compute(
    runs_root: Path,
    methods: list[str],
    regroup_trajectory_lap_cost: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_dir in sorted(path for path in runs_root.glob("*/*") if path.is_dir()):
        expected_path = run_dir / "expected_source_frames.csv"
        expected = pd.read_csv(expected_path) if expected_path.exists() else pd.DataFrame(columns=["FRAME", "source_id"])
        metadata = _metadata_from_run(run_dir)
        for method in methods:
            row = _strict_metrics_for_method(
                run_dir,
                method,
                expected,
                regroup_trajectory_lap_cost=regroup_trajectory_lap_cost,
            )
            rows.append({**metadata, **row})

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["condition", "dropout_prob", "center_separation_px", "clutter_points_per_frame", "method"])
        .agg(
            n=("seed", "nunique"),
            strict_overlap_association_accuracy_mean=("strict_overlap_association_accuracy", "mean"),
            strict_overlap_association_accuracy_sd=("strict_overlap_association_accuracy", "std"),
            strict_overlap_real_source_recall_mean=("strict_overlap_real_source_recall", "mean"),
            strict_overlap_real_source_recall_sd=("strict_overlap_real_source_recall", "std"),
            strict_overlap_real_group_count_mean=("strict_overlap_real_group_count", "mean"),
            strict_overlap_pure_clutter_group_count_mean=("strict_overlap_pure_clutter_group_count", "mean"),
        )
        .reset_index()
    )
    return per_run, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--regroup-trajectory-lap-cost", action="store_true")
    args = parser.parse_args()

    runs_root = args.runs_root or OUT_DIR / f"trajectory_matching_advantage_runs_{args.tag}"
    output_tag = args.output_tag or args.tag
    per_run, summary = compute(
        runs_root,
        args.methods,
        regroup_trajectory_lap_cost=args.regroup_trajectory_lap_cost,
    )

    per_run_path = OUT_DIR / f"strict_overlap_tolerant_per_run_{output_tag}.csv"
    summary_path = OUT_DIR / f"strict_overlap_tolerant_summary_{output_tag}.csv"
    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Per-run: {per_run_path}")
    print(f"Summary: {summary_path}")
    print(
        summary[summary["method"].isin(["framewise_geometry", "geometry_previous_frame_linked", "trajectory_lap_cost"])]
        .sort_values(["condition", "method"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
