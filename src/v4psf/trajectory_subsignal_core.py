from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage import feature, filters, io
import tifffile

from .csv_io import read_csv as read_pipeline_csv


ROOT = Path(__file__).resolve().parent
PSF_DIR = ROOT / "data" / "psf_20x_roi16bit"
OUT_DIR = ROOT / "outputs"
logging.getLogger("tifffile").setLevel(logging.CRITICAL)

FULL_PSF_Z_RANGE_UM = (0.0, 20.0)
DEFAULT_SIM_Z_RANGE_UM = (4.0, 16.0)

PRODUCTION_TRACK_CLASSIFICATION_PARAMS = {
    "MAX_DIAGONAL": 150.0,
    "SQUARE_TOLERANCE": 0.2,
    "NEIGHBORS_K": 20,
    "LIMIT_SQUARE": 4.0,
    "LIMIT_TRIANGLE": 0.3,
}

SPOT_APPEARANCE_COLUMNS = [
    "log_peak_intensity",
    "log_integrated_intensity",
    "spot_sigma_x",
    "spot_sigma_y",
    "spot_shape_ratio",
    "local_contrast",
]


def _runtime_float(names: str | tuple[str, ...], default: float) -> float:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            try:
                return float(value)
            except Exception:
                pass
    return float(default)


def _runtime_bool(names: str | tuple[str, ...], default: bool) -> bool:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _runtime_camera_pixel_um() -> float:
    value = _runtime_float(
        (
            "AUTODETECT_PHYSICAL_SCALE",
            "TRAJ_ASSOC_CAMERA_PIXEL_UM",
            "POSITION_XY_PHYSICAL_SCALE",
        ),
        4.5,
    )
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Camera pixel size must be positive, got {value!r}")
    return float(value)


def runtime_sensor_distance_px(
    sensor_names: str | tuple[str, ...],
    legacy_pixel_names: str | tuple[str, ...],
    default_sensor_um: float,
) -> float:
    """Resolve one spatial gate in current-image pixels.

    Physical camera-sensor distances are authoritative. Legacy pixel-valued
    environment variables remain a compatibility fallback, but a missing
    environment variable can no longer resurrect a fixed 4.5-um-camera pixel
    default on a binned or different-pixel camera.
    """
    if isinstance(sensor_names, str):
        sensor_names = (sensor_names,)
    if isinstance(legacy_pixel_names, str):
        legacy_pixel_names = (legacy_pixel_names,)
    for name in sensor_names:
        value = os.getenv(name)
        if value not in (None, ""):
            distance_um = float(value)
            if not np.isfinite(distance_um) or distance_um <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            return distance_um / _runtime_camera_pixel_um()
    for name in legacy_pixel_names:
        value = os.getenv(name)
        if value not in (None, ""):
            distance_px = float(value)
            if not np.isfinite(distance_px) or distance_px <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            return distance_px
    return float(default_sensor_um) / _runtime_camera_pixel_um()


def _reference_pixels_in_current_image(reference_pixels: float) -> float:
    """Convert an old 4.5-um-camera pixel constant to current image pixels."""
    return float(reference_pixels) * 4.5 / _runtime_camera_pixel_um()


def _runtime_diag_bounds() -> tuple[float, float]:
    bounds = (
        runtime_sensor_distance_px(
            ("TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM", "AUTODETECT_MIN_DIAGONAL_SENSOR_UM"),
            ("TRAJ_ASSOC_MIN_DIAGONAL", "AUTODETECT_MIN_DIAGONAL"),
            202.5,
        ),
        runtime_sensor_distance_px(
            ("TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM", "AUTODETECT_MAX_DIAGONAL_SENSOR_UM"),
            ("TRAJ_ASSOC_MAX_DIAGONAL", "AUTODETECT_MAX_DIAGONAL"),
            585.0,
        ),
    )
    if bounds[1] <= bounds[0]:
        raise ValueError(f"Invalid trajectory diagonal bounds in current pixels: {bounds!r}")
    return bounds


@dataclass
class PsfTemplate:
    file_index: int
    z_um: float
    image: np.ndarray
    source_center_rc: np.ndarray
    offsets_rc: np.ndarray
    mean_diag_px: float


@dataclass
class TrajectoryParams:
    base_y: float
    p1_x0: float
    p1_x1: float
    p2_x0: float
    p2_x1: float
    p1_phase: float
    p2_phase: float
    p3_phase: float
    p1_y_amp: float
    p2_y_amp: float
    p3_x0: float
    p3_y0: float
    p3_y1: float
    p3_x_amp: float
    z_phase1: float
    z_phase2: float
    z_phase3: float
    clutter_cx: float
    clutter_cy: float


def compress_z_to_range(z: np.ndarray | float, z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM):
    z_min, z_max = map(float, z_range_um)
    full_min, full_max = FULL_PSF_Z_RANGE_UM
    if z_max <= z_min:
        raise ValueError(f"Invalid z_range_um={z_range_um}; expected z_max > z_min")
    z_arr = np.asarray(z, dtype=float)
    z_norm = (np.clip(z_arr, full_min, full_max) - full_min) / (full_max - full_min)
    mapped = z_min + z_norm * (z_max - z_min)
    if np.isscalar(z):
        return float(mapped)
    return mapped


def make_trajectory_params(rng: np.random.Generator) -> TrajectoryParams:
    return TrajectoryParams(
        base_y=float(120 + rng.normal(0, 5.0)),
        p1_x0=float(82 + rng.normal(0, 5.0)),
        p1_x1=float(188 + rng.normal(0, 5.0)),
        p2_x0=float(188 + rng.normal(0, 5.0)),
        p2_x1=float(82 + rng.normal(0, 5.0)),
        p1_phase=float(rng.uniform(-0.7, 0.7)),
        p2_phase=float(rng.uniform(-0.7, 0.7)),
        p3_phase=float(rng.uniform(0, 2 * np.pi)),
        p1_y_amp=float(rng.uniform(4.0, 9.0)),
        p2_y_amp=float(rng.uniform(4.0, 9.0)),
        p3_x0=float(300 + rng.normal(0, 8.0)),
        p3_y0=float(100 + rng.normal(0, 6.0)),
        p3_y1=float(178 + rng.normal(0, 6.0)),
        p3_x_amp=float(rng.uniform(12.0, 24.0)),
        z_phase1=float(rng.uniform(-0.5, 0.5)),
        z_phase2=float(rng.uniform(-0.5, 0.5)),
        z_phase3=float(rng.uniform(0, 2 * np.pi)),
        clutter_cx=float(135 + rng.normal(0, 9.0)),
        clutter_cy=float(120 + rng.normal(0, 7.0)),
    )


def generate_clutter_paths(
    rng: np.random.Generator,
    n_frames: int,
    clutter_points_per_frame: int,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> list[dict]:
    """Fast drifting PSF-like clutter crossing the field of view."""
    n_paths = max(1, int(np.ceil(clutter_points_per_frame / 4)))
    paths = []
    for j in range(n_paths):
        side = int(rng.integers(0, 4))
        if side == 0:  # left to right
            x0, y0 = -40.0, float(rng.uniform(50, 250))
            x1, y1 = 430.0, float(rng.uniform(50, 250))
        elif side == 1:  # right to left
            x0, y0 = 430.0, float(rng.uniform(50, 250))
            x1, y1 = -40.0, float(rng.uniform(50, 250))
        elif side == 2:  # top to bottom
            x0, y0 = float(rng.uniform(45, 345)), -35.0
            x1, y1 = float(rng.uniform(45, 345)), 335.0
        else:  # bottom to top
            x0, y0 = float(rng.uniform(45, 345)), 335.0
            x1, y1 = float(rng.uniform(45, 345)), -35.0
        start = int(rng.integers(0, max(1, n_frames // 3)))
        duration = int(rng.integers(max(5, n_frames // 5), max(6, n_frames // 2)))
        paths.append(
            {
                "path_id": j,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "start": start,
                "duration": duration,
                "z": float(rng.uniform(float(z_range_um[0]), float(z_range_um[1]))),
                "amp": float(rng.uniform(0.35, 0.8)),
                "phase": float(rng.uniform(0, 2 * np.pi)),
            }
        )
    return paths


def generate_particle_trajectories(
    rng: np.random.Generator,
    n_frames: int,
    n_particles: int,
    center_separation_px: float,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate seed-specific 3D trajectories with varied morphologies."""
    t = np.linspace(0, 1, n_frames)
    rows = []
    param_rows = []

    # Two deliberately interacting particles create the ambiguous crossing zone.
    base_y = float(118 + rng.normal(0, 7))
    crossing_specs = [
        {
            "pid": 1,
            "x0": float(70 + rng.normal(0, 8)),
            "x1": float(205 + rng.normal(0, 8)),
            "y_offset": -center_separation_px / 2,
            "phase": float(rng.uniform(-np.pi, np.pi)),
            "amp": float(rng.uniform(6, 18)),
            "z0": float(rng.uniform(2, 6)),
            "z1": float(rng.uniform(13, 19)),
        },
        {
            "pid": 2,
            "x0": float(205 + rng.normal(0, 8)),
            "x1": float(70 + rng.normal(0, 8)),
            "y_offset": center_separation_px / 2,
            "phase": float(rng.uniform(-np.pi, np.pi)),
            "amp": float(rng.uniform(6, 18)),
            "z0": float(rng.uniform(14, 19)),
            "z1": float(rng.uniform(2, 7)),
        },
    ]
    for spec in crossing_specs[: min(2, n_particles)]:
        pid = spec["pid"]
        x = spec["x0"] + (spec["x1"] - spec["x0"]) * t
        y = base_y + spec["y_offset"] + spec["amp"] * np.sin(2 * np.pi * t + spec["phase"])
        z = spec["z0"] + (spec["z1"] - spec["z0"]) * t + rng.uniform(0.8, 2.4) * np.sin(
            2 * np.pi * t + rng.uniform(-np.pi, np.pi)
        )
        z = compress_z_to_range(z, z_range_um)
        for frame in range(n_frames):
            rows.append(
                {
                    "FRAME": frame,
                    "particle_id": pid,
                    "X_true": float(np.clip(x[frame], 45, 345)),
                    "Y_true": float(np.clip(y[frame], 55, 245)),
                    "Z_true": float(z[frame]),
                }
            )
        param_rows.append(
            {
                "particle_id": pid,
                "trajectory_type": "crossing",
                **spec,
                "sim_z_min_um": float(z_range_um[0]),
                "sim_z_max_um": float(z_range_um[1]),
                "z_mean": float(np.mean(z)),
                "z_range": float(np.max(z) - np.min(z)),
            }
        )

    # Additional particles are intentionally heterogeneous: drift, arc, oscillation.
    for pid in range(3, n_particles + 1):
        kind = ["drift", "arc", "oscillatory", "loop"][int(rng.integers(0, 4))]
        x0 = float(rng.uniform(55, 330))
        y0 = float(rng.uniform(65, 225))
        if kind == "drift":
            x = x0 + rng.uniform(-55, 55) * t + rng.uniform(5, 18) * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
            y = y0 + rng.uniform(-45, 45) * t + rng.uniform(4, 14) * np.cos(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
        elif kind == "arc":
            radius = rng.uniform(18, 45)
            theta0 = rng.uniform(0, 2 * np.pi)
            theta1 = theta0 + rng.uniform(0.8 * np.pi, 1.8 * np.pi)
            theta = theta0 + (theta1 - theta0) * t
            x = x0 + radius * np.cos(theta)
            y = y0 + 0.65 * radius * np.sin(theta)
        elif kind == "oscillatory":
            x = x0 + rng.uniform(-20, 20) * t + rng.uniform(18, 38) * np.sin(3 * np.pi * t + rng.uniform(0, 2 * np.pi))
            y = y0 + rng.uniform(-20, 20) * t + rng.uniform(10, 28) * np.sin(5 * np.pi * t + rng.uniform(0, 2 * np.pi))
        else:
            radius_x = rng.uniform(16, 36)
            radius_y = rng.uniform(12, 30)
            theta = rng.uniform(0, 2 * np.pi) + rng.uniform(1.2, 2.3) * 2 * np.pi * t
            x = x0 + radius_x * np.cos(theta)
            y = y0 + radius_y * np.sin(theta)
        z = rng.uniform(3, 15) + rng.uniform(2, 6) * np.sin(
            rng.uniform(1.5, 4.5) * np.pi * t + rng.uniform(0, 2 * np.pi)
        )
        z = compress_z_to_range(z, z_range_um)
        for frame in range(n_frames):
            rows.append(
                {
                    "FRAME": frame,
                    "particle_id": pid,
                    "X_true": float(np.clip(x[frame], 45, 345)),
                    "Y_true": float(np.clip(y[frame], 55, 245)),
                    "Z_true": float(z[frame]),
                }
            )
        param_rows.append(
            {
                "particle_id": pid,
                "trajectory_type": kind,
                "x0": x0,
                "y0": y0,
                "z_mean": float(np.mean(z)),
                "z_range": float(np.max(z) - np.min(z)),
                "sim_z_min_um": float(z_range_um[0]),
                "sim_z_max_um": float(z_range_um[1]),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def intensity_centroid(roi: np.ndarray) -> tuple[float, float]:
    roi = roi.astype(float)
    bg = min(float(np.percentile(roi, 20)), float(np.median(roi)))
    signal = np.clip(roi - bg, 0, None)
    total = float(signal.sum())
    if total <= 1e-12:
        return (roi.shape[0] - 1) / 2.0, (roi.shape[1] - 1) / 2.0
    rr, cc = np.indices(signal.shape)
    return float((rr * signal).sum() / total), float((cc * signal).sum() / total)


def square_score(points_xy: np.ndarray) -> float:
    dists = np.sort(pdist(points_xy))
    if len(dists) != 6:
        return np.inf
    sides = dists[:4]
    diags = dists[4:]
    mean_side = float(np.mean(sides))
    mean_diag = float(np.mean(diags))
    if mean_side <= 0 or mean_diag <= 0:
        return np.inf
    return (
        float(np.std(sides) / mean_side)
        + float(np.std(diags) / mean_diag)
        + abs(mean_diag / mean_side - np.sqrt(2))
    )


def mean_diag(points_xy: np.ndarray) -> float:
    dists = np.sort(pdist(points_xy))
    if len(dists) != 6:
        return np.nan
    return float(np.mean(dists[4:]))


def four_spot_shape_metrics(points_xy: np.ndarray) -> dict[str, float]:
    dists = np.sort(pdist(points_xy))
    if len(dists) != 6:
        return {
            "score": np.inf,
            "mean_side": np.nan,
            "mean_diag": np.nan,
            "side_cv": np.inf,
            "diag_cv": np.inf,
            "diag_side_ratio_error": np.inf,
        }
    sides = dists[:4]
    diags = dists[4:]
    mean_side = float(np.mean(sides))
    mean_diag_value = float(np.mean(diags))
    if mean_side <= 0 or mean_diag_value <= 0:
        return {
            "score": np.inf,
            "mean_side": mean_side,
            "mean_diag": mean_diag_value,
            "side_cv": np.inf,
            "diag_cv": np.inf,
            "diag_side_ratio_error": np.inf,
        }
    side_cv = float(np.std(sides) / mean_side)
    diag_cv = float(np.std(diags) / mean_diag_value)
    ratio_error = abs(mean_diag_value / mean_side - np.sqrt(2))
    return {
        "score": side_cv + diag_cv + ratio_error,
        "mean_side": mean_side,
        "mean_diag": mean_diag_value,
        "side_cv": side_cv,
        "diag_cv": diag_cv,
        "diag_side_ratio_error": float(ratio_error),
    }


def is_four_spot_shape(
    points_xy: np.ndarray,
    diag_range: tuple[float, float] | None = None,
    max_score: float = 0.18,
    max_side_cv: float = 0.13,
    max_diag_cv: float = 0.07,
    max_ratio_error: float = 0.13,
) -> bool:
    if diag_range is None:
        diag_range = _runtime_diag_bounds()
    metrics = four_spot_shape_metrics(points_xy)
    return (
        diag_range[0] <= metrics["mean_diag"] <= diag_range[1]
        and metrics["score"] <= max_score
        and metrics["side_cv"] <= max_side_cv
        and metrics["diag_cv"] <= max_diag_cv
        and metrics["diag_side_ratio_error"] <= max_ratio_error
    )


def triangle_missing_spot_metrics(points_xy: np.ndarray) -> dict[str, float | np.ndarray]:
    if len(points_xy) != 3:
        return {"score": np.inf, "mean_diag": np.nan, "center": np.array([np.nan, np.nan])}
    dmat = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    if i == j:
        return {"score": np.inf, "mean_diag": np.nan, "center": np.array([np.nan, np.nan])}
    right_idx = int(({0, 1, 2} - {int(i), int(j)}).pop())
    short = np.array([dmat[right_idx, int(i)], dmat[right_idx, int(j)]], dtype=float)
    hyp = float(dmat[int(i), int(j)])
    if hyp <= 0 or np.any(short <= 0):
        return {"score": np.inf, "mean_diag": np.nan, "center": np.array([np.nan, np.nan])}
    short_mean = float(np.mean(short))
    side_cv = float(np.std(short) / short_mean)
    ratio_error = abs(hyp / short_mean - np.sqrt(2))
    center = (points_xy[int(i)] + points_xy[int(j)]) / 2.0
    return {
        "score": float(side_cv + ratio_error),
        "mean_diag": hyp,
        "center": center,
        "side_cv": side_cv,
        "diag_side_ratio_error": float(ratio_error),
    }


def is_missing_one_spot_triangle(
    points_xy: np.ndarray,
    diag_range: tuple[float, float] | None = None,
    max_score: float = 0.16,
    max_side_cv: float = 0.10,
    max_ratio_error: float = 0.12,
) -> bool:
    if diag_range is None:
        diag_range = _runtime_diag_bounds()
    metrics = triangle_missing_spot_metrics(points_xy)
    return (
        diag_range[0] <= float(metrics["mean_diag"]) <= diag_range[1]
        and float(metrics["score"]) <= max_score
        and float(metrics["side_cv"]) <= max_side_cv
        and float(metrics["diag_side_ratio_error"]) <= max_ratio_error
    )


def association_stats(pids: list[int]) -> dict[str, object]:
    """Score source consistency: clutter and partially unlabeled same-source groups are correct."""
    if not pids:
        return {"assigned_particle_id": 0, "group_purity": 0.0, "is_correct": False, "association_status": "BAD"}
    nonzero = [int(pid) for pid in pids if int(pid) > 0]
    if not nonzero:
        return {"assigned_particle_id": 0, "group_purity": 1.0, "is_correct": True, "association_status": "CLUTTER"}
    counts = {pid: nonzero.count(pid) for pid in sorted(set(nonzero))}
    majority_pid = max(counts, key=counts.get)
    unique_nonzero = set(nonzero)
    if len(unique_nonzero) == 1:
        status = "OK" if len(nonzero) == len(pids) else "PARTIAL"
        return {
            "assigned_particle_id": int(majority_pid),
            "group_purity": 1.0,
            "is_correct": True,
            "association_status": status,
        }
    return {
        "assigned_particle_id": int(majority_pid),
        "group_purity": float(counts[majority_pid] / len(nonzero)),
        "is_correct": False,
        "association_status": "MIXED",
    }


def parse_source_set(value: object, fallback: object = "") -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = fallback
    text = str(value)
    ids = {part.strip() for part in text.split(";") if part.strip()}
    if not ids and fallback not in {None, ""}:
        ids = {str(fallback)}
    return ids


def source_to_particle_id(source_id: str) -> int:
    if source_id.startswith("P"):
        try:
            return int(source_id[1:])
        except ValueError:
            return 0
    return 0


def association_stats_from_source_sets(possible_sets: list[set[str]], hard_sources: list[str]) -> dict[str, object]:
    if not possible_sets:
        return {"assigned_particle_id": 0, "group_purity": 0.0, "is_correct": False, "association_status": "BAD"}
    informative = [set(s) for s in possible_sets if s]
    if not informative:
        return {"assigned_particle_id": 0, "group_purity": 1.0, "is_correct": True, "association_status": "CLUTTER"}

    common = set.intersection(*informative)
    if common:
        hard_counts = {sid: hard_sources.count(sid) for sid in common}
        assigned_source = max(common, key=lambda sid: (hard_counts.get(sid, 0), sid.startswith("P"), sid))
        assigned_pid = source_to_particle_id(assigned_source)
        has_overlap = any(len(s) > 1 for s in informative)
        if assigned_pid == 0 and all(not sid.startswith("P") for sid in common):
            status = "CLUTTER"
        else:
            status = "OVERLAP" if has_overlap else "OK"
        return {
            "assigned_particle_id": int(assigned_pid),
            "assigned_source_id": assigned_source,
            "group_purity": 1.0,
            "is_correct": True,
            "association_status": status,
        }

    counts: dict[str, int] = {}
    for source_set in informative:
        for sid in source_set:
            counts[sid] = counts.get(sid, 0) + 1
    assigned_source = max(counts, key=counts.get)
    return {
        "assigned_particle_id": int(source_to_particle_id(assigned_source)),
        "assigned_source_id": assigned_source,
        "group_purity": float(counts[assigned_source] / len(informative)),
        "is_correct": False,
        "association_status": "MIXED",
    }


def association_stats_from_group(group: pd.DataFrame) -> dict[str, object]:
    if "source_id" in group.columns and "possible_source_ids" not in group.columns:
        hard_sources = [str(x) if str(x) != "nan" else "" for x in group["source_id"].tolist()]
        possible_sets = [parse_source_set(source, fallback=source) for source in hard_sources]
        return association_stats_from_source_sets(possible_sets, hard_sources)
    if "possible_source_ids" not in group.columns:
        pids = group["particle_id"].astype(int).tolist()
        stats = association_stats(pids)
        stats["assigned_source_id"] = f"P{stats['assigned_particle_id']}" if int(stats["assigned_particle_id"]) > 0 else ""
        return stats
    hard_sources = [str(x) if str(x) != "nan" else "" for x in group.get("source_id", pd.Series([""] * len(group))).tolist()]
    possible_sets = [
        parse_source_set(value, fallback=source)
        for value, source in zip(group["possible_source_ids"].tolist(), hard_sources)
    ]
    return association_stats_from_source_sets(possible_sets, hard_sources)


def local_four_combinations(points_xy: np.ndarray, neighbors_k: int = 7):
    """Yield unique local four-point combinations instead of all global C(n, 4)."""
    n = len(points_xy)
    if n < 4:
        return
    dist = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    seen: set[tuple[int, ...]] = set()
    for i in range(n):
        local_ids = np.argsort(dist[i])[: min(neighbors_k, n)]
        for combo in combinations(local_ids, 4):
            combo = tuple(sorted(int(x) for x in combo))
            if combo in seen:
                continue
            seen.add(combo)
            yield combo


def local_three_combinations(points_xy: np.ndarray, neighbors_k: int = 7):
    n = len(points_xy)
    if n < 3:
        return
    dist = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    seen: set[tuple[int, ...]] = set()
    for i in range(n):
        local_ids = np.argsort(dist[i])[: min(neighbors_k, n)]
        for combo in combinations(local_ids, 3):
            combo = tuple(sorted(int(x) for x in combo))
            if combo in seen:
                continue
            seen.add(combo)
            yield combo


def order_four_spots(points_rc: np.ndarray) -> np.ndarray:
    center = points_rc.mean(axis=0)
    angles = np.arctan2(points_rc[:, 0] - center[0], points_rc[:, 1] - center[1])
    return points_rc[np.argsort(angles)]


def load_psf_templates() -> list[PsfTemplate]:
    templates: list[PsfTemplate] = []
    for file_index in range(10015, 10036):
        path = PSF_DIR / f"{file_index}.tif"
        raw = tifffile.imread(str(path)).astype(float)
        bg = float(np.percentile(raw, 50))
        img = np.clip(raw - bg, 0, None)
        if img.max() > 0:
            img = img / img.max()

        smoothed = filters.gaussian(raw, sigma=1.0)
        peaks = feature.peak_local_max(
            smoothed,
            min_distance=20,
            threshold_abs=float(np.percentile(smoothed, 99.25)),
            num_peaks=8,
        )
        if len(peaks) < 4:
            raise RuntimeError(f"Could not detect four PSF spots in {path}")
        values = np.array([raw[int(r), int(c)] for r, c in peaks])
        peaks = peaks[np.argsort(values)[::-1][:4]].astype(float)

        refined = []
        for r, c in peaks:
            r_i, c_i = int(round(r)), int(round(c))
            r0, r1 = max(0, r_i - 7), min(raw.shape[0], r_i + 8)
            c0, c1 = max(0, c_i - 7), min(raw.shape[1], c_i + 8)
            dr, dc = intensity_centroid(raw[r0:r1, c0:c1])
            refined.append([r0 + dr, c0 + dc])
        spots_rc = order_four_spots(np.asarray(refined, dtype=float))
        center_rc = spots_rc.mean(axis=0)
        offsets_rc = spots_rc - center_rc
        templates.append(
            PsfTemplate(
                file_index=file_index,
                z_um=float(file_index - 10015),
                image=img,
                source_center_rc=center_rc,
                offsets_rc=offsets_rc,
                mean_diag_px=mean_diag(spots_rc[:, ::-1]),
            )
        )
    return templates


def template_for_z(templates: list[PsfTemplate], z_um: float) -> PsfTemplate:
    return min(templates, key=lambda t: abs(t.z_um - z_um))


def add_shifted_template(canvas: np.ndarray, template: np.ndarray, center_rc: tuple[float, float], amp: float) -> None:
    h, w = template.shape
    cr, cc = center_rc
    r0 = int(round(cr - h / 2))
    c0 = int(round(cc - w / 2))
    dst_r0, dst_c0 = max(0, r0), max(0, c0)
    dst_r1, dst_c1 = min(canvas.shape[0], r0 + h), min(canvas.shape[1], c0 + w)
    if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
        return
    src_r0, src_c0 = dst_r0 - r0, dst_c0 - c0
    src_r1, src_c1 = src_r0 + (dst_r1 - dst_r0), src_c0 + (dst_c1 - dst_c0)
    canvas[dst_r0:dst_r1, dst_c0:dst_c1] += amp * template[src_r0:src_r1, src_c0:src_c1]


def add_shifted_template_spots(
    canvas: np.ndarray,
    template: np.ndarray,
    source_center_rc: np.ndarray,
    offsets_rc: np.ndarray,
    center_rc: tuple[float, float],
    amp: float,
    include_spots: list[bool] | np.ndarray,
    patch_radius: int = 17,
) -> None:
    """Render selected measured sub-spot patches so simulated dropout is visible in the image."""
    h, w = template.shape
    template_center = np.asarray(source_center_rc, dtype=float)
    target_center = np.array(center_rc, dtype=float)
    for include, offset in zip(include_spots, offsets_rc):
        if not include:
            continue
        src_center = template_center + np.asarray(offset, dtype=float)
        dst_center = target_center + np.asarray(offset, dtype=float)
        src_r0 = int(round(src_center[0] - patch_radius))
        src_c0 = int(round(src_center[1] - patch_radius))
        src_r1 = int(round(src_center[0] + patch_radius + 1))
        src_c1 = int(round(src_center[1] + patch_radius + 1))
        dst_r0 = int(round(dst_center[0] - patch_radius))
        dst_c0 = int(round(dst_center[1] - patch_radius))
        dst_r1 = int(round(dst_center[0] + patch_radius + 1))
        dst_c1 = int(round(dst_center[1] + patch_radius + 1))

        src_clip_r0, src_clip_c0 = max(0, src_r0), max(0, src_c0)
        src_clip_r1, src_clip_c1 = min(h, src_r1), min(w, src_c1)
        dst_clip_r0, dst_clip_c0 = max(0, dst_r0), max(0, dst_c0)
        dst_clip_r1, dst_clip_c1 = min(canvas.shape[0], dst_r1), min(canvas.shape[1], dst_c1)

        rows = min(src_clip_r1 - src_clip_r0, dst_clip_r1 - dst_clip_r0)
        cols = min(src_clip_c1 - src_clip_c0, dst_clip_c1 - dst_clip_c0)
        if rows <= 0 or cols <= 0:
            continue
        canvas[dst_clip_r0 : dst_clip_r0 + rows, dst_clip_c0 : dst_clip_c0 + cols] += (
            amp * template[src_clip_r0 : src_clip_r0 + rows, src_clip_c0 : src_clip_c0 + cols]
        )


def particle_state(
    pid: int,
    frame: int,
    n_frames: int,
    center_separation_px: float = 18.0,
    traj: TrajectoryParams | None = None,
) -> tuple[float, float, float]:
    if traj is None:
        traj = TrajectoryParams(
            base_y=120.0,
            p1_x0=88.0,
            p1_x1=182.0,
            p2_x0=182.0,
            p2_x1=88.0,
            p1_phase=0.0,
            p2_phase=0.3,
            p3_phase=0.0,
            p1_y_amp=6.0,
            p2_y_amp=6.0,
            p3_x0=300.0,
            p3_y0=112.0,
            p3_y1=180.0,
            p3_x_amp=18.0,
            z_phase1=0.0,
            z_phase2=0.0,
            z_phase3=0.0,
            clutter_cx=135.0,
            clutter_cy=120.0,
        )
    t = frame / (n_frames - 1)
    if pid == 1:
        x = traj.p1_x0 + (traj.p1_x1 - traj.p1_x0) * t
        y = traj.base_y - center_separation_px / 2 + traj.p1_y_amp * np.sin(2 * np.pi * t + traj.p1_phase)
        z = 4 + 12 * t + 1.5 * np.sin(2 * np.pi * t + traj.z_phase1)
    elif pid == 2:
        x = traj.p2_x0 + (traj.p2_x1 - traj.p2_x0) * t
        y = traj.base_y + center_separation_px / 2 - traj.p2_y_amp * np.sin(2 * np.pi * t + traj.p2_phase)
        z = 16 - 12 * t + 1.2 * np.cos(2 * np.pi * t + traj.z_phase2)
    else:
        x = traj.p3_x0 + traj.p3_x_amp * np.sin(2 * np.pi * t + traj.p3_phase)
        y = traj.p3_y0 + (traj.p3_y1 - traj.p3_y0) * t
        z = 8 + 4 * np.sin(4 * np.pi * t + traj.z_phase3)
    return float(x), float(y), float(np.clip(z, 0, 20))


def make_truth_and_detections(
    templates: list[PsfTemplate],
    seed: int,
    n_frames: int,
    dropout_prob: float = 0.045,
    center_separation_px: float = 18.0,
    clutter_points_per_frame: int = 4,
    n_particles: int = 6,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stack = []
    truth_rows = []
    det_rows = []
    h, w = 300, 390
    noise_sigma = 0.018
    truth_df, trajectory_params = generate_particle_trajectories(
        rng,
        n_frames=n_frames,
        n_particles=n_particles,
        center_separation_px=center_separation_px,
        z_range_um=z_range_um,
    )
    clutter_paths = generate_clutter_paths(rng, n_frames, clutter_points_per_frame, z_range_um=z_range_um)

    for frame in range(n_frames):
        crossing_start = int(round(0.38 * n_frames))
        crossing_end = int(round(0.63 * n_frames))
        canvas = rng.normal(0.0, noise_sigma, size=(h, w))
        frame_truth = truth_df[truth_df["FRAME"] == frame]
        crossing_extra_dropout_prob = 0.09 if dropout_prob > 0 else 0.0
        for _, truth_row in frame_truth.iterrows():
            pid = int(truth_row["particle_id"])
            x = float(truth_row["X_true"])
            y = float(truth_row["Y_true"])
            z = float(truth_row["Z_true"])
            tmpl = template_for_z(templates, z)
            amp = (0.72 + 0.24 * rng.random()) * (0.95 if pid <= 2 else rng.uniform(0.72, 0.95))
            truth_rows.append({"FRAME": frame, "particle_id": pid, "X_true": x, "Y_true": y, "Z_true": z})

            include_spots = []
            for spot_idx, (dr, dc) in enumerate(tmpl.offsets_rc, start=1):
                sx = x + dc
                sy = y + dr
                missing = rng.random() < dropout_prob
                if (
                    pid in {1, 2}
                    and crossing_start <= frame <= crossing_end
                    and rng.random() < crossing_extra_dropout_prob
                ):
                    missing = True
                include_spots.append(not missing)
                if missing:
                    continue
                loc_noise_x = rng.normal(0, 0.75)
                loc_noise_y = rng.normal(0, 0.75)
                det_rows.append(
                    {
                        "FRAME": frame,
                        "detection_id": f"f{frame}_p{pid}_s{spot_idx}",
                        "TRACK_ID": pid * 10 + spot_idx,
                        "particle_id": pid,
                        "source_id": f"P{pid}",
                        "spot_index": spot_idx,
                        "X": sx + loc_noise_x,
                        "Y": sy + loc_noise_y,
                        "X_render": sx,
                        "Y_render": sy,
                        "localization_noise_x": loc_noise_x,
                        "localization_noise_y": loc_noise_y,
                        "is_noise": False,
                    }
                )
            add_shifted_template_spots(
                canvas,
                tmpl.image,
                tmpl.source_center_rc,
                tmpl.offsets_rc,
                (y, x),
                amp,
                include_spots,
            )

        if crossing_start <= frame <= crossing_end and clutter_points_per_frame > 0:
            # Fast drifting fake four-spot objects rendered from measured PSF templates.
            # They cross the field rapidly, so they may disturb frame-wise grouping
            # but do not form long-lived trajectories.
            for path in clutter_paths:
                u = (frame - int(path["start"])) / max(1, int(path["duration"]))
                if not -0.05 <= u <= 1.05:
                    continue
                wiggle = 10.0 * np.sin(2 * np.pi * u + path["phase"])
                fake_x = float(path["x0"] + (path["x1"] - path["x0"]) * u)
                fake_y = float(path["y0"] + (path["y1"] - path["y0"]) * u + wiggle)
                if not (-60 <= fake_x <= 450 and -60 <= fake_y <= 360):
                    continue
                fake_tmpl = template_for_z(templates, float(path["z"]))
                include_spots = []
                for fake_spot_idx, (dr, dc) in enumerate(fake_tmpl.offsets_rc, start=1):
                    sx = fake_x + dc
                    sy = fake_y + dr
                    include_spot = bool(0 <= sx < w and 0 <= sy < h and rng.random() < 0.92)
                    include_spots.append(include_spot)
                    if include_spot:
                        loc_noise_x = rng.normal(0, 0.9)
                        loc_noise_y = rng.normal(0, 0.9)
                        det_rows.append(
                            {
                                "FRAME": frame,
                                "detection_id": f"f{frame}_drift{path['path_id']}_s{fake_spot_idx}",
                                "TRACK_ID": -100000 - int(path["path_id"]) * 10 - fake_spot_idx,
                                "particle_id": 0,
                                "source_id": f"C{int(path['path_id'])}",
                                "spot_index": 0,
                                "X": sx + loc_noise_x,
                                "Y": sy + loc_noise_y,
                                "X_render": sx,
                                "Y_render": sy,
                                "localization_noise_x": loc_noise_x,
                                "localization_noise_y": loc_noise_y,
                                "is_noise": True,
                            }
                        )
                add_shifted_template_spots(
                    canvas,
                    fake_tmpl.image,
                    fake_tmpl.source_center_rc,
                    fake_tmpl.offsets_rc,
                    (fake_y, fake_x),
                    float(path["amp"]),
                    include_spots,
                )
                if rng.random() < 0.08:
                    fx = float(np.clip(fake_x + rng.normal(0, 20), 55, 335))
                    fy = float(np.clip(fake_y + rng.normal(0, 20), 60, 235))
                    det_rows.append(
                        {
                            "FRAME": frame,
                            "detection_id": f"f{frame}_artifact{path['path_id']}",
                            "TRACK_ID": -200000 - frame * 100 - int(path["path_id"]),
                            "particle_id": 0,
                            "source_id": f"A{frame}_{int(path['path_id'])}",
                            "spot_index": 0,
                            "X": fx,
                            "Y": fy,
                            "X_render": fx,
                            "Y_render": fy,
                            "localization_noise_x": 0.0,
                            "localization_noise_y": 0.0,
                            "is_noise": True,
                        }
                    )
                    rr, cc = int(round(fy)), int(round(fx))
                    if 0 <= rr < h and 0 <= cc < w:
                        canvas[max(0, rr - 2) : min(h, rr + 3), max(0, cc - 2) : min(w, cc + 3)] += 0.22

        stack.append(np.clip(canvas, 0, None))

    stack_arr = np.asarray(stack, dtype=float)
    stack_u16 = np.asarray(250 + 14000 * stack_arr / max(stack_arr.max(), 1e-12), dtype=np.uint16)
    return pd.DataFrame(truth_rows), pd.DataFrame(det_rows), stack_u16, trajectory_params


def detect_spots_from_stack(stack: np.ndarray) -> pd.DataFrame:
    """Detect sub-spots from the rendered image stack; no simulator track IDs are used."""
    rows = []
    for frame, raw in enumerate(stack):
        img = raw.astype(float)
        smooth = filters.gaussian(img, sigma=1.0)
        threshold = max(float(np.percentile(smooth, 99.25)), float(np.median(smooth) + 4.0 * np.std(smooth)))
        peaks = feature.peak_local_max(
            smooth,
            min_distance=7,
            threshold_abs=threshold,
            num_peaks=96,
            exclude_border=8,
        )
        if len(peaks) == 0:
            continue
        values = np.array([smooth[int(r), int(c)] for r, c in peaks])
        order = np.argsort(values)[::-1]
        for local_idx, peak_idx in enumerate(order):
            r, c = peaks[peak_idx]
            r_i, c_i = int(round(r)), int(round(c))
            r0, r1 = max(0, r_i - 6), min(img.shape[0], r_i + 7)
            c0, c1 = max(0, c_i - 6), min(img.shape[1], c_i + 7)
            patch = img[r0:r1, c0:c1]
            dr, dc = intensity_centroid(patch)
            border = np.concatenate(
                [
                    patch[0, :],
                    patch[-1, :],
                    patch[:, 0],
                    patch[:, -1],
                ]
            )
            local_bg = float(np.median(border)) if len(border) else float(np.median(patch))
            local_noise = float(np.std(border)) if len(border) else 0.0
            signal = np.clip(patch - local_bg, 0, None)
            signal_sum = float(signal.sum())
            if signal_sum > 0:
                yy, xx = np.indices(signal.shape)
                cx = float(dc)
                cy = float(dr)
                sigma_x = float(np.sqrt(np.sum(signal * (xx - cx) ** 2) / signal_sum))
                sigma_y = float(np.sqrt(np.sum(signal * (yy - cy) ** 2) / signal_sum))
                area = int(np.sum(signal > max(1.0, 0.25 * float(signal.max()))))
            else:
                sigma_x = 0.0
                sigma_y = 0.0
                area = 0
            shape_ratio = float(max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-6))
            peak_value = float(img[r_i, c_i])
            rows.append(
                {
                    "FRAME": int(frame),
                    "detection_id": f"img_f{frame}_d{local_idx}",
                    "X": float(c0 + dc),
                    "Y": float(r0 + dr),
                    "intensity": peak_value,
                    "integrated_intensity": signal_sum,
                    "local_bg": local_bg,
                    "local_contrast": float((peak_value - local_bg) / max(local_noise, 1.0)),
                    "spot_area": area,
                    "spot_sigma_x": sigma_x,
                    "spot_sigma_y": sigma_y,
                    "spot_shape_ratio": shape_ratio,
                    "log_peak_intensity": float(np.log1p(max(peak_value - local_bg, 0.0))),
                    "log_integrated_intensity": float(np.log1p(max(signal_sum, 0.0))),
                    "TRACK_ID": -1,
                    "particle_id": 0,
                    "source_id": "",
                    "possible_source_ids": "",
                    "possible_particle_ids": "",
                    "spot_index": 0,
                    "truth_track_id": 0,
                    "is_noise": True,
                    "is_interpolated": False,
                }
            )
    detected = pd.DataFrame(rows)
    if not detected.empty:
        detected["image_detection_index"] = np.arange(len(detected), dtype=int)
        detected["TRACK_ID"] = detected["image_detection_index"].astype(int)
    return detected


def label_detected_spots(detected: pd.DataFrame, truth_detections: pd.DataFrame, max_label_dist: float = 8.0) -> pd.DataFrame:
    """Attach nearest truth labels for scoring only; grouping/tracking do not use these labels."""
    if detected.empty:
        return detected.copy()
    labeled = detected.copy()
    for frame, frame_df in labeled.groupby("FRAME"):
        truth_frame = truth_detections[truth_detections["FRAME"] == frame]
        if truth_frame.empty:
            continue
        det_idx = frame_df.index.to_numpy()
        det_pts = frame_df[["X", "Y"]].to_numpy(float)
        truth_xy_cols = ["X_render", "Y_render"] if {"X_render", "Y_render"}.issubset(truth_frame.columns) else ["X", "Y"]
        truth_pts = truth_frame[truth_xy_cols].to_numpy(float)
        cost = np.linalg.norm(det_pts[:, None, :] - truth_pts[None, :, :], axis=2)
        rr, cc = _stable_linear_sum_assignment(cost)
        truth_index = truth_frame.index.to_numpy()
        for local_i, row_index in enumerate(det_idx):
            compatible = truth_frame.iloc[np.where(cost[local_i] <= max_label_dist)[0]]
            if compatible.empty:
                continue
            source_ids = sorted(str(x) for x in compatible["source_id"].dropna().unique())
            particle_ids = sorted(int(x) for x in compatible["particle_id"].dropna().unique())
            labeled.loc[row_index, "possible_source_ids"] = ";".join(source_ids)
            labeled.loc[row_index, "possible_particle_ids"] = ";".join(map(str, particle_ids))
        for r, c in zip(rr, cc):
            if cost[r, c] > max_label_dist:
                continue
            src = truth_detections.loc[truth_index[c]]
            row_index = det_idx[r]
            labeled.loc[row_index, "particle_id"] = int(src["particle_id"])
            labeled.loc[row_index, "source_id"] = str(src["source_id"])
            labeled.loc[row_index, "spot_index"] = int(src["spot_index"])
            labeled.loc[row_index, "truth_track_id"] = int(src["TRACK_ID"])
            labeled.loc[row_index, "is_noise"] = bool(src["is_noise"])
    return labeled


def ensure_spot_appearance_columns(detections: pd.DataFrame) -> pd.DataFrame:
    """Make image-derived spot appearance columns available for custom tracking costs."""
    out = detections.copy()
    if "intensity" not in out.columns:
        out["intensity"] = 0.0
    if "integrated_intensity" not in out.columns:
        out["integrated_intensity"] = out["intensity"].astype(float)
    if "log_peak_intensity" not in out.columns:
        out["log_peak_intensity"] = np.log1p(np.clip(out["intensity"].astype(float), 0, None))
    if "log_integrated_intensity" not in out.columns:
        out["log_integrated_intensity"] = np.log1p(np.clip(out["integrated_intensity"].astype(float), 0, None))
    for col in ["spot_sigma_x", "spot_sigma_y", "spot_shape_ratio", "local_contrast"]:
        if col not in out.columns:
            out[col] = 0.0
    for col in SPOT_APPEARANCE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def robust_feature_scales(detections: pd.DataFrame, columns: list[str] = SPOT_APPEARANCE_COLUMNS) -> np.ndarray:
    if detections.empty:
        return np.ones(len(columns), dtype=float)
    arr = detections[columns].to_numpy(float)
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med[None, :]), axis=0)
    sd = np.nanstd(arr, axis=0)
    scales = np.where(mad > 1e-6, 1.4826 * mad, sd)
    return np.maximum(scales, 1.0)


def add_frame_geometry_context(detections: pd.DataFrame) -> pd.DataFrame:
    """Annotate detections with per-frame four-aperture relative geometry, without truth labels."""
    out = detections.copy()
    out["has_geometry_context"] = False
    out["geom_center_x"] = np.nan
    out["geom_center_y"] = np.nan
    out["geom_slot_cos"] = 0.0
    out["geom_slot_sin"] = 0.0
    out["geom_radius"] = 0.0
    out["geom_group_score"] = np.nan
    if out.empty:
        return out
    groups = framewise_group(out)
    if groups.empty:
        return out
    id_to_index = {str(row["detection_id"]): idx for idx, row in out.iterrows()}
    for _, group in groups.iterrows():
        if float(group.get("score", np.inf)) > 0.55:
            continue
        center = np.asarray([float(group["X_rec"]), float(group["Y_rec"])], dtype=float)
        ids = str(group["member_detection_ids"]).split(";")
        for det_id in ids:
            idx = id_to_index.get(str(det_id))
            if idx is None:
                continue
            pos = out.loc[idx, ["X", "Y"]].to_numpy(float)
            rel = pos - center
            radius = float(np.linalg.norm(rel))
            if radius <= 1e-6:
                continue
            out.loc[idx, "has_geometry_context"] = True
            out.loc[idx, "geom_center_x"] = center[0]
            out.loc[idx, "geom_center_y"] = center[1]
            out.loc[idx, "geom_slot_cos"] = float(rel[0] / radius)
            out.loc[idx, "geom_slot_sin"] = float(rel[1] / radius)
            out.loc[idx, "geom_radius"] = radius
            out.loc[idx, "geom_group_score"] = float(group.get("score", np.nan))
    return out


def _cosine_direction_cost(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float((1.0 - cosine) / 2.0)


def _stable_linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve exact/near LAP ties independently of the installed SciPy version."""
    stable_cost = np.asarray(cost, dtype=float).copy()
    if stable_cost.size == 0:
        return linear_sum_assignment(stable_cost)
    n_rows, n_cols = stable_cost.shape
    row_rank = np.arange(n_rows, dtype=float)[:, None]
    col_rank = np.arange(n_cols, dtype=float)[None, :]
    row_scale = max(1.0, float(n_rows - 1))
    col_scale = max(1.0, float(n_cols - 1))
    # Non-separable perturbation: equal-cost assignments prefer matching stable
    # row/column ranks. 1e-10 is far below every physical cost tolerance here.
    tie_break = np.abs(row_rank / row_scale - col_rank / col_scale)
    tie_break += (row_rank * (col_rank + 1.0)) * 1e-3 / max(1.0, float(n_rows * n_cols))
    finite = np.isfinite(stable_cost) & (stable_cost < 1e5)
    stable_cost[finite] += 1e-10 * tie_break[finite]
    return linear_sum_assignment(stable_cost)


def link_detections_to_tracks_lap_cost(
    detections: pd.DataFrame,
    max_link_dist: float | None = None,
    max_gap: int = 1,
    max_total_cost: float = 1.45,
) -> pd.DataFrame:
    """Link detections with a LapTrack-style custom cost instead of distance alone."""
    if detections.empty:
        return detections.copy()
    if max_link_dist is None:
        max_link_dist = runtime_sensor_distance_px((), (), 60.75)
    max_link_dist = float(max_link_dist)
    gap_growth_dist = _reference_pixels_in_current_image(4.0)
    velocity_scale = _reference_pixels_in_current_image(7.0)
    radius_scale = _reference_pixels_in_current_image(18.0)
    linked = ensure_spot_appearance_columns(detections).copy()
    if "detection_id" not in linked.columns:
        linked["detection_id"] = [f"det_{i}" for i in range(len(linked))]
    linked = linked.sort_values(
        ["FRAME", "intensity", "X", "Y", "detection_id"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).copy()
    linked = add_frame_geometry_context(linked)
    linked["RAW_TRACK_ID"] = -1
    linked["lap_link_cost"] = np.nan
    linked["lap_link_status"] = "new_track"
    feature_scales = robust_feature_scales(linked)
    tracks: dict[int, dict[str, object]] = {}
    next_tid = 1

    for frame in sorted(linked["FRAME"].unique()):
        frame_idx = linked.index[linked["FRAME"] == frame].to_numpy()
        frame_pts = linked.loc[frame_idx, ["X", "Y"]].to_numpy(float)
        frame_feat = linked.loc[frame_idx, SPOT_APPEARANCE_COLUMNS].to_numpy(float)
        active_ids = sorted(
            tid for tid, state in tracks.items() if int(frame) - int(state["last_frame"]) <= max_gap + 1
        )
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()

        if active_ids and len(frame_idx):
            cost = np.full((len(active_ids), len(frame_idx)), 1e6, dtype=float)
            raw_dist = np.full_like(cost, np.inf, dtype=float)
            for r, tid in enumerate(active_ids):
                state = tracks[tid]
                dt = max(1, int(frame) - int(state["last_frame"]))
                last_xy = np.asarray([float(state["x"]), float(state["y"])], dtype=float)
                velocity = np.asarray([float(state.get("vx", 0.0)), float(state.get("vy", 0.0))], dtype=float)
                predicted = last_xy + velocity * dt
                expected_feature = np.asarray(state["feature"], dtype=float)
                for c, pt in enumerate(frame_pts):
                    distance = float(np.linalg.norm(pt - predicted))
                    raw_dist[r, c] = distance
                    if distance > max_link_dist + gap_growth_dist * max(0, dt - 1):
                        continue
                    step_velocity = (pt - last_xy) / dt
                    velocity_cost = float(np.linalg.norm(step_velocity - velocity) / velocity_scale)
                    direction_cost = _cosine_direction_cost(step_velocity, velocity)
                    feature_cost = float(
                        np.linalg.norm((frame_feat[c] - expected_feature) / feature_scales)
                        / np.sqrt(len(SPOT_APPEARANCE_COLUMNS))
                    )
                    slot_cost = 0.0
                    radius_cost = 0.0
                    current_has_context = bool(linked.loc[frame_idx[c], "has_geometry_context"])
                    if bool(state.get("has_geometry_context", False)) and current_has_context:
                        current_slot = linked.loc[frame_idx[c], ["geom_slot_cos", "geom_slot_sin"]].to_numpy(float)
                        expected_slot = np.asarray(state.get("geometry_slot", np.array([0.0, 0.0])), dtype=float)
                        slot_cost = float(np.linalg.norm(current_slot - expected_slot) / 2.0)
                        radius_cost = float(abs(float(linked.loc[frame_idx[c], "geom_radius"]) - float(state.get("geom_radius", 0.0))) / radius_scale)
                    distance_cost = distance / max_link_dist
                    gap_cost = 0.12 * max(0, dt - 1)
                    cost[r, c] = (
                        0.50 * distance_cost
                        + 0.19 * velocity_cost
                        + 0.07 * direction_cost
                        + 0.12 * feature_cost
                        + 0.10 * slot_cost
                        + 0.02 * radius_cost
                        + gap_cost
                    )
            rr, cc = _stable_linear_sum_assignment(cost)
            pairs = sorted(
                [(float(cost[r, c]), int(r), int(c)) for r, c in zip(rr, cc) if cost[r, c] <= max_total_cost],
                key=lambda x: (x[0], active_ids[x[1]], x[2]),
            )
            for pair_cost, r, det_local in pairs:
                tid = active_ids[r]
                if tid in assigned_tracks or det_local in assigned_dets:
                    continue
                finite_track_costs = np.sort(cost[r, cost[r] < 1e5])
                finite_det_costs = np.sort(cost[cost[:, det_local] < 1e5, det_local])
                ambiguous_track = len(finite_track_costs) > 1 and float(finite_track_costs[1] - finite_track_costs[0]) < 0.10
                ambiguous_det = len(finite_det_costs) > 1 and float(finite_det_costs[1] - finite_det_costs[0]) < 0.10
                if pair_cost > 0.75 and ambiguous_track and ambiguous_det:
                    continue
                row_index = frame_idx[det_local]
                prev_x = float(tracks[tid]["x"])
                prev_y = float(tracks[tid]["y"])
                prev_frame = int(tracks[tid]["last_frame"])
                dt = max(1, int(frame) - prev_frame)
                x = float(linked.loc[row_index, "X"])
                y = float(linked.loc[row_index, "Y"])
                feat = linked.loc[row_index, SPOT_APPEARANCE_COLUMNS].to_numpy(float)
                old_feat = np.asarray(tracks[tid]["feature"], dtype=float)
                current_has_context = bool(linked.loc[row_index, "has_geometry_context"])
                if current_has_context:
                    current_slot = linked.loc[row_index, ["geom_slot_cos", "geom_slot_sin"]].to_numpy(float)
                    old_slot = np.asarray(tracks[tid].get("geometry_slot", current_slot), dtype=float)
                    new_slot = 0.75 * current_slot + 0.25 * old_slot
                    norm = float(np.linalg.norm(new_slot))
                    if norm > 1e-6:
                        new_slot = new_slot / norm
                    current_radius = float(linked.loc[row_index, "geom_radius"])
                    previous_radius = float(tracks[tid].get("geom_radius", current_radius))
                    if not np.isfinite(previous_radius):
                        previous_radius = current_radius
                    geom_radius = 0.75 * current_radius + 0.25 * previous_radius
                else:
                    new_slot = np.asarray(tracks[tid].get("geometry_slot", np.array([0.0, 0.0])), dtype=float)
                    geom_radius = float(tracks[tid].get("geom_radius", 0.0))
                    if not np.isfinite(geom_radius):
                        geom_radius = 0.0
                linked.loc[row_index, "RAW_TRACK_ID"] = tid
                linked.loc[row_index, "lap_link_cost"] = pair_cost
                linked.loc[row_index, "lap_link_status"] = "linked"
                tracks[tid] = {
                    "x": x,
                    "y": y,
                    "last_frame": int(frame),
                    "vx": 0.65 * ((x - prev_x) / dt) + 0.35 * float(tracks[tid].get("vx", 0.0)),
                    "vy": 0.65 * ((y - prev_y) / dt) + 0.35 * float(tracks[tid].get("vy", 0.0)),
                    "feature": 0.70 * feat + 0.30 * old_feat,
                    "has_geometry_context": bool(current_has_context or tracks[tid].get("has_geometry_context", False)),
                    "geometry_slot": new_slot,
                    "geom_radius": geom_radius,
                }
                assigned_tracks.add(tid)
                assigned_dets.add(det_local)

        for det_local, row_index in enumerate(frame_idx):
            if det_local in assigned_dets:
                continue
            x = float(linked.loc[row_index, "X"])
            y = float(linked.loc[row_index, "Y"])
            linked.loc[row_index, "RAW_TRACK_ID"] = next_tid
            tracks[next_tid] = {
                "x": x,
                "y": y,
                "last_frame": int(frame),
                "vx": 0.0,
                "vy": 0.0,
                "feature": linked.loc[row_index, SPOT_APPEARANCE_COLUMNS].to_numpy(float),
                "has_geometry_context": bool(linked.loc[row_index, "has_geometry_context"]),
                "geometry_slot": linked.loc[row_index, ["geom_slot_cos", "geom_slot_sin"]].to_numpy(float),
                "geom_radius": float(linked.loc[row_index, "geom_radius"])
                if np.isfinite(float(linked.loc[row_index, "geom_radius"]))
                else 0.0,
            }
            next_tid += 1
    linked["TRACK_ID"] = linked["RAW_TRACK_ID"].astype(int)
    return linked.sort_values(["FRAME", "TRACK_ID"]).reset_index(drop=True)


def link_detections_to_tracks(
    detections: pd.DataFrame,
    max_link_dist: float | None = None,
    max_gap: int = 1,
) -> pd.DataFrame:
    """Nearest-prediction linking on detected image spots, allowing occasional wrong links."""
    if detections.empty:
        return detections.copy()
    if max_link_dist is None:
        max_link_dist = runtime_sensor_distance_px((), (), 60.75)
    max_link_dist = float(max_link_dist)
    linked = detections.sort_values(["FRAME", "intensity"], ascending=[True, False]).copy()
    linked["RAW_TRACK_ID"] = -1
    tracks: dict[int, dict[str, float | int | None]] = {}
    next_tid = 1

    for frame in sorted(linked["FRAME"].unique()):
        frame_idx = linked.index[linked["FRAME"] == frame].to_numpy()
        frame_pts = linked.loc[frame_idx, ["X", "Y"]].to_numpy(float)
        active_ids = [
            tid for tid, state in tracks.items() if int(frame) - int(state["last_frame"]) <= max_gap + 1
        ]
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()
        if active_ids and len(frame_idx):
            preds = []
            for tid in active_ids:
                state = tracks[tid]
                dt = int(frame) - int(state["last_frame"])
                vx = float(state.get("vx", 0.0))
                vy = float(state.get("vy", 0.0))
                preds.append([float(state["x"]) + vx * dt, float(state["y"]) + vy * dt])
            cost = np.linalg.norm(np.asarray(preds)[:, None, :] - frame_pts[None, :, :], axis=2)
            rr, cc = _stable_linear_sum_assignment(cost)
            pairs = sorted(
                [(float(cost[r, c]), active_ids[r], int(c)) for r, c in zip(rr, cc) if cost[r, c] <= max_link_dist],
                key=lambda x: x[0],
            )
            for _, tid, det_local in pairs:
                if tid in assigned_tracks or det_local in assigned_dets:
                    continue
                row_index = frame_idx[det_local]
                prev_x = float(tracks[tid]["x"])
                prev_y = float(tracks[tid]["y"])
                prev_frame = int(tracks[tid]["last_frame"])
                dt = max(1, int(frame) - prev_frame)
                x = float(linked.loc[row_index, "X"])
                y = float(linked.loc[row_index, "Y"])
                linked.loc[row_index, "RAW_TRACK_ID"] = tid
                tracks[tid] = {
                    "x": x,
                    "y": y,
                    "last_frame": int(frame),
                    "vx": (x - prev_x) / dt,
                    "vy": (y - prev_y) / dt,
                }
                assigned_tracks.add(tid)
                assigned_dets.add(det_local)

        for det_local, row_index in enumerate(frame_idx):
            if det_local in assigned_dets:
                continue
            x = float(linked.loc[row_index, "X"])
            y = float(linked.loc[row_index, "Y"])
            linked.loc[row_index, "RAW_TRACK_ID"] = next_tid
            tracks[next_tid] = {"x": x, "y": y, "last_frame": int(frame), "vx": 0.0, "vy": 0.0}
            next_tid += 1
    linked["TRACK_ID"] = linked["RAW_TRACK_ID"].astype(int)
    return linked.sort_values(["FRAME", "TRACK_ID"]).reset_index(drop=True)


def _split_track_on_jumps(group: pd.DataFrame) -> list[pd.DataFrame]:
    group = group.sort_values("FRAME").copy()
    if len(group) <= 2:
        return [group]
    pts = group[["X", "Y"]].to_numpy(float)
    frames = group["FRAME"].to_numpy(int)
    disp = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    finite_disp = disp[np.isfinite(disp)]
    median = float(np.median(finite_disp)) if len(finite_disp) else 0.0
    mad = float(np.median(np.abs(finite_disp - median))) if len(finite_disp) else 0.0
    one_reference_px = _reference_pixels_in_current_image(1.0)
    jump_threshold = max(
        _reference_pixels_in_current_image(16.0),
        median + 5.0 * mad,
        3.0 * max(median, one_reference_px),
    )
    break_after = set()
    for i, d in enumerate(disp):
        gap = int(frames[i + 1] - frames[i])
        if gap > 2 or d > jump_threshold:
            break_after.add(i)
            continue
        if i >= 1:
            v_prev = pts[i] - pts[i - 1]
            v_next = pts[i + 1] - pts[i]
            accel = float(np.linalg.norm(v_next - v_prev))
            if accel > max(
                _reference_pixels_in_current_image(18.0),
                4.0 * max(median, one_reference_px),
            ):
                break_after.add(i)
    parts = []
    start = 0
    for i in range(len(group) - 1):
        if i in break_after:
            parts.append(group.iloc[start : i + 1].copy())
            start = i + 1
    parts.append(group.iloc[start:].copy())
    return [p for p in parts if not p.empty]


def repair_and_interpolate_tracks(
    raw_tracks: pd.DataFrame,
    n_frames: int,
    max_relink_gap: int = 3,
    max_relink_dist: float | None = None,
) -> pd.DataFrame:
    """Split likely mislinks, relink compatible fragments, and interpolate short gaps."""
    if raw_tracks.empty:
        return raw_tracks.copy()
    if max_relink_dist is None:
        max_relink_dist = runtime_sensor_distance_px((), (), 63.0)
    max_relink_dist = float(max_relink_dist)

    segments = []
    segment_id = 1
    for _, group in raw_tracks.groupby("RAW_TRACK_ID"):
        for part in _split_track_on_jumps(group):
            part = part.copy()
            part["TRACK_SEGMENT_ID"] = segment_id
            segments.append(part)
            segment_id += 1
    seg_df = pd.concat(segments, ignore_index=True) if segments else raw_tracks.copy()

    seg_meta = []
    for sid, group in seg_df.groupby("TRACK_SEGMENT_ID"):
        group = group.sort_values("FRAME")
        start = group.iloc[0]
        end = group.iloc[-1]
        pts = group[["X", "Y"]].to_numpy(float)
        frames = group["FRAME"].to_numpy(int)
        if len(group) >= 2:
            v_start = (pts[1] - pts[0]) / max(1, frames[1] - frames[0])
            v_end = (pts[-1] - pts[-2]) / max(1, frames[-1] - frames[-2])
        else:
            v_start = np.array([0.0, 0.0])
            v_end = np.array([0.0, 0.0])
        seg_meta.append(
            {
                "sid": int(sid),
                "start_frame": int(start["FRAME"]),
                "end_frame": int(end["FRAME"]),
                "start_x": float(start["X"]),
                "start_y": float(start["Y"]),
                "end_x": float(end["X"]),
                "end_y": float(end["Y"]),
                "v_start_x": float(v_start[0]),
                "v_start_y": float(v_start[1]),
                "v_end_x": float(v_end[0]),
                "v_end_y": float(v_end[1]),
                "duration": int(group["FRAME"].nunique()),
            }
        )
    meta = pd.DataFrame(seg_meta)
    parent = {int(sid): int(sid) for sid in meta["sid"].tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(meta) > 1:
        for _, a in meta.iterrows():
            candidates = meta[
                (meta["start_frame"] > int(a["end_frame"]))
                & (meta["start_frame"] - int(a["end_frame"]) <= max_relink_gap)
            ]
            if candidates.empty:
                continue
            dt = candidates["start_frame"].to_numpy(float) - float(a["end_frame"])
            pred_x = float(a["end_x"]) + float(a["v_end_x"]) * dt
            pred_y = float(a["end_y"]) + float(a["v_end_y"]) * dt
            dist = np.hypot(candidates["start_x"].to_numpy(float) - pred_x, candidates["start_y"].to_numpy(float) - pred_y)
            best_pos = int(np.argmin(dist))
            if float(dist[best_pos]) <= max_relink_dist:
                union(int(a["sid"]), int(candidates.iloc[best_pos]["sid"]))

    root_to_track: dict[int, int] = {}
    next_tid = 1
    track_ids = []
    for sid in seg_df["TRACK_SEGMENT_ID"].astype(int):
        root = find(int(sid))
        if root not in root_to_track:
            root_to_track[root] = next_tid
            next_tid += 1
        track_ids.append(root_to_track[root])
    repaired = seg_df.copy()
    repaired["TRACK_ID"] = track_ids
    repaired["is_interpolated"] = False

    interp_rows = []
    for tid, group in repaired.groupby("TRACK_ID"):
        group = group.sort_values("FRAME")
        records = list(group.to_dict("records"))
        for a, b in zip(records[:-1], records[1:]):
            gap = int(b["FRAME"]) - int(a["FRAME"])
            if gap <= 1 or gap > max_relink_gap + 1:
                continue
            same_label = int(a.get("particle_id", 0)) == int(b.get("particle_id", 0)) and int(a.get("spot_index", 0)) == int(
                b.get("spot_index", 0)
            )
            for step in range(1, gap):
                alpha = step / gap
                row = dict(a)
                row["FRAME"] = int(a["FRAME"]) + step
                row["X"] = float((1 - alpha) * float(a["X"]) + alpha * float(b["X"]))
                row["Y"] = float((1 - alpha) * float(a["Y"]) + alpha * float(b["Y"]))
                row["intensity"] = float((float(a.get("intensity", 0)) + float(b.get("intensity", 0))) / 2)
                row["detection_id"] = f"interp_t{int(tid)}_f{int(row['FRAME'])}"
                row["is_interpolated"] = True
                row["TRACK_ID"] = int(tid)
                if not same_label:
                    row["particle_id"] = 0
                    row["spot_index"] = 0
                    row["truth_track_id"] = 0
                    row["is_noise"] = True
                interp_rows.append(row)
    if interp_rows:
        repaired = pd.concat([repaired, pd.DataFrame(interp_rows)], ignore_index=True)
    repaired["TRACK_ID"] = repaired["TRACK_ID"].astype(int)
    return repaired.sort_values(["FRAME", "TRACK_ID", "detection_id"]).reset_index(drop=True)


def _split_track_on_lap_cost_jumps(group: pd.DataFrame, feature_scales: np.ndarray) -> list[pd.DataFrame]:
    group = ensure_spot_appearance_columns(group).sort_values("FRAME").copy()
    if len(group) <= 2:
        return [group]
    pts = group[["X", "Y"]].to_numpy(float)
    frames = group["FRAME"].to_numpy(int)
    feats = group[SPOT_APPEARANCE_COLUMNS].to_numpy(float)
    disp = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    finite_disp = disp[np.isfinite(disp)]
    median = float(np.median(finite_disp)) if len(finite_disp) else 0.0
    mad = float(np.median(np.abs(finite_disp - median))) if len(finite_disp) else 0.0
    one_reference_px = _reference_pixels_in_current_image(1.0)
    jump_threshold = max(
        _reference_pixels_in_current_image(18.0),
        median + 5.0 * mad,
        3.2 * max(median, one_reference_px),
    )
    break_after = set()
    link_costs = pd.to_numeric(group.get("lap_link_cost", pd.Series([np.nan] * len(group))), errors="coerce").to_numpy(float)
    for i, d in enumerate(disp):
        gap = int(frames[i + 1] - frames[i])
        feature_jump = float(
            np.linalg.norm((feats[i + 1] - feats[i]) / feature_scales)
            / np.sqrt(len(SPOT_APPEARANCE_COLUMNS))
        )
        next_link_cost = float(link_costs[i + 1]) if np.isfinite(link_costs[i + 1]) else 0.0
        if gap > 2 or d > jump_threshold:
            break_after.add(i)
            continue
        if next_link_cost > 1.75 and (
            d > _reference_pixels_in_current_image(6.0) or feature_jump > 1.5
        ):
            break_after.add(i)
            continue
        if feature_jump > 2.8 and d > max(
            _reference_pixels_in_current_image(5.0),
            1.5 * max(median, one_reference_px),
        ):
            break_after.add(i)
            continue
        if i >= 1:
            v_prev = pts[i] - pts[i - 1]
            v_next = pts[i + 1] - pts[i]
            accel = float(np.linalg.norm(v_next - v_prev))
            dir_cost = _cosine_direction_cost(v_prev, v_next)
            if accel > max(
                _reference_pixels_in_current_image(20.0),
                4.2 * max(median, one_reference_px),
            ) and dir_cost > 0.35:
                break_after.add(i)
    parts = []
    start = 0
    for i in range(len(group) - 1):
        if i in break_after:
            parts.append(group.iloc[start : i + 1].copy())
            start = i + 1
    parts.append(group.iloc[start:].copy())
    return [p for p in parts if not p.empty]


def repair_and_interpolate_tracks_lap_cost(
    raw_tracks: pd.DataFrame,
    n_frames: int,
    max_relink_gap: int = 4,
    max_relink_dist: float | None = None,
    max_relink_cost: float = 0.95,
) -> pd.DataFrame:
    """Split and reconnect fragments with a multi-feature custom assignment cost."""
    if raw_tracks.empty:
        return raw_tracks.copy()
    if max_relink_dist is None:
        max_relink_dist = runtime_sensor_distance_px((), (), 63.0)
    max_relink_dist = float(max_relink_dist)
    gap_growth_dist = _reference_pixels_in_current_image(4.0)
    velocity_scale = _reference_pixels_in_current_image(8.0)
    raw_tracks = ensure_spot_appearance_columns(raw_tracks)
    feature_scales = robust_feature_scales(raw_tracks)

    segments = []
    segment_id = 1
    for _, group in raw_tracks.groupby("RAW_TRACK_ID"):
        for part in _split_track_on_lap_cost_jumps(group, feature_scales):
            part = part.copy()
            part["TRACK_SEGMENT_ID"] = segment_id
            segments.append(part)
            segment_id += 1
    seg_df = pd.concat(segments, ignore_index=True) if segments else raw_tracks.copy()

    seg_meta = []
    for sid, group in seg_df.groupby("TRACK_SEGMENT_ID"):
        group = group.sort_values("FRAME").drop_duplicates("FRAME", keep="last")
        start = group.iloc[0]
        end = group.iloc[-1]
        pts = group[["X", "Y"]].to_numpy(float)
        frames = group["FRAME"].to_numpy(int)
        feats = group[SPOT_APPEARANCE_COLUMNS].to_numpy(float)
        if len(group) >= 2:
            v_start = (pts[1] - pts[0]) / max(1, frames[1] - frames[0])
            v_end = (pts[-1] - pts[-2]) / max(1, frames[-1] - frames[-2])
        else:
            v_start = np.array([0.0, 0.0])
            v_end = np.array([0.0, 0.0])
        seg_meta.append(
            {
                "sid": int(sid),
                "start_frame": int(start["FRAME"]),
                "end_frame": int(end["FRAME"]),
                "start_x": float(start["X"]),
                "start_y": float(start["Y"]),
                "end_x": float(end["X"]),
                "end_y": float(end["Y"]),
                "v_start_x": float(v_start[0]),
                "v_start_y": float(v_start[1]),
                "v_end_x": float(v_end[0]),
                "v_end_y": float(v_end[1]),
                "duration": int(group["FRAME"].nunique()),
                "feature_start": feats[0],
                "feature_end": feats[-1],
            }
        )
    meta = pd.DataFrame(seg_meta)
    parent = {int(sid): int(sid) for sid in meta["sid"].tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(meta) > 1:
        sids = meta["sid"].astype(int).tolist()
        cost = np.full((len(meta), len(meta)), 1e6, dtype=float)
        for r, a in meta.iterrows():
            end_frame = int(a["end_frame"])
            end_xy = np.asarray([float(a["end_x"]), float(a["end_y"])], dtype=float)
            end_vel = np.asarray([float(a["v_end_x"]), float(a["v_end_y"])], dtype=float)
            end_feature = np.asarray(a["feature_end"], dtype=float)
            for c, b in meta.iterrows():
                gap = int(b["start_frame"]) - end_frame
                if gap <= 0 or gap > max_relink_gap + 1:
                    continue
                start_xy = np.asarray([float(b["start_x"]), float(b["start_y"])], dtype=float)
                start_vel = np.asarray([float(b["v_start_x"]), float(b["v_start_y"])], dtype=float)
                predicted = end_xy + end_vel * gap
                distance = float(np.linalg.norm(start_xy - predicted))
                if distance > max_relink_dist + gap_growth_dist * max(0, gap - 1):
                    continue
                bridge_velocity = (start_xy - end_xy) / gap
                velocity_cost = float(np.linalg.norm(bridge_velocity - end_vel) / velocity_scale)
                start_velocity_cost = float(np.linalg.norm(bridge_velocity - start_vel) / velocity_scale)
                direction_cost = max(_cosine_direction_cost(bridge_velocity, end_vel), _cosine_direction_cost(bridge_velocity, start_vel))
                feature_cost = float(
                    np.linalg.norm((np.asarray(b["feature_start"], dtype=float) - end_feature) / feature_scales)
                    / np.sqrt(len(SPOT_APPEARANCE_COLUMNS))
                )
                distance_cost = distance / max_relink_dist
                gap_cost = 0.10 * max(0, gap - 1)
                short_segment_penalty = 0.05 if int(a["duration"]) < 2 or int(b["duration"]) < 2 else 0.0
                edge_cost = (
                    0.45 * distance_cost
                    + 0.18 * velocity_cost
                    + 0.12 * start_velocity_cost
                    + 0.08 * direction_cost
                    + 0.17 * feature_cost
                    + gap_cost
                    + short_segment_penalty
                )
                if edge_cost <= max_relink_cost:
                    cost[int(r), int(c)] = edge_cost
        rr, cc = _stable_linear_sum_assignment(cost)
        accepted = sorted(
            [(float(cost[r, c]), int(r), int(c)) for r, c in zip(rr, cc) if cost[r, c] <= max_relink_cost],
            key=lambda x: (x[0], sids[x[1]], sids[x[2]]),
        )
        for edge_cost, r, c in accepted:
            sid_a = int(sids[r])
            sid_b = int(sids[c])
            if sid_a == sid_b:
                continue
            union(sid_a, sid_b)

    root_to_track: dict[int, int] = {}
    next_tid = 1
    track_ids = []
    for sid in seg_df["TRACK_SEGMENT_ID"].astype(int):
        root = find(int(sid))
        if root not in root_to_track:
            root_to_track[root] = next_tid
            next_tid += 1
        track_ids.append(root_to_track[root])
    repaired = seg_df.copy()
    repaired["TRACK_ID"] = track_ids
    repaired["is_interpolated"] = False
    repaired["repair_model"] = "lap_custom_cost"

    interp_rows = []
    for tid, group in repaired.groupby("TRACK_ID"):
        group = group.sort_values("FRAME").drop_duplicates("FRAME", keep="last")
        records = list(group.to_dict("records"))
        for a, b in zip(records[:-1], records[1:]):
            gap = int(b["FRAME"]) - int(a["FRAME"])
            if gap <= 1 or gap > max_relink_gap + 1:
                continue
            same_label = int(a.get("truth_track_id", 0)) == int(b.get("truth_track_id", 0)) and int(a.get("truth_track_id", 0)) != 0
            for step in range(1, gap):
                alpha = step / gap
                row = dict(a)
                row["FRAME"] = int(a["FRAME"]) + step
                row["X"] = float((1 - alpha) * float(a["X"]) + alpha * float(b["X"]))
                row["Y"] = float((1 - alpha) * float(a["Y"]) + alpha * float(b["Y"]))
                for col in ["intensity", "integrated_intensity", "log_peak_intensity", "log_integrated_intensity", "local_contrast"]:
                    row[col] = float((1 - alpha) * float(a.get(col, 0.0)) + alpha * float(b.get(col, 0.0)))
                for col in ["spot_sigma_x", "spot_sigma_y", "spot_shape_ratio"]:
                    row[col] = float((1 - alpha) * float(a.get(col, 0.0)) + alpha * float(b.get(col, 0.0)))
                row["detection_id"] = f"lap_interp_t{int(tid)}_f{int(row['FRAME'])}"
                row["is_interpolated"] = True
                row["TRACK_ID"] = int(tid)
                row["repair_model"] = "lap_custom_cost"
                if not same_label:
                    row["particle_id"] = 0
                    row["spot_index"] = 0
                    row["truth_track_id"] = 0
                    row["source_id"] = ""
                    row["is_noise"] = True
                interp_rows.append(row)
    if interp_rows:
        repaired = pd.concat([repaired, pd.DataFrame(interp_rows)], ignore_index=True)
    repaired["TRACK_ID"] = repaired["TRACK_ID"].astype(int)
    return repaired.sort_values(["FRAME", "TRACK_ID", "detection_id"]).reset_index(drop=True)


def framewise_group(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    diag_range = _runtime_diag_bounds()
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 3:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            if not is_four_spot_shape(group_pts, diag_range=diag_range, max_score=0.18, max_side_cv=0.13, max_diag_cv=0.07, max_ratio_error=0.13):
                continue
            score = square_score(group_pts)
            center = group_pts.mean(axis=0)
            candidates.append((score, combo, center, mean_diag(group_pts), "four_spot"))
        for combo in local_three_combinations(pts):
            group_pts = pts[list(combo)]
            if not is_missing_one_spot_triangle(group_pts, diag_range=diag_range):
                continue
            tri = triangle_missing_spot_metrics(group_pts)
            center = np.asarray(tri["center"], dtype=float)
            # Penalize degraded three-spot groups so complete four-spot groups win when both are possible.
            candidates.append((float(tri["score"]) + 0.12, combo, center, float(tri["mean_diag"]), "three_spot_missing_one"))
        candidates.sort(key=lambda x: x[0])
        used = set()
        gid = 1
        for score, combo, center, diag_value, geometry_type in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            pids = group["particle_id"].astype(int).tolist()
            stats = association_stats_from_group(group)
            rows.append(
                {
                    "FRAME": int(frame),
                    "group_id": gid,
                    "method": "framewise_geometry",
                    "score": float(score),
                    "member_detection_ids": ";".join(group["detection_id"].astype(str)),
                    "member_track_ids": ";".join(group["TRACK_ID"].astype(str)),
                    "member_particle_ids": ";".join(map(str, pids)),
                    "member_source_ids": ";".join(group.get("source_id", pd.Series([""] * len(group))).astype(str)),
                    "member_possible_source_ids": "|".join(group.get("possible_source_ids", pd.Series([""] * len(group))).astype(str)),
                    "assigned_particle_id": int(stats["assigned_particle_id"]),
                    "assigned_source_id": str(stats.get("assigned_source_id", "")),
                    "group_purity": float(stats["group_purity"]),
                    "is_correct": bool(stats["is_correct"]),
                    "association_status": str(stats["association_status"]),
                    "X_rec": float(center[0]),
                    "Y_rec": float(center[1]),
                    "mean_diag_px": float(diag_value),
                    "mean_diag_sensor_um": float(diag_value) * _runtime_camera_pixel_um(),
                    "n_spots": int(len(group)),
                    "geometry_type": geometry_type,
                }
            )
            used.update(combo)
            gid += 1
    return pd.DataFrame(rows)


def rows_from_frame_group(
    frame: int,
    group_id: int,
    method: str,
    group: pd.DataFrame,
    score: float,
    center_xy: np.ndarray | None = None,
    diag_value: float | None = None,
    geometry_type: str | None = None,
) -> dict:
    pids = group["particle_id"].astype(int).tolist()
    stats = association_stats_from_group(group)
    pts = group[["X", "Y"]].to_numpy(float)
    if center_xy is None:
        center_xy = pts.mean(axis=0)
    if diag_value is None:
        diag_value = mean_diag(pts) if len(pts) == 4 else float(triangle_missing_spot_metrics(pts)["mean_diag"])
    if geometry_type is None:
        geometry_type = "four_spot" if len(group) == 4 else "three_spot_missing_one"
    return {
        "FRAME": int(frame),
        "group_id": int(group_id),
        "method": method,
        "score": float(score),
        "member_detection_ids": ";".join(group["detection_id"].astype(str)),
        "member_track_ids": ";".join(group["TRACK_ID"].astype(str)),
        "member_particle_ids": ";".join(map(str, pids)),
        "member_source_ids": ";".join(group.get("source_id", pd.Series([""] * len(group))).astype(str)),
        "member_possible_source_ids": "|".join(group.get("possible_source_ids", pd.Series([""] * len(group))).astype(str)),
        "assigned_particle_id": int(stats["assigned_particle_id"]),
        "assigned_source_id": str(stats.get("assigned_source_id", "")),
        "group_purity": float(stats["group_purity"]),
        "is_correct": bool(stats["is_correct"]),
        "association_status": str(stats["association_status"]),
        "X_rec": float(center_xy[0]),
        "Y_rec": float(center_xy[1]),
        "mean_diag_px": float(diag_value),
        "mean_diag_sensor_um": float(diag_value) * _runtime_camera_pixel_um(),
        "n_spots": int(len(group)),
        "geometry_type": geometry_type,
    }


def nearest_neighbor_framewise_group(detections: pd.DataFrame) -> pd.DataFrame:
    """A simple per-frame baseline that favors compact nearest-neighbor quartets."""
    rows = []
    min_diag, max_diag = _runtime_diag_bounds()
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 4:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            dists = np.sort(pdist(group_pts))
            diag = float(np.mean(dists[4:]))
            if not min_diag <= diag <= max_diag:
                continue
            compactness = float(np.mean(dists[:4]))
            diag_spread = float(np.std(dists[4:]))
            candidates.append((compactness + 0.25 * diag_spread, combo))
        candidates.sort(key=lambda x: x[0])
        used = set()
        gid = 1
        for score, combo in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            rows.append(rows_from_frame_group(int(frame), gid, "nearest_neighbor_framewise", group, score))
            used.update(combo)
            gid += 1
    return pd.DataFrame(rows)


def geometry_previous_frame_group(detections: pd.DataFrame) -> pd.DataFrame:
    """A stronger frame-wise baseline with a previous-frame center continuity term."""
    rows = []
    diag_range = _runtime_diag_bounds()
    prev_centers: np.ndarray | None = None
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 3:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            if not is_four_spot_shape(group_pts, diag_range=diag_range, max_score=0.22, max_side_cv=0.16, max_diag_cv=0.09, max_ratio_error=0.16):
                continue
            geom = square_score(group_pts)
            center = group_pts.mean(axis=0)
            temporal_penalty = 0.0
            if prev_centers is not None and len(prev_centers):
                temporal_penalty = 0.02 * float(np.min(np.linalg.norm(prev_centers - center[None, :], axis=1)))
            candidates.append((geom + temporal_penalty, combo, center, mean_diag(group_pts), "four_spot"))
        for combo in local_three_combinations(pts):
            group_pts = pts[list(combo)]
            if not is_missing_one_spot_triangle(group_pts, diag_range=diag_range, max_score=0.18, max_side_cv=0.12, max_ratio_error=0.14):
                continue
            tri = triangle_missing_spot_metrics(group_pts)
            center = np.asarray(tri["center"], dtype=float)
            temporal_penalty = 0.0
            if prev_centers is not None and len(prev_centers):
                temporal_penalty = 0.02 * float(np.min(np.linalg.norm(prev_centers - center[None, :], axis=1)))
            candidates.append(
                (float(tri["score"]) + 0.12 + temporal_penalty, combo, center, float(tri["mean_diag"]), "three_spot_missing_one")
            )
        candidates.sort(key=lambda x: x[0])
        used = set()
        frame_rows = []
        centers = []
        gid = 1
        for score, combo, center, diag_value, geometry_type in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            frame_rows.append(
                rows_from_frame_group(
                    int(frame),
                    gid,
                    "geometry_previous_frame",
                    group,
                    score,
                    center_xy=center,
                    diag_value=diag_value,
                    geometry_type=geometry_type,
                )
            )
            centers.append(center)
            used.update(combo)
            gid += 1
        rows.extend(frame_rows)
        if centers:
            prev_centers = np.asarray(centers, dtype=float)
    return pd.DataFrame(rows)


def link_framewise_groups_to_trajectories(
    groups: pd.DataFrame,
    method_name: str = "geometry_previous_frame_linked",
    max_link_dist: float = 30.0,
    max_gap: int = 3,
) -> pd.DataFrame:
    """Link per-frame geometric groups into object trajectories using only recovered centers."""
    if groups.empty:
        return groups.copy()
    linked = groups.sort_values(["FRAME", "score"]).copy()
    linked["FRAMEWISE_TRAJECTORY_ID"] = -1
    linked["prelink_assigned_source_id"] = linked.get("assigned_source_id", pd.Series([""] * len(linked))).astype(str)
    linked["prelink_is_correct"] = linked.get("is_correct", pd.Series([False] * len(linked))).astype(bool)
    tracks: dict[int, dict[str, float | int]] = {}
    next_tid = 1

    for frame in sorted(linked["FRAME"].astype(int).unique()):
        frame_idx = linked.index[linked["FRAME"].astype(int) == int(frame)].to_numpy()
        centers = linked.loc[frame_idx, ["X_rec", "Y_rec"]].to_numpy(float)
        active_ids = [
            tid for tid, state in tracks.items() if int(frame) - int(state["last_frame"]) <= max_gap + 1
        ]
        assigned_tracks: set[int] = set()
        assigned_rows: set[int] = set()
        if active_ids and len(frame_idx):
            cost = np.full((len(active_ids), len(frame_idx)), 1e6, dtype=float)
            for r, tid in enumerate(active_ids):
                state = tracks[tid]
                dt = max(1, int(frame) - int(state["last_frame"]))
                pred = np.asarray([float(state["x"]) + float(state["vx"]) * dt, float(state["y"]) + float(state["vy"]) * dt])
                distances = np.linalg.norm(centers - pred[None, :], axis=1)
                cost[r, :] = distances
            rr, cc = _stable_linear_sum_assignment(cost)
            pairs = sorted(
                [(float(cost[r, c]), int(active_ids[r]), int(c)) for r, c in zip(rr, cc) if cost[r, c] <= max_link_dist],
                key=lambda item: item[0],
            )
            for _, tid, local_idx in pairs:
                if tid in assigned_tracks or local_idx in assigned_rows:
                    continue
                row_idx = frame_idx[local_idx]
                state = tracks[tid]
                dt = max(1, int(frame) - int(state["last_frame"]))
                x = float(linked.loc[row_idx, "X_rec"])
                y = float(linked.loc[row_idx, "Y_rec"])
                vx_new = (x - float(state["x"])) / dt
                vy_new = (y - float(state["y"])) / dt
                linked.loc[row_idx, "FRAMEWISE_TRAJECTORY_ID"] = tid
                tracks[tid] = {
                    "x": x,
                    "y": y,
                    "last_frame": int(frame),
                    "vx": 0.65 * vx_new + 0.35 * float(state["vx"]),
                    "vy": 0.65 * vy_new + 0.35 * float(state["vy"]),
                }
                assigned_tracks.add(tid)
                assigned_rows.add(local_idx)

        for local_idx, row_idx in enumerate(frame_idx):
            if local_idx in assigned_rows:
                continue
            x = float(linked.loc[row_idx, "X_rec"])
            y = float(linked.loc[row_idx, "Y_rec"])
            linked.loc[row_idx, "FRAMEWISE_TRAJECTORY_ID"] = next_tid
            tracks[next_tid] = {"x": x, "y": y, "last_frame": int(frame), "vx": 0.0, "vy": 0.0}
            next_tid += 1

    linked["FRAMEWISE_TRAJECTORY_ID"] = linked["FRAMEWISE_TRAJECTORY_ID"].astype(int)
    linked["linked_assigned_source_id"] = ""
    linked["linked_assigned_particle_id"] = 0
    linked["linked_is_correct"] = False
    linked["linked_association_status"] = "BAD_OR_MIXED"

    for tid, group in linked.groupby("FRAMEWISE_TRAJECTORY_ID"):
        correct = group[group["prelink_is_correct"].astype(bool)]
        source_series = correct["prelink_assigned_source_id"].astype(str)
        source_series = source_series[source_series != ""]
        if source_series.empty:
            source_series = group["prelink_assigned_source_id"].astype(str)
            source_series = source_series[source_series != ""]
        linked_source = str(source_series.mode().iloc[0]) if not source_series.empty else ""
        linked_pid = source_to_particle_id(linked_source)
        for idx, row in group.iterrows():
            pre_source = str(row.get("prelink_assigned_source_id", ""))
            pre_correct = bool(row.get("prelink_is_correct", False))
            row_correct = bool(pre_correct and linked_source != "" and pre_source == linked_source)
            linked.loc[idx, "linked_assigned_source_id"] = linked_source
            linked.loc[idx, "linked_assigned_particle_id"] = int(linked_pid)
            linked.loc[idx, "linked_is_correct"] = row_correct
            if row_correct:
                linked.loc[idx, "linked_association_status"] = "LINKED_OK"
            elif pre_correct:
                linked.loc[idx, "linked_association_status"] = "LINKED_ID_SWITCH"
            else:
                linked.loc[idx, "linked_association_status"] = "BAD_OR_MIXED"

    linked["method"] = method_name
    linked["assigned_source_id"] = linked["linked_assigned_source_id"].astype(str)
    linked["assigned_particle_id"] = linked["linked_assigned_particle_id"].astype(int)
    linked["is_correct"] = linked["linked_is_correct"].astype(bool)
    linked["association_status"] = linked["linked_association_status"].astype(str)
    return linked.sort_values(["FRAMEWISE_TRAJECTORY_ID", "FRAME"]).reset_index(drop=True)


def oracle_group(detections: pd.DataFrame) -> pd.DataFrame:
    """Upper bound baseline: uses true source labels to group sub-spots."""
    rows = []
    if "source_id" not in detections.columns:
        return pd.DataFrame(rows)
    valid = detections[detections["source_id"].astype(str) != ""]
    for (frame, source_id), group in valid.groupby(["FRAME", "source_id"]):
        if len(group) < 3:
            continue
        rows.append(rows_from_frame_group(int(frame), len(rows) + 1, "oracle_grouping", group, 0.0))
    return pd.DataFrame(rows)


def track_features(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, group in detections.groupby("TRACK_ID"):
        group = group.sort_values("FRAME")
        pts = group[["X", "Y"]].to_numpy(float)
        disp = np.diff(pts, axis=0)
        speed = np.linalg.norm(disp, axis=1)
        hard_sources = [str(x) if str(x) != "nan" else "" for x in group.get("source_id", pd.Series([""] * len(group))).tolist()]
        possible_sources: set[str] = set()
        for value, source in zip(group.get("possible_source_ids", pd.Series([""] * len(group))).tolist(), hard_sources):
            possible_sources.update(parse_source_set(value, fallback=source))
        rows.append(
            {
                "TRACK_ID": int(tid),
                "mean_x": float(np.mean(pts[:, 0])),
                "mean_y": float(np.mean(pts[:, 1])),
                "vx": float(np.mean(disp[:, 0])) if len(disp) else 0.0,
                "vy": float(np.mean(disp[:, 1])) if len(disp) else 0.0,
                "speed_mean": float(np.mean(speed)) if len(speed) else 0.0,
                "speed_std": float(np.std(speed)) if len(speed) else 0.0,
                "duration": int(group["FRAME"].nunique()),
                "true_particle_id": int(group["particle_id"].mode().iloc[0]),
                "source_id": str(group["source_id"].mode().iloc[0]) if "source_id" in group and not group["source_id"].mode().empty else "",
                "possible_source_ids": ";".join(sorted(possible_sources)),
            }
        )
    return pd.DataFrame(rows)


def trackmate_style_track_features(detections: pd.DataFrame) -> pd.DataFrame:
    """Build TrackMate-like track statistics for the production PCA classifier."""
    rows = []
    for tid, group in detections.groupby("TRACK_ID"):
        group = group.sort_values("FRAME").drop_duplicates("FRAME", keep="last")
        if group.empty:
            continue
        pts = group[["X", "Y"]].to_numpy(float)
        frames = group["FRAME"].to_numpy(int)
        disp = np.diff(pts, axis=0)
        dt = np.diff(frames).astype(float)
        valid_dt = np.where(dt > 0, dt, 1.0)
        step_dist = np.linalg.norm(disp, axis=1)
        speed = step_dist / valid_dt if len(step_dist) else np.asarray([], dtype=float)
        total_distance = float(np.sum(step_dist)) if len(step_dist) else 0.0
        net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
        if len(pts) > 1:
            max_distance = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
        else:
            max_distance = 0.0
        if len(disp) >= 2:
            angles = np.arctan2(disp[:, 1], disp[:, 0])
            dtheta = np.diff(np.unwrap(angles))
            mean_directional_change = float(np.mean(np.abs(dtheta)))
        else:
            mean_directional_change = 0.0
        hard_sources = [
            str(x) if str(x) != "nan" else ""
            for x in group.get("source_id", pd.Series([""] * len(group))).tolist()
        ]
        possible_sources: set[str] = set()
        for value, source in zip(group.get("possible_source_ids", pd.Series([""] * len(group))).tolist(), hard_sources):
            possible_sources.update(parse_source_set(value, fallback=source))
        rows.append(
            {
                "TRACK_ID": int(tid),
                "LABEL": f"T{int(tid)}",
                "TRACK_DURATION": int(group["FRAME"].nunique()),
                "TRACK_MEAN_SPEED": float(np.mean(speed)) if len(speed) else 0.0,
                "TRACK_STD_SPEED": float(np.std(speed)) if len(speed) else 0.0,
                "CONFINEMENT_RATIO": float(net_disp / total_distance) if total_distance > 0 else 0.0,
                "LINEARITY_OF_FORWARD_PROGRESSION": float(net_disp / total_distance) if total_distance > 0 else 0.0,
                "MEAN_DIRECTIONAL_CHANGE_RATE": mean_directional_change,
                "TRACK_X_LOCATION": float(np.mean(pts[:, 0])),
                "TRACK_Y_LOCATION": float(np.mean(pts[:, 1])),
                "MAX_DISTANCE_TRAVELED": max_distance,
                "true_particle_id": int(group["particle_id"].mode().iloc[0]),
                "source_id": str(group["source_id"].mode().iloc[0]) if "source_id" in group and not group["source_id"].mode().empty else "",
                "possible_source_ids": ";".join(sorted(possible_sources)),
            }
        )
    return pd.DataFrame(rows)


def _production_square_score(points_xy: np.ndarray) -> float:
    dists = sorted(pdist(points_xy))
    if len(dists) != 6:
        return np.inf
    sides, diags = np.asarray(dists[:4], dtype=float), np.asarray(dists[4:], dtype=float)
    mean_side = float(np.mean(sides))
    mean_diag_value = float(np.mean(diags))
    if mean_side <= 0 or mean_diag_value <= 0:
        return np.inf
    return float(
        np.std(sides) / mean_side
        + np.std(diags) / mean_diag_value
        + abs(mean_diag_value / mean_side - 1.414)
    )


def trajectory_level_group(detections: pd.DataFrame, method_name: str = "trajectory_level") -> pd.DataFrame:
    """Production-style TrackMate track classification: PCA motion features + track-mean geometry."""
    params = PRODUCTION_TRACK_CLASSIFICATION_PARAMS
    feats = trackmate_style_track_features(detections)
    feat_cols = [
        "TRACK_MEAN_SPEED",
        "TRACK_STD_SPEED",
        "CONFINEMENT_RATIO",
        "LINEARITY_OF_FORWARD_PROGRESSION",
        "MEAN_DIRECTIONAL_CHANGE_RATE",
    ]
    pos_cols = ["TRACK_X_LOCATION", "TRACK_Y_LOCATION"]
    required = pos_cols + feat_cols + ["MAX_DISTANCE_TRAVELED", "TRACK_ID"]
    feats = feats.dropna(subset=required).reset_index(drop=True)
    if len(feats) < 3:
        return pd.DataFrame()

    x_pos = feats[pos_cols].to_numpy(float)
    x_feat = feats[feat_cols].to_numpy(float)
    x_feat_scaled = StandardScaler().fit_transform(x_feat)
    if np.allclose(np.nanvar(x_feat_scaled, axis=0), 0.0):
        x_feat_pca = np.zeros((len(feats), 1), dtype=float)
    else:
        n_comp = min(0.9, 1.0)
        x_feat_pca = PCA(n_components=n_comp).fit_transform(x_feat_scaled)

    n_neighbors = min(int(params["NEIGHBORS_K"]), len(feats))
    indices = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(x_pos).kneighbors(x_pos, return_distance=False)

    candidates_4 = []
    seen_4: set[tuple[int, ...]] = set()
    for i in range(len(feats)):
        for combo in combinations(indices[i], 4):
            combo = tuple(sorted(int(x) for x in combo))
            if combo in seen_4:
                continue
            seen_4.add(combo)
            pts = x_pos[list(combo)]
            dists = sorted(pdist(pts))
            if len(dists) != 6 or dists[-1] > float(params["MAX_DIAGONAL"]):
                continue
            sq_score = _production_square_score(pts)
            if sq_score > float(params["SQUARE_TOLERANCE"]):
                continue
            feat_dist = float(np.mean(pdist(x_feat_pca[list(combo)])))
            if feat_dist > float(params["LIMIT_SQUARE"]):
                continue
            candidates_4.append(
                {
                    "indices": combo,
                    "feature_score": feat_dist,
                    "geometry_score": sq_score,
                    "type": "4-Square",
                }
            )
    candidates_4.sort(key=lambda x: x["feature_score"])

    used: set[int] = set()
    selected = []
    for cand in candidates_4:
        if set(cand["indices"]).isdisjoint(used):
            selected.append(cand)
            used.update(cand["indices"])

    candidates_3 = []
    seen_3: set[tuple[int, ...]] = set()
    remaining = [i for i in range(len(feats)) if i not in used]
    for i in remaining:
        current_neighbors = [int(n) for n in indices[i] if int(n) not in used]
        for combo in combinations(current_neighbors[:15], 3):
            combo = tuple(sorted(int(x) for x in combo))
            if combo in seen_3:
                continue
            seen_3.add(combo)
            pts = x_pos[list(combo)]
            dists = sorted(pdist(pts))
            if len(dists) != 3 or dists[-1] > float(params["MAX_DIAGONAL"]):
                continue
            if dists[-1] <= 0 or (dists[0] + dists[1]) / dists[2] < 1.05:
                continue
            feat_dist = float(np.mean(pdist(x_feat_pca[list(combo)])))
            if feat_dist > float(params["LIMIT_TRIANGLE"]):
                continue
            candidates_3.append(
                {
                    "indices": combo,
                    "feature_score": feat_dist,
                    "geometry_score": float((dists[0] + dists[1]) / dists[2]),
                    "type": "3-Triangle",
                }
            )
    candidates_3.sort(key=lambda x: x["feature_score"])
    for cand in candidates_3:
        if set(cand["indices"]).isdisjoint(used):
            selected.append(cand)
            used.update(cand["indices"])

    groups = []
    for gid, cand in enumerate(selected, start=1):
        combo = list(cand["indices"])
        n_tracks = len(combo)
        tids = feats.iloc[combo]["TRACK_ID"].astype(int).tolist()
        pids = feats.iloc[combo]["true_particle_id"].astype(int).tolist()
        hard_sources = feats.iloc[combo]["source_id"].astype(str).tolist()
        source_sets = [
            parse_source_set(value, fallback=source)
            for value, source in zip(feats.iloc[combo]["possible_source_ids"], hard_sources)
        ]
        stats = association_stats_from_source_sets(source_sets, hard_sources)
        groups.append(
            {
                "trajectory_group_id": gid,
                "method": method_name,
                "score": float(cand["feature_score"]),
                "feature_score": float(cand["feature_score"]),
                "geometry_score": float(cand["geometry_score"]),
                "n_tracks": int(n_tracks),
                "trajectory_geometry_type": "production_4_square" if n_tracks == 4 else "production_3_triangle",
                "member_track_ids": ";".join(map(str, tids)),
                "member_particle_ids": ";".join(map(str, pids)),
                "member_source_ids": ";".join(hard_sources),
                "member_possible_source_ids": "|".join(feats.iloc[combo]["possible_source_ids"].astype(str)),
                "assigned_particle_id": int(stats["assigned_particle_id"]),
                "assigned_source_id": str(stats.get("assigned_source_id", "")),
                "group_purity": float(stats["group_purity"]),
                "is_correct": bool(stats["is_correct"]),
                "association_status": str(stats["association_status"]),
            }
        )
    return pd.DataFrame(groups)


def _track_source_summary(
    detections: pd.DataFrame,
    track_ids: list[int],
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[list[int], list[str], list[str], list[set[str]]]:
    pids = []
    hard_sources = []
    possible_texts = []
    possible_sets = []
    for tid in track_ids:
        group = detections[detections["TRACK_ID"].astype(int) == int(tid)]
        if start_frame is not None:
            group = group[group["FRAME"].astype(int) >= int(start_frame)]
        if end_frame is not None:
            group = group[group["FRAME"].astype(int) <= int(end_frame)]
        if group.empty:
            pids.append(0)
            hard_sources.append("")
            possible_texts.append("")
            possible_sets.append(set())
            continue
        pid_mode = group["particle_id"].mode() if "particle_id" in group.columns else pd.Series([], dtype=int)
        source_mode = group["source_id"].astype(str).mode() if "source_id" in group.columns else pd.Series([], dtype=str)
        pid = int(pid_mode.iloc[0]) if not pid_mode.empty else 0
        hard_source = str(source_mode.iloc[0]) if not source_mode.empty and str(source_mode.iloc[0]) != "nan" else ""
        source_set: set[str] = set()
        possible_col = group.get("possible_source_ids", pd.Series([""] * len(group)))
        hard_col = group.get("source_id", pd.Series([""] * len(group)))
        for value, source in zip(possible_col.tolist(), hard_col.tolist()):
            source_set.update(parse_source_set(value, fallback=source))
        if not source_set and hard_source:
            source_set = {hard_source}
        pids.append(pid)
        hard_sources.append(hard_source)
        possible_texts.append(";".join(sorted(source_set)))
        possible_sets.append(source_set)
    return pids, hard_sources, possible_texts, possible_sets


def _recoverable_frames_for_tracks(
    track_frames: dict[int, set[int]],
    track_ids: list[int],
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[int]:
    if not track_ids:
        return []
    frame_union = set().union(*(track_frames[tid] for tid in track_ids if tid in track_frames))
    if start_frame is not None:
        frame_union = {frame for frame in frame_union if frame >= int(start_frame)}
    if end_frame is not None:
        frame_union = {frame for frame in frame_union if frame <= int(end_frame)}
    return [
        frame
        for frame in sorted(frame_union)
        if sum(frame in track_frames.get(tid, set()) for tid in track_ids) >= 3
    ]


def _trajectory_common_window_candidate(
    combo: tuple[int, ...],
    track_frames: dict[int, set[int]],
    track_rows: dict[int, dict[int, object]],
) -> dict[str, object] | None:
    common_frames = set.intersection(*(track_frames[int(tid)] for tid in combo))
    if len(common_frames) < 3:
        return None
    n_tracks = len(combo)
    geometry_scores = []
    diag_values = []
    min_diag, max_diag = _runtime_diag_bounds()
    for frame in sorted(common_frames):
        pts = np.asarray([[float(track_rows[int(tid)][frame].X), float(track_rows[int(tid)][frame].Y)] for tid in combo], dtype=float)
        if n_tracks == 4:
            metrics = four_spot_shape_metrics(pts)
            if not min_diag <= float(metrics["mean_diag"]) <= max_diag:
                return None
            geometry_scores.append(float(metrics["score"]))
            diag_values.append(float(metrics["mean_diag"]))
        elif n_tracks == 3:
            metrics = triangle_missing_spot_metrics(pts)
            if not min_diag <= float(metrics["mean_diag"]) <= max_diag:
                return None
            geometry_scores.append(float(metrics["score"]))
            diag_values.append(float(metrics["mean_diag"]))
        else:
            return None
    if not geometry_scores:
        return None

    geometry_scores_arr = np.asarray(geometry_scores, dtype=float)
    diag_arr = np.asarray(diag_values, dtype=float)
    median_geometry = float(np.median(geometry_scores_arr))
    p75_geometry = float(np.percentile(geometry_scores_arr, 75))
    diag_cv = float(np.std(diag_arr) / max(np.mean(diag_arr), 1e-6))
    motion_residuals = []
    common_sorted = sorted(common_frames)
    common_set = set(common_sorted)
    for frame in common_sorted:
        if frame + 1 not in common_set:
            continue
        velocities = []
        for tid in combo:
            a = track_rows[int(tid)][frame]
            b = track_rows[int(tid)][frame + 1]
            velocities.append([float(b.X) - float(a.X), float(b.Y) - float(a.Y)])
        velocities_arr = np.asarray(velocities, dtype=float)
        center_velocity = velocities_arr.mean(axis=0)
        motion_residuals.extend(np.linalg.norm(velocities_arr - center_velocity[None, :], axis=1).tolist())
    motion_residual = float(np.median(motion_residuals)) if motion_residuals else 0.0

    if n_tracks == 4:
        if median_geometry > 0.12 or p75_geometry > 0.16:
            return None
    else:
        if median_geometry > 0.16 or p75_geometry > 0.22:
            return None
    if diag_cv > 0.18 or motion_residual > 3.5:
        return None

    score = float(median_geometry + 0.15 * diag_cv + 0.04 * motion_residual - 0.002 * min(len(common_frames), 50))
    return {
        "combo": tuple(int(tid) for tid in combo),
        "n_tracks": int(n_tracks),
        "common_n": int(len(common_frames)),
        "start_frame": int(min(common_frames)),
        "end_frame": int(max(common_frames)),
        "geometry_score": median_geometry,
        "geometry_p75": p75_geometry,
        "diag_cv": diag_cv,
        "mean_diag_px": float(np.median(diag_arr)),
        "mean_diag_sensor_um": float(np.median(diag_arr)) * _runtime_camera_pixel_um(),
        "motion_residual": motion_residual,
        "score": score,
    }


def _contiguous_frame_intervals(frames: list[int], min_len: int) -> list[list[int]]:
    if not frames:
        return []
    frames = sorted({int(frame) for frame in frames})
    intervals: list[list[int]] = []
    current = [frames[0]]
    for frame in frames[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            if len(current) >= int(min_len):
                intervals.append(current)
            current = [frame]
    if len(current) >= int(min_len):
        intervals.append(current)
    return intervals


def _shape_metrics_for_track_combo(
    combo: tuple[int, ...],
    track_rows: dict[int, dict[int, object]],
    frame: int,
) -> dict[str, float] | None:
    try:
        pts = np.asarray(
            [[float(track_rows[int(tid)][int(frame)].X), float(track_rows[int(tid)][int(frame)].Y)] for tid in combo],
            dtype=float,
        )
    except Exception:
        return None
    if len(combo) == 4:
        return four_spot_shape_metrics(pts)
    if len(combo) == 3:
        return triangle_missing_spot_metrics(pts)
    return None


def _track_combo_interval_candidate(
    combo: tuple[int, ...],
    track_rows: dict[int, dict[int, object]],
    frames: list[int],
) -> dict[str, object] | None:
    frames = sorted({int(frame) for frame in frames})
    if not frames:
        return None
    n_tracks = len(combo)
    geometry_scores = []
    diag_values = []
    for frame in frames:
        metrics = _shape_metrics_for_track_combo(combo, track_rows, frame)
        if metrics is None:
            return None
        geometry_scores.append(float(metrics["score"]))
        diag_values.append(float(metrics["mean_diag"]))

    geometry_scores_arr = np.asarray(geometry_scores, dtype=float)
    diag_arr = np.asarray(diag_values, dtype=float)
    median_geometry = float(np.median(geometry_scores_arr))
    p75_geometry = float(np.percentile(geometry_scores_arr, 75))
    diag_cv = float(np.std(diag_arr) / max(np.mean(diag_arr), 1e-6))
    motion_residuals = []
    frame_set = set(frames)
    for frame in frames:
        if frame + 1 not in frame_set:
            continue
        velocities = []
        for tid in combo:
            a = track_rows[int(tid)][frame]
            b = track_rows[int(tid)][frame + 1]
            velocities.append([float(b.X) - float(a.X), float(b.Y) - float(a.Y)])
        velocities_arr = np.asarray(velocities, dtype=float)
        center_velocity = velocities_arr.mean(axis=0)
        motion_residuals.extend(np.linalg.norm(velocities_arr - center_velocity[None, :], axis=1).tolist())
    motion_residual = float(np.median(motion_residuals)) if motion_residuals else 0.0

    if n_tracks == 4:
        if median_geometry > 0.14 or p75_geometry > 0.18:
            return None
    elif n_tracks == 3:
        if median_geometry > 0.16 or p75_geometry > 0.22:
            return None
    else:
        return None
    if diag_cv > 0.18 or motion_residual > 3.5:
        return None

    four_track_bonus = _runtime_float("TRAJ_ASSOC_FOUR_TRACK_INTERVAL_BONUS", 0.06) if n_tracks == 4 else 0.0
    score = float(
        median_geometry
        + 0.15 * diag_cv
        + 0.04 * motion_residual
        - 0.003 * min(len(frames), 40)
        - four_track_bonus
    )
    return {
        "combo": tuple(int(tid) for tid in combo),
        "n_tracks": int(n_tracks),
        "common_n": int(len(frames)),
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "geometry_score": median_geometry,
        "geometry_p75": p75_geometry,
        "diag_cv": diag_cv,
        "mean_diag_px": float(np.median(diag_arr)),
        "mean_diag_sensor_um": float(np.median(diag_arr)) * _runtime_camera_pixel_um(),
        "motion_residual": motion_residual,
        "score": score,
        "valid_frames": tuple(int(frame) for frame in frames),
    }


def _interval_candidate_sort_key(candidate: dict[str, object]) -> tuple[object, ...]:
    prefer_four = _runtime_bool("TRAJ_ASSOC_PREFER_FOUR_TRACK_INTERVALS", True)
    n_tracks = int(candidate.get("n_tracks", 0))
    score = float(candidate.get("score", np.inf))
    common_n = int(candidate.get("common_n", 0))
    start_frame = int(candidate.get("start_frame", 0))
    if prefer_four:
        return (-n_tracks, score, -common_n, start_frame)
    return (score, -n_tracks, -common_n, start_frame)


def _trajectory_segment_candidates(
    combo: tuple[int, ...],
    track_frames: dict[int, set[int]],
    track_rows: dict[int, dict[int, object]],
    start_frame: int,
    end_frame: int,
) -> list[dict[str, object]]:
    common_frames = set.intersection(*(track_frames[int(tid)] for tid in combo))
    common_frames = {frame for frame in common_frames if int(start_frame) <= frame <= int(end_frame)}
    min_interval_frames = int(_runtime_float("TRAJ_ASSOC_LOCAL_MIN_INTERVAL_FRAMES", 5.0))
    if len(common_frames) < min_interval_frames:
        return []
    n_tracks = len(combo)
    min_diag, max_diag = _runtime_diag_bounds()
    max_four_frame_score = _runtime_float("TRAJ_ASSOC_MAX_FOUR_FRAME_SCORE", 0.18)
    max_three_frame_score = _runtime_float("TRAJ_ASSOC_MAX_THREE_FRAME_SCORE", 0.18)
    good_frames = []
    for frame in sorted(common_frames):
        metrics = _shape_metrics_for_track_combo(combo, track_rows, frame)
        if metrics is None:
            continue
        diag = float(metrics["mean_diag"])
        score = float(metrics["score"])
        if not min_diag <= diag <= max_diag:
            continue
        if n_tracks == 4 and score <= max_four_frame_score:
            good_frames.append(int(frame))
        elif n_tracks == 3 and score <= max_three_frame_score:
            good_frames.append(int(frame))
        else:
            continue

    candidates = []
    for interval_frames in _contiguous_frame_intervals(good_frames, min_interval_frames):
        candidate = _track_combo_interval_candidate(tuple(int(tid) for tid in combo), track_rows, interval_frames)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=_interval_candidate_sort_key)
    return candidates


def _trajectory_segment_candidate(
    combo: tuple[int, ...],
    track_frames: dict[int, set[int]],
    track_rows: dict[int, dict[int, object]],
    start_frame: int,
    end_frame: int,
) -> dict[str, object] | None:
    candidates = _trajectory_segment_candidates(combo, track_frames, track_rows, start_frame, end_frame)
    return candidates[0] if candidates else None


def _local_segment_track_groups(
    track_ids: list[int],
    track_frames: dict[int, set[int]],
    track_rows: dict[int, dict[int, object]],
    start_frame: int,
    end_frame: int,
) -> list[dict[str, object]]:
    active_tracks = [
        tid
        for tid in track_ids
        if sum(int(start_frame) <= frame <= int(end_frame) for frame in track_frames.get(tid, set())) >= 5
    ]
    if len(active_tracks) < 3:
        return []

    candidates = []
    for n_tracks in (4, 3):
        if len(active_tracks) < n_tracks:
            continue
        for combo in combinations(active_tracks, n_tracks):
            candidates.extend(
                _trajectory_segment_candidates(
                    tuple(int(tid) for tid in combo),
                    track_frames,
                    track_rows,
                    start_frame,
                    end_frame,
                )
            )
    if not candidates:
        return []

    min_interval_frames = int(_runtime_float("TRAJ_ASSOC_LOCAL_MIN_INTERVAL_FRAMES", 5.0))
    candidates.sort(key=_interval_candidate_sort_key)
    selected = []
    used_track_frames: set[tuple[int, int]] = set()
    seen_trimmed: set[tuple[tuple[int, ...], int, int]] = set()
    queue = list(candidates)
    while queue:
        cand = queue.pop(0)
        combo = tuple(int(tid) for tid in cand["combo"])
        frames = [int(frame) for frame in cand.get("valid_frames", [])]
        if not frames:
            frames = list(range(int(cand["start_frame"]), int(cand["end_frame"]) + 1))
        available = [
            frame
            for frame in frames
            if not any((tid, frame) in used_track_frames for tid in combo)
        ]
        if len(available) < min_interval_frames:
            continue
        if len(available) != len(frames):
            for interval_frames in _contiguous_frame_intervals(available, min_interval_frames):
                key = (combo, int(interval_frames[0]), int(interval_frames[-1]))
                if key in seen_trimmed:
                    continue
                seen_trimmed.add(key)
                trimmed = _track_combo_interval_candidate(combo, track_rows, interval_frames)
                if trimmed is not None:
                    queue.append(trimmed)
            queue.sort(key=_interval_candidate_sort_key)
            continue
        selected.append(cand)
        for tid in combo:
            for frame in available:
                used_track_frames.add((tid, int(frame)))
    return selected


def _parse_member_track_ids(value: object) -> tuple[int, ...]:
    tids: list[int] = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            tids.append(int(float(token)))
        except Exception:
            continue
    return tuple(sorted(set(tids)))


def _seed_combos_from_framewise_evidence(
    evidence: pd.DataFrame,
    track_frames: dict[int, set[int]],
) -> set[tuple[int, ...]]:
    seed_combos: set[tuple[int, ...]] = set()
    if evidence.empty or "member_track_ids" not in evidence.columns:
        return seed_combos
    add_three_subsets = _runtime_bool("TRAJ_ASSOC_ADD_THREE_SUBSETS_FROM_FOUR", True)
    for _, group in evidence.iterrows():
        tids = _parse_member_track_ids(group.get("member_track_ids", ""))
        if len(tids) in {3, 4} and all(tid in track_frames for tid in tids):
            seed_combos.add(tids)
            if len(tids) == 4 and add_three_subsets:
                for subset in combinations(tids, 3):
                    seed_combos.add(tuple(int(tid) for tid in subset))
    return seed_combos


def _select_interval_candidates(
    candidates: list[dict[str, object]],
    track_rows: dict[int, dict[int, object]],
) -> list[dict[str, object]]:
    min_interval_frames = int(_runtime_float("TRAJ_ASSOC_LOCAL_MIN_INTERVAL_FRAMES", 5.0))
    queue = sorted(candidates, key=_interval_candidate_sort_key)
    selected: list[dict[str, object]] = []
    used_track_frames: set[tuple[int, int]] = set()
    seen_trimmed: set[tuple[tuple[int, ...], int, int]] = set()

    while queue:
        cand = queue.pop(0)
        combo = tuple(int(tid) for tid in cand["combo"])
        frames = [int(frame) for frame in cand.get("valid_frames", [])]
        if not frames:
            frames = list(range(int(cand["start_frame"]), int(cand["end_frame"]) + 1))
        available = [
            frame
            for frame in frames
            if not any((tid, frame) in used_track_frames for tid in combo)
        ]
        if len(available) < min_interval_frames:
            continue
        if len(available) != len(frames):
            for interval_frames in _contiguous_frame_intervals(available, min_interval_frames):
                key = (combo, int(interval_frames[0]), int(interval_frames[-1]))
                if key in seen_trimmed:
                    continue
                seen_trimmed.add(key)
                trimmed = _track_combo_interval_candidate(combo, track_rows, interval_frames)
                if trimmed is not None:
                    queue.append(trimmed)
            queue.sort(key=_interval_candidate_sort_key)
            continue

        selected.append(cand)
        for tid in combo:
            for frame in available:
                used_track_frames.add((tid, int(frame)))
    return selected


def _interval_endpoint_points(
    candidate: dict[str, object],
    track_rows: dict[int, dict[int, object]],
    *,
    start: bool,
) -> np.ndarray | None:
    frame = int(candidate["start_frame"] if start else candidate["end_frame"])
    points = []
    for tid in candidate["combo"]:
        row = track_rows.get(int(tid), {}).get(frame)
        if row is None:
            continue
        points.append([float(row.X), float(row.Y)])
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=float)


def _track_turnover_endpoint_cost(
    previous: dict[str, object],
    current: dict[str, object],
    track_rows: dict[int, dict[int, object]],
    max_point_distance_px: float,
) -> tuple[float, float] | None:
    """Match physical subspots when an upstream tracker has replaced their IDs."""
    previous_points = _interval_endpoint_points(previous, track_rows, start=False)
    current_points = _interval_endpoint_points(current, track_rows, start=True)
    if previous_points is None or current_points is None:
        return None
    distances = np.linalg.norm(previous_points[:, None, :] - current_points[None, :, :], axis=2)
    row_indices, column_indices = _stable_linear_sum_assignment(distances)
    matched = distances[row_indices, column_indices]
    if len(matched) < 3 or float(np.max(matched)) > float(max_point_distance_px):
        return None
    return float(np.sqrt(np.mean(matched**2))), float(np.max(matched))


def _assign_interval_parent_ids(
    candidates: list[dict[str, object]],
    track_rows: dict[int, dict[int, object]],
) -> None:
    max_gap = int(_runtime_float("TRAJ_ASSOC_PARENT_LINK_MAX_GAP", 12.0))
    min_shared = int(_runtime_float("TRAJ_ASSOC_PARENT_LINK_MIN_SHARED_TRACKS", 2.0))
    turnover_enabled = _runtime_bool("TRAJ_ASSOC_PARENT_TRACK_TURNOVER_ENABLED", True)
    turnover_max_distance_px = _runtime_float("TRAJ_ASSOC_PARENT_TRACK_TURNOVER_MAX_DISTANCE_PX", 400.0 / 4.5)
    parents: list[dict[str, object]] = []

    for cand in sorted(candidates, key=lambda c: (int(c["start_frame"]), int(c["end_frame"]), -int(c["n_tracks"]))):
        combo = set(int(tid) for tid in cand["combo"])
        start_frame = int(cand["start_frame"])
        best_index = None
        best_key = None
        best_method = "new_parent"
        for index, parent in enumerate(parents):
            last_end = int(parent["last_end"])
            gap = start_frame - last_end - 1
            if gap < 0 or gap > max_gap:
                continue
            shared = len(combo & set(parent["last_tracks"]))
            method = "shared_track_ids"
            if shared >= min_shared:
                key = (2.0, float(shared), float(-gap), 0.0, float(last_end))
            elif turnover_enabled:
                turnover_cost = _track_turnover_endpoint_cost(
                    parent["last_candidate"], cand, track_rows, turnover_max_distance_px
                )
                if turnover_cost is None:
                    continue
                rms_distance, max_distance = turnover_cost
                method = "track_id_turnover_spatial"
                key = (1.0, float(-max_distance), float(-rms_distance), float(-gap), float(last_end))
            else:
                continue
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
                best_method = method
        if best_index is None:
            parent_id = len(parents) + 1
            parents.append(
                {
                    "parent_id": parent_id,
                    "last_end": int(cand["end_frame"]),
                    "last_tracks": set(combo),
                    "last_candidate": cand,
                }
            )
        else:
            parent = parents[best_index]
            parent_id = int(parent["parent_id"])
            parent["last_end"] = int(cand["end_frame"])
            parent["last_tracks"] = set(combo)
            parent["last_candidate"] = cand
        cand["trajectory_parent_group_id"] = int(parent_id)
        cand["trajectory_parent_link_method"] = best_method


def _refine_interval_boundaries(
    candidates: list[dict[str, object]],
    track_rows: dict[int, dict[int, object]],
) -> list[dict[str, object]]:
    """Adjust segment boundaries so that contested frames go to the combo
    with better per-frame geometry score.

    When two adjacent segments share >= 2 tracks and their frame ranges
    are contiguous or near-contiguous, check each frame near the boundary
    to see if an alternative track combo would produce better geometry.
    If so, reassign the contested frame to the segment with the better combo.
    """
    max_boundary_gap = int(_runtime_float("TRAJ_ASSOC_BOUNDARY_REFINE_MAX_GAP", 2.0))
    min_shared = int(_runtime_float("TRAJ_ASSOC_BOUNDARY_REFINE_MIN_SHARED", 2.0))
    boundary_frames = int(_runtime_float("TRAJ_ASSOC_BOUNDARY_REFINE_N_FRAMES", 3.0))

    if len(candidates) < 2:
        return candidates

    # Index by trajectory_parent_group_id after initial assignment
    parent_groups: dict[int, list[dict[str, object]]] = {}
    for cand in candidates:
        pid = int(cand.get("trajectory_parent_group_id", cand.get("parent_id", 0)))
        parent_groups.setdefault(pid, []).append(cand)

    for pid, group_cands in parent_groups.items():
        if len(group_cands) < 2:
            continue
        group_cands.sort(key=lambda c: int(c["start_frame"]))
        for idx in range(len(group_cands) - 1):
            cand_a = group_cands[idx]
            cand_b = group_cands[idx + 1]
            end_a = int(cand_a["end_frame"])
            start_b = int(cand_b["start_frame"])
            gap = start_b - end_a - 1
            if gap > max_boundary_gap:
                continue

            combo_a = set(int(tid) for tid in cand_a["combo"])
            combo_b = set(int(tid) for tid in cand_b["combo"])
            if len(combo_a & combo_b) < min_shared:
                continue

            # Check frames at the boundary: last N of A and first N of B.
            # If a frame in B's range would have better geometry with A's combo
            # (or vice versa), reassign.
            boundary_range = range(max(0, start_b - boundary_frames),
                                   min(int(cand_b["end_frame"]) + 1, start_b + boundary_frames + 1))

            frames_to_reassign: dict[int, tuple[int, ...]] = {}
            for frame in sorted(boundary_range):
                # Try A's combo and B's combo on this frame
                metrics_a = _shape_metrics_for_track_combo(tuple(sorted(combo_a)), track_rows, frame)
                metrics_b = _shape_metrics_for_track_combo(tuple(sorted(combo_b)), track_rows, frame)
                if metrics_a is None or metrics_b is None:
                    continue
                score_a = float(metrics_a["score"])
                score_b = float(metrics_b["score"])
                diag_a = float(metrics_a.get("mean_diag", 0))
                diag_b = float(metrics_b.get("mean_diag", 0))

                min_diag, max_diag = _runtime_diag_bounds()
                a_valid = min_diag <= diag_a <= max_diag
                b_valid = min_diag <= diag_b <= max_diag

                if frame < start_b:
                    # Frame in A's range. If B's combo is much better, offer to B.
                    if b_valid and score_b < score_a - 0.03:
                        frames_to_reassign[frame] = tuple(sorted(combo_b))
                else:
                    # Frame in B's range. If A's combo is much better, offer to A.
                    if a_valid and score_a < score_b - 0.03:
                        frames_to_reassign[frame] = tuple(sorted(combo_a))

            if not frames_to_reassign:
                continue

            # Apply reassignments: shift boundary frames in both directions.
            # When frame < start_b but B's combo is better: slide B's start earlier.
            # When frame >= start_b but A's combo is better: slide A's end later.
            new_end_a = end_a
            new_start_b = start_b
            for frame in sorted(frames_to_reassign):
                target_combo = set(frames_to_reassign[frame])
                if frame < start_b and target_combo == combo_b:
                    # Frame was in A's range but B's combo is better.
                    # Give this frame to B: shrink A's end, expand B's start.
                    new_end_a = min(new_end_a, frame - 1)
                    new_start_b = min(new_start_b, frame)
                elif frame >= start_b and target_combo == combo_a:
                    # Frame was in B's range but A's combo is better.
                    # Give this frame to A: expand A's end, shrink B's start.
                    new_end_a = max(new_end_a, frame)
                    new_start_b = max(new_start_b, frame + 1)

            if new_end_a != end_a:
                cand_a["end_frame"] = int(new_end_a)
                # Recompute geometry metadata for shortened segment
                frames_a = list(range(int(cand_a["start_frame"]), int(new_end_a) + 1))
                _recompute_candidate_metadata(cand_a, combo_a, frames_a, track_rows)
            if new_start_b != start_b:
                cand_b["start_frame"] = int(new_start_b)
                frames_b = list(range(int(new_start_b), int(cand_b["end_frame"]) + 1))
                _recompute_candidate_metadata(cand_b, combo_b, frames_b, track_rows)

    result = []
    for group_cands in parent_groups.values():
        result.extend(group_cands)
    return result


def _recompute_candidate_metadata(
    cand: dict[str, object],
    combo: set[int],
    frames: list[int],
    track_rows: dict[int, dict[int, object]],
) -> None:
    """Update candidate geometry_score, diag_cv, etc. for a modified frame range."""
    if not frames:
        return
    geometry_scores = []
    diag_values = []
    for frame in frames:
        metrics = _shape_metrics_for_track_combo(tuple(sorted(combo)), track_rows, frame)
        if metrics is not None:
            geometry_scores.append(float(metrics["score"]))
            diag_values.append(float(metrics.get("mean_diag", np.nan)))
    if geometry_scores:
        cand["geometry_score"] = float(np.median(geometry_scores))
        cand["geometry_p75"] = float(np.percentile(geometry_scores, 75))
    if diag_values:
        cand["diag_cv"] = float(np.std(diag_values) / max(np.mean(diag_values), 1e-9))
        cand["mean_diag_px"] = float(np.median(diag_values))
        cand["mean_diag_sensor_um"] = float(np.median(diag_values)) * _runtime_camera_pixel_um()
    cand["valid_frames"] = frames
    cand["common_n"] = int(len(frames))


def _interval_local_groups_from_evidence(
    detections: pd.DataFrame,
    evidence: pd.DataFrame,
    track_frames: dict[int, set[int]],
    track_rows: dict[int, dict[int, object]],
) -> pd.DataFrame:
    seed_combos = _seed_combos_from_framewise_evidence(evidence, track_frames)
    if not seed_combos:
        return pd.DataFrame()

    frame_values = pd.to_numeric(detections["FRAME"], errors="coerce").dropna().astype(int)
    if frame_values.empty:
        return pd.DataFrame()
    start_frame = int(frame_values.min())
    end_frame = int(frame_values.max())

    candidates: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, ...], int, int]] = set()
    for combo in sorted(seed_combos):
        for candidate in _trajectory_segment_candidates(combo, track_frames, track_rows, start_frame, end_frame):
            key = (
                tuple(int(tid) for tid in candidate["combo"]),
                int(candidate["start_frame"]),
                int(candidate["end_frame"]),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    if not candidates:
        return pd.DataFrame()

    selected = _select_interval_candidates(candidates, track_rows)
    if not selected:
        return pd.DataFrame()
    _assign_interval_parent_ids(selected, track_rows)

    # Boundary refinement: where two adjacent segments of the *same* parent
    # share ≥2 tracks, evaluate per-frame geometry at the boundary to decide
    # which combo owns contested frames. This handles cases like F171 where
    # the framewise group correctly prefers {8,15,18,23} over {15,18,23,39}
    # but the segment-level combo assignment splits them on different sides.
    if _runtime_bool("TRAJ_ASSOC_REFINE_BOUNDARY_FRAMES", True):
        selected = _refine_interval_boundaries(selected, track_rows)

    rows: list[dict[str, object]] = []
    for gid, cand in enumerate(sorted(selected, key=lambda c: (int(c["trajectory_parent_group_id"]), int(c["start_frame"]), int(c["end_frame"]))), start=1):
        tids = [int(tid) for tid in cand["combo"]]
        start = int(cand["start_frame"])
        end = int(cand["end_frame"])
        pids, hard_sources, possible_texts, possible_sets = _track_source_summary(
            detections,
            tids,
            start_frame=start,
            end_frame=end,
        )
        stats = association_stats_from_source_sets(possible_sets, hard_sources)
        rows.append(
            {
                "score": float(cand["score"]),
                "geometry_score": float(cand["geometry_score"]),
                "geometry_p75": float(cand.get("geometry_p75", np.nan)),
                "diag_cv": float(cand.get("diag_cv", np.nan)),
                "mean_diag_px": float(cand.get("mean_diag_px", np.nan)),
                "mean_diag_sensor_um": float(cand.get("mean_diag_sensor_um", np.nan)),
                "motion_residual": float(cand.get("motion_residual", np.nan)),
                "fragment_candidate_count": 1,
                "common_window_frames": int(cand["common_n"]),
                "recoverable_track_frames": int(cand["common_n"]),
                "member_track_ids": ";".join(map(str, tids)),
                "member_particle_ids": ";".join(map(str, pids)),
                "member_source_ids": ";".join(hard_sources),
                "member_possible_source_ids": "|".join(possible_texts),
                "assigned_particle_id": int(stats["assigned_particle_id"]),
                "assigned_source_id": str(stats.get("assigned_source_id", "")),
                "group_purity": float(stats["group_purity"]),
                "is_correct": bool(stats["is_correct"]),
                "association_status": str(stats["association_status"]),
                "segment_start_frame": start,
                "segment_end_frame": end,
                "trajectory_parent_group_id": int(cand["trajectory_parent_group_id"]),
                "trajectory_parent_link_method": str(cand.get("trajectory_parent_link_method", "unknown")),
                "trajectory_group_id": gid,
                "method": "trajectory_lap_cost",
                "feature_score": float(cand["score"]),
                "n_tracks": int(len(tids)),
                "trajectory_geometry_type": "interval_framewise_geometry_temporal_segment",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["trajectory_parent_group_id", "segment_start_frame", "segment_end_frame"]).reset_index(drop=True)
    out["trajectory_group_id"] = np.arange(1, len(out) + 1, dtype=int)
    out["local_segment_group_index"] = 1
    for parent_id, index in out.groupby("trajectory_parent_group_id").groups.items():
        ordered = list(index)
        out.loc[ordered, "segment_index"] = np.arange(1, len(ordered) + 1, dtype=int)
        out.loc[ordered, "segment_count"] = int(len(ordered))
    return out


def trajectory_lap_cost_group(detections: pd.DataFrame) -> pd.DataFrame:
    """Fragment-aware trajectory grouping using common-window four-aperture geometry."""
    if detections.empty or "TRACK_ID" not in detections.columns:
        return pd.DataFrame()
    detections = detections.sort_values(["TRACK_ID", "FRAME"]).copy()

    track_frames: dict[int, set[int]] = {}
    track_rows: dict[int, dict[int, object]] = {}
    for tid, group in detections.groupby("TRACK_ID"):
        group = group.sort_values("FRAME").drop_duplicates("FRAME", keep="last")
        if len(group) < 3:
            continue
        track_frames[int(tid)] = set(group["FRAME"].astype(int).tolist())
        track_rows[int(tid)] = {int(row.FRAME): row for row in group.itertuples(index=False)}
    if len(track_frames) < 3:
        return pd.DataFrame()

    evidence = framewise_group(detections)
    if _runtime_bool("TRAJ_ASSOC_INTERVAL_LOCAL_GROUPING", True):
        interval_groups = _interval_local_groups_from_evidence(detections, evidence, track_frames, track_rows)
        if not interval_groups.empty:
            return interval_groups

    seed_combos: set[tuple[int, ...]] = set()
    if not evidence.empty:
        for _, group in evidence.iterrows():
            tids = tuple(sorted({int(x) for x in str(group["member_track_ids"]).split(";") if str(x).strip()}))
            if len(tids) in {3, 4} and all(tid in track_frames for tid in tids):
                seed_combos.add(tids)
    if not seed_combos:
        for combo in combinations(sorted(track_frames), 4):
            seed_combos.add(tuple(int(x) for x in combo))

    candidates = []
    for combo in sorted(seed_combos):
        candidate = _trajectory_common_window_candidate(combo, track_frames, track_rows)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return pd.DataFrame()

    parent = {i: i for i in range(len(candidates))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, cand_a in enumerate(candidates):
        tracks_a = set(cand_a["combo"])
        for j in range(i + 1, len(candidates)):
            tracks_b = set(candidates[j]["combo"])
            if len(tracks_a & tracks_b) >= 2:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        components.setdefault(find(i), []).append(i)

    component_rows = []
    for comp_indices in components.values():
        member_tracks = sorted(set().union(*(set(candidates[i]["combo"]) for i in comp_indices)))
        if len(member_tracks) < 3:
            continue
        recoverable_frames = _recoverable_frames_for_tracks(track_frames, member_tracks)
        if len(recoverable_frames) < 5:
            continue
        score = float(np.mean([float(candidates[i]["score"]) for i in comp_indices]))
        geometry_score = float(np.median([float(candidates[i]["geometry_score"]) for i in comp_indices]))
        pids, hard_sources, possible_texts, possible_sets = _track_source_summary(detections, member_tracks)
        stats = association_stats_from_source_sets(possible_sets, hard_sources)
        component_rows.append(
            {
                "score": score,
                "geometry_score": geometry_score,
                "mean_diag_px": float(np.median([float(candidates[i].get("mean_diag_px", np.nan)) for i in comp_indices])),
                "mean_diag_sensor_um": float(np.median([float(candidates[i].get("mean_diag_sensor_um", np.nan)) for i in comp_indices])),
                "fragment_candidate_count": int(len(comp_indices)),
                "common_window_frames": int(sum(int(candidates[i]["common_n"]) for i in comp_indices)),
                "recoverable_track_frames": int(len(recoverable_frames)),
                "member_track_ids": ";".join(map(str, member_tracks)),
                "member_particle_ids": ";".join(map(str, pids)),
                "member_source_ids": ";".join(hard_sources),
                "member_possible_source_ids": "|".join(possible_texts),
                "assigned_particle_id": int(stats["assigned_particle_id"]),
                "assigned_source_id": str(stats.get("assigned_source_id", "")),
                "group_purity": float(stats["group_purity"]),
                "is_correct": bool(stats["is_correct"]),
                "association_status": str(stats["association_status"]),
            }
        )

    component_rows.sort(key=lambda row: (-row["recoverable_track_frames"], row["score"]))
    selected = []
    used_tracks: set[int] = set()
    for row in component_rows:
        tids = [int(x) for x in str(row["member_track_ids"]).split(";") if x]
        overlap = used_tracks & set(tids)
        if overlap:
            tids = [tid for tid in tids if tid not in overlap]
            if len(tids) < 3:
                continue
            recoverable_frames = _recoverable_frames_for_tracks(track_frames, tids)
            if len(recoverable_frames) < 5:
                continue
            pids, hard_sources, possible_texts, possible_sets = _track_source_summary(detections, tids)
            stats = association_stats_from_source_sets(possible_sets, hard_sources)
            row = dict(row)
            row["member_track_ids"] = ";".join(map(str, tids))
            row["member_particle_ids"] = ";".join(map(str, pids))
            row["member_source_ids"] = ";".join(hard_sources)
            row["member_possible_source_ids"] = "|".join(possible_texts)
            row["recoverable_track_frames"] = int(len(recoverable_frames))
            row["assigned_particle_id"] = int(stats["assigned_particle_id"])
            row["assigned_source_id"] = str(stats.get("assigned_source_id", ""))
            row["group_purity"] = float(stats["group_purity"])
            row["is_correct"] = bool(stats["is_correct"])
            row["association_status"] = str(stats["association_status"])
        used_tracks.update(tids)
        selected.append(row)

    def temporal_segment_rows(row: dict[str, object], max_segment_frames: int = 40, split_threshold: int = 50) -> list[dict[str, object]]:
        tids = [int(x) for x in str(row["member_track_ids"]).split(";") if x]
        recoverable_frames = _recoverable_frames_for_tracks(track_frames, tids)
        if len(recoverable_frames) <= split_threshold:
            return [row]

        segments = []
        chunks = [
            recoverable_frames[i : i + max_segment_frames]
            for i in range(0, len(recoverable_frames), max_segment_frames)
        ]
        for segment_index, chunk in enumerate(chunks, start=1):
            if len(chunk) < 5:
                continue
            start_frame = int(chunk[0])
            end_frame = int(chunk[-1])
            segment_tids = [
                tid
                for tid in tids
                if sum(frame in track_frames.get(tid, set()) for frame in chunk) >= 3
            ]
            if len(segment_tids) < 3:
                continue
            local_candidates = _local_segment_track_groups(
                segment_tids,
                track_frames,
                track_rows,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            if not local_candidates:
                local_candidates = [
                    {
                        "combo": tuple(segment_tids),
                        "score": row["score"],
                        "geometry_score": row["geometry_score"],
                        "common_n": len(chunk),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                ]
            for local_index, local_candidate in enumerate(local_candidates, start=1):
                local_tids = [int(tid) for tid in local_candidate["combo"]]
                local_start = int(local_candidate.get("start_frame", start_frame))
                local_end = int(local_candidate.get("end_frame", end_frame))
                segment_recoverable = _recoverable_frames_for_tracks(
                    track_frames,
                    local_tids,
                    start_frame=local_start,
                    end_frame=local_end,
                )
                if len(segment_recoverable) < 5:
                    continue
                pids, hard_sources, possible_texts, possible_sets = _track_source_summary(
                    detections,
                    local_tids,
                    start_frame=local_start,
                    end_frame=local_end,
                )
                stats = association_stats_from_source_sets(possible_sets, hard_sources)
                segment_row = dict(row)
                segment_row["score"] = float(local_candidate["score"])
                segment_row["geometry_score"] = float(local_candidate["geometry_score"])
                segment_row["common_window_frames"] = int(local_candidate.get("common_n", len(segment_recoverable)))
                segment_row["member_track_ids"] = ";".join(map(str, local_tids))
                segment_row["member_particle_ids"] = ";".join(map(str, pids))
                segment_row["member_source_ids"] = ";".join(hard_sources)
                segment_row["member_possible_source_ids"] = "|".join(possible_texts)
                segment_row["recoverable_track_frames"] = int(len(segment_recoverable))
                segment_row["assigned_particle_id"] = int(stats["assigned_particle_id"])
                segment_row["assigned_source_id"] = str(stats.get("assigned_source_id", ""))
                segment_row["group_purity"] = float(stats["group_purity"])
                segment_row["is_correct"] = bool(stats["is_correct"])
                segment_row["association_status"] = str(stats["association_status"])
                segment_row["segment_start_frame"] = local_start
                segment_row["segment_end_frame"] = local_end
                segment_row["segment_index"] = int(segment_index)
                segment_row["segment_count"] = int(len(chunks))
                segment_row["local_segment_group_index"] = int(local_index)
                segments.append(segment_row)
        return segments if segments else [row]

    segmented = []
    for parent_gid, row in enumerate(selected, start=1):
        row = dict(row)
        row["trajectory_parent_group_id"] = int(parent_gid)
        segmented.extend(temporal_segment_rows(dict(row)))

    groups = []
    for gid, row in enumerate(segmented, start=1):
        row = dict(row)
        row.update(
            {
                "trajectory_group_id": gid,
                "method": "trajectory_lap_cost",
                "feature_score": float(row["score"]),
                "n_tracks": int(len([x for x in str(row["member_track_ids"]).split(";") if x])),
                "trajectory_geometry_type": "fragment_aware_common_window"
                if int(row.get("segment_count", 1)) <= 1
                else "fragment_aware_common_window_temporal_segment",
            }
        )
        groups.append(row)
    return pd.DataFrame(groups)


def stabilize_track_ids_from_parent_groups(
    detections: pd.DataFrame,
    groups: pd.DataFrame,
    max_normalized_slot_error: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join subspot IDs only when an already accepted parent keeps its geometry."""
    if detections.empty or groups.empty or "TRACK_ID" not in detections.columns:
        return detections.copy(), pd.DataFrame()
    required = {
        "trajectory_parent_group_id",
        "segment_start_frame",
        "segment_end_frame",
        "member_track_ids",
    }
    if not required.issubset(groups.columns):
        return detections.copy(), pd.DataFrame()

    out = detections.copy()
    out["PRE_STABILIZATION_TRACK_ID"] = out["TRACK_ID"].astype(int)
    all_ids = sorted(out["TRACK_ID"].astype(int).unique().tolist())
    parent = {tid: tid for tid in all_ids}

    def find(tid: int) -> int:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        keep, discard = min(ra, rb), max(ra, rb)
        parent[discard] = keep

    def endpoint(track_id: int, start: int, end: int, take_end: bool) -> np.ndarray | None:
        rows = out[
            (out["TRACK_ID"].astype(int) == int(track_id))
            & (out["FRAME"].astype(int) >= int(start))
            & (out["FRAME"].astype(int) <= int(end))
        ].sort_values(["FRAME", "detection_id"], kind="mergesort")
        if rows.empty:
            return None
        row = rows.iloc[-1] if take_end else rows.iloc[0]
        return row[["X", "Y"]].to_numpy(float)

    def geometry_center(points: dict[int, np.ndarray]) -> np.ndarray:
        array = np.asarray(list(points.values()), dtype=float)
        if len(array) == 3:
            try:
                center = np.asarray(triangle_missing_spot_metrics(array)["center"], dtype=float)
                if center.shape == (2,) and np.all(np.isfinite(center)):
                    return center
            except Exception:
                pass
        return np.mean(array, axis=0)

    audit_rows: list[dict[str, object]] = []
    for parent_id, parent_groups in groups.groupby("trajectory_parent_group_id", sort=True):
        ordered = parent_groups.sort_values(
            ["segment_start_frame", "segment_end_frame", "member_track_ids"],
            kind="mergesort",
        )
        previous: pd.Series | None = None
        for _, current in ordered.iterrows():
            if previous is None:
                previous = current
                continue
            prev_ids = list(_parse_member_track_ids(previous["member_track_ids"]))
            curr_ids = list(_parse_member_track_ids(current["member_track_ids"]))
            prev_start = int(previous["segment_start_frame"])
            prev_end = int(previous["segment_end_frame"])
            curr_start = int(current["segment_start_frame"])
            curr_end = int(current["segment_end_frame"])
            shared = set(prev_ids) & set(curr_ids)

            prev_points = {
                tid: endpoint(tid, prev_start, prev_end, True)
                for tid in prev_ids
            }
            curr_points = {
                tid: endpoint(tid, curr_start, curr_end, False)
                for tid in curr_ids
            }
            prev_points = {tid: pt for tid, pt in prev_points.items() if pt is not None}
            curr_points = {tid: pt for tid, pt in curr_points.items() if pt is not None}
            if not prev_points or not curr_points:
                previous = current
                continue

            prev_center = geometry_center(prev_points)
            curr_center = geometry_center(curr_points)
            centered_prev = {tid: pt - prev_center for tid, pt in prev_points.items()}
            centered_curr = {tid: pt - curr_center for tid, pt in curr_points.items()}
            radii = [
                float(np.linalg.norm(pt))
                for pt in list(centered_prev.values()) + list(centered_curr.values())
            ]
            geometry_scale = max(float(np.median(radii)), 1.0)

            left_ids = [tid for tid in prev_ids if tid not in shared and tid in centered_prev]
            right_ids = [tid for tid in curr_ids if tid not in shared and tid in centered_curr]
            if left_ids and right_ids:
                cost = np.asarray(
                    [
                        [
                            np.linalg.norm(centered_prev[left] - centered_curr[right])
                            / geometry_scale
                            for right in right_ids
                        ]
                        for left in left_ids
                    ],
                    dtype=float,
                )
                rr, cc = _stable_linear_sum_assignment(cost)
                for r, c in zip(rr, cc):
                    normalized_error = float(cost[r, c])
                    if normalized_error > float(max_normalized_slot_error):
                        continue
                    old_id = int(left_ids[r])
                    new_id = int(right_ids[c])
                    union(old_id, new_id)
                    audit_rows.append(
                        {
                            "trajectory_parent_group_id": int(parent_id),
                            "previous_segment_end_frame": prev_end,
                            "current_segment_start_frame": curr_start,
                            "previous_track_id": old_id,
                            "current_track_id": new_id,
                            "normalized_slot_error": normalized_error,
                            "method": "parent_geometry_slot_assignment",
                        }
                    )
            previous = current

    mapping = {tid: find(tid) for tid in all_ids}
    out["TRACK_ID"] = out["PRE_STABILIZATION_TRACK_ID"].map(mapping).astype(int)
    changed = out["TRACK_ID"] != out["PRE_STABILIZATION_TRACK_ID"]
    out["TRACK_ID_STABILIZATION_METHOD"] = np.where(
        changed,
        "parent_geometry_slot_assignment",
        "unchanged",
    )
    if "repair_model" in out.columns:
        out.loc[changed, "repair_model"] = (
            out.loc[changed, "repair_model"].astype(str)
            + "+stable_parent_track_id"
        )
    out = out.sort_values(["FRAME", "TRACK_ID", "detection_id"], kind="mergesort").reset_index(drop=True)
    return out, pd.DataFrame(audit_rows)


def recover_from_groups(groups: pd.DataFrame, detections: pd.DataFrame, z_cal: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    if method == "framewise":
        for _, group in groups.iterrows():
            ids = str(group["member_detection_ids"]).split(";")
            sub = detections[detections["detection_id"].astype(str).isin(ids)]
            if len(sub) < 3:
                continue
            diag_value = float(group["mean_diag_px"]) if "mean_diag_px" in group and np.isfinite(group["mean_diag_px"]) else np.nan
            rows.append(
                {
                    "FRAME": int(group["FRAME"]),
                    "assigned_particle_id": int(group["assigned_particle_id"]),
                    "assigned_source_id": str(group["assigned_source_id"]) if "assigned_source_id" in group else "",
                    "X_rec": float(group["X_rec"]) if "X_rec" in group else float(sub["X"].mean()),
                    "Y_rec": float(group["Y_rec"]) if "Y_rec" in group else float(sub["Y"].mean()),
                    "Z_rec": diag_to_z(diag_value, z_cal),
                    "is_correct": bool(group["is_correct"]),
                    "n_spots": int(group["n_spots"]) if "n_spots" in group else int(len(sub)),
                    "geometry_type": str(group["geometry_type"]) if "geometry_type" in group else "unknown",
                }
            )
    else:
        group_map = []
        for _, group in groups.iterrows():
            tids = [int(x) for x in str(group["member_track_ids"]).split(";")]
            source_id = str(group["assigned_source_id"]) if "assigned_source_id" in group else ""
            start_frame = group.get("segment_start_frame", np.nan)
            end_frame = group.get("segment_end_frame", np.nan)
            start_value = int(start_frame) if pd.notna(start_frame) else None
            end_value = int(end_frame) if pd.notna(end_frame) else None
            group_map.append((int(group["assigned_particle_id"]), source_id, tids, bool(group["is_correct"]), start_value, end_value))
        for pid, source_id, tids, is_correct, start_frame, end_frame in group_map:
            for frame, frame_df in detections[detections["TRACK_ID"].isin(tids)].groupby("FRAME"):
                frame_int = int(frame)
                if start_frame is not None and frame_int < start_frame:
                    continue
                if end_frame is not None and frame_int > end_frame:
                    continue
                if len(frame_df) < 3:
                    continue
                selected_frame = frame_df
                pts = frame_df[["X", "Y"]].to_numpy(float)
                geometry_type = "trajectory_tracks"
                if len(frame_df) > 4:
                    best_score = np.inf
                    best_combo = None
                    min_diag, max_diag = _runtime_diag_bounds()
                    for combo in combinations(range(len(frame_df)), 4):
                        combo_pts = pts[list(combo)]
                        metrics = four_spot_shape_metrics(combo_pts)
                        score = float(metrics["score"])
                        if min_diag <= float(metrics["mean_diag"]) <= max_diag and score < best_score:
                            best_score = score
                            best_combo = combo
                    if best_combo is not None:
                        selected_frame = frame_df.iloc[list(best_combo)]
                        pts = selected_frame[["X", "Y"]].to_numpy(float)
                        geometry_type = "trajectory_tracks_best_four"
                center_xy = pts.mean(axis=0)
                diag_value = np.nan
                if len(pts) == 4:
                    diag_value = mean_diag(pts)
                elif len(pts) == 3:
                    tri = triangle_missing_spot_metrics(pts)
                    if np.all(np.isfinite(tri["center"])):
                        center_xy = np.asarray(tri["center"], dtype=float)
                    diag_value = np.nan
                rows.append(
                    {
                        "FRAME": frame_int,
                        "assigned_particle_id": pid,
                        "assigned_source_id": source_id,
                        "X_rec": float(center_xy[0]),
                        "Y_rec": float(center_xy[1]),
                        "Z_rec": diag_to_z(diag_value, z_cal),
                        "is_correct": is_correct,
                        "n_spots": int(len(selected_frame)),
                        "geometry_type": geometry_type,
                    }
                )
    return pd.DataFrame(rows).sort_values(["assigned_particle_id", "FRAME"]).reset_index(drop=True)


def diag_to_z(diag: float, z_cal: pd.DataFrame) -> float:
    if not np.isfinite(diag):
        return np.nan
    return float(np.interp(diag, z_cal["mean_diag_px"], z_cal["z_um"], left=np.nan, right=np.nan))


def attach_errors(recovered: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = recovered.merge(
        truth,
        left_on=["FRAME", "assigned_particle_id"],
        right_on=["FRAME", "particle_id"],
        how="left",
    )
    merged["xy_error"] = np.hypot(merged["X_rec"] - merged["X_true"], merged["Y_rec"] - merged["Y_true"])
    merged["z_error"] = merged["Z_rec"] - merged["Z_true"]
    merged["xyz_error"] = np.sqrt(merged["xy_error"] ** 2 + merged["z_error"] ** 2)
    return merged


def count_recoverable_source_frames(truth_detections: pd.DataFrame, min_spots: int = 3) -> int:
    if truth_detections.empty or "source_id" not in truth_detections.columns:
        return 0
    valid = truth_detections.copy()
    if "particle_id" in valid.columns:
        valid = valid[pd.to_numeric(valid["particle_id"], errors="coerce").fillna(0).astype(int) > 0]
    valid = valid[valid["source_id"].astype(str).str.startswith("P")]
    counts = valid.groupby(["FRAME", "source_id"]).size()
    return int((counts >= min_spots).sum())


def recoverable_source_frames(truth_detections: pd.DataFrame, min_spots: int = 3) -> pd.DataFrame:
    if truth_detections.empty or "source_id" not in truth_detections.columns:
        return pd.DataFrame(columns=["FRAME", "source_id", "visible_spots"])
    valid = truth_detections.copy()
    if "particle_id" in valid.columns:
        valid = valid[pd.to_numeric(valid["particle_id"], errors="coerce").fillna(0).astype(int) > 0]
    valid = valid[valid["source_id"].astype(str).str.startswith("P")]
    counts = valid.groupby(["FRAME", "source_id"]).size().reset_index(name="visible_spots")
    return counts[counts["visible_spots"] >= min_spots].sort_values(["source_id", "FRAME"]).reset_index(drop=True)


def recovered_source_frames(recovered: pd.DataFrame) -> pd.DataFrame:
    if recovered.empty or "assigned_source_id" not in recovered.columns:
        return pd.DataFrame(columns=["FRAME", "source_id"])
    valid = recovered[recovered.get("is_correct", False).astype(bool)].copy()
    valid = valid[valid["assigned_source_id"].astype(str) != ""]
    return (
        valid[["FRAME", "assigned_source_id"]]
        .drop_duplicates()
        .rename(columns={"assigned_source_id": "source_id"})
        .sort_values(["source_id", "FRAME"])
        .reset_index(drop=True)
    )


def missing_source_frames(expected: pd.DataFrame, recovered: pd.DataFrame, method: str) -> pd.DataFrame:
    if expected.empty:
        return pd.DataFrame(columns=["method", "FRAME", "source_id", "visible_spots"])
    got = recovered_source_frames(recovered)
    missing = expected.merge(got, on=["FRAME", "source_id"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    missing.insert(0, "method", method)
    return missing.sort_values(["source_id", "FRAME"]).reset_index(drop=True)


def summarize(method: str, groups: pd.DataFrame, recovered: pd.DataFrame, expected_source_frame_df: pd.DataFrame) -> dict:
    expected_source_frames = int(len(expected_source_frame_df))
    if groups.empty or recovered.empty:
        return {
            "method": method,
            "association_accuracy": 0.0 if groups.empty else float(groups["is_correct"].mean()),
            "mean_group_purity": 0.0 if groups.empty else float(groups["group_purity"].mean()),
            "source_recall": 0.0,
            "source_recall_recovered": 0,
            "source_recall_expected": int(expected_source_frames),
            "missing_or_failed_frames": expected_source_frames,
        }
    missing = missing_source_frames(expected_source_frame_df, recovered, method)
    valid_slots = int(expected_source_frames - len(missing))
    source_recall = float(valid_slots / expected_source_frames) if expected_source_frames > 0 else 0.0
    return {
        "method": method,
        "association_accuracy": float(groups["is_correct"].mean()),
        "mean_group_purity": float(groups["group_purity"].mean()),
        "source_recall": source_recall,
        "source_recall_recovered": int(valid_slots),
        "source_recall_expected": int(expected_source_frames),
        "missing_or_failed_frames": int(expected_source_frames - valid_slots),
    }


def summarize_track_quality(track_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in track_tables.items():
        if df.empty:
            rows.append(
                {
                    "track_table": name,
                    "n_tracks": 0,
                    "n_detections": 0,
                    "mean_track_purity": 0.0,
                    "track_purity_ge_0p9_fraction": 0.0,
                    "mixed_tracks_ge_3": 0,
                    "mean_fragments_per_truth_track": 0.0,
                    "median_fragments_per_truth_track": 0.0,
                    "interpolated_points": 0,
                }
            )
            continue
        table = df.copy()
        if "truth_track_id" not in table.columns:
            table["truth_track_id"] = 0
        if "is_interpolated" not in table.columns:
            table["is_interpolated"] = False
        real = table[
            (pd.to_numeric(table["truth_track_id"], errors="coerce").fillna(0).astype(int) != 0)
            & (~table["is_interpolated"].astype(bool))
        ]
        purities = []
        for _, group in real.groupby("TRACK_ID"):
            if len(group) < 3:
                continue
            counts = group["truth_track_id"].astype(int).value_counts()
            purities.append(float(counts.iloc[0] / len(group)))
        fragment_counts = [int(group["TRACK_ID"].nunique()) for _, group in real.groupby("truth_track_id")]
        purity_arr = np.asarray(purities, dtype=float)
        rows.append(
            {
                "track_table": name,
                "n_tracks": int(table["TRACK_ID"].nunique()) if "TRACK_ID" in table.columns else 0,
                "n_detections": int(len(table)),
                "mean_track_purity": float(np.mean(purity_arr)) if len(purity_arr) else 0.0,
                "track_purity_ge_0p9_fraction": float(np.mean(purity_arr >= 0.9)) if len(purity_arr) else 0.0,
                "mixed_tracks_ge_3": int(np.sum(purity_arr < 0.9)) if len(purity_arr) else 0,
                "mean_fragments_per_truth_track": float(np.mean(fragment_counts)) if fragment_counts else 0.0,
                "median_fragments_per_truth_track": float(np.median(fragment_counts)) if fragment_counts else 0.0,
                "interpolated_points": int(table["is_interpolated"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_grouping(framewise: pd.DataFrame, traj_rows: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    frame_acc = framewise.groupby("FRAME")["is_correct"].mean()
    axes[0].plot(frame_acc.index, frame_acc.values, color="#d62728", lw=1.7, label="Frame-wise")
    if len(frame_acc.index):
        max_frame = int(max(frame_acc.index))
        axes[0].axvspan(
            int(round(0.38 * max_frame)),
            int(round(0.63 * max_frame)),
            color="#f4a261",
            alpha=0.25,
            label="crossing/clutter",
        )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Per-frame group correctness")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Correct group fraction")
    axes[0].legend(fontsize=8)

    counts = [
        float(framewise["group_purity"].mean()) if not framewise.empty else 0,
        float(traj_rows["group_purity"].mean()) if not traj_rows.empty else 0,
    ]
    axes[1].bar(["Frame-wise", "Trajectory-level"], counts, color=["#d62728", "#2ca02c"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Mean group purity")

    acc = [
        float(framewise["is_correct"].mean()) if not framewise.empty else 0,
        float(traj_rows["is_correct"].mean()) if not traj_rows.empty else 0,
    ]
    axes[2].bar(["Frame-wise", "Trajectory-level"], acc, color=["#d62728", "#2ca02c"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Association accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_3d(truth: pd.DataFrame, rec_frame: pd.DataFrame, rec_traj: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for pid, sub in truth.groupby("particle_id"):
        ax.plot(sub["X_true"], sub["Y_true"], sub["Z_true"], lw=2, label=f"truth P{pid}")
        ax2.plot(sub["X_true"], sub["Y_true"], sub["Z_true"], lw=2, label=f"truth P{pid}")
    for pid, sub in rec_frame.groupby("assigned_particle_id"):
        ax.scatter(sub["X_rec"], sub["Y_rec"], sub["Z_rec"], s=4, alpha=0.45)
    for pid, sub in rec_traj.groupby("assigned_particle_id"):
        ax2.scatter(sub["X_rec"], sub["Y_rec"], sub["Z_rec"], s=4, alpha=0.45)
    ax.set_title("Frame-wise geometry")
    ax2.set_title("Trajectory-level association")
    for a in [ax, ax2]:
        a.set_xlabel("X px")
        a.set_ylabel("Y px")
        a.set_zlabel("Z um")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_failure_examples(
    detections: pd.DataFrame,
    framewise: pd.DataFrame,
    traj_groups: pd.DataFrame,
    out: Path,
    stack: np.ndarray | None = None,
) -> None:
    all_frames = sorted(int(x) for x in detections["FRAME"].drop_duplicates())
    if all_frames:
        max_frame = max(all_frames)
        crossing_range = (
            int(round(0.38 * max_frame)),
            int(round(0.63 * max_frame)),
        )
    else:
        crossing_range = (0, 0)
    if framewise.empty:
        frames = list(detections["FRAME"].drop_duplicates().head(3))
    else:
        bad = framewise[~framewise["is_correct"]]
        bad_crossing = bad[(bad["FRAME"] >= crossing_range[0]) & (bad["FRAME"] <= crossing_range[1])]
        frames = list(bad_crossing["FRAME"].drop_duplicates().head(3))
        if len(frames) < 3:
            for frame in bad["FRAME"].drop_duplicates():
                if int(frame) not in frames:
                    frames.append(int(frame))
                if len(frames) == 3:
                    break
        if len(frames) < 3:
            correctness = framewise.groupby("FRAME")["is_correct"].mean().sort_values()
            for frame in correctness.index:
                if int(frame) not in frames:
                    frames.append(int(frame))
                if len(frames) == 3:
                    break
    if not frames:
        frames = list(detections["FRAME"].drop_duplicates().head(3))
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]
    unique_pids = sorted(int(x) for x in detections["particle_id"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_pids))))
    colors = {0: "#999999"}
    for i, pid in enumerate([p for p in unique_pids if p != 0]):
        colors[pid] = palette[i % len(palette)]
    for ax, frame in zip(axes, frames):
        sub = detections[detections["FRAME"] == frame]
        if stack is not None and 0 <= int(frame) < len(stack):
            raw = stack[int(frame)].astype(float)
            ax.imshow(raw, cmap="gray", origin="upper", vmin=np.percentile(raw, 5), vmax=np.percentile(raw, 99.7))
        ax.scatter(sub["X"], sub["Y"], c=[colors.get(int(pid), "#555555") for pid in sub["particle_id"]], s=26, alpha=0.85)
        for _, g in framewise[framewise["FRAME"] == frame].iterrows():
            ids = str(g["member_detection_ids"]).split(";")
            pts = sub[sub["detection_id"].isin(ids)][["X", "Y"]].to_numpy(float)
            if len(pts) == 4:
                order = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()))
                poly = np.vstack([pts[order], pts[order][0]])
                ax.plot(poly[:, 0], poly[:, 1], color="#d62728", lw=1.2, alpha=0.8)
        ax.set_title(f"Frame {frame}: red = frame-wise groups")
        ax.set_aspect("equal")
        if stack is None:
            ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def z_calibration_from_templates(templates: list[PsfTemplate]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"file_index": t.file_index, "z_um": t.z_um, "mean_diag_px": t.mean_diag_px} for t in templates]
    ).sort_values("mean_diag_px")


def analyze_stack_and_truth(
    templates: list[PsfTemplate],
    truth: pd.DataFrame,
    detections: pd.DataFrame,
    stack: np.ndarray,
    seed: int,
    dropout_prob: float,
    center_separation_px: float,
    clutter_points_per_frame: int,
    trajectory_params: pd.DataFrame | None = None,
) -> dict[str, object]:
    z_cal = z_calibration_from_templates(templates)
    n_frames = int(stack.shape[0]) if hasattr(stack, "shape") and len(stack.shape) >= 1 else int(truth["FRAME"].max()) + 1
    image_detections = detect_spots_from_stack(stack)
    labeled_detections = label_detected_spots(image_detections, detections)
    raw_tracks = link_detections_to_tracks(labeled_detections)
    tracked_detections = repair_and_interpolate_tracks(raw_tracks, n_frames=n_frames)
    # LapTrack-style mode: use a custom multi-feature cost for initial sub-spot
    # tracklet linking, then reuse the same feature-aware repair stage.
    lap_raw_tracks = link_detections_to_tracks_lap_cost(labeled_detections)
    lap_raw_tracks["repair_model"] = "lap_custom_initial_linking"
    lap_tracked_detections = repair_and_interpolate_tracks_lap_cost(lap_raw_tracks, n_frames=n_frames)

    geometry_previous_groups = geometry_previous_frame_group(labeled_detections)
    groups: dict[str, pd.DataFrame] = {
        "nearest_neighbor_framewise": nearest_neighbor_framewise_group(labeled_detections),
        "framewise_geometry": framewise_group(labeled_detections),
        "geometry_previous_frame": geometry_previous_groups,
        "geometry_previous_frame_linked": link_framewise_groups_to_trajectories(geometry_previous_groups),
        "trajectory_level": trajectory_level_group(tracked_detections),
        "trajectory_lap_cost": trajectory_lap_cost_group(lap_tracked_detections),
        "oracle_grouping": oracle_group(detections),
    }
    recovered: dict[str, pd.DataFrame] = {}
    summaries = []
    expected_source_frame_df = recoverable_source_frames(detections)
    for name, group_df in groups.items():
        mode = "trajectory" if name.startswith("trajectory") else "framewise"
        if name == "trajectory_level":
            source_detections = tracked_detections
        elif name == "trajectory_lap_cost":
            source_detections = lap_tracked_detections
        elif name == "oracle_grouping":
            source_detections = detections
        else:
            source_detections = labeled_detections
        rec = recover_from_groups(group_df, source_detections, z_cal, mode)
        rec = attach_errors(rec, truth) if not rec.empty else rec
        recovered[name] = rec
        summaries.append(summarize(name, group_df, rec, expected_source_frame_df))
    missing_sources = {
        name: missing_source_frames(expected_source_frame_df, rec, name) for name, rec in recovered.items()
    }
    summary = pd.DataFrame(summaries)
    summary.insert(0, "seed", seed)
    summary.insert(1, "dropout_prob", dropout_prob)
    summary.insert(2, "center_separation_px", center_separation_px)
    summary.insert(3, "clutter_points_per_frame", clutter_points_per_frame)
    track_quality = summarize_track_quality(
        {
            "distance_raw": raw_tracks,
            "distance_repaired": tracked_detections,
            "lap_cost_raw": lap_raw_tracks,
            "lap_cost_repaired": lap_tracked_detections,
        }
    )
    if trajectory_params is None:
        trajectory_params = pd.DataFrame()
    return {
        "z_cal": z_cal,
        "truth": truth,
        "detections": detections,
        "image_detections": labeled_detections,
        "raw_tracks": raw_tracks,
        "tracked_detections": tracked_detections,
        "lap_raw_tracks": lap_raw_tracks,
        "lap_tracked_detections": lap_tracked_detections,
        "stack": stack,
        "trajectory_params": trajectory_params,
        "groups": groups,
        "recovered": recovered,
        "expected_source_frames": expected_source_frame_df,
        "missing_source_frames": missing_sources,
        "summary": summary,
        "track_quality": track_quality,
    }


def run_experiment(
    templates: list[PsfTemplate],
    seed: int,
    n_frames: int,
    dropout_prob: float,
    center_separation_px: float,
    clutter_points_per_frame: int,
    n_particles: int,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> dict[str, object]:
    truth, detections, stack, trajectory_params = make_truth_and_detections(
        templates,
        seed,
        n_frames,
        dropout_prob=dropout_prob,
        center_separation_px=center_separation_px,
        clutter_points_per_frame=clutter_points_per_frame,
        n_particles=n_particles,
        z_range_um=z_range_um,
    )
    return analyze_stack_and_truth(
        templates,
        truth,
        detections,
        stack,
        seed=seed,
        dropout_prob=dropout_prob,
        center_separation_px=center_separation_px,
        clutter_points_per_frame=clutter_points_per_frame,
        trajectory_params=trajectory_params,
    )


def summarize_sweep(per_run: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sweep_name", "sweep_value", "method"]
    metric_cols = [
        "association_accuracy",
        "mean_group_purity",
        "source_recall",
        "source_recall_recovered",
        "source_recall_expected",
        "missing_or_failed_frames",
    ]
    rows = []
    for keys, sub in per_run.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(sub))
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sd"] = float(vals.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def plot_sweep(summary: pd.DataFrame, out: Path) -> None:
    methods = [
        "oracle_grouping",
        "nearest_neighbor_framewise",
        "framewise_geometry",
        "geometry_previous_frame",
        "geometry_previous_frame_linked",
        "trajectory_level",
        "trajectory_lap_cost",
    ]
    colors = {
        "nearest_neighbor_framewise": "#9467bd",
        "framewise_geometry": "#d62728",
        "geometry_previous_frame": "#ff7f0e",
        "geometry_previous_frame_linked": "#a65628",
        "trajectory_level": "#2ca02c",
        "trajectory_lap_cost": "#0072b2",
        "oracle_grouping": "#4d4d4d",
    }
    labels = {
        "nearest_neighbor_framewise": "Nearest-neighbor frame-wise",
        "framewise_geometry": "Frame-wise geometry",
        "geometry_previous_frame": "Geometry + previous frame",
        "geometry_previous_frame_linked": "Geometry + linked trajectories",
        "trajectory_level": "Trajectory-level",
        "trajectory_lap_cost": "Trajectory + custom link cost",
        "oracle_grouping": "Oracle upper bound",
    }
    sweep_labels = {
        "dropout_prob": "Dropout probability",
        "center_separation_px": "Minimum center separation (px)",
        "clutter_points_per_frame": "Clutter detections per crossing frame",
    }
    sweep_order = ["dropout_prob", "center_separation_px", "clutter_points_per_frame"]
    available_sweeps = [name for name in sweep_order if name in set(summary["sweep_name"].astype(str))]
    if not available_sweeps:
        return
    fig, axes = plt.subplots(len(available_sweeps), 2, figsize=(12, 4 * len(available_sweeps)))
    axes = np.atleast_2d(axes)
    for row_idx, sweep_name in enumerate(available_sweeps):
        xlabel = sweep_labels.get(sweep_name, sweep_name)
        sub_all = summary[summary["sweep_name"] == sweep_name]
        for method in methods:
            sub = sub_all[sub_all["method"] == method].sort_values("sweep_value")
            if sub.empty:
                continue
            x = sub["sweep_value"].astype(float).to_numpy()
            y = sub["association_accuracy_mean"].astype(float).to_numpy()
            e = sub["association_accuracy_sd"].astype(float).fillna(0).to_numpy()
            axes[row_idx, 0].errorbar(
                x,
                y,
                yerr=e,
                marker="o",
                lw=1.5,
                capsize=3,
                color=colors[method],
                label=labels[method],
                linestyle="--" if method == "oracle_grouping" else "-",
                alpha=0.65 if method == "oracle_grouping" else 1.0,
            )
            y2 = sub["source_recall_mean"].astype(float).to_numpy()
            e2 = sub["source_recall_sd"].astype(float).fillna(0).to_numpy()
            axes[row_idx, 1].errorbar(
                x,
                y2,
                yerr=e2,
                marker="o",
                lw=1.5,
                capsize=3,
                color=colors[method],
                label=labels[method],
                linestyle="--" if method == "oracle_grouping" else "-",
                alpha=0.65 if method == "oracle_grouping" else 1.0,
            )
        axes[row_idx, 0].set_xlabel(xlabel)
        axes[row_idx, 0].set_ylabel("Association accuracy")
        axes[row_idx, 0].set_ylim(-0.03, 1.05)
        axes[row_idx, 1].set_xlabel(xlabel)
        axes[row_idx, 1].set_ylabel("Source recall")
        axes[row_idx, 1].set_ylim(-0.03, 1.05)
    axes[0, 0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_sweep_conditions(include_fast_clutter: bool) -> list[tuple[str, float, float, float, int]]:
    default_dropout = 0.045
    default_sep = 18.0
    default_clutter = 4 if include_fast_clutter else 0
    conditions = []
    for value in [0.0, 0.05, 0.10, 0.20]:
        conditions.append(("dropout_prob", value, value, default_sep, default_clutter))
    for value in [8.0, 18.0, 32.0, 55.0]:
        conditions.append(("center_separation_px", value, default_dropout, value, default_clutter))
    if include_fast_clutter:
        for value in [0, 4, 8, 12]:
            conditions.append(("clutter_points_per_frame", value, default_dropout, default_sep, value))
    return conditions


def run_sweep(
    templates: list[PsfTemplate],
    n_seeds: int,
    n_frames: int,
    n_particles: int,
    include_fast_clutter: bool = True,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = build_sweep_conditions(include_fast_clutter)
    rows = []
    for sweep_name, sweep_value, dropout, sep, clutter in conditions:
        for seed in range(n_seeds):
            result = run_experiment(
                templates,
                seed=seed,
                n_frames=n_frames,
                dropout_prob=float(dropout),
                center_separation_px=float(sep),
                clutter_points_per_frame=int(clutter),
                n_particles=n_particles,
                z_range_um=z_range_um,
            )
            summary = result["summary"].copy()
            summary.insert(0, "sweep_name", sweep_name)
            summary.insert(1, "sweep_value", sweep_value)
            rows.append(summary)
            print(f"sweep={sweep_name}:{sweep_value} seed={seed + 1}/{n_seeds}", flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    return per_run, summarize_sweep(per_run)


def safe_condition_name(sweep_name: str, sweep_value: float) -> str:
    value_text = str(sweep_value).replace(".", "p").replace("-", "m")
    return f"{sweep_name}_{value_text}"


def save_sweep_run_details(result: dict[str, object], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(str(run_dir / "simulated_stack.tif"), result["stack"], check_contrast=False)
    result["truth"].to_csv(run_dir / "ground_truth_3d.csv", index=False)
    result["detections"].to_csv(run_dir / "subspot_tracks_truth.csv", index=False)
    result["image_detections"].to_csv(run_dir / "image_spot_detections.csv", index=False)
    result["raw_tracks"].to_csv(run_dir / "raw_image_spot_tracks.csv", index=False)
    result["tracked_detections"].to_csv(run_dir / "repaired_interpolated_image_spot_tracks.csv", index=False)
    result["lap_raw_tracks"].to_csv(run_dir / "lap_cost_raw_image_spot_tracks.csv", index=False)
    result["lap_tracked_detections"].to_csv(run_dir / "lap_cost_repaired_interpolated_image_spot_tracks.csv", index=False)
    result["z_cal"].to_csv(run_dir / "psf_z_calibration.csv", index=False)
    result["summary"].to_csv(run_dir / "metrics_summary.csv", index=False)
    result["track_quality"].to_csv(run_dir / "track_linking_quality_summary.csv", index=False)
    result["expected_source_frames"].to_csv(run_dir / "expected_source_frames.csv", index=False)
    result["trajectory_params"].to_csv(run_dir / "trajectory_params.csv", index=False)

    groups_dir = run_dir / "groups"
    recovered_dir = run_dir / "recovered_3d"
    missing_dir = run_dir / "missing_source_frames"
    groups_dir.mkdir(exist_ok=True)
    recovered_dir.mkdir(exist_ok=True)
    missing_dir.mkdir(exist_ok=True)
    for method, group_df in result["groups"].items():
        group_df.to_csv(groups_dir / f"{method}.csv", index=False)
    for method, rec_df in result["recovered"].items():
        rec_df.to_csv(recovered_dir / f"{method}.csv", index=False)
    for method, missing_df in result["missing_source_frames"].items():
        missing_df.to_csv(missing_dir / f"{method}.csv", index=False)


def run_sweep_with_saved_runs(
    templates: list[PsfTemplate],
    n_seeds: int,
    n_frames: int,
    n_particles: int,
    include_fast_clutter: bool = True,
    runs_root: Path | None = None,
    force_rerun: bool = False,
    z_range_um: tuple[float, float] = DEFAULT_SIM_Z_RANGE_UM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = build_sweep_conditions(include_fast_clutter)
    if runs_root is None:
        runs_root = OUT_DIR / ("sweep_runs" if include_fast_clutter else "sweep_runs_no_fast_clutter")
    rows = []
    for sweep_name, sweep_value, dropout, sep, clutter in conditions:
        condition_dir = runs_root / safe_condition_name(sweep_name, sweep_value)
        for seed in range(n_seeds):
            run_dir = condition_dir / f"seed_{seed:03d}"
            required_outputs = [
                run_dir / "simulated_stack.tif",
                run_dir / "metrics_summary.csv",
                run_dir / "image_spot_detections.csv",
                run_dir / "repaired_interpolated_image_spot_tracks.csv",
                run_dir / "lap_cost_repaired_interpolated_image_spot_tracks.csv",
            ]
            if not force_rerun and all(path.exists() for path in required_outputs):
                summary = read_pipeline_csv(run_dir / "metrics_summary.csv")
                summary.insert(0, "sweep_name", sweep_name)
                summary.insert(1, "sweep_value", sweep_value)
                rows.append(summary)
                print(f"reused sweep={sweep_name}:{sweep_value} seed={seed + 1}/{n_seeds}", flush=True)
                continue
            result = run_experiment(
                templates,
                seed=seed,
                n_frames=n_frames,
                dropout_prob=float(dropout),
                center_separation_px=float(sep),
                clutter_points_per_frame=int(clutter),
                n_particles=n_particles,
                z_range_um=z_range_um,
            )
            save_sweep_run_details(result, run_dir)
            summary = result["summary"].copy()
            summary.insert(0, "sweep_name", sweep_name)
            summary.insert(1, "sweep_value", sweep_value)
            rows.append(summary)
            print(f"saved sweep={sweep_name}:{sweep_value} seed={seed + 1}/{n_seeds}", flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    return per_run, summarize_sweep(per_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dropout-prob", type=float, default=0.045)
    parser.add_argument("--center-separation-px", type=float, default=18.0)
    parser.add_argument("--clutter-points-per-frame", type=int, default=4)
    parser.add_argument("--z-min-um", type=float, default=DEFAULT_SIM_Z_RANGE_UM[0])
    parser.add_argument("--z-max-um", type=float, default=DEFAULT_SIM_Z_RANGE_UM[1])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--save-sweep-runs", action="store_true")
    parser.add_argument("--no-fast-clutter", action="store_true")
    parser.add_argument("--force-sweep-rerun", action="store_true")
    parser.add_argument("--particles", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = load_psf_templates()

    if args.sweep:
        include_fast_clutter = not args.no_fast_clutter
        base_suffix = "" if include_fast_clutter else "_no_fast_clutter"
        tag = args.sweep_output_tag.strip().replace(" ", "_")
        suffix = f"{base_suffix}_{tag}" if tag else base_suffix
        runs_root = OUT_DIR / f"sweep_runs{suffix}"
        if args.save_sweep_runs:
            per_run, sweep_summary = run_sweep_with_saved_runs(
                templates,
                n_seeds=args.n_seeds,
                n_frames=args.frames,
                n_particles=args.particles,
                include_fast_clutter=include_fast_clutter,
                runs_root=runs_root,
                force_rerun=args.force_sweep_rerun,
                z_range_um=(args.z_min_um, args.z_max_um),
            )
        else:
            per_run, sweep_summary = run_sweep(
                templates,
                n_seeds=args.n_seeds,
                n_frames=args.frames,
                n_particles=args.particles,
                include_fast_clutter=include_fast_clutter,
                z_range_um=(args.z_min_um, args.z_max_um),
            )
        per_run.to_csv(OUT_DIR / f"sweep_per_run_metrics{suffix}.csv", index=False)
        sweep_summary.to_csv(OUT_DIR / f"sweep_summary_mean_sd{suffix}.csv", index=False)
        plot_sweep(sweep_summary, OUT_DIR / f"supp_fig_s6_algorithm_validation{suffix}.png")
        print(sweep_summary.to_string(index=False))
        print(f"Saved sweep outputs to {OUT_DIR}")
        return

    result = run_experiment(
        templates,
        seed=args.seed,
        n_frames=args.frames,
        dropout_prob=args.dropout_prob,
        center_separation_px=args.center_separation_px,
        clutter_points_per_frame=args.clutter_points_per_frame,
        n_particles=args.particles,
        z_range_um=(args.z_min_um, args.z_max_um),
    )
    z_cal = result["z_cal"]
    truth = result["truth"]
    detections = result["detections"]
    image_detections = result["image_detections"]
    raw_tracks = result["raw_tracks"]
    tracked_detections = result["tracked_detections"]
    lap_raw_tracks = result["lap_raw_tracks"]
    lap_tracked_detections = result["lap_tracked_detections"]
    stack = result["stack"]
    trajectory_params = result["trajectory_params"]
    groups = result["groups"]
    recovered = result["recovered"]
    missing_sources = result["missing_source_frames"]
    summary = result["summary"].drop(columns=["seed", "dropout_prob", "center_separation_px", "clutter_points_per_frame"])
    track_quality = result["track_quality"]

    io.imsave(str(OUT_DIR / "simulated_stack.tif"), stack, check_contrast=False)
    z_cal.to_csv(OUT_DIR / "psf_z_calibration.csv", index=False)
    truth.to_csv(OUT_DIR / "ground_truth_3d.csv", index=False)
    detections.to_csv(OUT_DIR / "subspot_tracks_truth.csv", index=False)
    image_detections.to_csv(OUT_DIR / "image_spot_detections.csv", index=False)
    raw_tracks.to_csv(OUT_DIR / "raw_image_spot_tracks.csv", index=False)
    tracked_detections.to_csv(OUT_DIR / "repaired_interpolated_image_spot_tracks.csv", index=False)
    lap_raw_tracks.to_csv(OUT_DIR / "lap_cost_raw_image_spot_tracks.csv", index=False)
    lap_tracked_detections.to_csv(OUT_DIR / "lap_cost_repaired_interpolated_image_spot_tracks.csv", index=False)
    trajectory_params.to_csv(OUT_DIR / "trajectory_params.csv", index=False)
    result["expected_source_frames"].to_csv(OUT_DIR / "expected_source_frames.csv", index=False)
    groups["framewise_geometry"].to_csv(OUT_DIR / "framewise_groups.csv", index=False)
    groups["nearest_neighbor_framewise"].to_csv(OUT_DIR / "nearest_neighbor_framewise_groups.csv", index=False)
    groups["geometry_previous_frame"].to_csv(OUT_DIR / "geometry_previous_frame_groups.csv", index=False)
    groups["geometry_previous_frame_linked"].to_csv(OUT_DIR / "geometry_previous_frame_linked_groups.csv", index=False)
    groups["trajectory_level"].to_csv(OUT_DIR / "trajectory_level_groups.csv", index=False)
    groups["trajectory_lap_cost"].to_csv(OUT_DIR / "trajectory_lap_cost_groups.csv", index=False)
    groups["oracle_grouping"].to_csv(OUT_DIR / "oracle_groups.csv", index=False)
    recovered["framewise_geometry"].to_csv(OUT_DIR / "recovered_3d_framewise.csv", index=False)
    recovered["nearest_neighbor_framewise"].to_csv(OUT_DIR / "recovered_3d_nearest_neighbor_framewise.csv", index=False)
    recovered["geometry_previous_frame"].to_csv(OUT_DIR / "recovered_3d_geometry_previous_frame.csv", index=False)
    recovered["geometry_previous_frame_linked"].to_csv(OUT_DIR / "recovered_3d_geometry_previous_frame_linked.csv", index=False)
    recovered["trajectory_level"].to_csv(OUT_DIR / "recovered_3d_trajectory_level.csv", index=False)
    recovered["trajectory_lap_cost"].to_csv(OUT_DIR / "recovered_3d_trajectory_lap_cost.csv", index=False)
    recovered["oracle_grouping"].to_csv(OUT_DIR / "recovered_3d_oracle.csv", index=False)
    missing_dir = OUT_DIR / "missing_source_frames"
    missing_dir.mkdir(exist_ok=True)
    for method, missing_df in missing_sources.items():
        missing_df.to_csv(missing_dir / f"{method}.csv", index=False)
    summary.to_csv(OUT_DIR / "association_accuracy_summary.csv", index=False)
    track_quality.to_csv(OUT_DIR / "track_linking_quality_summary.csv", index=False)

    plot_grouping(groups["framewise_geometry"], groups["trajectory_level"], OUT_DIR / "grouping_comparison.png")
    plot_grouping(groups["framewise_geometry"], groups["trajectory_lap_cost"], OUT_DIR / "grouping_comparison_lap_cost.png")
    plot_3d(truth, recovered["framewise_geometry"], recovered["trajectory_level"], OUT_DIR / "trajectory_3d_comparison.png")
    plot_3d(truth, recovered["framewise_geometry"], recovered["trajectory_lap_cost"], OUT_DIR / "trajectory_3d_comparison_lap_cost.png")
    plot_failure_examples(
        image_detections,
        groups["framewise_geometry"],
        groups["trajectory_lap_cost"],
        OUT_DIR / "example_failure_frames.png",
        stack=stack,
    )

    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
