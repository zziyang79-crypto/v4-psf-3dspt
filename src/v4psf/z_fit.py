from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from .csv_io import read_csv as read_pipeline_csv

try:
    import tifffile
except Exception:  # pragma: no cover - optional runtime dependency
    tifffile = None


REPO_DIR = Path(__file__).resolve().parent

target_file = "spots_refined_interpolate.csv"
output_file = "Axial positions.csv"

col_cluster = "Cluster_ID"
col_frame = "FRAME"
col_x = "X"
col_y = "Y"

DEFAULT_MODEL_FILE = REPO_DIR / "calibration_models" / "phasor_length_backup_models.json"
DEFAULT_ANGLE_MODEL_FILE = REPO_DIR / "calibration_models" / "phasor_angle_tracked_models.json"

# Conservative thresholds from centroid localization on the selected real PSF
# calibration + independent-check ranges. The selected PSF ranges are treated
# as normal data, so the defaults sit above the observed PSF maxima with margin.
DIAG_QC_DEFAULT_PERCENT_THRESHOLDS = {
    "20x_0.45NA_air": 10.0,
    "40x_0.6NA_air": 10.0,
    "60x_1.0NA_water": 10.0,
}

# Median mean(Diag1, Diag2) / DiagX factors from centroid localization on
# selected real PSF calibration + independent-check ranges. Runtime data can
# override these when enough non-triggered frames are available.
DIAG_QC_DEFAULT_MEAN_FACTORS = {
    "20x_0.45NA_air": {"Diag1": 0.994135324, "Diag2": 1.005934705},
    "40x_0.6NA_air": {"Diag1": 0.969159871, "Diag2": 1.032867398},
    "60x_1.0NA_water": {"Diag1": 0.981266896, "Diag2": 1.019462282},
}


def load_default_objective_config() -> str:
    env_value = os.getenv("Z_OBJECTIVE_CONFIG", "").strip()
    if env_value:
        return env_value
    settings_path = REPO_DIR / "gui_settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            return str(data.get("step_parameters", {}).get("_global", {}).get("objective_config", "60x_NA1.0_n1.333"))
        except Exception:
            pass
    return "60x_NA1.0_n1.333"


def load_model_payload() -> dict:
    model_file = Path(os.getenv("Z_LENGTH_BACKUP_MODEL_FILE", os.getenv("Z_MODEL_FILE", str(DEFAULT_MODEL_FILE))))
    if not model_file.is_absolute():
        model_file = REPO_DIR / model_file
    if not model_file.exists():
        raise FileNotFoundError(f"Z model file not found: {model_file}")
    return json.loads(model_file.read_text(encoding="utf-8"))


def resolve_objective(payload: dict, objective_config: str) -> tuple[str, dict]:
    aliases = payload.get("objective_aliases", {})
    objective_key = aliases.get(objective_config, objective_config)
    models = payload.get("models", {})
    if objective_key not in models:
        supported = ", ".join(sorted(models))
        raise ValueError(
            f"No optimized spline Z model for objective '{objective_config}'. "
            f"Supported objectives: {supported}"
        )
    return objective_key, models[objective_key]


def load_angle_payload() -> dict | None:
    model_file = Path(os.getenv("Z_ANGLE_MODEL_FILE", str(DEFAULT_ANGLE_MODEL_FILE)))
    if not model_file.is_absolute():
        model_file = REPO_DIR / model_file
    if not model_file.exists():
        print(f"WARNING: angle primary model file not found: {model_file}")
        return None
    try:
        return json.loads(model_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to load angle primary model {model_file}: {exc}")
        return None


def resolve_angle_model(payload: dict | None, objective_config: str) -> tuple[str | None, dict | None]:
    if not payload:
        return None, None
    aliases = payload.get("objective_aliases", {})
    objective_key = aliases.get(objective_config, objective_config)
    models = payload.get("models", {})
    if objective_key not in models:
        print(f"WARNING: no angle primary model for objective {objective_config!r}")
        return None, None
    return objective_key, models[objective_key]


def angle_mod180(p: np.ndarray, q: np.ndarray) -> float:
    vector = q - p
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)


def unwrap_period180(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).all():
        return np.rad2deg(np.unwrap(np.deg2rad(values * 2.0))) / 2.0
    out = np.full(len(values), np.nan, dtype=float)
    reference = np.nan
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        out[idx] = unwrap_near_period180(value, reference)
        reference = out[idx]
    return out


def mean_diagonal(coords: np.ndarray) -> tuple[float, float, float, float, float]:
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_coords = coords[np.argsort(angles)]
    p0, p1, p2, p3 = sorted_coords[0], sorted_coords[1], sorted_coords[2], sorted_coords[3]
    diag1 = float(np.linalg.norm(p0 - p2))
    diag2 = float(np.linalg.norm(p1 - p3))
    diag1_angle = angle_mod180(p0, p2)
    diag2_angle = angle_mod180(p1, p3)
    return (diag1 + diag2) / 2.0, diag1, diag2, diag1_angle, diag2_angle


def diagonal_geometry(coords: np.ndarray, track_ids: np.ndarray | None = None) -> dict:
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_coords = coords[order]
    p0, p1, p2, p3 = sorted_coords[0], sorted_coords[1], sorted_coords[2], sorted_coords[3]
    diag1 = float(np.linalg.norm(p0 - p2))
    diag2 = float(np.linalg.norm(p1 - p3))
    out = {
        "Calculated_Size": (diag1 + diag2) / 2.0,
        "Diag1": diag1,
        "Diag2": diag2,
        "Diag1_Angle_Deg": angle_mod180(p0, p2),
        "Diag2_Angle_Deg": angle_mod180(p1, p3),
        "sorted_coords": sorted_coords,
    }
    if track_ids is not None:
        out["sorted_track_ids"] = np.asarray(track_ids)[order]
    return out


def normalize_track_id(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(float(value))
    text = str(value)
    return text if text else None


def sorted_pair_key(a, b) -> tuple[str, str] | None:
    aa = normalize_track_id(a)
    bb = normalize_track_id(b)
    if aa is None or bb is None:
        return None
    return tuple(sorted((aa, bb)))


def resolve_z_tracking_group_column(target_df: pd.DataFrame) -> str:
    """Use the parent trajectory for Z branch tracking when auto-detect split it."""
    for candidate in ("Trajectory_Parent_Group_ID", "Center_Interpolation_Group"):
        if candidate in target_df.columns and target_df[candidate].notna().any():
            return candidate
    return col_cluster


def add_parent_qc_group_columns(res_df: pd.DataFrame) -> pd.DataFrame:
    """Choose a one-row-per-frame parent timeline for temporal Z QC.

    Z estimation remains segment-level so every source group can be audited. The
    temporal QC, however, must see frames on both sides of a source switch. A
    parent is usable only when it has at most one selected segment per frame;
    ambiguous parent/frame duplicates fall back to their individual segments
    rather than inventing an ordering between two simultaneous alternatives.
    """
    out = res_df.copy()
    if "Z_Tracking_Group_ID" not in out.columns or not out["Z_Tracking_Group_ID"].notna().any():
        out["Z_QC_Group_ID"] = "cluster:" + out[col_cluster].astype(str)
        out["Z_QC_Group_Basis"] = "cluster_no_parent_tracking"
        out["Z_QC_Parent_Segment_Boundary"] = False
        return out

    tracking = out["Z_Tracking_Group_ID"]
    valid_parent = tracking.notna()
    parent_frame = pd.DataFrame(
        {
            "parent": tracking.loc[valid_parent].astype(str),
            "frame": out.loc[valid_parent, col_frame].to_numpy(),
        },
        index=out.index[valid_parent],
    )
    duplicated_parent_ids = set(
        parent_frame.loc[parent_frame.duplicated(["parent", "frame"], keep=False), "parent"].tolist()
    )
    parent_key = tracking.astype(str)
    use_parent = valid_parent & (~parent_key.isin(duplicated_parent_ids))

    out["Z_QC_Group_ID"] = "cluster:" + out[col_cluster].astype(str)
    out["Z_QC_Group_Basis"] = "cluster_duplicate_parent_frame"
    out.loc[use_parent, "Z_QC_Group_ID"] = "parent:" + parent_key.loc[use_parent]
    out.loc[use_parent, "Z_QC_Group_Basis"] = "parent"
    out["Z_QC_Parent_Segment_Boundary"] = False

    for _, group in out.loc[use_parent].groupby("Z_QC_Group_ID", sort=False):
        ordered = group.sort_values([col_frame, col_cluster], kind="stable")
        cluster_ids = ordered[col_cluster].to_numpy()
        boundary = np.zeros(len(ordered), dtype=bool)
        if len(ordered) > 1:
            boundary[1:] |= cluster_ids[1:] != cluster_ids[:-1]
            boundary[:-1] |= cluster_ids[:-1] != cluster_ids[1:]
        out.loc[ordered.index, "Z_QC_Parent_Segment_Boundary"] = boundary

    return out


def tracking_group_value(group: pd.DataFrame, cluster_id, tracking_group_col: str):
    if tracking_group_col in group.columns:
        values = group[tracking_group_col].dropna()
        if not values.empty:
            return values.iloc[0]
    return cluster_id


def resolve_z_coordinate_columns(target_df: pd.DataFrame) -> tuple[str, str]:
    if "Z_Localization_X" in target_df.columns and "Z_Localization_Y" in target_df.columns:
        return "Z_Localization_X", "Z_Localization_Y"
    return col_x, col_y


def apply_spline(z_uncorrected: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correction = model["real_correction"]
    knots_x = np.asarray(correction["knots_x_z_uncorrected_um"], dtype=float)
    knots_y = np.asarray(correction["knots_y_z_corrected_relative_um"], dtype=float)
    order = np.argsort(knots_x)
    knots_x = knots_x[order]
    knots_y = knots_y[order]

    out_of_domain_mode = os.getenv("Z_SPLINE_OUT_OF_DOMAIN", "extrapolate").strip().lower()
    domain_ok = (z_uncorrected >= knots_x[0]) & (z_uncorrected <= knots_x[-1])

    if out_of_domain_mode == "extrapolate":
        eval_x = z_uncorrected
    elif out_of_domain_mode == "nan":
        eval_x = z_uncorrected.copy()
    else:
        out_of_domain_mode = "clip"
        eval_x = np.clip(z_uncorrected, knots_x[0], knots_x[-1])

    spline = PchipInterpolator(knots_x, knots_y, extrapolate=(out_of_domain_mode == "extrapolate"))
    z_corrected = np.asarray(spline(eval_x), dtype=float)
    if out_of_domain_mode == "nan":
        z_corrected[~domain_ok] = np.nan
    return z_corrected, domain_ok, eval_x


def boundary_polynomial_extrapolate(
    x_values: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    degree: int,
    n_points: int,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    result = np.full(len(x_values), np.nan, dtype=float)
    finite = np.isfinite(x_values)
    if len(knots_x) == 0:
        return result

    domain_ok = finite & (x_values >= knots_x[0]) & (x_values <= knots_x[-1])
    if np.any(domain_ok):
        spline = PchipInterpolator(knots_x, knots_y, extrapolate=False)
        result[domain_ok] = np.asarray(spline(x_values[domain_ok]), dtype=float)

    if len(knots_x) < 2:
        result[finite & ~domain_ok] = knots_y[0]
        return result

    degree = int(max(1, min(degree, 2)))
    n_points = int(max(degree + 1, min(len(knots_x), n_points)))

    def eval_side(mask: np.ndarray, side: str) -> None:
        if not np.any(mask):
            return
        if side == "left":
            fit_x = knots_x[:n_points]
            fit_y = knots_y[:n_points]
            origin = knots_x[0]
        else:
            fit_x = knots_x[-n_points:]
            fit_y = knots_y[-n_points:]
            origin = knots_x[-1]
        unique_count = len(np.unique(fit_x))
        fit_degree = int(max(1, min(degree, unique_count - 1)))
        coeff = np.polyfit(fit_x - origin, fit_y, fit_degree)
        result[mask] = np.polyval(coeff, x_values[mask] - origin)

    eval_side(finite & (x_values < knots_x[0]), "left")
    eval_side(finite & (x_values > knots_x[-1]), "right")
    return result


def unwrap_near_period180(value: float, reference: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if not np.isfinite(reference):
        return float(value)
    return float(value + 180.0 * np.round((reference - value) / 180.0))


def add_tracked_angle_columns(res_df: pd.DataFrame) -> pd.DataFrame:
    tracking_col = "Z_Tracking_Group_ID" if "Z_Tracking_Group_ID" in res_df.columns else "Cluster_ID"
    sort_cols = [tracking_col, "FRAME"]
    if "Cluster_ID" not in sort_cols:
        sort_cols.append("Cluster_ID")
    df = res_df.sort_values(sort_cols).copy()
    tracked_groups: list[pd.DataFrame] = []
    for _, group in df.groupby(tracking_col, sort=False, dropna=False):
        group = group.sort_values("FRAME").copy()
        raw_a = group["Diag1_Angle_Deg"].to_numpy(dtype=float)
        raw_b = group["Diag2_Angle_Deg"].to_numpy(dtype=float)
        group["Theta_Raw_Diag1_Deg"] = raw_a
        group["Theta_Raw_Diag2_Deg"] = raw_b
        group["Theta_Raw_Diag1_Unwrapped_Deg"] = unwrap_period180(raw_a)
        group["Theta_Raw_Diag2_Unwrapped_Deg"] = unwrap_period180(raw_b)

        tracked0 = np.full(len(group), np.nan, dtype=float)
        tracked1 = np.full(len(group), np.nan, dtype=float)
        tracked_simple_average = np.full(len(group), np.nan, dtype=float)
        average_source = np.array(["unavailable"] * len(group), dtype=object)

        def cost_to_reference(values: list[float], references: list[float], avg_value: float, prev_avg: float) -> float:
            costs = [
                abs(value - reference)
                for value, reference in zip(values, references)
                if np.isfinite(value) and np.isfinite(reference)
            ]
            if np.isfinite(avg_value) and np.isfinite(prev_avg):
                costs.append(abs(avg_value - prev_avg))
            return float(np.sum(costs)) if costs else 0.0

        def single_diag_average(branch_value: float, branch_index: int, prev_avg: float) -> float:
            if not np.isfinite(branch_value):
                return np.nan
            base = branch_value + 45.0 if branch_index == 0 else branch_value - 45.0
            return unwrap_near_period180(base, prev_avg)

        prev0 = np.nan
        prev1 = np.nan
        prev_avg = np.nan
        for idx in range(len(group)):
            a_finite = np.isfinite(raw_a[idx])
            b_finite = np.isfinite(raw_b[idx])
            if a_finite and b_finite:
                if np.isfinite(prev0) or np.isfinite(prev1):
                    same0 = unwrap_near_period180(raw_a[idx], prev0)
                    same1 = unwrap_near_period180(raw_b[idx], prev1)
                    same_avg = unwrap_near_period180((same0 + same1) / 2.0, prev_avg)
                    swap0 = unwrap_near_period180(raw_b[idx], prev0)
                    swap1 = unwrap_near_period180(raw_a[idx], prev1)
                    swap_avg = unwrap_near_period180((swap0 + swap1) / 2.0, prev_avg)
                    same_cost = cost_to_reference([same0, same1], [prev0, prev1], same_avg, prev_avg)
                    swap_cost = cost_to_reference([swap0, swap1], [prev0, prev1], swap_avg, prev_avg)
                    if np.isfinite(swap_cost) and (not np.isfinite(same_cost) or swap_cost < same_cost):
                        tracked0[idx], tracked1[idx], tracked_simple_average[idx] = swap0, swap1, swap_avg
                    else:
                        tracked0[idx], tracked1[idx], tracked_simple_average[idx] = same0, same1, same_avg
                else:
                    tracked0[idx] = raw_a[idx]
                    tracked1[idx] = raw_b[idx]
                    tracked_simple_average[idx] = (tracked0[idx] + tracked1[idx]) / 2.0
                average_source[idx] = "two_diagonal"
            elif a_finite or b_finite:
                raw_value = raw_a[idx] if a_finite else raw_b[idx]
                candidates = []
                for branch_index, previous_branch in [(0, prev0), (1, prev1)]:
                    unwrapped = unwrap_near_period180(raw_value, previous_branch)
                    avg_value = single_diag_average(unwrapped, branch_index, prev_avg)
                    default_penalty = 0.0 if ((a_finite and branch_index == 0) or (b_finite and branch_index == 1)) else 1.0
                    score = cost_to_reference([unwrapped], [previous_branch], avg_value, prev_avg) + default_penalty
                    candidates.append((score, branch_index, unwrapped, avg_value))
                _, branch_index, unwrapped, avg_value = min(candidates, key=lambda item: item[0])
                if branch_index == 0:
                    tracked0[idx] = unwrapped
                    average_source[idx] = "single_diag0"
                else:
                    tracked1[idx] = unwrapped
                    average_source[idx] = "single_diag1"
                tracked_simple_average[idx] = avg_value

            if np.isfinite(tracked0[idx]):
                prev0 = tracked0[idx]
            if np.isfinite(tracked1[idx]):
                prev1 = tracked1[idx]
            if np.isfinite(tracked_simple_average[idx]):
                prev_avg = tracked_simple_average[idx]

        group["Theta_Tracked_Diag0_Unwrapped_Deg"] = tracked0
        group["Theta_Tracked_Diag1_Unwrapped_Deg"] = tracked1
        group["Theta_Tracked_Diag0_Deg"] = np.mod(tracked0, 180.0)
        group["Theta_Tracked_Diag1_Deg"] = np.mod(tracked1, 180.0)
        group["Theta_Tracked_Simple_Average_Unwrapped_Deg"] = tracked_simple_average
        group["Theta_Tracked_Simple_Average_Deg"] = np.mod(tracked_simple_average, 180.0)
        group["Theta_Tracked_Simple_Average_Source"] = average_source
        diag_angle_diff = np.abs(
            ((group["Theta_Raw_Diag2_Deg"].to_numpy(dtype=float) - group["Theta_Raw_Diag1_Deg"].to_numpy(dtype=float) + 90.0) % 180.0)
            - 90.0
        )
        group["Diag_Orthogonality_Error_Deg"] = np.abs(diag_angle_diff - 90.0)
        tracked_groups.append(group)
    return pd.concat(tracked_groups, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)


def angle_distance_to_domain(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    dist = np.full(len(arr), np.nan, dtype=float)
    finite = np.isfinite(arr)
    dist[finite] = np.maximum.reduce([lo - arr[finite], arr[finite] - hi, np.zeros(np.sum(finite), dtype=float)])
    return dist


def select_tracked_branch(group: pd.DataFrame, preferred: str, theta_lo: float, theta_hi: float) -> tuple[str, str]:
    if preferred == "tracked_diag1":
        candidates = [
            ("Theta_Tracked_Diag1_Unwrapped_Deg", "tracked_diag1"),
            ("Theta_Tracked_Diag0_Unwrapped_Deg", "tracked_diag0"),
        ]
    else:
        candidates = [
            ("Theta_Tracked_Diag0_Unwrapped_Deg", "tracked_diag0"),
            ("Theta_Tracked_Diag1_Unwrapped_Deg", "tracked_diag1"),
        ]

    best_col, best_signal = candidates[0]
    best_score = (-1.0, -np.inf)
    for col, signal in candidates:
        values = group[col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            score = (-1.0, -np.inf)
        else:
            inside = finite & (values >= theta_lo) & (values <= theta_hi)
            inside_fraction = float(np.sum(inside) / np.sum(finite))
            domain_dist = angle_distance_to_domain(values[finite], theta_lo, theta_hi)
            median_dist = float(np.nanmedian(domain_dist)) if len(domain_dist) else np.inf
            score = (inside_fraction, -median_dist)
        if score > best_score:
            best_score = score
            best_col, best_signal = col, signal
    return best_col, best_signal


def angle_signal_column(group: pd.DataFrame, angle_signal: str, theta_lo: float, theta_hi: float) -> tuple[str, str]:
    signal = str(angle_signal).strip().lower()
    if signal in {"diag_a", "diag1", "a"}:
        return "Theta_Raw_Diag1_Unwrapped_Deg", "diag_a"
    if signal in {"diag_b", "diag2", "b"}:
        return "Theta_Raw_Diag2_Unwrapped_Deg", "diag_b"
    if signal in {"tracked_diag0", "tracked_diag1"}:
        return select_tracked_branch(group, signal, theta_lo, theta_hi)
    if signal in {"tracked_simple_average", "tracked_average", "tracked_mean", "joint_average"}:
        return "Theta_Tracked_Simple_Average_Unwrapped_Deg", "tracked_simple_average"
    return "Theta_Raw_Diag2_Unwrapped_Deg", "diag_b"


def apply_angle_spline(theta_values: np.ndarray, angle_model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spline_model = angle_model["angle_spline"]
    knots_x = np.asarray(spline_model["knots_x_theta_deg"], dtype=float)
    knots_y = np.asarray(spline_model["knots_y_z_corrected_relative_um"], dtype=float)
    order = np.argsort(knots_x)
    knots_x = knots_x[order]
    knots_y = knots_y[order]

    domain_ok = (theta_values >= knots_x[0]) & (theta_values <= knots_x[-1])
    out_of_domain_mode = os.getenv("Z_ANGLE_SPLINE_OUT_OF_DOMAIN", "linear").strip().lower()
    if out_of_domain_mode == "nan":
        eval_x = theta_values.copy()
    elif out_of_domain_mode == "clip":
        eval_x = np.clip(theta_values, knots_x[0], knots_x[-1])
    elif out_of_domain_mode in {"linear", "boundary_linear", "quadratic", "boundary_quadratic"}:
        eval_x = theta_values
    else:
        out_of_domain_mode = "extrapolate"
        eval_x = theta_values

    if out_of_domain_mode in {"linear", "boundary_linear", "quadratic", "boundary_quadratic"}:
        degree = 2 if out_of_domain_mode in {"quadratic", "boundary_quadratic"} else 1
        n_points = int(os.getenv("Z_ANGLE_EXTRAPOLATION_POINTS", "5"))
        z_angle = boundary_polynomial_extrapolate(eval_x, knots_x, knots_y, degree, n_points)
    else:
        spline = PchipInterpolator(knots_x, knots_y, extrapolate=(out_of_domain_mode == "extrapolate"))
        z_angle = np.asarray(spline(eval_x), dtype=float)
    if out_of_domain_mode == "nan":
        z_angle[~domain_ok] = np.nan
    return z_angle, domain_ok, eval_x


def selected_theta_from_geometry(geom: dict, selected_signal: str, current_theta: float) -> float:
    signal = str(selected_signal).strip().lower()
    if signal in {"diag_a", "diag1", "a"}:
        return unwrap_near_period180(float(geom["Diag1_Angle_Deg"]), current_theta)
    if signal in {"diag_b", "diag2", "b"}:
        return unwrap_near_period180(float(geom["Diag2_Angle_Deg"]), current_theta)
    if signal in {"tracked_simple_average", "tracked_average", "tracked_mean", "joint_average"}:
        angle_a = float(geom["Diag1_Angle_Deg"])
        angle_b = float(geom["Diag2_Angle_Deg"])
        candidates = []
        for ka in [-1, 0, 1]:
            for kb in [-1, 0, 1]:
                candidates.append((angle_a + 180.0 * ka + angle_b + 180.0 * kb) / 2.0)
        candidates = np.asarray(candidates, dtype=float)
        return float(candidates[np.nanargmin(np.abs(candidates - current_theta))])
    cand_a = unwrap_near_period180(float(geom["Diag1_Angle_Deg"]), current_theta)
    cand_b = unwrap_near_period180(float(geom["Diag2_Angle_Deg"]), current_theta)
    return cand_a if abs(cand_a - current_theta) <= abs(cand_b - current_theta) else cand_b


def row_track_coord_map(row: pd.Series) -> dict:
    coord_map = {}
    for idx in range(4):
        tid_col = f"Spot{idx}_Track_ID"
        x_col = f"Spot{idx}_X_Px"
        y_col = f"Spot{idx}_Y_Px"
        if tid_col not in row.index or x_col not in row.index or y_col not in row.index:
            continue
        tid = row[tid_col]
        x = float(row[x_col])
        y = float(row[y_col])
        if np.isfinite(x + y):
            coord_map[tid] = np.array([x, y], dtype=float)
    return coord_map


def apply_geometry_to_group_row(group: pd.DataFrame, row_index, geom: dict) -> None:
    group.at[row_index, "Calculated_Size"] = float(geom["Calculated_Size"])
    group.at[row_index, "Diag1"] = float(geom["Diag1"])
    group.at[row_index, "Diag2"] = float(geom["Diag2"])
    group.at[row_index, "Diag1_Angle_Deg"] = float(geom["Diag1_Angle_Deg"])
    group.at[row_index, "Diag2_Angle_Deg"] = float(geom["Diag2_Angle_Deg"])
    group.at[row_index, "Theta_Raw_Diag1_Deg"] = float(geom["Diag1_Angle_Deg"])
    group.at[row_index, "Theta_Raw_Diag2_Deg"] = float(geom["Diag2_Angle_Deg"])
    sorted_coords = geom["sorted_coords"]
    sorted_track_ids = geom.get("sorted_track_ids", np.arange(4, dtype=float))
    for spot_idx in range(4):
        group.at[row_index, f"Spot{spot_idx}_X_Px"] = float(sorted_coords[spot_idx, 0])
        group.at[row_index, f"Spot{spot_idx}_Y_Px"] = float(sorted_coords[spot_idx, 1])
        group.at[row_index, f"Spot{spot_idx}_Track_ID"] = sorted_track_ids[spot_idx]


def try_single_point_soft_domain_correction(
    group: pd.DataFrame,
    local_idx: int,
    selected_signal: str,
    angle_model: dict,
    theta_lo: float,
    theta_hi: float,
    original_theta: float,
    original_domain_distance: float,
    min_point_error_px: float,
    min_domain_improvement_deg: float,
    max_frame_gap: float,
) -> tuple[bool, str]:
    if local_idx <= 0 or local_idx >= len(group) - 1:
        return False, "no_bracketing_frames"

    prev_row = group.iloc[local_idx - 1]
    curr_row = group.iloc[local_idx]
    next_row = group.iloc[local_idx + 1]
    prev_frame = float(prev_row["FRAME"])
    curr_frame = float(curr_row["FRAME"])
    next_frame = float(next_row["FRAME"])
    if not (prev_frame < curr_frame < next_frame):
        return False, "non_monotonic_frame_order"
    if (curr_frame - prev_frame) > max_frame_gap or (next_frame - curr_frame) > max_frame_gap:
        return False, "frame_gap_too_large"

    prev_map = row_track_coord_map(prev_row)
    curr_map = row_track_coord_map(curr_row)
    next_map = row_track_coord_map(next_row)
    common_ids = [tid for tid in curr_map if tid in prev_map and tid in next_map]
    if len(common_ids) < 4:
        return False, "track_ids_not_available"

    ratio = (curr_frame - prev_frame) / (next_frame - prev_frame)
    errors = []
    predicted = {}
    for tid in common_ids:
        pred = prev_map[tid] + ratio * (next_map[tid] - prev_map[tid])
        predicted[tid] = pred
        errors.append((float(np.linalg.norm(curr_map[tid] - pred)), tid))
    errors.sort(reverse=True, key=lambda item: item[0])
    max_error, suspect_tid = errors[0]
    if max_error < min_point_error_px:
        return False, f"max_point_error={max_error:.3f}px_below_threshold"

    corrected_map = dict(curr_map)
    corrected_map[suspect_tid] = predicted[suspect_tid]
    corrected_track_ids = np.array(list(corrected_map.keys()), dtype=object)
    corrected_coords = np.vstack([corrected_map[tid] for tid in corrected_track_ids]).astype(float)
    geom = diagonal_geometry(corrected_coords, track_ids=corrected_track_ids)
    corrected_theta = selected_theta_from_geometry(geom, selected_signal, original_theta)
    corrected_distance = float(angle_distance_to_domain(np.array([corrected_theta]), theta_lo, theta_hi)[0])
    if not np.isfinite(corrected_distance):
        return False, "corrected_theta_not_finite"
    if corrected_distance > max(0.0, original_domain_distance - min_domain_improvement_deg):
        return False, f"domain_not_improved_enough:{corrected_distance:.3f}>{original_domain_distance:.3f}"

    z_corrected, corrected_domain_ok, eval_theta = apply_angle_spline(np.array([corrected_theta], dtype=float), angle_model)
    row_index = group.index[local_idx]
    apply_geometry_to_group_row(group, row_index, geom)
    group.at[row_index, "Z_Angle_Primary"] = float(z_corrected[0])
    group.at[row_index, "Z_Angle_Spline_Input"] = float(eval_theta[0])
    group.at[row_index, "Z_Angle_Model_Domain_OK"] = True
    group.at[row_index, "Z_Angle_Domain_Distance_Deg"] = float(corrected_distance)
    group.at[row_index, "Z_Angle_Out_Of_Domain_Level"] = "soft_corrected"
    if str(selected_signal).strip().lower() in {"tracked_simple_average", "tracked_average", "tracked_mean", "joint_average"}:
        group.at[row_index, "Theta_Tracked_Simple_Average_Unwrapped_Deg"] = float(corrected_theta)
        group.at[row_index, "Theta_Tracked_Simple_Average_Deg"] = float(corrected_theta % 180.0)
    group.at[row_index, "Z_Angle_Soft_Domain_Corrected"] = True
    group.at[row_index, "Z_Angle_Soft_Domain_Method"] = "single_point_coordinate_interp"
    group.at[row_index, "Z_Angle_Soft_Domain_Original_Theta_Deg"] = float(original_theta)
    group.at[row_index, "Z_Angle_Soft_Domain_Corrected_Theta_Deg"] = float(corrected_theta)
    group.at[row_index, "Z_Angle_Soft_Domain_Distance_Deg"] = float(original_domain_distance)
    group.at[row_index, "Z_Angle_Soft_Domain_Corrected_Distance_Deg"] = float(corrected_distance)
    group.at[row_index, "Z_Angle_Soft_Domain_Replaced_Track_ID"] = suspect_tid
    group.at[row_index, "Z_Angle_Soft_Domain_Replaced_Point_Error_Px"] = float(max_error)
    group.at[row_index, "Z_Angle_QC_Message"] = "soft_domain_single_point_coordinate_interp"
    return True, "single_point_coordinate_interp"


def apply_soft_domain_angle_corrections(group: pd.DataFrame, angle_model: dict, theta_lo: float, theta_hi: float) -> pd.DataFrame:
    margin_deg = env_float("Z_ANGLE_SOFT_DOMAIN_MARGIN_DEG", 1.5)
    min_point_error_px = env_float("Z_SOFT_DOMAIN_POINT_INTERP_MIN_ERROR_PX", 2.0)
    min_domain_improvement_deg = env_float("Z_SOFT_DOMAIN_MIN_DOMAIN_IMPROVEMENT_DEG", 0.2)
    max_frame_gap = env_float("Z_SOFT_DOMAIN_POINT_INTERP_MAX_FRAME_GAP", 2.0)
    trend_max_search = int(max(1, env_float("Z_SOFT_DOMAIN_TREND_MAX_SEARCH_FRAMES", 8.0)))
    keep_extrapolated = env_bool("Z_SOFT_DOMAIN_KEEP_ANGLE_EXTRAPOLATED", True)

    group = group.sort_values("FRAME").copy()
    for column, default in [
        ("Z_Angle_Soft_Domain_Corrected", False),
        ("Z_Angle_Soft_Domain_Method", ""),
        ("Z_Angle_Soft_Domain_Original_Theta_Deg", np.nan),
        ("Z_Angle_Soft_Domain_Corrected_Theta_Deg", np.nan),
        ("Z_Angle_Soft_Domain_Distance_Deg", np.nan),
        ("Z_Angle_Soft_Domain_Corrected_Distance_Deg", np.nan),
        ("Z_Angle_Soft_Domain_Replaced_Track_ID", np.nan),
        ("Z_Angle_Soft_Domain_Replaced_Point_Error_Px", np.nan),
    ]:
        group[column] = default

    selected_signal = str(group["Z_Angle_Selected_Signal"].iloc[0]) if len(group) else "diag_b"
    theta = group["Z_Angle_Spline_Input"].to_numpy(dtype=float)
    domain_ok = group["Z_Angle_Model_Domain_OK"].to_numpy(dtype=bool)
    tracking_ok = group["Z_Angle_Tracking_Success"].to_numpy(dtype=bool)
    domain_distance = angle_distance_to_domain(theta, theta_lo, theta_hi)
    soft_mask = (~domain_ok) & tracking_ok & np.isfinite(theta) & np.isfinite(domain_distance) & (domain_distance <= margin_deg)

    for local_idx in np.where(soft_mask)[0]:
        original_theta = float(theta[local_idx])
        original_distance = float(domain_distance[local_idx])
        ok, _ = try_single_point_soft_domain_correction(
            group,
            local_idx,
            selected_signal,
            angle_model,
            theta_lo,
            theta_hi,
            original_theta,
            original_distance,
            min_point_error_px,
            min_domain_improvement_deg,
            max_frame_gap,
        )
        if ok:
            continue

        current_domain_ok = group["Z_Angle_Model_Domain_OK"].to_numpy(dtype=bool)
        z_angle = group["Z_Angle_Primary"].to_numpy(dtype=float)
        left = None
        right = None
        for step in range(1, trend_max_search + 1):
            cand = local_idx - step
            if cand >= 0 and current_domain_ok[cand] and np.isfinite(z_angle[cand]):
                left = cand
                break
        for step in range(1, trend_max_search + 1):
            cand = local_idx + step
            if cand < len(group) and current_domain_ok[cand] and np.isfinite(z_angle[cand]):
                right = cand
                break
        row_index = group.index[local_idx]
        if left is not None and right is not None:
            f0 = float(group["FRAME"].iloc[left])
            f1 = float(group["FRAME"].iloc[right])
            ratio = float((float(group["FRAME"].iloc[local_idx]) - f0) / max(f1 - f0, 1e-12))
            z_interp = float(z_angle[left] + ratio * (z_angle[right] - z_angle[left]))
            group.at[row_index, "Z_Angle_Primary"] = z_interp
            group.at[row_index, "Z_Angle_Model_Domain_OK"] = True
            group.at[row_index, "Z_Angle_Out_Of_Domain_Level"] = "soft_corrected"
            group.at[row_index, "Z_Angle_Soft_Domain_Corrected"] = True
            group.at[row_index, "Z_Angle_Soft_Domain_Method"] = "neighbor_angle_z_trend_interp"
            group.at[row_index, "Z_Angle_Soft_Domain_Original_Theta_Deg"] = original_theta
            group.at[row_index, "Z_Angle_Soft_Domain_Distance_Deg"] = original_distance
            group.at[row_index, "Z_Angle_QC_Message"] = "soft_domain_neighbor_angle_z_trend_interp"
        elif keep_extrapolated:
            group.at[row_index, "Z_Angle_Model_Domain_OK"] = True
            group.at[row_index, "Z_Angle_Out_Of_Domain_Level"] = "soft_extrapolated"
            group.at[row_index, "Z_Angle_Soft_Domain_Corrected"] = True
            group.at[row_index, "Z_Angle_Soft_Domain_Method"] = "soft_domain_keep_angle_extrapolated"
            group.at[row_index, "Z_Angle_Soft_Domain_Original_Theta_Deg"] = original_theta
            group.at[row_index, "Z_Angle_Soft_Domain_Distance_Deg"] = original_distance
            group.at[row_index, "Z_Angle_QC_Message"] = "soft_domain_keep_angle_extrapolated"

    return group


def apply_angle_primary_model(res_df: pd.DataFrame, angle_model: dict | None) -> pd.DataFrame:
    df = add_tracked_angle_columns(res_df)
    df["Z_Angle_Primary"] = np.nan
    df["Z_Angle_Signal"] = ""
    df["Z_Angle_Selected_Signal"] = ""
    df["Z_Angle_Spline_Input"] = np.nan
    df["Z_Angle_Model_Domain_OK"] = False
    df["Z_Angle_Domain_Distance_Deg"] = np.nan
    df["Z_Angle_Out_Of_Domain_Level"] = "unavailable"
    df["Z_Angle_Tracking_Success"] = False

    if not angle_model:
        df["Z_Angle_QC_Message"] = "missing_angle_model"
        return df

    spline_model = angle_model["angle_spline"]
    theta_lo, theta_hi = map(float, spline_model["domain_theta_deg"])
    configured_signal = str(angle_model.get("angle_signal", "diag_b"))
    groups: list[pd.DataFrame] = []
    for _, group in df.groupby("Cluster_ID", sort=False):
        group = group.sort_values("FRAME").copy()
        theta_col, selected_signal = angle_signal_column(group, configured_signal, theta_lo, theta_hi)
        theta = group[theta_col].to_numpy(dtype=float)
        z_angle, domain_ok, eval_theta = apply_angle_spline(theta, angle_model)
        domain_distance = angle_distance_to_domain(theta, theta_lo, theta_hi)
        group["Z_Angle_Primary"] = z_angle
        group["Z_Angle_Signal"] = configured_signal
        group["Z_Angle_Selected_Signal"] = selected_signal
        group["Z_Angle_Spline_Input"] = eval_theta
        group["Z_Angle_Model_Domain_OK"] = domain_ok
        group["Z_Angle_Domain_Distance_Deg"] = domain_distance
        group["Z_Angle_Out_Of_Domain_Level"] = np.where(
            np.isfinite(theta),
            np.where(domain_ok, "in_domain", "out_of_domain"),
            "unavailable",
        )
        group["Z_Angle_Tracking_Success"] = np.isfinite(theta)
        group["Z_Angle_QC_Message"] = "ok"
        group = apply_soft_domain_angle_corrections(group, angle_model, theta_lo, theta_hi)
        groups.append(group)

    return pd.concat(groups, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)


def apply_length_backup_model(res_df: pd.DataFrame, length_model: dict) -> pd.DataFrame:
    df = res_df.copy()
    poly_coeff = np.asarray(length_model["simulation_poly2_coefficients"], dtype=float)
    z_uncorrected = np.polyval(poly_coeff, df["Calculated_Size"].to_numpy(dtype=float))
    z_length, domain_ok, eval_x = apply_spline(z_uncorrected, length_model)
    df["Z_Length_Uncorrected_SimulationPoly2"] = z_uncorrected
    df["Z_Length_Spline_Input"] = eval_x
    df["Z_Length_Model_Domain_OK"] = domain_ok
    df["Z_Length_Backup_Raw"] = z_length
    df["Z_Length_Backup"] = z_length
    return df


def local_length_alignment_offsets(
    frames: np.ndarray,
    z_angle: np.ndarray,
    z_length_raw: np.ndarray,
    valid_alignment: np.ndarray,
    window_frames: float,
    min_local_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local constant offset to put length backup on the angle-Z origin."""
    frames = np.asarray(frames, dtype=float)
    offsets = np.asarray(z_angle, dtype=float) - np.asarray(z_length_raw, dtype=float)
    valid_alignment = np.asarray(valid_alignment, dtype=bool) & np.isfinite(offsets) & np.isfinite(frames)
    out = np.full(len(frames), np.nan, dtype=float)
    source = np.array(["none"] * len(frames), dtype=object)

    if valid_alignment.any():
        fallback_offset = float(np.nanmedian(offsets[valid_alignment]))
    else:
        fallback_offset = 0.0

    valid_indices = np.where(valid_alignment)[0]
    for idx in range(len(frames)):
        if len(valid_indices):
            local_mask = valid_alignment & (np.abs(frames - frames[idx]) <= window_frames)
            if int(np.sum(local_mask)) >= min_local_samples:
                out[idx] = float(np.nanmedian(offsets[local_mask]))
                source[idx] = "local_median"
                continue

            left_candidates = valid_indices[valid_indices < idx]
            right_candidates = valid_indices[valid_indices > idx]
            left = int(left_candidates[-1]) if len(left_candidates) else None
            right = int(right_candidates[0]) if len(right_candidates) else None
            if left is not None and right is not None and frames[right] != frames[left]:
                ratio = float((frames[idx] - frames[left]) / (frames[right] - frames[left]))
                ratio = float(np.clip(ratio, 0.0, 1.0))
                out[idx] = float(offsets[left] + ratio * (offsets[right] - offsets[left]))
                source[idx] = "bracket_interp"
                continue
            if left is not None:
                out[idx] = float(offsets[left])
                source[idx] = "left_nearest"
                continue
            if right is not None:
                out[idx] = float(offsets[right])
                source[idx] = "right_nearest"
                continue

        out[idx] = fallback_offset
        source[idx] = "group_median"

    return out, source


def interpolate_by_length_trend(z_angle: np.ndarray, z_length: np.ndarray, idx: int, left: int, right: int) -> tuple[float, str, float]:
    denom = float(z_length[right] - z_length[left])
    if np.isfinite(denom) and abs(denom) >= 0.2:
        ratio = float((z_length[idx] - z_length[left]) / denom)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        z_value = float(z_angle[left] + ratio * (z_angle[right] - z_angle[left]))
        return z_value, "length_trend", ratio
    ratio = float((idx - left) / max(right - left, 1))
    z_value = float(z_angle[left] + ratio * (z_angle[right] - z_angle[left]))
    return z_value, "frame_linear_length_flat", ratio


def nonmax_suppress_candidates(candidate: np.ndarray, score: np.ndarray, radius: int) -> np.ndarray:
    keep = np.zeros(len(candidate), dtype=bool)
    for idx in np.where(candidate)[0]:
        lo = max(0, idx - radius)
        hi = min(len(candidate), idx + radius + 1)
        local = score[lo:hi]
        local_max = np.nanmax(local) if np.isfinite(local).any() else np.nan
        if np.isfinite(local_max) and score[idx] >= local_max - 1e-12:
            keep[idx] = True
    return keep


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))


def finite01(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(arr), arr, fill)
    return np.clip(out, 0.0, 1.0)


def local_offset_statistics(
    z_angle: np.ndarray,
    z_length: np.ndarray,
    excluded: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angle = np.asarray(z_angle, dtype=float)
    length = np.asarray(z_length, dtype=float)
    excluded = np.asarray(excluded, dtype=bool)
    offset = length - angle
    local_median = np.full(len(offset), np.nan, dtype=float)
    local_mad = np.full(len(offset), np.nan, dtype=float)
    local_std = np.full(len(offset), np.nan, dtype=float)
    local_deviation = np.full(len(offset), np.nan, dtype=float)
    for idx in range(len(offset)):
        lo = max(0, idx - window)
        hi = min(len(offset), idx + window + 1)
        local_idx = np.arange(lo, hi)
        mask = (
            (local_idx != idx)
            & (~excluded[lo:hi])
            & np.isfinite(offset[lo:hi])
        )
        vals = offset[lo:hi][mask]
        if len(vals) == 0:
            continue
        med = float(np.nanmedian(vals))
        local_median[idx] = med
        local_mad[idx] = float(np.nanmedian(np.abs(vals - med))) if len(vals) >= 2 else 0.0
        local_std[idx] = float(np.nanstd(vals)) if len(vals) >= 2 else 0.0
        if np.isfinite(offset[idx]):
            local_deviation[idx] = abs(float(offset[idx] - med))
    return local_median, local_mad, local_std, local_deviation


def discover_image_stack_path() -> Path | None:
    env_value = os.getenv("Z_SPOT_QUALITY_IMAGE_FILE", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
    candidates: list[Path] = []
    for pattern in ("selected_image_stack*.tif", "selected_image_stack*.tiff", "*.tif", "*.tiff"):
        candidates.extend(sorted(Path.cwd().glob(pattern)))
    seen = set()
    unique = []
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique[0] if unique else None


def load_spot_quality_stack() -> tuple[np.ndarray | None, str]:
    if not env_bool("Z_V4_SPOT_QUALITY_USE_IMAGE", True):
        return None, "disabled"
    image_path = discover_image_stack_path()
    if image_path is None:
        return None, "not_found"
    if tifffile is None:
        return None, "tifffile_unavailable"
    try:
        stack = tifffile.imread(str(image_path))
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        if stack.ndim < 3:
            return None, f"invalid_shape_{stack.shape}"
        return np.asarray(stack), str(image_path)
    except Exception as exc:
        return None, f"read_failed:{exc}"


def angle_diff_180(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(abs(((a - b + 90.0) % 180.0) - 90.0))


def square_quality(coords: np.ndarray) -> dict[str, float]:
    if coords.shape != (4, 2) or not np.isfinite(coords).all():
        return {
            "side_cv": np.nan,
            "diag_rel_diff": np.nan,
            "diag_midpoint_gap_rel": np.nan,
            "diag_angle_orth_error": np.nan,
            "square_residual": np.nan,
            "mean_diag_px_feature": np.nan,
        }
    center = coords.mean(axis=0)
    ordered = coords[np.argsort(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))]
    sides = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    diag0_vec = ordered[2] - ordered[0]
    diag1_vec = ordered[3] - ordered[1]
    diag0 = float(np.linalg.norm(diag0_vec))
    diag1 = float(np.linalg.norm(diag1_vec))
    mean_diag = 0.5 * (diag0 + diag1)
    midpoint0 = 0.5 * (ordered[0] + ordered[2])
    midpoint1 = 0.5 * (ordered[1] + ordered[3])
    angle0 = math.degrees(math.atan2(diag0_vec[1], diag0_vec[0])) % 180.0
    angle1 = math.degrees(math.atan2(diag1_vec[1], diag1_vec[0])) % 180.0
    orth_error = abs(angle_diff_180(angle0 - angle1, 90.0)) / 90.0
    side_cv = float(np.std(sides) / max(np.mean(sides), 1e-9))
    diag_rel = float(abs(diag0 - diag1) / max(mean_diag, 1e-9))
    midpoint_gap = float(np.linalg.norm(midpoint0 - midpoint1) / max(mean_diag, 1e-9))
    return {
        "side_cv": side_cv,
        "diag_rel_diff": diag_rel,
        "diag_midpoint_gap_rel": midpoint_gap,
        "diag_angle_orth_error": orth_error,
        "square_residual": float(max(side_cv, diag_rel, midpoint_gap, orth_error)),
        "mean_diag_px_feature": float(mean_diag),
    }


def row_spot_coords(row: pd.Series) -> np.ndarray:
    coords = np.full((4, 2), np.nan, dtype=float)
    for idx in range(4):
        coords[idx, 0] = pd.to_numeric(row.get(f"Spot{idx}_X_Px", np.nan), errors="coerce")
        coords[idx, 1] = pd.to_numeric(row.get(f"Spot{idx}_Y_Px", np.nan), errors="coerce")
    return coords


def radial_center(roi: np.ndarray, sigma_px: float = 1.0) -> tuple[float, float]:
    img = ndimage.gaussian_filter(np.asarray(roi, dtype=float), max(0.0, float(sigma_px)))
    ny, nx = img.shape
    ymid, xmid = (ny - 1) / 2.0, (nx - 1) / 2.0
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]
    gy, gx = np.gradient(img)
    g2 = gx**2 + gy**2 + 1e-10
    sum_gx2 = np.sum(g2 * gx**2)
    sum_gy2 = np.sum(g2 * gy**2)
    sum_gxgy = np.sum(g2 * gx * gy)
    sum_b1 = np.sum(g2 * gx * ((grid_x - xmid) * gx + (grid_y - ymid) * gy))
    sum_b2 = np.sum(g2 * gy * ((grid_x - xmid) * gx + (grid_y - ymid) * gy))
    try:
        offset = np.linalg.solve(
            np.array([[sum_gx2, sum_gxgy], [sum_gxgy, sum_gy2]], dtype=float),
            np.array([sum_b1, sum_b2], dtype=float),
        )
        return float(offset[0] + xmid), float(offset[1] + ymid)
    except Exception:
        return float(xmid), float(ymid)


def radial_residual_features(
    image_stack: np.ndarray | None,
    row: pd.Series,
    roi_radius: int,
    coordinate_to_image_scale: float = 1.0,
    image_to_reference_scale: float = 1.0,
) -> tuple[float, float]:
    if image_stack is None:
        return np.nan, np.nan
    try:
        frame = int(row[col_frame])
    except Exception:
        return np.nan, np.nan
    if frame < 0 or frame >= image_stack.shape[0]:
        return np.nan, np.nan
    image = np.asarray(image_stack[frame], dtype=float)
    coords = row_spot_coords(row) * float(coordinate_to_image_scale)
    current_roi_radius = max(2, int(round(float(roi_radius) * float(coordinate_to_image_scale))))
    residuals = []
    height, width = image.shape
    for x, y in coords:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        ix, iy = int(round(x)), int(round(y))
        r0 = max(0, iy - current_roi_radius)
        r1 = min(height, iy + current_roi_radius + 1)
        c0 = max(0, ix - current_roi_radius)
        c1 = min(width, ix + current_roi_radius + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        roi = image[r0:r1, c0:c1]
        rx, ry = radial_center(roi, sigma_px=max(0.25, float(coordinate_to_image_scale)))
        residual_current_px = float(np.linalg.norm(np.array([rx, ry]) - np.array([x - c0, y - r0])))
        residuals.append(residual_current_px * float(image_to_reference_scale))
    if not residuals:
        return np.nan, np.nan
    return float(np.nanmedian(residuals)), float(np.nanmax(residuals))


def add_spot_quality_raw_features(
    res_df: pd.DataFrame,
    target_df: pd.DataFrame,
    coord_scale: float,
    z_x_col: str,
    z_y_col: str,
    image_coord_scale: float | None = None,
) -> pd.DataFrame:
    out = res_df.copy()
    image_stack, image_source = load_spot_quality_stack()
    image_scale = float(image_coord_scale if image_coord_scale is not None else coord_scale)
    if coord_scale <= 0 or image_scale <= 0:
        raise ValueError("Z model-reference and current camera pixel sizes must be positive.")
    coordinate_to_image_scale = float(coord_scale) / image_scale
    image_to_reference_scale = image_scale / float(coord_scale)
    roi_radius = int(max(2, env_float("Z_V4_SPOT_QUALITY_ROI_RADIUS_PX", 7.0)))
    out["Z_QC_V4_Spot_Quality_Image_Source"] = image_source

    square_rows = []
    radial_medians = []
    radial_maxes = []
    for _, row in out.iterrows():
        geom = square_quality(row_spot_coords(row))
        square_rows.append(geom)
        median_res, max_res = radial_residual_features(
            image_stack,
            row,
            roi_radius,
            coordinate_to_image_scale=coordinate_to_image_scale,
            image_to_reference_scale=image_to_reference_scale,
        )
        radial_medians.append(median_res)
        radial_maxes.append(max_res)
    out["spot_angle_radial_phasor_median_px"] = radial_medians
    out["spot_angle_radial_phasor_max_px"] = radial_maxes
    out["spot_angle_square_residual"] = [row["square_residual"] for row in square_rows]
    out["spot_angle_side_cv"] = [row["side_cv"] for row in square_rows]
    out["spot_angle_diag_rel_diff"] = [row["diag_rel_diff"] for row in square_rows]
    out["spot_angle_diag_midpoint_gap_rel"] = [row["diag_midpoint_gap_rel"] for row in square_rows]
    out["spot_angle_diag_angle_orth_error"] = [row["diag_angle_orth_error"] for row in square_rows]
    out["spot_angle_mean_diag_px_feature"] = [row["mean_diag_px_feature"] for row in square_rows]

    blur_features: dict[tuple, dict[str, float]] = {}
    if col_x in target_df.columns and col_y in target_df.columns and z_x_col in target_df.columns and z_y_col in target_df.columns:
        for key, group in target_df.groupby([col_cluster, col_frame], sort=False):
            z_coords = group[[z_x_col, z_y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / coord_scale
            xy_coords = group[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / coord_scale
            finite = np.isfinite(z_coords).all(axis=1) & np.isfinite(xy_coords).all(axis=1)
            z_clean = z_coords[finite]
            xy_clean = xy_coords[finite]
            if len(z_clean) == 0:
                continue
            shift = np.linalg.norm(z_clean - xy_clean, axis=1)
            center_shift = float(np.linalg.norm(np.nanmean(z_clean, axis=0) - np.nanmean(xy_clean, axis=0)))
            z_geom = square_quality(z_clean) if len(z_clean) == 4 else square_quality(np.full((4, 2), np.nan))
            xy_geom = square_quality(xy_clean) if len(xy_clean) == 4 else square_quality(np.full((4, 2), np.nan))
            blur_features[key] = {
                "spot_blur_no_blur_mean_spot_shift_px": float(np.nanmean(shift)),
                "spot_blur_no_blur_center_shift_px": center_shift,
                "spot_blur_no_blur_mean_diag_diff_px": abs(
                    float(z_geom["mean_diag_px_feature"]) - float(xy_geom["mean_diag_px_feature"])
                ) if np.isfinite(z_geom["mean_diag_px_feature"]) and np.isfinite(xy_geom["mean_diag_px_feature"]) else np.nan,
                "spot_blur_no_blur_orth_error_diff": abs(
                    float(z_geom["diag_angle_orth_error"]) - float(xy_geom["diag_angle_orth_error"])
                ) if np.isfinite(z_geom["diag_angle_orth_error"]) and np.isfinite(xy_geom["diag_angle_orth_error"]) else np.nan,
            }

    blur_cols = [
        "spot_blur_no_blur_mean_spot_shift_px",
        "spot_blur_no_blur_center_shift_px",
        "spot_blur_no_blur_mean_diag_diff_px",
        "spot_blur_no_blur_orth_error_diff",
    ]
    for col in blur_cols:
        out[col] = np.nan
    for row_index, row in out.iterrows():
        key = (row[col_cluster], row[col_frame])
        features = blur_features.get(key)
        if not features:
            continue
        for col, value in features.items():
            out.at[row_index, col] = value
    return out


def normalize_feature_for_candidates(values: pd.Series, candidate: np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.zeros(len(arr), dtype=float)
    finite_mask = np.isfinite(arr)
    train = arr[np.asarray(candidate, dtype=bool) & finite_mask]
    if len(train) == 0:
        return out
    lo = float(np.nanpercentile(train, 5))
    hi = float(np.nanpercentile(train, 95))
    scale = hi - lo
    if not np.isfinite(scale) or scale <= 1e-12:
        lo = 0.0
        scale = max(float(np.nanmedian(np.abs(train))), 1.0)
    out[finite_mask] = np.maximum((arr[finite_mask] - lo) / scale, 0.0)
    out[~np.isfinite(out)] = 0.0
    return out


def nanmax_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arr = df[columns].to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    all_nan = np.all(~np.isfinite(arr), axis=1)
    safe = np.nanmax(np.where(all_nan[:, None], 0.0, arr), axis=1)
    safe[~np.isfinite(safe)] = 0.0
    return safe


def add_spot_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate = out.get("Z_Hard_QC_Candidate", pd.Series(False, index=out.index)).to_numpy(dtype=bool)
    feature_cols = [
        "spot_angle_radial_phasor_median_px",
        "spot_angle_radial_phasor_max_px",
        "spot_angle_square_residual",
        "spot_angle_side_cv",
        "spot_angle_diag_rel_diff",
        "spot_angle_diag_midpoint_gap_rel",
        "spot_angle_diag_angle_orth_error",
        "spot_blur_no_blur_mean_spot_shift_px",
        "spot_blur_no_blur_center_shift_px",
        "spot_blur_no_blur_mean_diag_diff_px",
        "spot_blur_no_blur_orth_error_diff",
    ]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[f"n_{col}"] = normalize_feature_for_candidates(out[col], candidate)

    out["spot_radial_residual_score"] = nanmax_columns(
        out,
        [
            "n_spot_angle_radial_phasor_median_px",
            "n_spot_angle_radial_phasor_max_px",
        ],
    )
    out["spot_square_residual_score"] = nanmax_columns(
        out,
        [
            "n_spot_angle_square_residual",
            "n_spot_angle_side_cv",
            "n_spot_angle_diag_rel_diff",
            "n_spot_angle_diag_midpoint_gap_rel",
            "n_spot_angle_diag_angle_orth_error",
        ],
    )
    out["spot_blur_difference_score"] = nanmax_columns(
        out,
        [
            "n_spot_blur_no_blur_mean_spot_shift_px",
            "n_spot_blur_no_blur_center_shift_px",
            "n_spot_blur_no_blur_mean_diag_diff_px",
            "n_spot_blur_no_blur_orth_error_diff",
        ],
    )
    out["spot_quality_combined_score"] = nanmax_columns(
        out,
        [
            "spot_radial_residual_score",
            "spot_square_residual_score",
            "spot_blur_difference_score",
        ],
    )
    out["Z_QC_V4_Spot_Radial_Residual_Score"] = out["spot_radial_residual_score"]
    out["Z_QC_V4_Spot_Square_Residual_Score"] = out["spot_square_residual_score"]
    out["Z_QC_V4_Spot_Blur_Difference_Score"] = out["spot_blur_difference_score"]
    out["Z_QC_V4_Spot_Quality_Combined_Score"] = out["spot_quality_combined_score"]
    return out


def add_qc_v2_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_parent_qc_group_columns(df)
    qc_group_col = "Z_QC_Group_ID"
    out_parts = []
    angle_norm_scale = env_float("Z_QC_V2_ANGLE_NORM_SCALE", 0.35)
    relative_scale = env_float("Z_QC_V2_RELATIVE_SCALE", 12.0)
    length_norm_scale = env_float("Z_QC_V2_LENGTH_NORM_SCALE", 1.6)
    offset_deviation_scale = env_float("Z_QC_V2_OFFSET_DEVIATION_SCALE_UM", 0.9)
    offset_std_scale = env_float("Z_QC_V2_OFFSET_STD_SCALE_UM", 0.35)
    correction_gap_scale = env_float("Z_QC_V2_CORRECTION_GAP_SCALE_UM", 1.1)
    length_smooth_power = env_float("Z_QC_V2_LENGTH_SMOOTH_POWER", 0.5)
    offset_consistency_power = env_float("Z_QC_V2_OFFSET_CONSISTENCY_POWER", 0.5)
    local_stability_power = env_float("Z_QC_V2_LOCAL_STABILITY_POWER", 0.25)
    gap_power = env_float("Z_QC_V2_GAP_POWER", 0.5)
    offset_window = int(max(1, env_float("Z_QC_V2_LOCAL_OFFSET_WINDOW_FRAMES", 5.0)))

    for _, group in df.groupby(qc_group_col, sort=False, dropna=False):
        group = group.sort_values("FRAME").copy()
        z_angle = group["Z_Angle_Primary"].to_numpy(dtype=float)
        z_length = group.get(
            "Z_Length_Backup_Aligned",
            group.get("Z_Length_Backup", pd.Series(np.nan, index=group.index)),
        ).to_numpy(dtype=float)
        z_hard = group.get("Z_Hard_QC_Target", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
        angle_norm = group.get("Z_Angle_Second_Diff_Normalized", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
        length_norm = group.get("Z_Length_Second_Diff_Normalized", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
        rel = group.get("Angle_Length_D2_Relative_Percent", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
        angle_jump = group.get("Z_Angle_Jump_Flag", pd.Series(False, index=group.index)).to_numpy(dtype=bool)
        correction = np.abs(z_hard - z_angle)
        local_median, local_mad, local_std, local_deviation = local_offset_statistics(
            z_angle,
            z_length,
            angle_jump,
            offset_window,
        )

        angle_evidence = sigmoid((angle_norm - 1.0) / max(angle_norm_scale, 1e-9)) * sigmoid(
            (rel - 90.0) / max(relative_scale, 1e-9)
        )
        length_smooth = 1.0 / (1.0 + (np.maximum(length_norm, 0.0) / max(length_norm_scale, 1e-9)) ** 2)
        offset_consistency = 1.0 / (
            1.0 + (np.maximum(local_deviation, 0.0) / max(offset_deviation_scale, 1e-9)) ** 2
        )
        local_stability = 1.0 / (1.0 + (np.maximum(local_std, 0.0) / max(offset_std_scale, 1e-9)) ** 2)
        gap_credibility = 1.0 / (1.0 + (np.maximum(correction, 0.0) / max(correction_gap_scale, 1e-9)) ** 2)
        spot_count = group.get("Available_Spot_Count", pd.Series(4.0, index=group.index)).to_numpy(dtype=float)
        length_domain = group.get("Z_Length_Model_Domain_OK", pd.Series(True, index=group.index)).to_numpy(dtype=bool)
        length_success = np.isfinite(z_length)
        model_quality = np.clip(spot_count / 4.0, 0.0, 1.0)
        model_quality *= np.where(length_success, 1.0, 0.15)
        model_quality *= np.where(length_domain, 1.0, 0.35)

        components = {
            "qc_v2_angle_evidence": finite01(angle_evidence),
            "qc_v2_length_smooth": finite01(length_smooth),
            "qc_v2_offset_consistency": finite01(offset_consistency),
            "qc_v2_local_stability": finite01(local_stability),
            "qc_v2_gap_credibility": finite01(gap_credibility),
            "qc_v2_model_quality": finite01(model_quality),
        }
        for key, value in components.items():
            group[key] = value
        group["length_minus_angle_um"] = z_length - z_angle
        group["local_length_angle_offset_median_um"] = local_median
        group["local_length_angle_offset_mad_um"] = local_mad
        group["local_length_angle_offset_std_um"] = local_std
        group["local_length_angle_offset_deviation_um"] = local_deviation
        group["current_qc_correction_um"] = z_hard - z_angle
        group["current_qc_correction_abs_um"] = correction

        candidate = group.get("Z_Hard_QC_Candidate", pd.Series(False, index=group.index)).to_numpy(dtype=bool)
        weight = (
            components["qc_v2_angle_evidence"]
            * components["qc_v2_length_smooth"] ** length_smooth_power
            * components["qc_v2_offset_consistency"] ** offset_consistency_power
            * components["qc_v2_local_stability"] ** local_stability_power
            * components["qc_v2_gap_credibility"] ** gap_power
            * components["qc_v2_model_quality"]
        )
        weight = finite01(weight)
        weight[~candidate] = 0.0
        group["qc_v2_weight"] = weight
        group["Z_QC_V2_Weight"] = weight
        group["Z_QC_V2_Prediction"] = z_angle + weight * (z_hard - z_angle)
        group["Z_QC_V2_Params_ID"] = "best_validation_rmse"
        out_parts.append(group)
    return pd.concat(out_parts, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)


def apply_v4_gate_v2_correction(df: pd.DataFrame) -> pd.DataFrame:
    mode = os.getenv("Z_QC_CORRECTION_MODE", "v4_gate_v2").strip().lower()
    out = df.copy()
    out["Z_QC_Correction_Mode"] = mode
    out["Z_Final_Hard_QC_PreGate"] = out["Z_Final"].to_numpy(dtype=float)
    out["Z_Final_Source_Hard_QC_PreGate"] = out["Z_Final_Source"].astype(str)
    if mode in {"hard", "legacy", "angle_length_hard_qc"}:
        return out
    if mode not in {"v4_gate_v2", "v4+v2", "gate_v2"}:
        print(f"WARNING: unknown Z_QC_CORRECTION_MODE={mode!r}; using v4_gate_v2")

    out = add_qc_v2_weight_columns(out)
    out = add_spot_quality_scores(out)

    score_col = os.getenv("Z_QC_V4_SCORE_COL", "spot_quality_combined_score").strip()
    if score_col not in out.columns:
        score_col = "spot_quality_combined_score"
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score_low = env_float("Z_QC_V4_SCORE_LOW", 0.9)
    score_high = env_float("Z_QC_V4_SCORE_HIGH", 1.2)
    score_power = env_float("Z_QC_V4_SCORE_POWER", 0.5)
    w_min = env_float("Z_QC_V4_W_MIN", 0.0)
    spot_confidence = np.clip((score - score_low) / max(score_high - score_low, 1e-9), 0.0, 1.0) ** score_power

    hard_candidate = out.get("Z_Hard_QC_Candidate", pd.Series(False, index=out.index)).to_numpy(dtype=bool)
    z_angle = out["Z_Angle_Primary"].to_numpy(dtype=float)
    z_hard = out["Z_Hard_QC_Target"].to_numpy(dtype=float)
    weight = out["qc_v2_weight"].to_numpy(dtype=float)
    z_v2 = z_angle + weight * (z_hard - z_angle)
    accepted = (
        hard_candidate
        & np.isfinite(z_v2)
        & (weight >= w_min)
        & (spot_confidence > 0.0)
    )

    pre_final = out["Z_Final_Hard_QC_PreGate"].to_numpy(dtype=float)
    pre_source = out["Z_Final_Source_Hard_QC_PreGate"].astype(str).to_numpy(dtype=object)
    final = pre_final.copy()
    source = pre_source.copy()
    confidence = out["Z_Confidence"].astype(str).to_numpy(dtype=object)
    reason = out["Angle_Length_QC_Reason"].astype(str).to_numpy(dtype=object)

    rejected = hard_candidate & (~accepted)
    final[rejected] = z_angle[rejected]
    source[rejected] = "angle_v4_gate_rejected"
    confidence[rejected] = "high"
    for idx in np.where(rejected)[0]:
        reason[idx] = (
            f"{reason[idx]}; v4_gate_rejected; "
            f"score={score[idx]:.3f}; spot_confidence={spot_confidence[idx]:.3f}; "
            f"qc_v2_weight={weight[idx]:.3f}"
        )

    final[accepted] = z_v2[accepted]
    source[accepted] = "v4_gate_v2_correction"
    confidence[accepted] = "medium_high"
    for idx in np.where(accepted)[0]:
        reason[idx] = (
            f"{reason[idx]}; v4_gate_v2_accepted; "
            f"score={score[idx]:.3f}; spot_confidence={spot_confidence[idx]:.3f}; "
            f"qc_v2_weight={weight[idx]:.3f}"
        )

    out["Z_QC_V4_Trigger_Score_Col"] = score_col
    out["Z_QC_V4_Trigger_Score"] = score
    out["Z_QC_V4_Spot_Confidence"] = spot_confidence
    out["Z_QC_V4_Gate_Accepted"] = accepted
    out["qc_v4_trigger_accepted"] = accepted
    out["Z_QC_V4_Gate_V2_Prediction"] = np.where(accepted, z_v2, z_angle)
    out["Z_Final"] = final
    out["Z_Final_Source"] = source
    out["Z_Confidence"] = confidence
    out["Angle_Length_QC_Reason"] = reason
    return out


def apply_parent_stitching_boundary_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Align Z across parent stitching boundaries where segment membership changes.

    When two segments belonging to the same parent trajectory share >=2 tracks
    but swap a corner track (e.g. C17={15,18,23,39} -> C18={8,15,18,23}),
    the four-point angle can shift by enough to cause a Z discontinuity even
    though all four spots are observed and in-domain on both sides.

    This function detects those boundaries and applies a constant Z-offset to
    the later segment so that its Z median (over trusted four-spot frames near
    the boundary) aligns with the earlier segment's Z level.
    """
    # This changes measured Z values and is therefore opt-in. The default
    # production output remains the directly reconstructed parent trajectory.
    enabled = env_bool("Z_PARENT_STITCHING_ALIGNMENT_ENABLED", False)
    if not enabled:
        df["Z_Parent_Stitching_Alignment_Offset"] = 0.0
        df["Z_Parent_Stitching_Alignment_Applied"] = False
        df["Z_Parent_Stitching_Alignment_Message"] = "disabled"
        return df

    if "Z_Tracking_Group_ID" not in df.columns:
        df["Z_Parent_Stitching_Alignment_Offset"] = 0.0
        df["Z_Parent_Stitching_Alignment_Applied"] = False
        df["Z_Parent_Stitching_Alignment_Message"] = "no_tracking_group"
        return df

    max_boundary_gap = int(env_float("Z_PARENT_STITCHING_MAX_BOUNDARY_GAP", 2.0))
    min_shared = int(env_float("Z_PARENT_STITCHING_MIN_SHARED", 2.0))
    anchor_radius = int(env_float("Z_PARENT_STITCHING_ANCHOR_RADIUS", 10.0))
    max_offset_um = env_float("Z_PARENT_STITCHING_MAX_OFFSET_UM", 5.0)
    min_anchor_frames = int(env_float("Z_PARENT_STITCHING_MIN_ANCHOR_FRAMES", 3.0))

    df = df.sort_values(["Z_Tracking_Group_ID", "Cluster_ID", "FRAME"]).copy()

    df["Z_Parent_Stitching_Alignment_Offset"] = 0.0
    df["Z_Parent_Stitching_Alignment_Applied"] = False
    df["Z_Parent_Stitching_Alignment_Message"] = "ok"

    for tracking_id, tracking_group in df.groupby("Z_Tracking_Group_ID", sort=False):
        seg_ids = (
            tracking_group.groupby("Cluster_ID", sort=False)["FRAME"]
            .min()
            .sort_values(kind="stable")
            .index.astype(int)
            .tolist()
        )
        if len(seg_ids) < 2:
            continue

        # For each pair of adjacent segments, check boundary conditions
        for idx in range(len(seg_ids) - 1):
            seg_a = seg_ids[idx]
            seg_b = seg_ids[idx + 1]
            # Re-read from df after every earlier boundary correction so the
            # accumulated offset propagates correctly across 3+ segments.
            current_tracking = df[df["Z_Tracking_Group_ID"] == tracking_id]
            group_a = current_tracking[current_tracking["Cluster_ID"].astype(int) == seg_a].sort_values("FRAME")
            group_b = current_tracking[current_tracking["Cluster_ID"].astype(int) == seg_b].sort_values("FRAME")
            if group_a.empty or group_b.empty:
                continue

            last_frame_a = int(group_a["FRAME"].max())
            first_frame_b = int(group_b["FRAME"].min())
            gap = first_frame_b - last_frame_a - 1
            if gap > max_boundary_gap:
                continue

            # Check shared tracks
            tracks_a = set(
                str(int(float(v)))
                for v in group_a.get("Single_Diagonal_Track_ID_A", pd.Series(dtype=object))
                if pd.notna(v)
            )
            tracks_a |= set(
                str(int(float(v)))
                for v in group_a.get("Single_Diagonal_Track_ID_B", pd.Series(dtype=object))
                if pd.notna(v)
            )
            tracks_b = set(
                str(int(float(v)))
                for v in group_b.get("Single_Diagonal_Track_ID_A", pd.Series(dtype=object))
                if pd.notna(v)
            )
            tracks_b |= set(
                str(int(float(v)))
                for v in group_b.get("Single_Diagonal_Track_ID_B", pd.Series(dtype=object))
                if pd.notna(v)
            )
            # Also use SpotX_Track_ID columns if available
            for spot_idx in range(4):
                col = f"Spot{spot_idx}_Track_ID"
                if col in group_a.columns:
                    tracks_a |= set(
                        str(int(float(v)))
                        for v in group_a[col]
                        if pd.notna(v)
                    )
                if col in group_b.columns:
                    tracks_b |= set(
                        str(int(float(v)))
                        for v in group_b[col]
                        if pd.notna(v)
                    )

            shared = tracks_a & tracks_b
            if len(shared) < min_shared:
                continue

            # Find anchor frames: trusted four-spot in-domain angle frames near boundary
            anchor_a = group_a[
                (group_a["Geometry_Mode"].astype(str) == "four_spot")
                & (group_a["Z_Angle_Model_Domain_OK"].astype(bool))
                & (group_a["FRAME"].astype(int) >= last_frame_a - anchor_radius)
            ]
            anchor_b = group_b[
                (group_b["Geometry_Mode"].astype(str) == "four_spot")
                & (group_b["Z_Angle_Model_Domain_OK"].astype(bool))
                & (group_b["FRAME"].astype(int) <= first_frame_b + anchor_radius)
            ]
            if len(anchor_a) < min_anchor_frames or len(anchor_b) < min_anchor_frames:
                continue

            z_med_a = float(anchor_a["Z_Final"].median())
            z_med_b = float(anchor_b["Z_Final"].median())
            offset = z_med_b - z_med_a

            if abs(offset) < 0.05 or abs(offset) > max_offset_um:
                continue

            # Apply: subtract offset from all Z_Final values in segment B
            # and all subsequent segments in this tracking group
            offset_applied = 0
            for later_idx in range(idx + 1, len(seg_ids)):
                later_seg = seg_ids[later_idx]
                mask = df["Z_Tracking_Group_ID"] == tracking_id
                mask &= df["Cluster_ID"].astype(int) == later_seg
                if mask.any():
                    df.loc[mask, "Z_Final"] = (
                        df.loc[mask, "Z_Final"].to_numpy(dtype=float) - offset
                    )
                    existing_offset = pd.to_numeric(
                        df.loc[mask, "Z_Parent_Stitching_Alignment_Offset"], errors="coerce"
                    ).fillna(0.0)
                    df.loc[mask, "Z_Parent_Stitching_Alignment_Offset"] = (
                        existing_offset.to_numpy(dtype=float) - offset
                    )
                    df.loc[mask, "Z_Parent_Stitching_Alignment_Applied"] = True
                    n = int(mask.sum())
                    offset_applied += n
                    source_values = df.loc[mask, "Z_Final_Source"].astype(str)
                    df.loc[mask, "Z_Final_Source"] = source_values.apply(
                        lambda s: s if "parent_stitching_aligned" in s else f"{s};parent_stitching_aligned"
                    )
                    conf_values = df.loc[mask, "Z_Confidence"].astype(str)
                    df.loc[mask, "Z_Confidence"] = conf_values.apply(
                        lambda c: (
                            c
                            if c in {"low", "medium"}
                            else "medium_high"
                        )
                    )

            if offset_applied:
                boundary_mask = (df["Z_Tracking_Group_ID"] == tracking_id)
                boundary_mask &= df["Cluster_ID"].astype(int) == seg_b
                boundary_mask &= df["FRAME"].astype(int).between(
                    first_frame_b, first_frame_b + anchor_radius
                )
                df.loc[
                    boundary_mask,
                    "Z_Parent_Stitching_Alignment_Message",
                ] = (
                    f"boundary_C{seg_a}_C{seg_b}:"
                    f"gap={gap} "
                    f"shared={len(shared)} "
                    f"anchor_a_median={z_med_a:.3f} "
                    f"anchor_b_median={z_med_b:.3f} "
                    f"offset={-offset:.3f}um "
                    f"frames_aligned={offset_applied}"
                )

    return df.sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)


def apply_two_point_trusted_frame_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Replace two-spot Z/diagonal outputs using bracketing trusted frames."""
    if df.empty or "Available_Spot_Count" not in df.columns or "Z_Final" not in df.columns:
        return df

    out = df.copy()
    tracking_col = "Z_Tracking_Group_ID" if "Z_Tracking_Group_ID" in out.columns else "Cluster_ID"
    for col, default in [
        ("Z_Two_Point_Interpolated", False),
        ("Z_Two_Point_Interp_Left_Frame", np.nan),
        ("Z_Two_Point_Interp_Right_Frame", np.nan),
        ("Z_Two_Point_Interp_Ratio", np.nan),
        ("Z_Two_Point_Interp_Message", ""),
    ]:
        if col not in out.columns:
            out[col] = default

    interp_cols = [
        "Calculated_Size",
        "Diag1",
        "Diag2",
        "Single_Diagonal_Length_Px",
        "Theta_Tracked_Simple_Average_Unwrapped_Deg",
        "Z_Angle_Spline_Input",
        "Z_Angle_Primary",
        "Z_Length_Backup",
        "Z_Length_Backup_Aligned",
        "Z_Hard_QC_Target",
        "Z_QC_V2_Prediction",
        "Z_QC_V4_Gate_V2_Prediction",
        "Z_Final",
        "Predicted_Z",
    ]

    for _, group in out.sort_values([tracking_col, "FRAME", "Cluster_ID"]).groupby(tracking_col, sort=False, dropna=False):
        order = list(group.sort_values(["FRAME", "Cluster_ID"]).index)
        frames = out.loc[order, "FRAME"].to_numpy(dtype=float)
        spot_count = out.loc[order, "Available_Spot_Count"].to_numpy(dtype=float)
        z_final = out.loc[order, "Z_Final"].to_numpy(dtype=float)
        if "Z_Angle_Model_Domain_OK" in out.columns:
            domain_ok = out.loc[order, "Z_Angle_Model_Domain_OK"].to_numpy(dtype=bool)
        else:
            domain_ok = np.isfinite(z_final)
        if "Z_Confidence" in out.columns:
            confidence = out.loc[order, "Z_Confidence"].astype(str).to_numpy()
        else:
            confidence = np.array([""] * len(order), dtype=object)
        if "Z_Final_Source" in out.columns:
            source = out.loc[order, "Z_Final_Source"].astype(str).to_numpy()
        else:
            source = np.array([""] * len(order), dtype=object)
        trusted = (
            (spot_count >= 3)
            & np.isfinite(z_final)
            & domain_ok
            & (confidence != "low")
            & np.asarray(["extrapolated" not in value for value in source], dtype=bool)
        )
        targets = spot_count <= 2
        for pos, row_index in enumerate(order):
            if not targets[pos]:
                continue
            left_candidates = np.flatnonzero(trusted[:pos])
            right_candidates = np.flatnonzero(trusted[pos + 1 :]) + pos + 1
            if len(left_candidates) == 0 or len(right_candidates) == 0:
                out.at[row_index, "Z_Two_Point_Interp_Message"] = "missing_bracketing_trusted_frames"
                continue
            left_pos = int(left_candidates[-1])
            right_pos = int(right_candidates[0])
            left_frame = float(frames[left_pos])
            right_frame = float(frames[right_pos])
            if right_frame == left_frame:
                out.at[row_index, "Z_Two_Point_Interp_Message"] = "invalid_bracketing_frames"
                continue
            ratio = float((frames[pos] - left_frame) / (right_frame - left_frame))
            left_index = order[left_pos]
            right_index = order[right_pos]
            for col in interp_cols:
                if col not in out.columns:
                    continue
                left_value = out.at[left_index, col]
                right_value = out.at[right_index, col]
                if pd.notna(left_value) and pd.notna(right_value) and np.isfinite(float(left_value)) and np.isfinite(float(right_value)):
                    out.at[row_index, col] = float(left_value) + ratio * (float(right_value) - float(left_value))
            if "Theta_Tracked_Simple_Average_Deg" in out.columns and "Theta_Tracked_Simple_Average_Unwrapped_Deg" in out.columns:
                theta_value = out.at[row_index, "Theta_Tracked_Simple_Average_Unwrapped_Deg"]
                if pd.notna(theta_value) and np.isfinite(float(theta_value)):
                    out.at[row_index, "Theta_Tracked_Simple_Average_Deg"] = float(theta_value) % 180.0
            if "Z_Angle_Model_Domain_OK" in out.columns:
                out.at[row_index, "Z_Angle_Model_Domain_OK"] = True
            if "Z_Angle_Out_Of_Domain_Level" in out.columns:
                out.at[row_index, "Z_Angle_Out_Of_Domain_Level"] = "two_point_interpolated"
            if "Z_Angle_QC_Message" in out.columns:
                out.at[row_index, "Z_Angle_QC_Message"] = "two_point_trusted_frame_interpolation"
            if "Z_Final_Source" in out.columns:
                out.at[row_index, "Z_Final_Source"] = "two_point_trusted_frame_interpolated"
            if "Z_Confidence" in out.columns:
                out.at[row_index, "Z_Confidence"] = "medium"
            out.at[row_index, "Z_Two_Point_Interpolated"] = True
            out.at[row_index, "Z_Two_Point_Interp_Left_Frame"] = left_frame
            out.at[row_index, "Z_Two_Point_Interp_Right_Frame"] = right_frame
            out.at[row_index, "Z_Two_Point_Interp_Ratio"] = ratio
            out.at[row_index, "Z_Two_Point_Interp_Message"] = "ok"

    return out


def apply_angle_length_d2_qc(res_df: pd.DataFrame) -> pd.DataFrame:
    df = add_parent_qc_group_columns(res_df).sort_values(["Cluster_ID", "FRAME"]).copy()
    qc_group_col = "Z_QC_Group_ID"
    enabled = env_bool("Z_ANGLE_LENGTH_D2_QC_ENABLED", True)
    align_length = env_bool("Z_ALIGN_LENGTH_BACKUP_TO_ANGLE", True)
    align_window_frames = env_float("Z_LENGTH_BACKUP_ALIGNMENT_WINDOW_FRAMES", 10.0)
    align_min_samples = int(max(1, env_float("Z_LENGTH_BACKUP_ALIGNMENT_MIN_SAMPLES", 3.0)))
    trim_low_percent = env_float("Z_D2_NORMALIZATION_TRIM_LOW_PERCENT", 5.0)
    trim_high_percent = env_float("Z_D2_NORMALIZATION_TRIM_HIGH_PERCENT", 95.0)
    relative_threshold_percent = env_float("Z_ANGLE_LENGTH_D2_RELATIVE_THRESHOLD_PERCENT", 90.0)
    angle_norm_min = env_float("Z_ANGLE_D2_NORMALIZED_MIN", 1.0)
    soft_domain_margin_deg = env_float("Z_ANGLE_SOFT_DOMAIN_MARGIN_DEG", 1.5)
    max_search = int(max(1, env_float("Z_TREND_INTERP_MAX_SEARCH_FRAMES", 3.0)))
    nms_radius = int(max(0, env_float("Z_ANGLE_JUMP_NMS_RADIUS", 2.0)))

    df["Z_Final"] = df["Z_Angle_Primary"].to_numpy(dtype=float)
    df["Z_Final_Source"] = "angle"
    df["Z_Confidence"] = "high"
    df["Z_Length_Backup_Aligned"] = df["Z_Length_Backup_Raw"].to_numpy(dtype=float) if "Z_Length_Backup_Raw" in df.columns else df["Z_Length_Backup"].to_numpy(dtype=float)
    df["Z_Length_Backup_Alignment_Offset"] = 0.0
    df["Z_Length_Backup_Alignment_Source"] = "disabled"
    df["Z_Angle_Second_Diff"] = np.nan
    df["Z_Length_Second_Diff"] = np.nan
    df["Z_Angle_Second_Diff_Abs"] = np.nan
    df["Z_Length_Second_Diff_Abs"] = np.nan
    df["Z_Angle_Second_Diff_Normalized"] = np.nan
    df["Z_Length_Second_Diff_Normalized"] = np.nan
    df["Angle_Length_D2_Relative_Percent"] = np.nan
    df["Z_Angle_vs_Length_D2_Relative_Percent"] = np.nan
    df["Angle_Only_Jump_Flag"] = False
    df["Length_Only_Jump_Flag"] = False
    df["Z_Angle_Jump_Flag"] = False
    df["Z_Length_Jump_Flag"] = False
    df["Angle_Length_QC_Reason"] = "disabled" if not enabled else "ok"
    df["Z_Angle_Severe_Out_Of_Domain"] = False
    df["Z_Trend_Interp_Left_Frame"] = np.nan
    df["Z_Trend_Interp_Right_Frame"] = np.nan
    df["Z_Trend_Interp_Mode"] = ""
    df["Z_Trend_Interp_Ratio"] = np.nan
    df["Z_Angle_D2_Normalized_Min"] = angle_norm_min
    df["Z_Angle_Length_D2_Relative_Threshold_Percent"] = relative_threshold_percent
    df["Z_D2_Norm_Trim_Low_Percent"] = trim_low_percent
    df["Z_D2_Norm_Trim_High_Percent"] = trim_high_percent
    df["Z_Hard_QC_Candidate"] = False
    df["Z_Hard_QC_Target"] = df["Z_Final"].to_numpy(dtype=float)
    df["Z_Hard_QC_Target_Source"] = "angle"
    df["Z_Hard_QC_Target_Confidence"] = df["Z_Confidence"].astype(str)

    groups: list[pd.DataFrame] = []
    for _, group in df.groupby(qc_group_col, sort=False, dropna=False):
        group = group.sort_values(["FRAME", "Cluster_ID"], kind="stable").copy()
        z_angle = group["Z_Angle_Primary"].to_numpy(dtype=float)
        z_length_raw = group["Z_Length_Backup_Raw"].to_numpy(dtype=float) if "Z_Length_Backup_Raw" in group.columns else group["Z_Length_Backup"].to_numpy(dtype=float)
        angle_domain_ok = group.get("Z_Angle_Model_Domain_OK", pd.Series(False, index=group.index)).to_numpy(dtype=bool)
        angle_track_ok = group.get("Z_Angle_Tracking_Success", pd.Series(False, index=group.index)).to_numpy(dtype=bool)
        domain_distance = group.get("Z_Angle_Domain_Distance_Deg", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
        theta_average_source = group.get(
            "Theta_Tracked_Simple_Average_Source",
            pd.Series("", index=group.index),
        ).astype(str).to_numpy()
        single_diagonal_angle = np.asarray([value.startswith("single_diag") for value in theta_average_source], dtype=bool)
        valid_alignment = angle_domain_ok & angle_track_ok & np.isfinite(z_angle) & np.isfinite(z_length_raw)
        if align_length:
            align_offsets, align_sources = local_length_alignment_offsets(
                group["FRAME"].to_numpy(dtype=float),
                z_angle,
                z_length_raw,
                valid_alignment,
                align_window_frames,
                align_min_samples,
            )
        else:
            align_offsets = np.zeros(len(group), dtype=float)
            align_sources = np.array(["disabled"] * len(group), dtype=object)
        z_length_aligned = z_length_raw + align_offsets
        z_length_for_qc = z_length_raw
        group["Z_Length_Backup_Aligned"] = z_length_aligned
        group["Z_Length_Backup_Alignment_Offset"] = align_offsets
        group["Z_Length_Backup_Alignment_Source"] = align_sources
        group["Z_Length_Backup"] = z_length_aligned

        final = z_angle.copy()
        source = np.array(["angle"] * len(group), dtype=object)
        confidence = np.array(["high"] * len(group), dtype=object)
        parent_boundary = group.get(
            "Z_QC_Parent_Segment_Boundary",
            pd.Series(False, index=group.index),
        ).to_numpy(dtype=bool)
        reason = np.where(parent_boundary, "parent_segment_boundary_qc", "ok").astype(object)
        hard_qc_candidate = np.zeros(len(group), dtype=bool)
        hard_qc_target = final.copy()
        hard_qc_source = np.array(["angle"] * len(group), dtype=object)
        hard_qc_confidence = np.array(["high"] * len(group), dtype=object)
        angle_available = angle_track_ok & np.isfinite(z_angle)
        angle_unavailable = ~angle_available
        angle_out_of_domain = (~angle_domain_ok) & angle_available
        severe_out_of_domain = (
            angle_out_of_domain
            & (
                ~np.isfinite(domain_distance)
                | (domain_distance > soft_domain_margin_deg)
            )
        )
        out_of_domain_level = group.get(
            "Z_Angle_Out_Of_Domain_Level",
            pd.Series("unavailable", index=group.index),
        ).astype(str).to_numpy()
        out_of_domain_level[angle_available & angle_domain_ok & (out_of_domain_level == "out_of_domain")] = "in_domain"
        out_of_domain_level[angle_out_of_domain & (~severe_out_of_domain)] = "soft_unresolved_out_of_domain"
        out_of_domain_level[severe_out_of_domain] = "severe_out_of_domain"
        out_of_domain_level[angle_unavailable] = "unavailable"
        group["Z_Angle_Out_Of_Domain_Level"] = out_of_domain_level
        group["Z_Angle_Severe_Out_Of_Domain"] = severe_out_of_domain

        single_mask = angle_available & single_diagonal_angle
        source[single_mask] = "angle_single_diagonal"
        confidence[single_mask] = "medium"
        reason[single_mask] = [
            f"single_diagonal_degraded_angle:{src}"
            for src in theta_average_source[single_mask]
        ]
        soft_method = group.get("Z_Angle_Soft_Domain_Method", pd.Series("", index=group.index)).astype(str).to_numpy()
        soft_corrected = group.get("Z_Angle_Soft_Domain_Corrected", pd.Series(False, index=group.index)).to_numpy(dtype=bool)
        soft_mask = soft_corrected & (soft_method != "")
        source[soft_mask] = "angle_soft_domain"
        confidence[soft_mask] = np.where(
            soft_method[soft_mask] == "single_point_coordinate_interp",
            "medium_high",
            "medium",
        )
        reason[soft_mask] = [f"soft_domain_{method}" for method in soft_method[soft_mask]]

        length_finite = np.isfinite(z_length_aligned)
        source[angle_out_of_domain] = np.where(
            single_diagonal_angle[angle_out_of_domain],
            "angle_single_diagonal_extrapolated",
            "angle_extrapolated_out_of_domain",
        )
        confidence[angle_out_of_domain] = "low"
        reason[angle_out_of_domain] = "angle_out_of_domain_extrapolated; length_backup_kept_for_qc"

        invalidate_single_ood = env_bool("Z_INVALIDATE_SEVERE_SINGLE_DIAGONAL_OOD", True)
        severe_single_ood = invalidate_single_ood & severe_out_of_domain & single_diagonal_angle
        final[severe_single_ood] = np.nan
        source[severe_single_ood] = "angle_single_diagonal_severe_ood_invalid"
        confidence[severe_single_ood] = "low"
        reason[severe_single_ood] = "single_diagonal_severe_out_of_domain_invalidated"

        fallback = angle_unavailable & length_finite
        final[fallback] = z_length_aligned[fallback]
        source[fallback] = "length_fallback_angle_unavailable"
        confidence[fallback] = "low"
        reason[fallback] = "angle_unavailable; length_backup_aligned_to_angle"
        hard_qc_target[fallback] = final[fallback]
        hard_qc_source[fallback] = source[fallback]
        hard_qc_confidence[fallback] = confidence[fallback]
        no_fallback = angle_unavailable & (~length_finite)
        confidence[no_fallback] = "low"
        reason[no_fallback] = "angle_invalid_and_length_unavailable"

        frames = group["FRAME"].to_numpy(dtype=float)
        angle_d2, angle_d2_abs, d2_time_contiguous = local_second_difference_by_frame(z_angle, frames)
        length_d2, length_d2_abs, _ = local_second_difference_by_frame(z_length_for_qc, frames)
        angle_norm, angle_norm_low, angle_norm_high = robust_quantile_normalize(
            angle_d2_abs, trim_low_percent, trim_high_percent
        )
        length_norm, length_norm_low, length_norm_high = robust_quantile_normalize(
            length_d2_abs, trim_low_percent, trim_high_percent
        )
        d2_relative_percent = signed_symmetric_relative_percent(angle_norm, length_norm)

        angle_jump_candidate = (
            enabled
            & np.isfinite(d2_relative_percent)
            & np.isfinite(angle_norm)
            & (angle_norm >= angle_norm_min)
            & (d2_relative_percent >= relative_threshold_percent)
        )
        length_jump_candidate = (
            enabled
            & np.isfinite(d2_relative_percent)
            & np.isfinite(length_norm)
            & (length_norm >= angle_norm_min)
            & (d2_relative_percent <= -relative_threshold_percent)
        )
        angle_jump = nonmax_suppress_candidates(angle_jump_candidate, angle_norm, nms_radius)
        length_jump = nonmax_suppress_candidates(length_jump_candidate, length_norm, nms_radius)
        angle_only_jump = angle_jump & (~length_jump) & length_finite & (~fallback) & angle_available
        length_only_jump = length_jump & (~angle_jump)

        group["Z_Angle_Second_Diff"] = angle_d2
        group["Z_Length_Second_Diff"] = length_d2
        group["Z_Angle_Second_Diff_Abs"] = angle_d2_abs
        group["Z_Length_Second_Diff_Abs"] = length_d2_abs
        group["Z_Angle_Second_Diff_Normalized"] = angle_norm
        group["Z_Length_Second_Diff_Normalized"] = length_norm
        group["Angle_Length_D2_Relative_Percent"] = d2_relative_percent
        group["Z_Angle_vs_Length_D2_Relative_Percent"] = d2_relative_percent
        group["Z_Angle_Jump_Flag"] = angle_jump
        group["Z_Length_Jump_Flag"] = length_jump
        group["Angle_Only_Jump_Flag"] = angle_only_jump
        group["Length_Only_Jump_Flag"] = length_only_jump
        group["Z_Angle_D2_Norm_Low"] = angle_norm_low
        group["Z_Angle_D2_Norm_High"] = angle_norm_high
        group["Z_Length_D2_Norm_Low"] = length_norm_low
        group["Z_Length_D2_Norm_High"] = length_norm_high
        group["Z_QC_D2_Time_Contiguous"] = d2_time_contiguous

        n = len(group)
        for idx in np.where(angle_only_jump)[0]:
            left = None
            right = None
            for step in range(1, max_search + 1):
                cand = idx - step
                if (
                    cand >= 0
                    and not angle_jump[cand]
                    and not fallback[cand]
                    and np.isfinite(z_angle[cand])
                    and np.isfinite(z_length_for_qc[cand])
                ):
                    left = cand
                    break
            for step in range(1, max_search + 1):
                cand = idx + step
                if (
                    cand < n
                    and not angle_jump[cand]
                    and not fallback[cand]
                    and np.isfinite(z_angle[cand])
                    and np.isfinite(z_length_for_qc[cand])
                ):
                    right = cand
                    break
            if left is None or right is None:
                confidence[idx] = "low"
                reason[idx] = "angle_only_jump_no_bracketing_frames"
                continue

            z_value, mode, ratio = interpolate_by_length_trend(z_angle, z_length_for_qc, idx, left, right)
            final[idx] = z_value
            source[idx] = "angle_length_trend_interpolated"
            confidence[idx] = "medium_high"
            hard_qc_candidate[idx] = True
            hard_qc_target[idx] = z_value
            hard_qc_source[idx] = "angle_length_trend_interpolated"
            hard_qc_confidence[idx] = "medium_high"
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Left_Frame")] = group["FRAME"].iloc[left]
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Right_Frame")] = group["FRAME"].iloc[right]
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Mode")] = mode
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Ratio")] = ratio
            reason[idx] = (
                f"{'parent_segment_boundary; ' if parent_boundary[idx] else ''}"
                f"angle_only_jump; angle_d2_norm={angle_norm[idx]:.3f}; "
                f"length_d2_norm={length_norm[idx]:.3f}; "
                f"d2_relative={d2_relative_percent[idx]:.1f}%; mode={mode}"
            )

        both_jump = angle_jump & length_jump
        confidence[both_jump] = "low"
        reason[both_jump] = "angle_and_length_both_jump"
        length_only = length_only_jump & (~fallback)
        reason[length_only] = "length_only_jump_ignored"

        group["Z_Final"] = final
        group["Z_Final_Source"] = source
        group["Z_Confidence"] = confidence
        group["Angle_Length_QC_Reason"] = reason
        group["Z_Hard_QC_Candidate"] = hard_qc_candidate
        group["Z_Hard_QC_Target"] = hard_qc_target
        group["Z_Hard_QC_Target_Source"] = hard_qc_source
        group["Z_Hard_QC_Target_Confidence"] = hard_qc_confidence
        groups.append(group)

    out = pd.concat(groups, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)
    out = apply_v4_gate_v2_correction(out)
    out = apply_parent_stitching_boundary_alignment(out)
    out["Predicted_Z"] = out["Z_Final"]
    return out


def apply_angle_affine_model(res_df: pd.DataFrame, angle_model: dict | None) -> pd.DataFrame:
    df = res_df.sort_values(["Cluster_ID", "FRAME"]).copy()
    df["Z_Angle_Affine_Uncorrected"] = np.nan
    df["Z_Angle_Affine"] = np.nan
    df["Z_Angle_Primary_Branch"] = ""
    df["Z_Angle_Secondary_Branch"] = ""
    df["Theta_Primary_Deg"] = np.nan
    df["Theta_Secondary_Deg"] = np.nan
    df["Theta_Primary_Unwrapped_Deg"] = np.nan
    df["Theta_Secondary_Unwrapped_Deg"] = np.nan

    if not angle_model:
        return df

    primary = str(angle_model.get("primary_diag_branch", "a")).lower()
    secondary = str(angle_model.get("secondary_diag_branch", "b")).lower()
    primary_col = "Diag1_Angle_Deg" if primary == "a" else "Diag2_Angle_Deg"
    secondary_col = "Diag1_Angle_Deg" if secondary == "a" else "Diag2_Angle_Deg"
    params = angle_model.get("angle_hybrid_model", {})
    affine = angle_model.get("affine_correction", {})
    if not params or "primary_coefficients" not in params:
        return df

    groups: list[pd.DataFrame] = []
    for _, group in df.groupby("Cluster_ID", sort=False):
        group = group.sort_values("FRAME").copy()
        theta_primary = group[primary_col].to_numpy(dtype=float)
        theta_secondary = group[secondary_col].to_numpy(dtype=float)
        group["Z_Angle_Primary_Branch"] = primary
        group["Z_Angle_Secondary_Branch"] = secondary
        group["Theta_Primary_Deg"] = theta_primary
        group["Theta_Secondary_Deg"] = theta_secondary
        group["Theta_Primary_Unwrapped_Deg"] = unwrap_period180(theta_primary)
        group["Theta_Secondary_Unwrapped_Deg"] = unwrap_period180(theta_secondary)

        primary_coeff = np.asarray(params["primary_coefficients"], dtype=float)
        primary_pred = np.polyval(primary_coeff, group["Theta_Primary_Unwrapped_Deg"].to_numpy(dtype=float))
        residual_coeff = np.asarray(params.get("residual_coefficients", [0.0, 0.0, 0.0]), dtype=float)
        theta2_center = float(params.get("theta_secondary_center_deg", 0.0))
        size_center = float(params.get("mean_diag_center_px", 0.0))
        x_res = np.column_stack(
            [
                np.ones(len(group), dtype=float),
                group["Theta_Secondary_Unwrapped_Deg"].to_numpy(dtype=float) - theta2_center,
                group["Calculated_Size"].to_numpy(dtype=float) - size_center,
            ]
        )
        z_uncorrected = primary_pred + x_res @ residual_coeff
        a = float(affine.get("a", 1.0))
        b = float(affine.get("b", 0.0))
        group["Z_Angle_Affine_Uncorrected"] = z_uncorrected
        group["Z_Angle_Affine"] = a * z_uncorrected + b
        groups.append(group)

    return pd.concat(groups, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)


def local_residual(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    residual = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < 3:
        return residual
    prev_v = np.roll(arr, 1)
    next_v = np.roll(arr, -1)
    residual[1:-1] = np.abs(arr[1:-1] - (prev_v[1:-1] + next_v[1:-1]) / 2.0)
    return residual


def local_residual_percent(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    reference = np.full(len(arr), np.nan, dtype=float)
    residual = np.full(len(arr), np.nan, dtype=float)
    percent = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < 3:
        return residual, percent, reference
    prev_v = np.roll(arr, 1)
    next_v = np.roll(arr, -1)
    reference[1:-1] = (prev_v[1:-1] + next_v[1:-1]) / 2.0
    residual[1:-1] = np.abs(arr[1:-1] - reference[1:-1])
    denom = np.abs(arr) + np.abs(reference)
    valid = np.isfinite(residual) & np.isfinite(denom) & (denom > 1e-12)
    percent[valid] = 200.0 * residual[valid] / denom[valid]
    return residual, percent, reference


def local_second_difference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    d2 = np.full(len(arr), np.nan, dtype=float)
    abs_d2 = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < 3:
        return d2, abs_d2
    d2[1:-1] = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    abs_d2[1:-1] = np.abs(d2[1:-1])
    return d2, abs_d2


def local_second_difference_by_frame(
    values: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return second differences only where the parent timeline is consecutive."""
    arr = np.asarray(values, dtype=float)
    time = np.asarray(frames, dtype=float)
    d2 = np.full(len(arr), np.nan, dtype=float)
    abs_d2 = np.full(len(arr), np.nan, dtype=float)
    contiguous = np.zeros(len(arr), dtype=bool)
    if len(arr) < 3:
        return d2, abs_d2, contiguous

    valid = (
        np.isfinite(time[:-2])
        & np.isfinite(time[1:-1])
        & np.isfinite(time[2:])
        & np.isclose(time[1:-1] - time[:-2], 1.0)
        & np.isclose(time[2:] - time[1:-1], 1.0)
    )
    middle = np.arange(1, len(arr) - 1)
    valid_middle = middle[valid]
    d2[valid_middle] = arr[valid_middle + 1] - 2.0 * arr[valid_middle] + arr[valid_middle - 1]
    abs_d2[valid_middle] = np.abs(d2[valid_middle])
    contiguous[valid_middle] = True
    return d2, abs_d2, contiguous


def robust_jump_threshold(residual: np.ndarray, floor_value: float) -> float:
    finite = np.asarray(residual, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float(floor_value)
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    return float(max(floor_value, median + 6.0 * 1.4826 * mad))


def robust_quantile_normalize(values: np.ndarray, low_percent: float, high_percent: float) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    norm = np.full(len(arr), np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if len(finite) == 0:
        return norm, np.nan, np.nan

    low_p = float(np.clip(low_percent, 0.0, 49.0))
    high_p = float(np.clip(high_percent, low_p + 1.0, 100.0))
    low_value = float(np.nanpercentile(finite, low_p))
    high_value = float(np.nanpercentile(finite, high_p))
    scale = high_value - low_value
    if not np.isfinite(scale) or scale <= 1e-12:
        low_value = float(np.nanmin(finite))
        high_value = float(np.nanmax(finite))
        scale = high_value - low_value

    if np.isfinite(scale) and scale > 1e-12:
        norm_values = (arr[finite_mask] - low_value) / scale
        norm[finite_mask] = np.maximum(norm_values, 0.0)
    else:
        norm[finite_mask] = 0.0
    return norm, low_value, high_value


def signed_symmetric_relative_percent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    out = np.full(len(aa), np.nan, dtype=float)
    denom = np.abs(aa) + np.abs(bb)
    valid = np.isfinite(aa) & np.isfinite(bb) & np.isfinite(denom) & (denom > 1e-12)
    out[valid] = 200.0 * (aa[valid] - bb[valid]) / denom[valid]
    return out


def interpolate_by_angle_trend(z_length: np.ndarray, z_angle: np.ndarray, idx: int, left: int, right: int) -> tuple[float, str, float]:
    denom = float(z_angle[right] - z_angle[left])
    if np.isfinite(denom) and abs(denom) >= 0.2:
        ratio = float((z_angle[idx] - z_angle[left]) / denom)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        z_value = float(z_length[left] + ratio * (z_length[right] - z_length[left]))
        return z_value, "angle_trend", ratio
    ratio = float((idx - left) / max(right - left, 1))
    z_value = float(z_length[left] + ratio * (z_length[right] - z_length[left]))
    return z_value, "frame_linear_angle_flat", ratio


def apply_length_angle_trend_fusion(res_df: pd.DataFrame) -> pd.DataFrame:
    df = res_df.sort_values(["Cluster_ID", "FRAME"]).copy()
    enabled = env_bool("Z_LENGTH_ANGLE_TREND_FUSION_ENABLED", True)
    trim_low_percent = env_float("Z_D2_NORMALIZATION_TRIM_LOW_PERCENT", 5.0)
    trim_high_percent = env_float("Z_D2_NORMALIZATION_TRIM_HIGH_PERCENT", 95.0)
    relative_threshold_percent = env_float("Z_LENGTH_ANGLE_D2_RELATIVE_THRESHOLD_PERCENT", 90.0)
    length_norm_min = env_float("Z_LENGTH_D2_NORMALIZED_MIN", 1.0)
    max_search = int(max(1, env_float("Z_TREND_INTERP_MAX_SEARCH_FRAMES", 3.0)))
    nms_radius = int(max(0, env_float("Z_LENGTH_JUMP_NMS_RADIUS", 2.0)))

    df["Z_Length"] = df["Predicted_Z"].to_numpy(dtype=float)
    df["Z_Final"] = df["Z_Length"].to_numpy(dtype=float)
    df["Z_Final_Source"] = "length"
    df["Z_Confidence"] = "high"
    df["Z_Length_Local_Residual"] = np.nan
    df["Z_Angle_Local_Residual"] = np.nan
    df["Z_Length_Local_Reference"] = np.nan
    df["Z_Angle_Local_Reference"] = np.nan
    df["Z_Length_Local_Residual_Percent"] = np.nan
    df["Z_Angle_Local_Residual_Percent"] = np.nan
    df["Z_Length_Second_Diff"] = np.nan
    df["Z_Angle_Second_Diff"] = np.nan
    df["Z_Length_Second_Diff_Abs"] = np.nan
    df["Z_Angle_Second_Diff_Abs"] = np.nan
    df["Z_Length_Second_Diff_Normalized"] = np.nan
    df["Z_Angle_Second_Diff_Normalized"] = np.nan
    df["Z_Length_vs_Angle_D2_Relative_Percent"] = np.nan
    df["Z_Length_D2_Normalized_Min"] = length_norm_min
    df["Z_Length_Angle_D2_Relative_Threshold_Percent"] = relative_threshold_percent
    df["Z_D2_Norm_Trim_Low_Percent"] = trim_low_percent
    df["Z_D2_Norm_Trim_High_Percent"] = trim_high_percent
    df["Z_Length_D2_Norm_Low"] = np.nan
    df["Z_Length_D2_Norm_High"] = np.nan
    df["Z_Angle_D2_Norm_Low"] = np.nan
    df["Z_Angle_D2_Norm_High"] = np.nan
    df["Z_Length_Jump_Flag"] = False
    df["Z_Angle_Jump_Flag"] = False
    df["Z_Length_Only_Jump_Flag"] = False
    df["Z_Trend_Interp_Left_Frame"] = np.nan
    df["Z_Trend_Interp_Right_Frame"] = np.nan
    df["Z_Trend_Interp_Mode"] = ""
    df["Z_Trend_Interp_Ratio"] = np.nan
    df["Z_Trend_Fusion_Reason"] = "disabled" if not enabled else "ok"

    if not enabled or "Z_Angle_Affine" not in df.columns:
        df["Predicted_Z"] = df["Z_Final"]
        return df

    groups: list[pd.DataFrame] = []
    for _, group in df.groupby("Cluster_ID", sort=False):
        group = group.sort_values("FRAME").copy()
        z_length = group["Z_Length"].to_numpy(dtype=float)
        z_angle = group["Z_Angle_Affine"].to_numpy(dtype=float)
        domain_ok = group.get("Z_Model_Domain_OK", pd.Series(True, index=group.index)).to_numpy(dtype=bool)
        length_residual, length_residual_percent, length_reference = local_residual_percent(z_length)
        angle_residual, angle_residual_percent, angle_reference = local_residual_percent(z_angle)
        length_d2, length_d2_abs = local_second_difference(z_length)
        angle_d2, angle_d2_abs = local_second_difference(z_angle)
        length_d2_norm, length_norm_low, length_norm_high = robust_quantile_normalize(
            length_d2_abs, trim_low_percent, trim_high_percent
        )
        angle_d2_norm, angle_norm_low, angle_norm_high = robust_quantile_normalize(
            angle_d2_abs, trim_low_percent, trim_high_percent
        )
        d2_relative_percent = signed_symmetric_relative_percent(length_d2_norm, angle_d2_norm)

        length_jump_candidate = (
            np.isfinite(d2_relative_percent)
            & np.isfinite(length_d2_norm)
            & (length_d2_norm >= length_norm_min)
            & (d2_relative_percent >= relative_threshold_percent)
        )
        length_jump = np.zeros(len(group), dtype=bool)
        for idx in np.where(length_jump_candidate)[0]:
            lo = max(0, idx - nms_radius)
            hi = min(len(group), idx + nms_radius + 1)
            local = length_d2_norm[lo:hi]
            local_max = np.nanmax(local) if np.isfinite(local).any() else np.nan
            if np.isfinite(local_max) and length_d2_norm[idx] >= local_max - 1e-12:
                length_jump[idx] = True
        angle_jump = (
            np.isfinite(d2_relative_percent)
            & np.isfinite(angle_d2_norm)
            & (angle_d2_norm >= length_norm_min)
            & (d2_relative_percent <= -relative_threshold_percent)
        )
        only_length_jump = length_jump & (~angle_jump) & np.isfinite(z_angle)

        group["Z_Length_Local_Residual"] = length_residual
        group["Z_Angle_Local_Residual"] = angle_residual
        group["Z_Length_Local_Reference"] = length_reference
        group["Z_Angle_Local_Reference"] = angle_reference
        group["Z_Length_Local_Residual_Percent"] = length_residual_percent
        group["Z_Angle_Local_Residual_Percent"] = angle_residual_percent
        group["Z_Length_Second_Diff"] = length_d2
        group["Z_Angle_Second_Diff"] = angle_d2
        group["Z_Length_Second_Diff_Abs"] = length_d2_abs
        group["Z_Angle_Second_Diff_Abs"] = angle_d2_abs
        group["Z_Length_Second_Diff_Normalized"] = length_d2_norm
        group["Z_Angle_Second_Diff_Normalized"] = angle_d2_norm
        group["Z_Length_vs_Angle_D2_Relative_Percent"] = d2_relative_percent
        group["Z_Length_Jump_Flag"] = length_jump
        group["Z_Angle_Jump_Flag"] = angle_jump
        group["Z_Length_Only_Jump_Flag"] = only_length_jump
        group["Z_Length_D2_Normalized_Min"] = length_norm_min
        group["Z_Length_Angle_D2_Relative_Threshold_Percent"] = relative_threshold_percent
        group["Z_D2_Norm_Trim_Low_Percent"] = trim_low_percent
        group["Z_D2_Norm_Trim_High_Percent"] = trim_high_percent
        group["Z_Length_D2_Norm_Low"] = length_norm_low
        group["Z_Length_D2_Norm_High"] = length_norm_high
        group["Z_Angle_D2_Norm_Low"] = angle_norm_low
        group["Z_Angle_D2_Norm_High"] = angle_norm_high

        final = z_length.copy()
        source = group["Z_Final_Source"].to_numpy(dtype=object)
        confidence = group["Z_Confidence"].to_numpy(dtype=object)
        reason = group["Z_Trend_Fusion_Reason"].to_numpy(dtype=object)

        n = len(group)
        for idx in np.where(only_length_jump)[0]:
            left = None
            right = None
            for step in range(1, max_search + 1):
                cand = idx - step
                if cand >= 0 and not only_length_jump[cand] and not length_jump[cand] and np.isfinite(z_length[cand]) and np.isfinite(z_angle[cand]):
                    left = cand
                    break
            for step in range(1, max_search + 1):
                cand = idx + step
                if cand < n and not only_length_jump[cand] and not length_jump[cand] and np.isfinite(z_length[cand]) and np.isfinite(z_angle[cand]):
                    right = cand
                    break
            if left is None or right is None:
                source[idx] = "length"
                confidence[idx] = "low"
                reason[idx] = "length_only_jump_no_bracketing_frames"
                continue

            z_value, mode, ratio = interpolate_by_angle_trend(z_length, z_angle, idx, left, right)
            final[idx] = z_value
            source[idx] = "length_angle_trend_interpolated"
            confidence[idx] = "medium_high"
            left_frame = group["FRAME"].iloc[left]
            right_frame = group["FRAME"].iloc[right]
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Left_Frame")] = left_frame
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Right_Frame")] = right_frame
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Mode")] = mode
            group.iloc[idx, group.columns.get_loc("Z_Trend_Interp_Ratio")] = ratio
            reason[idx] = (
                f"length_only_jump; length_d2_norm={length_d2_norm[idx]:.3f}; "
                f"angle_d2_norm={angle_d2_norm[idx]:.3f}; "
                f"d2_relative={d2_relative_percent[idx]:.1f}%; "
                f"length_residual={length_residual[idx]:.3f}um/"
                f"{length_residual_percent[idx]:.1f}%; mode={mode}"
            )

        both_jump = length_jump & angle_jump
        confidence[both_jump] = "low"
        reason[both_jump] = "length_and_angle_both_jump"
        angle_only_jump = (~length_jump) & angle_jump
        confidence[angle_only_jump] = "medium"
        reason[angle_only_jump] = "angle_only_jump_ignored"

        group["Z_Final"] = final
        group["Z_Final_Source"] = source
        group["Z_Confidence"] = confidence
        group["Z_Trend_Fusion_Reason"] = reason
        groups.append(group)

    out = pd.concat(groups, ignore_index=True).sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True)
    out["Predicted_Z"] = out["Z_Final"]
    return out


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"WARNING: invalid {name}={value!r}; using {default}")
        return default


def finite_factor(value: float) -> bool:
    return np.isfinite(value) and 0.8 <= value <= 1.2


def size_in_model_domain(size_value: float, poly_coeff: np.ndarray | None, model: dict | None) -> bool:
    if poly_coeff is None or model is None or not np.isfinite(size_value):
        return True
    domain = model.get("real_correction", {}).get("domain_z_uncorrected_um")
    if not domain or len(domain) != 2:
        return True
    z_uncorrected = float(np.polyval(poly_coeff, size_value))
    return float(domain[0]) <= z_uncorrected <= float(domain[1])


def apply_diagonal_qc(
    res_df: pd.DataFrame,
    objective_key: str,
    poly_coeff: np.ndarray | None = None,
    model: dict | None = None,
) -> tuple[pd.DataFrame, int, float, float]:
    """Replace anomalous mean diagonal with an adjusted single-diagonal estimate.

    The trigger is based only on the percentage mismatch between Diag1 and
    Diag2. Adjacent frames are used only after triggering, to decide which one
    of Diag1/Diag2 is more credible for this frame. The selected raw diagonal
    is converted back to the expected mean(Diag1, Diag2) using objective-level
    PSF statistics or robust current-data ratios.
    """
    df = res_df.sort_values(["Cluster_ID", "FRAME"]).copy()
    enabled = env_bool("Z_DIAG_QC_ENABLED", False)
    manual_threshold = env_float("Z_DIAG_QC_PERCENT_THRESHOLD", 0.0)
    mean_multiplier = env_float("Z_DIAG_QC_MEAN_MULTIPLIER", 2.0)

    df["Calculated_Size_Raw"] = df["Calculated_Size"].to_numpy(dtype=float)
    df["Diag_QC_Applied"] = False
    df["Diag_QC_Trusted_Diagonal"] = "mean"
    df["Diag_QC_Neighbor_Reference"] = np.nan
    df["Diag_QC_Diag_Abs_Diff"] = np.abs(df["Diag1"].to_numpy(dtype=float) - df["Diag2"].to_numpy(dtype=float))
    denom = np.maximum((df["Diag1"].to_numpy(dtype=float) + df["Diag2"].to_numpy(dtype=float)) / 2.0, 1e-12)
    df["Diag_QC_Diag_Percent_Diff"] = df["Diag_QC_Diag_Abs_Diff"].to_numpy(dtype=float) / denom * 100.0
    percent_values = df["Diag_QC_Diag_Percent_Diff"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    current_mean_percent = float(np.mean(percent_values)) if len(percent_values) else np.nan
    objective_threshold = DIAG_QC_DEFAULT_PERCENT_THRESHOLDS.get(objective_key, 8.0)
    if manual_threshold > 0:
        percent_threshold = manual_threshold
        threshold_source = "manual"
    else:
        current_mean_threshold = current_mean_percent * mean_multiplier if np.isfinite(current_mean_percent) else objective_threshold
        percent_threshold = max(objective_threshold, current_mean_threshold)
        threshold_source = f"auto=max(objective={objective_threshold:.3f}, current_mean*{mean_multiplier:.3f}={current_mean_threshold:.3f})"
    df["Diag_QC_Percent_Threshold"] = percent_threshold
    df["Diag_QC_Current_Mean_Percent_Diff"] = current_mean_percent
    df["Diag_QC_Domain_Guard_Applied"] = False
    df["Diag_QC_Reason"] = "disabled" if not enabled else "ok"
    default_factors = DIAG_QC_DEFAULT_MEAN_FACTORS.get(objective_key, {"Diag1": 1.0, "Diag2": 1.0})
    diag1_factor = float(default_factors.get("Diag1", 1.0))
    diag2_factor = float(default_factors.get("Diag2", 1.0))
    ratio_source = "objective_real_psf_median"
    ratio_min_samples = int(max(3, env_float("Z_DIAG_QC_RATIO_MIN_SAMPLES", 10.0)))
    ratio_mask = (
        np.isfinite(df["Calculated_Size_Raw"].to_numpy(dtype=float))
        & np.isfinite(df["Diag1"].to_numpy(dtype=float))
        & np.isfinite(df["Diag2"].to_numpy(dtype=float))
        & np.isfinite(df["Diag_QC_Diag_Percent_Diff"].to_numpy(dtype=float))
        & (df["Diag_QC_Diag_Percent_Diff"].to_numpy(dtype=float) <= objective_threshold)
    )
    ratio_sample_count = int(np.sum(ratio_mask))
    if ratio_sample_count >= ratio_min_samples:
        current_diag1_factor = float(
            np.nanmedian(
                df.loc[ratio_mask, "Calculated_Size_Raw"].to_numpy(dtype=float)
                / np.maximum(df.loc[ratio_mask, "Diag1"].to_numpy(dtype=float), 1e-12)
            )
        )
        current_diag2_factor = float(
            np.nanmedian(
                df.loc[ratio_mask, "Calculated_Size_Raw"].to_numpy(dtype=float)
                / np.maximum(df.loc[ratio_mask, "Diag2"].to_numpy(dtype=float), 1e-12)
            )
        )
        if finite_factor(current_diag1_factor) and finite_factor(current_diag2_factor):
            diag1_factor = current_diag1_factor
            diag2_factor = current_diag2_factor
            ratio_source = "current_nontriggered_median"

    df["Diag_QC_Mean_From_Diag1_Factor"] = diag1_factor
    df["Diag_QC_Mean_From_Diag2_Factor"] = diag2_factor
    df["Diag_QC_Ratio_Source"] = ratio_source
    df["Diag_QC_Ratio_Sample_Count"] = ratio_sample_count
    df["Diag_QC_Adjusted_Size_From_Trusted_Diag"] = np.nan

    if not enabled or df.empty:
        return df.sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True), 0, percent_threshold, current_mean_percent

    corrected_count = 0
    corrected_groups: list[pd.DataFrame] = []
    for _, group in df.groupby("Cluster_ID", sort=False):
        group = group.sort_values("FRAME").copy()
        raw_size = group["Calculated_Size_Raw"].to_numpy(dtype=float)
        diag1 = group["Diag1"].to_numpy(dtype=float)
        diag2 = group["Diag2"].to_numpy(dtype=float)
        diag_percent_diff = group["Diag_QC_Diag_Percent_Diff"].to_numpy(dtype=float)
        prev_size = np.roll(raw_size, 1)
        next_size = np.roll(raw_size, -1)
        prev_size[0] = np.nan
        next_size[-1] = np.nan

        has_neighbors = np.isfinite(prev_size) & np.isfinite(next_size)
        neighbor_ref = np.full_like(raw_size, np.nan, dtype=float)
        neighbor_ref[has_neighbors] = (prev_size[has_neighbors] + next_size[has_neighbors]) / 2.0

        group["Diag_QC_Neighbor_Reference"] = neighbor_ref

        trigger = (
            has_neighbors
            & np.isfinite(raw_size)
            & np.isfinite(diag1)
            & np.isfinite(diag2)
            & np.isfinite(neighbor_ref)
            & np.isfinite(diag_percent_diff)
            & (diag_percent_diff >= percent_threshold)
        )

        for local_idx in np.where(trigger)[0]:
            ref = neighbor_ref[local_idx]
            d1_adjusted = diag1[local_idx] * diag1_factor
            d2_adjusted = diag2[local_idx] * diag2_factor
            d1_err = abs(d1_adjusted - ref)
            d2_err = abs(d2_adjusted - ref)
            if d1_err <= d2_err:
                trusted_name = "Diag1"
                trusted_size = d1_adjusted
            else:
                trusted_name = "Diag2"
                trusted_size = d2_adjusted

            row_index = group.index[local_idx]
            raw_domain_ok = size_in_model_domain(raw_size[local_idx], poly_coeff, model)
            trusted_domain_ok = size_in_model_domain(trusted_size, poly_coeff, model)
            reason_prefix = (
                f"diag_percent_diff={diag_percent_diff[local_idx]:.3f}%>={percent_threshold:.3f}%; "
                f"{threshold_source}; selected={trusted_name}; ratio_source={ratio_source}"
            )
            if raw_domain_ok and not trusted_domain_ok:
                group.at[row_index, "Diag_QC_Domain_Guard_Applied"] = True
                group.at[row_index, "Diag_QC_Trusted_Diagonal"] = trusted_name
                group.at[row_index, "Diag_QC_Adjusted_Size_From_Trusted_Diag"] = trusted_size
                group.at[row_index, "Diag_QC_Reason"] = (
                    reason_prefix
                    + "; selected candidate would leave spline domain, kept mean"
                )
                continue

            group.at[row_index, "Calculated_Size"] = trusted_size
            group.at[row_index, "Diag_QC_Applied"] = True
            group.at[row_index, "Diag_QC_Trusted_Diagonal"] = trusted_name
            group.at[row_index, "Diag_QC_Adjusted_Size_From_Trusted_Diag"] = trusted_size
            group.at[
                row_index,
                "Diag_QC_Reason",
            ] = reason_prefix
            corrected_count += 1

        corrected_groups.append(group)

    out = pd.concat(corrected_groups, ignore_index=True)
    return out.sort_values(["Cluster_ID", "FRAME"]).reset_index(drop=True), corrected_count, percent_threshold, current_mean_percent


def estimate_mean_from_diagonal_factors(res_df: pd.DataFrame, objective_key: str) -> tuple[float, float, str, int]:
    """Estimate raw single-diagonal -> expected mean-diagonal factors."""
    default_factors = DIAG_QC_DEFAULT_MEAN_FACTORS.get(objective_key, {"Diag1": 1.0, "Diag2": 1.0})
    diag1_factor = float(default_factors.get("Diag1", 1.0))
    diag2_factor = float(default_factors.get("Diag2", 1.0))
    ratio_source = "objective_real_psf_median"

    if res_df.empty or not {"Calculated_Size", "Diag1", "Diag2"}.issubset(res_df.columns):
        return diag1_factor, diag2_factor, ratio_source, 0

    objective_threshold = DIAG_QC_DEFAULT_PERCENT_THRESHOLDS.get(objective_key, 8.0)
    ratio_min_samples = int(max(3, env_float("Z_DIAG_QC_RATIO_MIN_SAMPLES", 10.0)))
    diag1 = res_df["Diag1"].to_numpy(dtype=float)
    diag2 = res_df["Diag2"].to_numpy(dtype=float)
    size = res_df["Calculated_Size"].to_numpy(dtype=float)
    denom = np.maximum((diag1 + diag2) / 2.0, 1e-12)
    percent_diff = np.abs(diag1 - diag2) / denom * 100.0
    ratio_mask = (
        np.isfinite(size)
        & np.isfinite(diag1)
        & np.isfinite(diag2)
        & np.isfinite(percent_diff)
        & (percent_diff <= objective_threshold)
    )
    ratio_sample_count = int(np.sum(ratio_mask))
    if ratio_sample_count >= ratio_min_samples:
        current_diag1_factor = float(np.nanmedian(size[ratio_mask] / np.maximum(diag1[ratio_mask], 1e-12)))
        current_diag2_factor = float(np.nanmedian(size[ratio_mask] / np.maximum(diag2[ratio_mask], 1e-12)))
        if finite_factor(current_diag1_factor) and finite_factor(current_diag2_factor):
            diag1_factor = current_diag1_factor
            diag2_factor = current_diag2_factor
            ratio_source = "current_nontriggered_median"
    return diag1_factor, diag2_factor, ratio_source, ratio_sample_count


def apply_single_diagonal_mean_scaling(res_df: pd.DataFrame, objective_key: str) -> tuple[pd.DataFrame, int]:
    """Scale single diagonal length to the expected two-diagonal mean size."""
    df = res_df.copy()
    df["Single_Diagonal_Raw_Length_Px"] = df.get("Single_Diagonal_Length_Px", np.nan)
    df["Single_Diagonal_Mean_Factor"] = np.nan
    df["Single_Diagonal_Equivalent_Mean_Size_Px"] = np.nan
    df["Single_Diagonal_Scale_Applied"] = False
    df["Single_Diagonal_Scale_Source"] = ""

    if df.empty or "Geometry_Mode" not in df.columns:
        return df, 0
    if not env_bool("Z_SINGLE_DIAGONAL_MEAN_SCALING_ENABLED", True):
        df["Single_Diagonal_Scale_Source"] = "disabled"
        return df, 0

    diag1_factor, diag2_factor, ratio_source, sample_count = estimate_mean_from_diagonal_factors(df, objective_key)
    single_mask = df["Geometry_Mode"].astype(str).eq("single_diagonal")
    if "Single_Diagonal_Name" not in df.columns:
        return df, 0

    raw_length = pd.to_numeric(df["Single_Diagonal_Length_Px"], errors="coerce")
    missing_raw = ~np.isfinite(raw_length.to_numpy(dtype=float))
    raw_length.loc[missing_raw] = pd.to_numeric(df.loc[missing_raw, "Calculated_Size"], errors="coerce")
    diag_name = df["Single_Diagonal_Name"].astype(str).str.lower()
    factors = np.where(diag_name.eq("diag2"), diag2_factor, diag1_factor).astype(float)
    scaled = raw_length.to_numpy(dtype=float) * factors
    scale_mask = single_mask.to_numpy(dtype=bool) & np.isfinite(raw_length.to_numpy(dtype=float)) & np.isfinite(scaled)

    df.loc[scale_mask, "Single_Diagonal_Mean_Factor"] = factors[scale_mask]
    df.loc[scale_mask, "Single_Diagonal_Equivalent_Mean_Size_Px"] = scaled[scale_mask]
    df.loc[scale_mask, "Calculated_Size"] = scaled[scale_mask]
    df.loc[scale_mask, "Calculated_Size_Source"] = "single_diagonal_mean_scaled_length"
    df.loc[scale_mask, "Single_Diagonal_Scale_Applied"] = True
    df.loc[scale_mask, "Single_Diagonal_Scale_Source"] = f"{ratio_source}; samples={sample_count}"
    return df, int(np.sum(scale_mask))


def infer_diagonal_pair_map(
    target_df: pd.DataFrame,
    z_x_col: str,
    z_y_col: str,
    coord_scale: float,
    tracking_group_col: str,
) -> dict:
    pair_counts = defaultdict(lambda: {"diag1": Counter(), "diag2": Counter()})
    group_cols = [tracking_group_col, col_cluster, col_frame] if tracking_group_col != col_cluster else [col_cluster, col_frame]
    for keys, group in target_df.groupby(group_cols, dropna=False):
        if tracking_group_col == col_cluster:
            tracking_id = keys[0]
            cluster_id = tracking_id
        else:
            tracking_id = keys[0]
            cluster_id = keys[1]
            if pd.isna(tracking_id):
                tracking_id = cluster_id
        coords = group[[z_x_col, z_y_col]].to_numpy(dtype=float) / coord_scale
        finite = np.isfinite(coords).all(axis=1)
        if "Z_Use_For_Geometry" in group.columns:
            finite &= group["Z_Use_For_Geometry"].fillna(True).astype(bool).to_numpy()
        elif "Subspot_Virtual" in group.columns:
            finite &= ~group["Subspot_Virtual"].fillna(False).astype(bool).to_numpy()
        if int(np.sum(finite)) != 4:
            continue
        clean_group = group.loc[finite]
        clean_coords = coords[finite]
        track_ids = clean_group["TRACK_ID"].to_numpy() if "TRACK_ID" in clean_group.columns else np.arange(len(clean_group))
        geom = diagonal_geometry(clean_coords, track_ids=track_ids)
        sorted_track_ids = geom.get("sorted_track_ids")
        if sorted_track_ids is None or len(sorted_track_ids) != 4:
            continue
        diag1_pair = sorted_pair_key(sorted_track_ids[0], sorted_track_ids[2])
        diag2_pair = sorted_pair_key(sorted_track_ids[1], sorted_track_ids[3])
        if diag1_pair:
            pair_counts[tracking_id]["diag1"][diag1_pair] += 1
        if diag2_pair:
            pair_counts[tracking_id]["diag2"][diag2_pair] += 1

    out = {}
    for cluster_id, counters in pair_counts.items():
        out[cluster_id] = {}
        for diag_name in ("diag1", "diag2"):
            if counters[diag_name]:
                out[cluster_id][diag_name] = counters[diag_name].most_common(1)[0][0]
    return out


def fill_present_spot_columns(row: dict, coords: np.ndarray, track_ids: np.ndarray) -> None:
    for idx in range(4):
        row[f"Spot{idx}_X_Px"] = np.nan
        row[f"Spot{idx}_Y_Px"] = np.nan
        row[f"Spot{idx}_Track_ID"] = np.nan
    if len(coords) == 0:
        return
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    order = np.argsort(angles)
    for spot_idx, coord_idx in enumerate(order[:4]):
        row[f"Spot{spot_idx}_X_Px"] = float(coords[coord_idx, 0])
        row[f"Spot{spot_idx}_Y_Px"] = float(coords[coord_idx, 1])
        row[f"Spot{spot_idx}_Track_ID"] = track_ids[coord_idx]


def single_diagonal_from_group(
    group: pd.DataFrame,
    coords: np.ndarray,
    track_ids: np.ndarray,
    known_pairs: dict,
) -> dict | None:
    coord_by_track = {}
    original_track_by_key = {}
    for coord, track_id in zip(coords, track_ids):
        key = normalize_track_id(track_id)
        if key is None or not np.isfinite(coord).all():
            continue
        coord_by_track[key] = coord
        original_track_by_key[key] = track_id

    candidates = []
    for diag_name, pair in known_pairs.items():
        if pair and pair[0] in coord_by_track and pair[1] in coord_by_track:
            p = coord_by_track[pair[0]]
            q = coord_by_track[pair[1]]
            candidates.append(
                {
                    "name": diag_name,
                    "p": p,
                    "q": q,
                    "track_a": original_track_by_key[pair[0]],
                    "track_b": original_track_by_key[pair[1]],
                    "source": "historical_track_pair",
                }
            )

    if not candidates and len(coords) >= 2:
        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if best is None or dist > best[0]:
                    best = (dist, i, j)
        if best is not None:
            _, i, j = best
            candidates.append(
                {
                    "name": "diag1",
                    "p": coords[i],
                    "q": coords[j],
                    "track_a": track_ids[i],
                    "track_b": track_ids[j],
                    "source": "farthest_available_pair",
                }
            )

    if not candidates:
        return None
    geom = candidates[0]
    length = float(np.linalg.norm(geom["p"] - geom["q"]))
    angle = angle_mod180(geom["p"], geom["q"])
    return {
        "name": geom["name"],
        "length": length,
        "angle": angle,
        "track_a": geom["track_a"],
        "track_b": geom["track_b"],
        "source": geom["source"],
    }


def extract_size_metrics(target_df: pd.DataFrame, coord_scale: float) -> tuple[pd.DataFrame, str, str]:
    z_x_col, z_y_col = resolve_z_coordinate_columns(target_df)
    required_cols = [col_cluster, col_frame, z_x_col, z_y_col]
    missing = [col for col in required_cols if col not in target_df.columns]
    if missing:
        raise ValueError(f"Target CSV missing required columns: {missing}; actual columns: {target_df.columns.tolist()}")

    results = []
    tracking_group_col = resolve_z_tracking_group_column(target_df)
    diagonal_pair_map = infer_diagonal_pair_map(target_df, z_x_col, z_y_col, coord_scale, tracking_group_col)
    grouped = target_df.groupby([col_cluster, col_frame])
    for (cluster_id, frame), group in grouped:
        z_tracking_group_id = tracking_group_value(group, cluster_id, tracking_group_col)
        coords = group[[z_x_col, z_y_col]].to_numpy(dtype=float) / coord_scale
        finite = np.isfinite(coords).all(axis=1)
        if "Z_Use_For_Geometry" in group.columns:
            finite &= group["Z_Use_For_Geometry"].fillna(True).astype(bool).to_numpy()
        elif "Subspot_Virtual" in group.columns:
            finite &= ~group["Subspot_Virtual"].fillna(False).astype(bool).to_numpy()
        if int(np.sum(finite)) < 2:
            continue
        clean_group = group.loc[finite]
        coords = coords[finite]
        track_ids = clean_group["TRACK_ID"].to_numpy() if "TRACK_ID" in clean_group.columns else np.arange(len(clean_group), dtype=float)
        spot_count = len(clean_group)
        if spot_count == 4:
            geom = diagonal_geometry(coords, track_ids=track_ids)
            calculated_size = geom["Calculated_Size"]
            diag1 = geom["Diag1"]
            diag2 = geom["Diag2"]
            diag1_angle = geom["Diag1_Angle_Deg"]
            diag2_angle = geom["Diag2_Angle_Deg"]
            geometry_mode = "four_spot"
            geometry_success = True
            calculated_size_source = "mean_two_diagonals"
            single_diag = None
        else:
            single_diag = single_diagonal_from_group(
                clean_group,
                coords,
                track_ids,
                diagonal_pair_map.get(z_tracking_group_id, {}),
            )
            if single_diag is None:
                continue
            calculated_size = single_diag["length"]
            diag1 = single_diag["length"] if single_diag["name"] != "diag2" else np.nan
            diag2 = single_diag["length"] if single_diag["name"] == "diag2" else np.nan
            diag1_angle = single_diag["angle"] if single_diag["name"] != "diag2" else np.nan
            diag2_angle = single_diag["angle"] if single_diag["name"] == "diag2" else np.nan
            geometry_mode = "single_diagonal"
            geometry_success = True
            calculated_size_source = "single_diagonal_length"
        row = {
            "Cluster_ID": cluster_id,
            "FRAME": frame,
            "Z_Tracking_Group_ID": z_tracking_group_id,
            "Z_Tracking_Group_Column": tracking_group_col,
            "Calculated_Size": calculated_size,
            "Diag1": diag1,
            "Diag2": diag2,
            "Diag1_Angle_Deg": diag1_angle,
            "Diag2_Angle_Deg": diag2_angle,
            "Z_Input_X_Column": z_x_col,
            "Z_Input_Y_Column": z_y_col,
            "Available_Spot_Count": spot_count,
            "Geometry_Mode": geometry_mode,
            "Geometry_Success": geometry_success,
            "Calculated_Size_Source": calculated_size_source,
            "Single_Diagonal_Name": single_diag["name"] if single_diag else "",
            "Single_Diagonal_Length_Px": single_diag["length"] if single_diag else np.nan,
            "Single_Diagonal_Angle_Deg": single_diag["angle"] if single_diag else np.nan,
            "Single_Diagonal_Track_ID_A": single_diag["track_a"] if single_diag else np.nan,
            "Single_Diagonal_Track_ID_B": single_diag["track_b"] if single_diag else np.nan,
            "Single_Diagonal_Source": single_diag["source"] if single_diag else "",
        }
        if spot_count == 4:
            sorted_coords = geom["sorted_coords"]
            sorted_track_ids = geom.get("sorted_track_ids", np.arange(4, dtype=float))
            for idx in range(4):
                row[f"Spot{idx}_X_Px"] = float(sorted_coords[idx, 0])
                row[f"Spot{idx}_Y_Px"] = float(sorted_coords[idx, 1])
                row[f"Spot{idx}_Track_ID"] = sorted_track_ids[idx]
        else:
            fill_present_spot_columns(row, coords, track_ids)
        results.append(row)
    return pd.DataFrame(results), z_x_col, z_y_col


def plot_distribution(res_df: pd.DataFrame) -> None:
    z_vals = res_df["Predicted_Z"].dropna().to_numpy(dtype=float)
    if len(z_vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z_vals, bins=min(30, max(5, len(z_vals) // 5)), color="#59a14f", alpha=0.75, edgecolor="white", lw=0.5)
    z_mean, z_std = float(np.mean(z_vals)), float(np.std(z_vals))
    ax.axvline(z_mean, color="#e15759", ls="--", lw=1.5, label=f"Mean = {z_mean:.2f} um")
    ax.axvline(z_mean - z_std, color="#e15759", ls=":", lw=1, alpha=0.5)
    ax.axvline(z_mean + z_std, color="#e15759", ls=":", lw=1, alpha=0.5, label=f"Std = {z_std:.2f} um")
    ax.set_title(f"Predicted Z-Depth Distribution (n={len(z_vals)})")
    ax.set_xlabel("Predicted Z Position (um, phasor angle primary)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.savefig("predicted_z_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("--- Z step: phasor angle-primary spline with phasor length backup QC ---")
    objective_config = load_default_objective_config()
    current_camera_pixel_um = float(os.getenv("Z_COORD_SCALE", "4.5"))
    model_reference_camera_pixel_um = float(os.getenv("Z_MODEL_REFERENCE_CAMERA_PIXEL_UM", "4.5"))
    if current_camera_pixel_um <= 0 or model_reference_camera_pixel_um <= 0:
        raise ValueError("Z_COORD_SCALE and Z_MODEL_REFERENCE_CAMERA_PIXEL_UM must both be positive.")
    length_payload = load_model_payload()
    objective_key, length_model = resolve_objective(length_payload, objective_config)
    angle_payload = load_angle_payload()
    angle_objective_key, angle_model = resolve_angle_model(angle_payload, objective_config)

    print(f"Objective config:        {objective_config}")
    print(f"Resolved length model:   {objective_key}")
    print(f"Resolved angle model:    {angle_objective_key or 'unavailable'}")
    print(f"Length backup model:     {length_payload.get('model_name', 'phasor_length_backup')}")
    print(f"Angle primary model:     {angle_payload.get('model_name', 'unavailable') if angle_payload else 'unavailable'}")
    print(f"Current camera scale:    {current_camera_pixel_um} um/current pixel")
    print(f"Model reference scale:   {model_reference_camera_pixel_um} um/reference pixel")
    print(
        "Z geometry conversion: sensor um / model reference scale; "
        "raw-image sampling: sensor um / current camera scale"
    )
    print(f"Length spline domain:    {length_model['real_correction']['domain_z_uncorrected_um']}")
    if angle_model:
        print(f"Angle signal:            {angle_model.get('angle_signal')} (sigma={angle_model.get('blur_sigma_px')} px)")
        print(f"Angle theta domain:      {angle_model['angle_spline']['domain_theta_deg']}")
    print(f"Spline out-of-domain:    {os.getenv('Z_SPLINE_OUT_OF_DOMAIN', 'extrapolate')}")
    print(f"Angle spline OOD:        {os.getenv('Z_ANGLE_SPLINE_OUT_OF_DOMAIN', 'linear')}")
    print(f"Angle-length D2 QC:      {os.getenv('Z_ANGLE_LENGTH_D2_QC_ENABLED', 'true')}")

    target_df = read_pipeline_csv(target_file)
    target_df.columns = [c.strip() for c in target_df.columns]
    print(f"Loaded target file: {target_file}, rows={len(target_df)}")

    res_df, z_x_col, z_y_col = extract_size_metrics(target_df, model_reference_camera_pixel_um)
    print(f"Z coordinates: {z_x_col}, {z_y_col}")
    print(f"Extracted {len(res_df)} valid Z-geometry groups.")
    if res_df.empty:
        print("WARNING: no valid 4-spot groups; cannot estimate Z.")
        return
    res_df, single_scale_count = apply_single_diagonal_mean_scaling(res_df, objective_key)
    length_poly_coeff = np.asarray(length_model["simulation_poly2_coefficients"], dtype=float)
    res_df, diag_qc_count, diag_qc_threshold, diag_qc_current_mean = apply_diagonal_qc(
        res_df,
        objective_key,
        poly_coeff=length_poly_coeff,
        model=length_model,
    )
    res_df = add_spot_quality_raw_features(
        res_df,
        target_df,
        model_reference_camera_pixel_um,
        z_x_col,
        z_y_col,
        image_coord_scale=current_camera_pixel_um,
    )
    res_df["Z_Input_Camera_Pixel_Um"] = current_camera_pixel_um
    res_df["Z_Model_Reference_Pixel_Um"] = model_reference_camera_pixel_um
    res_df["Z_Geometry_Input_Conversion"] = "sensor_um_div_model_reference_pixel_um"

    res_df = apply_length_backup_model(res_df, length_model)
    res_df = apply_angle_primary_model(res_df, angle_model)
    res_df["Z_Calibration_Objective"] = angle_objective_key or objective_key
    res_df["Z_Calibration_Method"] = angle_payload.get("model_name", "phasor_angle_tracked") if angle_payload else "missing_angle_model"
    res_df["Z_Localization_Method"] = "phasor_blur"
    res_df["Z_Length_Backup_Method"] = length_payload.get("model_name", "phasor_length_backup")

    res_df = apply_angle_length_d2_qc(res_df)
    res_df = apply_two_point_trusted_frame_interpolation(res_df)

    res_df.to_csv(output_file, index=False)
    angle_domain_fraction = float(np.mean(res_df["Z_Angle_Model_Domain_OK"].to_numpy(dtype=bool))) if len(res_df) else np.nan
    length_domain_fraction = float(np.mean(res_df["Z_Length_Model_Domain_OK"].to_numpy(dtype=bool))) if len(res_df) else np.nan
    interp_count = int((res_df.get("Z_Hard_QC_Candidate", pd.Series(dtype=bool))).sum())
    v4_accept_count = int((res_df.get("Z_QC_V4_Gate_Accepted", pd.Series(dtype=bool))).sum())
    length_fallback_count = int(res_df.get("Z_Final_Source", pd.Series(dtype=str)).astype(str).str.startswith("length_fallback").sum())
    angle_extrapolated_count = int(res_df.get("Z_Final_Source", pd.Series(dtype=str)).astype(str).str.contains("extrapolated", na=False).sum())
    single_diag_count = int((res_df.get("Geometry_Mode", pd.Series(dtype=str)) == "single_diagonal").sum())
    low_conf_count = int((res_df.get("Z_Confidence", "") == "low").sum())
    print(f"Saved axial positions to: {output_file}")
    print(f"Angle model-domain fraction:  {angle_domain_fraction:.3f}")
    print(f"Length model-domain fraction: {length_domain_fraction:.3f}")
    print(f"Hard QC candidate frames: {interp_count}")
    print(f"V4 gate + V2 accepted frames: {v4_accept_count}")
    print(f"Angle extrapolated frames: {angle_extrapolated_count}")
    print(f"Single-diagonal degraded frames: {single_diag_count}")
    print(f"Single-diagonal mean-scaled frames: {single_scale_count}")
    print(f"Diagonal QC corrected frames: {diag_qc_count} (threshold={diag_qc_threshold:.3f}%, current_mean={diag_qc_current_mean:.3f}%)")
    print(f"Length fallback frames: {length_fallback_count}")
    print(f"Low-confidence frames: {low_conf_count}")
    print(res_df.head())

    plot_distribution(res_df)
    print("Saved predicted_z_distribution.png")


if __name__ == "__main__":
    main()
