from __future__ import annotations

import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from .localization_methods import localize_four_spots, load_tiff_stack, sort_spots_by_angle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
OUTPUT_SUFFIX = "_tracked"

METHODS = [
    "centroid",
    "phasor",
    "gaussian2d",
    "elliptical_gaussian",
    "radial_symmetry",
    "trackmate_log",
]
BLUR_SIGMAS = [0.0, 0.6, 1.0, 1.4, 1.8, 2.4]
ANGLE_SIGNALS = ["tracked_diag0", "tracked_diag1", "diag_a", "diag_b", "quad_rotation", "pca"]


@dataclass(frozen=True)
class ObjectiveSpec:
    key: str
    sim_tif: Path
    real_dir: Path
    check_dir: Path
    z_values_um: np.ndarray
    sim_work_min_um: float
    sim_work_max_um: float


OBJECTIVES = [
    ObjectiveSpec(
        key="20x_0.45NA_air",
        sim_tif=DATA_DIR / "simulation" / "20x_0.45NA_air" / "1.tif",
        real_dir=DATA_DIR / "real" / "20x_0.45NA_air",
        check_dir=DATA_DIR / "independent_real" / "20x_0.45NA_air",
        z_values_um=np.arange(-10, 11, dtype=float),
        sim_work_min_um=-10.0,
        sim_work_max_um=10.0,
    ),
    ObjectiveSpec(
        key="40x_0.6NA_air",
        sim_tif=DATA_DIR / "simulation" / "40x_0.6NA_air" / "1.tif",
        real_dir=DATA_DIR / "real" / "40x_0.6NA_air",
        check_dir=DATA_DIR / "independent_real" / "40x_0.6NA_air",
        z_values_um=np.arange(-10, 11, dtype=float),
        sim_work_min_um=-5.0,
        sim_work_max_um=5.0,
    ),
    ObjectiveSpec(
        key="60x_1.0NA_water",
        sim_tif=DATA_DIR / "simulation" / "60x_1.0NA_water" / "1.tif",
        real_dir=DATA_DIR / "real" / "60x_1.0NA_water",
        check_dir=DATA_DIR / "independent_real" / "60x_1.0NA_water",
        z_values_um=np.arange(-10, 11, dtype=float),
        sim_work_min_um=-2.5,
        sim_work_max_um=2.5,
    ),
]


def ensure_dirs() -> None:
    for path in [
        RESULTS_DIR,
        FIGURES_DIR,
        FIGURES_DIR / "theta_curves",
        FIGURES_DIR / "z_predictions",
        FIGURES_DIR / "heatmaps",
        FIGURES_DIR / "overlays",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def numeric_file_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else 0, path.name.lower())


def load_single_tif(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.float32)


def preprocess_for_angle(image: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if sigma <= 0:
        return arr
    return ndimage.gaussian_filter(arr, sigma=float(sigma)).astype(np.float32, copy=False)


def line_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    if not np.isfinite(dx + dy) or (abs(dx) < 1e-12 and abs(dy) < 1e-12):
        return np.nan
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def periodic_diff_deg(curr: float, prev: float, period: float = 180.0) -> float:
    return float(((curr - prev + period / 2.0) % period) - period / 2.0)


def unwrap_period180(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    prev_raw = np.nan
    prev_unwrapped = np.nan
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        if not np.isfinite(prev_raw):
            out[idx] = value
        else:
            out[idx] = prev_unwrapped + periodic_diff_deg(value, prev_raw, 180.0)
        prev_raw = value
        prev_unwrapped = out[idx]
    return out


def unwrap_to_reference(raw_angle_deg: float, prev_unwrapped_deg: float) -> float:
    return float(prev_unwrapped_deg + periodic_diff_deg(raw_angle_deg, prev_unwrapped_deg % 180.0, 180.0))


def track_unordered_diagonal_angles(angle_a_deg: np.ndarray, angle_b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track two diagonal-angle branches when raw diag_a/diag_b labels can swap.

    The two diagonals are treated as an unlabeled pair at each frame. For every
    new frame, choose the assignment that minimizes the total 180-periodic angle
    change from the previous tracked branches.
    """
    a = np.asarray(angle_a_deg, dtype=float)
    b = np.asarray(angle_b_deg, dtype=float)
    tracked0 = np.full(len(a), np.nan, dtype=float)
    tracked1 = np.full(len(a), np.nan, dtype=float)
    prev0 = np.nan
    prev1 = np.nan

    for idx, (aval, bval) in enumerate(zip(a, b)):
        if not np.isfinite(aval + bval):
            continue
        if not np.isfinite(prev0 + prev1):
            tracked0[idx] = aval
            tracked1[idx] = bval
        else:
            a_to_0 = unwrap_to_reference(aval, prev0)
            b_to_1 = unwrap_to_reference(bval, prev1)
            b_to_0 = unwrap_to_reference(bval, prev0)
            a_to_1 = unwrap_to_reference(aval, prev1)
            cost_keep = abs(a_to_0 - prev0) + abs(b_to_1 - prev1)
            cost_swap = abs(b_to_0 - prev0) + abs(a_to_1 - prev1)
            if cost_swap < cost_keep:
                tracked0[idx] = b_to_0
                tracked1[idx] = a_to_1
            else:
                tracked0[idx] = a_to_0
                tracked1[idx] = b_to_1
        prev0 = tracked0[idx]
        prev1 = tracked1[idx]
    return tracked0, tracked1


def line_orientation_mean_deg(angles_deg: list[float]) -> float:
    vals = np.asarray([v for v in angles_deg if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return np.nan
    vec = np.exp(2j * np.deg2rad(vals))
    angle = 0.5 * np.rad2deg(np.angle(np.nanmean(vec)))
    return float(angle % 180.0)


def pca_angle_deg(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return np.nan
    pts = pts - np.nanmean(pts, axis=0)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, int(np.nanargmax(eigvals))]
        return float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)
    except Exception:
        return np.nan


def geometry_from_spots(spots: list[Any]) -> dict[str, float]:
    if len(spots) != 4:
        return {
            "diag_a_angle_deg": np.nan,
            "diag_b_angle_deg": np.nan,
            "quad_rotation_deg": np.nan,
            "pca_angle_deg": np.nan,
            "mean_diag_px": np.nan,
            "diag_orthogonality_error_deg": np.nan,
        }
    ordered = sort_spots_by_angle(spots)
    pts = np.asarray([[s.x, s.y] for s in ordered], dtype=float)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return {
            "diag_a_angle_deg": np.nan,
            "diag_b_angle_deg": np.nan,
            "quad_rotation_deg": np.nan,
            "pca_angle_deg": np.nan,
            "mean_diag_px": np.nan,
            "diag_orthogonality_error_deg": np.nan,
        }
    diag_a = float(np.linalg.norm(pts[0] - pts[2]))
    diag_b = float(np.linalg.norm(pts[1] - pts[3]))
    angle_a = line_angle_deg(pts[0], pts[2])
    angle_b = line_angle_deg(pts[1], pts[3])
    quad_rotation = line_orientation_mean_deg([angle_a, angle_b - 90.0])
    if np.isfinite(angle_a + angle_b):
        diff = abs(periodic_diff_deg(angle_b, angle_a, 180.0))
        diff = min(diff, 180.0 - diff)
        ortho_error = abs(90.0 - diff)
    else:
        ortho_error = np.nan
    return {
        "diag_a_angle_deg": angle_a,
        "diag_b_angle_deg": angle_b,
        "quad_rotation_deg": quad_rotation,
        "pca_angle_deg": pca_angle_deg(pts),
        "mean_diag_px": float((diag_a + diag_b) / 2.0),
        "diag_orthogonality_error_deg": ortho_error,
    }


def monotonic_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    d = np.diff(arr[np.isfinite(arr)])
    if len(d) == 0:
        return np.nan
    pos = float(np.mean(d >= 0))
    neg = float(np.mean(d <= 0))
    return max(pos, neg)


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(arr * arr)))


def mae(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.nanmean(np.abs(arr)))


def smoothness_rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return np.nan
    return rms(arr[2:] - 2 * arr[1:-1] + arr[:-2])


def fit_pchip_angle_to_z(theta: np.ndarray, z: np.ndarray) -> tuple[PchipInterpolator | None, str, int]:
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(theta) & np.isfinite(z)
    theta = theta[mask]
    z = z[mask]
    if len(theta) < 4:
        return None, "too_few_points", int(len(theta))
    order = np.argsort(theta)
    theta = theta[order]
    z = z[order]
    grouped = pd.DataFrame({"theta": theta, "z": z}).groupby("theta", as_index=False).mean()
    if len(grouped) < 4:
        return None, "too_few_unique_theta", int(len(grouped))
    try:
        return PchipInterpolator(grouped["theta"].to_numpy(float), grouped["z"].to_numpy(float), extrapolate=True), "ok", int(len(grouped))
    except Exception as exc:
        return None, str(exc), int(len(grouped))


def predict_with_model(model: PchipInterpolator | None, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    out = np.full(len(theta), np.nan, dtype=float)
    if model is None:
        return out
    mask = np.isfinite(theta)
    if mask.any():
        out[mask] = model(theta[mask])
    return out


def loocv_angle_to_z(theta: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)
    out = np.full(len(theta), np.nan, dtype=float)
    valid = np.where(np.isfinite(theta) & np.isfinite(z))[0]
    for idx in valid:
        train = valid[valid != idx]
        if len(train) < 4:
            continue
        model, status, _ = fit_pchip_angle_to_z(theta[train], z[train])
        if model is None or status != "ok":
            continue
        out[idx] = float(model(theta[idx]))
    err = out - z
    metrics = {
        "loocv_n_predicted": float(np.isfinite(out).sum()),
        "loocv_z_rmse_um": rms(err),
        "loocv_z_mae_um": mae(err),
        "loocv_z_max_abs_um": float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else np.nan,
    }
    return out, metrics


def step_metrics(frame: np.ndarray, z_pred: np.ndarray, start_frame: int, end_frame: int) -> dict[str, float]:
    frame = np.asarray(frame, dtype=int)
    z_pred = np.asarray(z_pred, dtype=float)
    mask = (frame >= start_frame) & (frame <= end_frame) & np.isfinite(z_pred)
    order = np.argsort(frame[mask])
    f = frame[mask][order]
    z = z_pred[mask][order]
    if len(z) < 2:
        return {
            "n_frames": float(len(z)),
            "n_steps": 0.0,
            "step_rmse_um": np.nan,
            "step_mae_um": np.nan,
            "step_std_um": np.nan,
            "direction_consistency": np.nan,
        }
    dz = np.diff(z)
    frame_step = np.diff(f)
    target = frame_step.astype(float)
    err = dz - target
    return {
        "n_frames": float(len(z)),
        "n_steps": float(len(dz)),
        "step_rmse_um": rms(err),
        "step_mae_um": mae(err),
        "step_std_um": float(np.nanstd(err)) if len(err) else np.nan,
        "direction_consistency": float(np.nanmean(dz > 0)) if len(dz) else np.nan,
    }


def read_ranges() -> tuple[pd.DataFrame, pd.DataFrame]:
    manual = pd.read_csv(ROOT / "manual_work_ranges.csv")
    check = pd.read_csv(ROOT / "independent_check_work_ranges.csv")
    return manual, check


def range_for_objective(manual: pd.DataFrame, objective: str, method: str) -> tuple[int, int]:
    rows = manual[(manual["objective"] == objective) & (manual["method"].isin([method, "*"]))]
    if rows.empty:
        rows = manual[manual["objective"] == objective]
    rows = rows[rows["use_for_correction"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if rows.empty:
        raise RuntimeError(f"No correction range for {objective}/{method}")
    row = rows.iloc[0]
    return int(row["use_start_frame"]), int(row["use_end_frame"])


def check_range_for_objective(check: pd.DataFrame, objective: str, method: str) -> tuple[int, int]:
    rows = check[(check["objective"] == objective) & (check["method"].isin([method, "*"]))]
    if rows.empty:
        rows = check[check["objective"] == objective]
    rows = rows[rows["use_for_validation"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if rows.empty:
        raise RuntimeError(f"No independent validation range for {objective}/{method}")
    row = rows.iloc[0]
    return int(row["start_frame"]), int(row["end_frame"])


def localize_image_records(
    *,
    objective: ObjectiveSpec,
    dataset: str,
    frame_index: int,
    image: np.ndarray,
    image_name: str,
    z_true_um: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for blur_sigma in BLUR_SIGMAS:
        blurred = preprocess_for_angle(image, blur_sigma)
        rough_peaks = None
        for method in METHODS:
            loc = localize_four_spots(blurred, objective.key, method, rough_peaks=rough_peaks)
            if rough_peaks is None and len(loc.spots) == 4:
                rough_peaks = [(float(s.x), float(s.y), float(s.quality if np.isfinite(s.quality) else 1.0)) for s in loc.spots]
            geom = geometry_from_spots(loc.spots)
            base: dict[str, Any] = {
                "objective": objective.key,
                "dataset": dataset,
                "frame_index": int(frame_index),
                "image_name": image_name,
                "z_true_um": float(z_true_um) if np.isfinite(z_true_um) else np.nan,
                "blur_sigma_px": float(blur_sigma),
                "method": method,
                "success": bool(loc.success),
                "center_x": loc.center_x,
                "center_y": loc.center_y,
                "runtime_ms": loc.runtime_ms,
                "quality": loc.quality,
                "residual": loc.residual,
                "message": loc.message,
                **geom,
            }
            for idx, spot in enumerate(loc.spots[:4]):
                base[f"spot{idx}_x"] = spot.x
                base[f"spot{idx}_y"] = spot.y
                base[f"spot{idx}_success"] = bool(spot.success)
            rows.append(base)
    return rows


def run_localization() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        print(f"localizing simulation: {objective.key}")
        for frame_idx, image in load_tiff_stack(objective.sim_tif):
            if frame_idx >= len(objective.z_values_um):
                break
            records.extend(
                localize_image_records(
                    objective=objective,
                    dataset="simulation",
                    frame_index=int(frame_idx),
                    image=image,
                    image_name=f"stack_frame_{frame_idx}",
                    z_true_um=float(objective.z_values_um[frame_idx]),
                )
            )
        print(f"localizing real calibration: {objective.key}")
        for frame_idx, path in enumerate(sorted(objective.real_dir.glob("*.tif"), key=numeric_file_key)):
            records.extend(
                localize_image_records(
                    objective=objective,
                    dataset="real_calibration",
                    frame_index=int(frame_idx),
                    image=load_single_tif(path),
                    image_name=path.name,
                    z_true_um=np.nan,
                )
            )
        print(f"localizing independent check: {objective.key}")
        for frame_idx, path in enumerate(sorted(objective.check_dir.glob("*.tif"), key=numeric_file_key)):
            records.extend(
                localize_image_records(
                    objective=objective,
                    dataset="independent_check",
                    frame_index=int(frame_idx),
                    image=load_single_tif(path),
                    image_name=path.name,
                    z_true_um=np.nan,
                )
            )
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "angle_blur_localization_results.csv", index=False)
    return df


def add_unwrapped_angles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    angle_map = {
        "diag_a": "diag_a_angle_deg",
        "diag_b": "diag_b_angle_deg",
        "quad_rotation": "quad_rotation_deg",
        "pca": "pca_angle_deg",
    }
    for signal, col in angle_map.items():
        out[f"theta_{signal}_deg"] = out[col].astype(float)
        out[f"theta_{signal}_unwrapped_deg"] = np.nan
    out["theta_tracked_diag0_deg"] = np.nan
    out["theta_tracked_diag1_deg"] = np.nan
    out["theta_tracked_diag0_unwrapped_deg"] = np.nan
    out["theta_tracked_diag1_unwrapped_deg"] = np.nan
    group_cols = ["objective", "dataset", "method", "blur_sigma_px"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        sub = out.loc[idx].sort_values("frame_index")
        tracked0, tracked1 = track_unordered_diagonal_angles(
            sub["diag_a_angle_deg"].to_numpy(dtype=float),
            sub["diag_b_angle_deg"].to_numpy(dtype=float),
        )
        out.loc[sub.index, "theta_tracked_diag0_deg"] = tracked0 % 180.0
        out.loc[sub.index, "theta_tracked_diag1_deg"] = tracked1 % 180.0
        out.loc[sub.index, "theta_tracked_diag0_unwrapped_deg"] = tracked0
        out.loc[sub.index, "theta_tracked_diag1_unwrapped_deg"] = tracked1
        for signal in ["diag_a", "diag_b", "quad_rotation", "pca"]:
            values = sub[f"theta_{signal}_deg"].to_numpy(dtype=float)
            out.loc[sub.index, f"theta_{signal}_unwrapped_deg"] = unwrap_period180(values)
    out.to_csv(RESULTS_DIR / "angle_blur_localization_with_angles.csv", index=False)
    return out


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual, check_ranges = read_ranges()
    model_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    for objective in OBJECTIVES:
        for method in METHODS:
            cal_start, cal_end = range_for_objective(manual, objective.key, method)
            chk_start, chk_end = check_range_for_objective(check_ranges, objective.key, method)
            for blur_sigma in BLUR_SIGMAS:
                base_mask = (df["objective"] == objective.key) & (df["method"] == method) & (df["blur_sigma_px"] == blur_sigma)
                for signal in ANGLE_SIGNALS:
                    theta_col = f"theta_{signal}_unwrapped_deg"
                    sim = df[base_mask & (df["dataset"] == "simulation")].sort_values("frame_index")
                    sim_work = sim[(sim["z_true_um"] >= objective.sim_work_min_um) & (sim["z_true_um"] <= objective.sim_work_max_um)]
                    sim_model, sim_status, sim_n = fit_pchip_angle_to_z(sim_work[theta_col].to_numpy(float), sim_work["z_true_um"].to_numpy(float))
                    sim_pred = predict_with_model(sim_model, sim_work[theta_col].to_numpy(float))
                    sim_err = sim_pred - sim_work["z_true_um"].to_numpy(float)
                    _, sim_loo = loocv_angle_to_z(sim_work[theta_col].to_numpy(float), sim_work["z_true_um"].to_numpy(float))
                    sim_rows.append(
                        {
                            "objective": objective.key,
                            "method": method,
                            "blur_sigma_px": blur_sigma,
                            "angle_signal": signal,
                            "model_status": sim_status,
                            "n_fit_points": sim_n,
                            "simulation_z_rmse_um": rms(sim_err),
                            "simulation_z_mae_um": mae(sim_err),
                            "simulation_z_loocv_rmse_um": sim_loo["loocv_z_rmse_um"],
                            "simulation_z_loocv_mae_um": sim_loo["loocv_z_mae_um"],
                            "simulation_z_loocv_max_abs_um": sim_loo["loocv_z_max_abs_um"],
                            "simulation_z_loocv_n_predicted": sim_loo["loocv_n_predicted"],
                            "simulation_theta_monotonic_fraction": monotonic_fraction(sim_work[theta_col].to_numpy(float)),
                            "simulation_theta_smoothness_rms_deg": smoothness_rms(sim_work[theta_col].to_numpy(float)),
                        }
                    )

                    cal = df[base_mask & (df["dataset"] == "real_calibration")].sort_values("frame_index")
                    cal_range = cal[(cal["frame_index"] >= cal_start) & (cal["frame_index"] <= cal_end)].copy()
                    cal_range["z_target_um"] = cal_range["frame_index"].astype(float) - float(cal_start)
                    model, status, n_fit = fit_pchip_angle_to_z(cal_range[theta_col].to_numpy(float), cal_range["z_target_um"].to_numpy(float))

                    cal_pred = predict_with_model(model, cal[theta_col].to_numpy(float))
                    chk = df[base_mask & (df["dataset"] == "independent_check")].sort_values("frame_index")
                    chk_pred = predict_with_model(model, chk[theta_col].to_numpy(float))
                    _, cal_loo = loocv_angle_to_z(cal_range[theta_col].to_numpy(float), cal_range["z_target_um"].to_numpy(float))

                    cal_metrics = step_metrics(cal["frame_index"].to_numpy(int), cal_pred, cal_start, cal_end)
                    chk_metrics = step_metrics(chk["frame_index"].to_numpy(int), chk_pred, chk_start, chk_end)
                    theta_cal = cal_range[theta_col].to_numpy(float)
                    model_rows.append(
                        {
                            "objective": objective.key,
                            "method": method,
                            "blur_sigma_px": blur_sigma,
                            "angle_signal": signal,
                            "calibration_start_frame": cal_start,
                            "calibration_end_frame": cal_end,
                            "independent_start_frame": chk_start,
                            "independent_end_frame": chk_end,
                            "model_status": status,
                            "n_fit_points": n_fit,
                            "theta_monotonic_fraction": monotonic_fraction(theta_cal),
                            "theta_smoothness_rms_deg": smoothness_rms(theta_cal),
                            "calibration_step_rmse_um": cal_metrics["step_rmse_um"],
                            "calibration_step_mae_um": cal_metrics["step_mae_um"],
                            "calibration_direction_consistency": cal_metrics["direction_consistency"],
                            "calibration_z_loocv_rmse_um": cal_loo["loocv_z_rmse_um"],
                            "calibration_z_loocv_mae_um": cal_loo["loocv_z_mae_um"],
                            "calibration_z_loocv_max_abs_um": cal_loo["loocv_z_max_abs_um"],
                            "calibration_z_loocv_n_predicted": cal_loo["loocv_n_predicted"],
                            "independent_step_rmse_um": chk_metrics["step_rmse_um"],
                            "independent_step_mae_um": chk_metrics["step_mae_um"],
                            "independent_step_std_um": chk_metrics["step_std_um"],
                            "independent_direction_consistency": chk_metrics["direction_consistency"],
                            "independent_n_steps": chk_metrics["n_steps"],
                        }
                    )

                    for source_df, z_pred, dataset_name in [(cal, cal_pred, "real_calibration"), (chk, chk_pred, "independent_check")]:
                        for row, pred in zip(source_df.to_dict("records"), z_pred):
                            in_range = False
                            if dataset_name == "real_calibration":
                                in_range = cal_start <= int(row["frame_index"]) <= cal_end
                            else:
                                in_range = chk_start <= int(row["frame_index"]) <= chk_end
                            pred_rows.append(
                                {
                                    "objective": objective.key,
                                    "dataset": dataset_name,
                                    "frame_index": int(row["frame_index"]),
                                    "image_name": row["image_name"],
                                    "method": method,
                                    "blur_sigma_px": blur_sigma,
                                    "angle_signal": signal,
                                    "theta_unwrapped_deg": row[theta_col],
                                    "z_pred_um": pred,
                                    "in_metric_range": bool(in_range),
                                    "model_status": status,
                                }
                            )

    models = pd.DataFrame(model_rows)
    predictions = pd.DataFrame(pred_rows)
    sim_metrics = pd.DataFrame(sim_rows)

    merged = models.merge(
        sim_metrics[
            [
                "objective",
                "method",
                "blur_sigma_px",
                "angle_signal",
                "simulation_z_rmse_um",
                "simulation_z_mae_um",
                "simulation_z_loocv_rmse_um",
                "simulation_z_loocv_mae_um",
                "simulation_z_loocv_max_abs_um",
                "simulation_z_loocv_n_predicted",
                "simulation_theta_monotonic_fraction",
                "simulation_theta_smoothness_rms_deg",
            ]
        ],
        on=["objective", "method", "blur_sigma_px", "angle_signal"],
        how="left",
    )
    for objective_key, sub in merged.groupby("objective"):
        ordered = sub.sort_values(
            [
                "independent_step_rmse_um",
                "calibration_z_loocv_rmse_um",
                "calibration_step_rmse_um",
                "simulation_z_loocv_rmse_um",
                "simulation_z_rmse_um",
                "theta_smoothness_rms_deg",
            ],
            na_position="last",
        ).reset_index(drop=True)
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        ranking_rows.extend(ordered.to_dict("records"))
    ranking = pd.DataFrame(ranking_rows)
    stable = ranking[
        (ranking["model_status"].eq("ok"))
        & (ranking["independent_direction_consistency"] >= 0.9)
        & (ranking["theta_monotonic_fraction"] >= 0.85)
        & (ranking["calibration_z_loocv_rmse_um"] <= 1.0)
    ].copy()
    stable_rows: list[dict[str, Any]] = []
    for _, sub in stable.groupby("objective"):
        ordered = sub.sort_values(
            ["independent_step_rmse_um", "calibration_z_loocv_rmse_um", "simulation_z_loocv_rmse_um"],
            na_position="last",
        ).reset_index(drop=True)
        ordered["stable_rank"] = np.arange(1, len(ordered) + 1)
        stable_rows.extend(ordered.to_dict("records"))
    stable_ranking = pd.DataFrame(stable_rows)

    models.to_csv(RESULTS_DIR / f"angle_only_real_calibrated_models{OUTPUT_SUFFIX}.csv", index=False)
    predictions.to_csv(RESULTS_DIR / f"angle_only_real_calibrated_predictions{OUTPUT_SUFFIX}.csv", index=False)
    sim_metrics.to_csv(RESULTS_DIR / f"angle_only_simulation_fit_metrics{OUTPUT_SUFFIX}.csv", index=False)
    ranking.to_csv(RESULTS_DIR / f"angle_only_method_blur_ranking{OUTPUT_SUFFIX}.csv", index=False)
    stable_ranking.to_csv(RESULTS_DIR / f"angle_only_stable_recommendation_ranking{OUTPUT_SUFFIX}.csv", index=False)
    return models, predictions, sim_metrics, ranking


def plot_heatmaps(ranking: pd.DataFrame) -> None:
    for objective_key in sorted(ranking["objective"].unique()):
        sub = ranking[ranking["objective"] == objective_key].copy()
        best_signal = (
            sub.sort_values("independent_step_rmse_um", na_position="last")
            .groupby(["method", "blur_sigma_px"], as_index=False)
            .first()
        )
        pivot = best_signal.pivot(index="method", columns="blur_sigma_px", values="independent_step_rmse_um")
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis_r")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Gaussian blur sigma (px)")
        ax.set_title(f"{objective_key}: independent step RMSE, best angle signal")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("RMSE (um)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "heatmaps" / f"{safe_name(objective_key)}_independent_rmse_heatmap.png")
        plt.close(fig)


def plot_best_curves(df: pd.DataFrame, predictions: pd.DataFrame, ranking: pd.DataFrame) -> None:
    for objective_key in sorted(ranking["objective"].unique()):
        best = ranking[ranking["objective"] == objective_key].sort_values("rank").iloc[0]
        method = best["method"]
        blur_sigma = float(best["blur_sigma_px"])
        signal = best["angle_signal"]
        theta_col = f"theta_{signal}_unwrapped_deg"

        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
        for dataset, style in [("real_calibration", "-"), ("independent_check", "--")]:
            sub = df[
                (df["objective"] == objective_key)
                & (df["dataset"] == dataset)
                & (df["method"] == method)
                & (df["blur_sigma_px"] == blur_sigma)
            ].sort_values("frame_index")
            if dataset == "real_calibration":
                sub = sub[(sub["frame_index"] >= int(best["calibration_start_frame"])) & (sub["frame_index"] <= int(best["calibration_end_frame"]))]
            else:
                sub = sub[(sub["frame_index"] >= int(best["independent_start_frame"])) & (sub["frame_index"] <= int(best["independent_end_frame"]))]
            ax.plot(sub["frame_index"], sub[theta_col], style, label=dataset)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Unwrapped angle (deg)")
        ax.set_title(f"{objective_key}: best pure-angle signal ({method}, sigma={blur_sigma:g}, {signal})")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "theta_curves" / f"{safe_name(objective_key)}_best_theta_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
        for dataset, style in [("real_calibration", "-"), ("independent_check", "--")]:
            pred = predictions[
                (predictions["objective"] == objective_key)
                & (predictions["dataset"] == dataset)
                & (predictions["method"] == method)
                & (predictions["blur_sigma_px"] == blur_sigma)
                & (predictions["angle_signal"] == signal)
                & (predictions["in_metric_range"])
            ].sort_values("frame_index")
            ax.plot(pred["frame_index"], pred["z_pred_um"], style, label=dataset)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Predicted relative Z (um)")
        ax.set_title(f"{objective_key}: pure-angle Z prediction ({method}, sigma={blur_sigma:g}, {signal})")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "z_predictions" / f"{safe_name(objective_key)}_best_z_prediction.png")
        plt.close(fig)


def plot_overlays(df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    for objective in OBJECTIVES:
        best = ranking[ranking["objective"] == objective.key].sort_values("rank").iloc[0]
        method = best["method"]
        blur_sigma = float(best["blur_sigma_px"])
        sample_path = sorted(objective.real_dir.glob("*.tif"), key=numeric_file_key)[int(best["calibration_start_frame"])]
        image = preprocess_for_angle(load_single_tif(sample_path), blur_sigma)
        loc = localize_four_spots(image, objective.key, method)
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
        lo, hi = np.nanpercentile(image, [0.5, 99.7])
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
        if len(loc.spots) == 4:
            ordered = sort_spots_by_angle(loc.spots)
            pts = np.asarray([[s.x, s.y] for s in ordered], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], c="tab:red", s=26)
            ax.plot([pts[0, 0], pts[2, 0]], [pts[0, 1], pts[2, 1]], color="tab:blue", lw=1.2)
            ax.plot([pts[1, 0], pts[3, 0]], [pts[1, 1], pts[3, 1]], color="tab:orange", lw=1.2)
        ax.set_title(f"{objective.key}: {method}, sigma={blur_sigma:g}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "overlays" / f"{safe_name(objective.key)}_best_overlay.png")
        plt.close(fig)


def write_report(ranking: pd.DataFrame) -> None:
    stable_path = RESULTS_DIR / f"angle_only_stable_recommendation_ranking{OUTPUT_SUFFIX}.csv"
    stable = pd.read_csv(stable_path) if stable_path.exists() else pd.DataFrame()
    lines = [
        "# Gaussian-blurred pure-angle Z validation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This isolated benchmark tests whether Z can be recovered from the planar rotation angle of the four-spot pattern after Gaussian smoothing.",
        "The Z model uses only angle signals: `tracked_diag0`, `tracked_diag1`, `diag_a`, `diag_b`, `quad_rotation`, and `pca`. Diagonal length is saved only as a diagnostic and is not used for Z fitting.",
        "",
        "Calibration ranges: 20X 11-34, 40X 5-16, 60X 8-17. Independent validation ranges: 20X 18-38, 40X 5-16, 60X 8-17.",
        "",
        "## Best Results By Objective",
        "",
    ]
    for objective_key in sorted(ranking["objective"].unique()):
        best = ranking[ranking["objective"] == objective_key].sort_values("rank").iloc[0]
        lines.extend(
            [
                f"### {objective_key}",
                "",
                f"- Best method: `{best['method']}`",
                f"- Blur sigma: `{float(best['blur_sigma_px']):g}` px",
                f"- Angle signal: `{best['angle_signal']}`",
                f"- Independent step RMSE: `{best['independent_step_rmse_um']:.4g}` um",
                f"- Independent step MAE: `{best['independent_step_mae_um']:.4g}` um",
                f"- Calibration leave-one-out Z RMSE: `{best['calibration_z_loocv_rmse_um']:.4g}` um",
                f"- Simulation leave-one-out Z RMSE: `{best['simulation_z_loocv_rmse_um']:.4g}` um",
                "",
            ]
        )
    if not stable.empty:
        lines.extend(["## Stable Recommendation Ranking", ""])
        lines.append("The table below rejects models with calibration leave-one-out Z RMSE above 1 um, direction consistency below 0.9, or calibration theta monotonic fraction below 0.85.")
        lines.append("")
        for objective_key in sorted(stable["objective"].unique()):
            best = stable[stable["objective"] == objective_key].sort_values("stable_rank").iloc[0]
            lines.extend(
                [
                    f"### {objective_key}",
                    "",
                    f"- Stable recommendation: `{best['method']}`",
                    f"- Blur sigma: `{float(best['blur_sigma_px']):g}` px",
                    f"- Angle signal: `{best['angle_signal']}`",
                    f"- Independent step RMSE: `{best['independent_step_rmse_um']:.4g}` um",
                    f"- Calibration leave-one-out Z RMSE: `{best['calibration_z_loocv_rmse_um']:.4g}` um",
                    "",
                ]
            )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `results/angle_blur_localization_results.csv`: raw four-spot localization results for each dataset, method, and blur sigma.",
            "- `results/angle_blur_localization_with_angles.csv`: localization results with raw and unwrapped angle signals.",
            f"- `results/angle_only_real_calibrated_models{OUTPUT_SUFFIX}.csv`: real-calibration pure-angle PCHIP model metrics.",
            f"- `results/angle_only_real_calibrated_predictions{OUTPUT_SUFFIX}.csv`: per-frame real calibration and independent validation predictions.",
            f"- `results/angle_only_simulation_fit_metrics{OUTPUT_SUFFIX}.csv`: simulation-only theta-to-Z fit metrics.",
            f"- `results/angle_only_method_blur_ranking{OUTPUT_SUFFIX}.csv`: ranking across method, blur sigma, and angle signal.",
            f"- `results/angle_only_stable_recommendation_ranking{OUTPUT_SUFFIX}.csv`: conservative ranking that rejects unstable interpolation cases.",
            "- `figures/heatmaps`: independent RMSE heatmaps.",
            "- `figures/theta_curves`: best theta curves.",
            "- `figures/z_predictions`: best pure-angle Z prediction curves.",
            "- `figures/overlays`: representative localization overlays.",
            "",
            "## Interpretation Notes",
            "",
            "A good pure-angle method should have low independent step RMSE and a smooth, mostly monotonic theta curve in the chosen working range.",
            "If smoothing improves a localization method, that supports the idea that angle extraction was limited by spot-coordinate noise.",
            "If all smoothed pure-angle results remain unstable, the angle-Z signal is likely weaker than the length-Z signal for this optical configuration.",
        ]
    )
    (ROOT / "angle_blur_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    objective_items = []
    for objective in OBJECTIVES:
        objective_items.append(
            {
                "key": objective.key,
                "sim_tif": str(objective.sim_tif),
                "real_dir": str(objective.real_dir),
                "check_dir": str(objective.check_dir),
                "z_values_um": objective.z_values_um.tolist(),
                "sim_work_min_um": objective.sim_work_min_um,
                "sim_work_max_um": objective.sim_work_max_um,
            }
        )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "methods": METHODS,
        "blur_sigmas": BLUR_SIGMAS,
        "angle_signals": ANGLE_SIGNALS,
        "objectives": objective_items,
        "notes": "Pure-angle validation with Gaussian smoothing; diagonal length is diagnostic only.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_manifest()
    loc_path = RESULTS_DIR / "angle_blur_localization_results.csv"
    if loc_path.exists():
        print("Using existing localization CSV.")
        loc_df = pd.read_csv(loc_path)
    else:
        loc_df = run_localization()
    angle_df = add_unwrapped_angles(loc_df)
    models, predictions, sim_metrics, ranking = evaluate_models(angle_df)
    plot_heatmaps(ranking)
    plot_best_curves(angle_df, predictions, ranking)
    plot_overlays(angle_df, ranking)
    write_report(ranking)
    print("Done.")
    print(ranking.sort_values(["objective", "rank"]).groupby("objective").head(3)[[
        "objective",
        "rank",
        "method",
        "blur_sigma_px",
        "angle_signal",
        "independent_step_rmse_um",
        "calibration_z_loocv_rmse_um",
        "calibration_step_rmse_um",
        "simulation_z_loocv_rmse_um",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
