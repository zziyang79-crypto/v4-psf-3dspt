from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qc_v2_credibility_shrinkage as qc_v2
import run_benchmark as rb


RUN = BASE / "production_equivalent_ec10ms"
V2 = BASE / "qc_v2_credibility_shrinkage"
OUT = BASE / "qc_v4_spot_quality_trigger"
DATA = OUT / "data"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"


def ensure_dirs() -> None:
    for path in [OUT, DATA, FIGURES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def radial_center(roi: np.ndarray) -> tuple[float, float]:
    img = ndimage.gaussian_filter(np.asarray(roi, dtype=float), 1.0)
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


def brute_match_mean_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    if a.shape != (4, 2) or b.shape != (4, 2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan, np.full((4, 2), np.nan)
    best = None
    best_perm = None
    for perm in itertools.permutations(range(4)):
        candidate = b[list(perm)]
        dist = float(np.mean(np.linalg.norm(a - candidate, axis=1)))
        if best is None or dist < best:
            best = dist
            best_perm = perm
    return float(best), b[list(best_perm)]


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


def feature_localize_four_spots(
    image: np.ndarray,
    *,
    blur_sigma: float,
    roi_radius: int,
    min_peak_distance: int,
    background_percentile: float,
) -> dict[str, Any]:
    work = np.asarray(image, dtype=np.float32)
    if blur_sigma > 0:
        work = ndimage.gaussian_filter(work, sigma=float(blur_sigma)).astype(np.float32, copy=False)
    rough = rb.find_four_peaks(work, min_peak_distance)
    if len(rough) != 4:
        return {
            "success": False,
            "coords": np.full((4, 2), np.nan),
            "radial_phasor_residual_median_px": np.nan,
            "radial_phasor_residual_max_px": np.nan,
            "radial_peak_residual_median_px": np.nan,
            "phasor_peak_shift_median_px": np.nan,
            "phasor_quality_median": np.nan,
            **{f"geom_{k}": v for k, v in square_quality(np.full((4, 2), np.nan)).items()},
        }

    coords = []
    radial_phasor = []
    radial_peak = []
    phasor_peak = []
    qualities = []
    for x, y, _score in rough:
        roi, x0, y0 = rb.crop(work, x, y, roi_radius)
        px, py, ok, _residual, quality = rb.phasor_center(roi, background_percentile)
        rx, ry = radial_center(roi)
        local_peak = np.array([x - x0, y - y0], dtype=float)
        phasor = np.array([px, py], dtype=float)
        radial = np.array([rx, ry], dtype=float)
        coords.append((float(x0 + px), float(y0 + py)))
        radial_phasor.append(float(np.linalg.norm(radial - phasor)))
        radial_peak.append(float(np.linalg.norm(radial - local_peak)))
        phasor_peak.append(float(np.linalg.norm(phasor - local_peak)))
        qualities.append(float(quality) if np.isfinite(quality) else np.nan)
    coords_arr = np.asarray(coords, dtype=float)
    geom = square_quality(coords_arr)
    return {
        "success": bool(np.isfinite(coords_arr).all()),
        "coords": coords_arr,
        "radial_phasor_residual_median_px": float(np.nanmedian(radial_phasor)),
        "radial_phasor_residual_max_px": float(np.nanmax(radial_phasor)),
        "radial_peak_residual_median_px": float(np.nanmedian(radial_peak)),
        "phasor_peak_shift_median_px": float(np.nanmedian(phasor_peak)),
        "phasor_quality_median": float(np.nanmedian(qualities)),
        **{f"geom_{k}": v for k, v in geom.items()},
    }


def candidate_spot_features(stack: np.ndarray, frame_index: int, cfg: dict[str, Any]) -> dict[str, float]:
    loc_cfg = cfg["localization"]
    frame = np.asarray(stack[frame_index], dtype=np.float32)
    angle = feature_localize_four_spots(
        frame,
        blur_sigma=float(cfg["angle_blur_sigma_px"]),
        roi_radius=int(loc_cfg["roi_radius_px"]),
        min_peak_distance=int(loc_cfg["min_peak_distance_px"]),
        background_percentile=float(loc_cfg["background_percentile"]),
    )
    length = feature_localize_four_spots(
        frame,
        blur_sigma=0.0,
        roi_radius=int(loc_cfg["roi_radius_px"]),
        min_peak_distance=int(loc_cfg["min_peak_distance_px"]),
        background_percentile=float(loc_cfg["background_percentile"]),
    )
    match_dist, matched_length = brute_match_mean_distance(angle["coords"], length["coords"])
    angle_coords = angle["coords"]
    if np.isfinite(match_dist):
        mean_diag_diff = abs(float(angle["geom_mean_diag_px_feature"]) - float(length["geom_mean_diag_px_feature"]))
        theta_diff = max(
            angle_diff_180(float(angle.get("geom_diag_angle_orth_error", np.nan)), float(length.get("geom_diag_angle_orth_error", np.nan))),
            0.0,
        )
        center_diff = float(np.linalg.norm(np.nanmean(angle_coords, axis=0) - np.nanmean(matched_length, axis=0)))
    else:
        mean_diag_diff = np.nan
        theta_diff = np.nan
        center_diff = np.nan

    return {
        "spot_angle_radial_phasor_median_px": float(angle["radial_phasor_residual_median_px"]),
        "spot_angle_radial_phasor_max_px": float(angle["radial_phasor_residual_max_px"]),
        "spot_angle_radial_peak_median_px": float(angle["radial_peak_residual_median_px"]),
        "spot_angle_phasor_peak_shift_median_px": float(angle["phasor_peak_shift_median_px"]),
        "spot_angle_square_residual": float(angle["geom_square_residual"]),
        "spot_angle_side_cv": float(angle["geom_side_cv"]),
        "spot_angle_diag_rel_diff": float(angle["geom_diag_rel_diff"]),
        "spot_angle_diag_midpoint_gap_rel": float(angle["geom_diag_midpoint_gap_rel"]),
        "spot_angle_diag_angle_orth_error": float(angle["geom_diag_angle_orth_error"]),
        "spot_length_radial_phasor_median_px": float(length["radial_phasor_residual_median_px"]),
        "spot_length_radial_peak_median_px": float(length["radial_peak_residual_median_px"]),
        "spot_length_square_residual": float(length["geom_square_residual"]),
        "spot_length_diag_rel_diff": float(length["geom_diag_rel_diff"]),
        "spot_blur_no_blur_mean_spot_shift_px": float(match_dist),
        "spot_blur_no_blur_center_shift_px": center_diff,
        "spot_blur_no_blur_mean_diag_diff_px": float(mean_diag_diff),
        "spot_blur_no_blur_orth_error_diff": float(theta_diff),
        "spot_angle_localize_success": bool(angle["success"]),
        "spot_length_localize_success": bool(length["success"]),
    }


def compute_candidate_features(pred: pd.DataFrame, manifest: pd.DataFrame, cfg: dict[str, Any], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    candidates = pred[pred["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")][
        ["stack_id", "frame_index"]
    ].drop_duplicates()
    stack_paths = dict(zip(manifest["stack_id"].astype(str), manifest["stack_path"].astype(str)))
    rows: list[dict[str, Any]] = []
    for stack_id, group in candidates.groupby("stack_id", sort=False):
        stack = tifffile.imread(stack_paths[str(stack_id)])
        for frame_index in group["frame_index"].astype(int):
            row: dict[str, Any] = {"stack_id": str(stack_id), "frame_index": int(frame_index)}
            row.update(candidate_spot_features(stack, int(frame_index), cfg))
            rows.append(row)
    features = pd.DataFrame(rows)
    features.to_csv(cache_path, index=False)
    return features


def normalize_feature(train: pd.Series, values: pd.Series) -> pd.Series:
    arr_train = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    finite = arr_train[np.isfinite(arr_train)]
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(finite) == 0:
        return pd.Series(np.zeros(len(vals), dtype=float), index=values.index)
    lo = float(np.nanpercentile(finite, 5))
    hi = float(np.nanpercentile(finite, 95))
    scale = hi - lo
    if not np.isfinite(scale) or scale <= 1e-12:
        lo = 0.0
        scale = max(float(np.nanmedian(np.abs(finite))), 1.0)
    out = np.maximum((vals - lo) / scale, 0.0)
    out[~np.isfinite(out)] = 0.0
    return pd.Series(out, index=values.index)


def add_spot_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate = out["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")
    train = candidate & (out["seed"].astype(int) < 50)
    feature_cols = [
        "spot_angle_radial_phasor_median_px",
        "spot_angle_radial_phasor_max_px",
        "spot_angle_radial_peak_median_px",
        "spot_angle_phasor_peak_shift_median_px",
        "spot_angle_square_residual",
        "spot_angle_side_cv",
        "spot_angle_diag_rel_diff",
        "spot_angle_diag_midpoint_gap_rel",
        "spot_angle_diag_angle_orth_error",
        "spot_length_radial_phasor_median_px",
        "spot_length_radial_peak_median_px",
        "spot_length_square_residual",
        "spot_length_diag_rel_diff",
        "spot_blur_no_blur_mean_spot_shift_px",
        "spot_blur_no_blur_center_shift_px",
        "spot_blur_no_blur_mean_diag_diff_px",
        "spot_blur_no_blur_orth_error_diff",
    ]
    for col in feature_cols:
        out[f"n_{col}"] = normalize_feature(out.loc[train, col], out[col])
    out["spot_radial_residual_score"] = np.nanmax(
        out[
            [
                "n_spot_angle_radial_phasor_median_px",
                "n_spot_angle_radial_phasor_max_px",
                "n_spot_angle_radial_peak_median_px",
                "n_spot_angle_phasor_peak_shift_median_px",
                "n_spot_length_radial_phasor_median_px",
                "n_spot_length_radial_peak_median_px",
            ]
        ].to_numpy(dtype=float),
        axis=1,
    )
    out["spot_square_residual_score"] = np.nanmax(
        out[
            [
                "n_spot_angle_square_residual",
                "n_spot_angle_side_cv",
                "n_spot_angle_diag_rel_diff",
                "n_spot_angle_diag_midpoint_gap_rel",
                "n_spot_angle_diag_angle_orth_error",
                "n_spot_length_square_residual",
                "n_spot_length_diag_rel_diff",
            ]
        ].to_numpy(dtype=float),
        axis=1,
    )
    out["spot_blur_difference_score"] = np.nanmax(
        out[
            [
                "n_spot_blur_no_blur_mean_spot_shift_px",
                "n_spot_blur_no_blur_center_shift_px",
                "n_spot_blur_no_blur_mean_diag_diff_px",
                "n_spot_blur_no_blur_orth_error_diff",
            ]
        ].to_numpy(dtype=float),
        axis=1,
    )
    out["spot_quality_combined_score"] = np.nanmax(
        out[
            [
                "spot_radial_residual_score",
                "spot_square_residual_score",
                "spot_blur_difference_score",
            ]
        ].to_numpy(dtype=float),
        axis=1,
    )
    out["spot_quality_weighted_score"] = (
        0.30 * out["spot_radial_residual_score"].fillna(0.0)
        + 0.40 * out["spot_square_residual_score"].fillna(0.0)
        + 0.30 * out["spot_blur_difference_score"].fillna(0.0)
    )
    return out


def apply_trigger_gate(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    score_col = str(params["score_col"])
    candidate = out["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")
    score = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    low = float(params["score_low"])
    high = float(params["score_high"])
    power = float(params["score_power"])
    span = max(high - low, 1e-9)
    spot_confidence = np.clip((score - low) / span, 0.0, 1.0) ** power
    accepted = (
        candidate
        & (pd.to_numeric(out["qc_v2_weight"], errors="coerce") >= float(params["w_min"]))
        & (spot_confidence > 0.0)
    )
    w = pd.to_numeric(out["qc_v2_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    effective_w = np.where(accepted.to_numpy(dtype=bool), w * spot_confidence, 0.0)
    out["qc_v4_trigger_score_col"] = score_col
    out["qc_v4_trigger_score"] = score
    out["qc_v4_spot_confidence"] = spot_confidence
    out["qc_v4_trigger_accepted"] = accepted.to_numpy(dtype=bool)
    out["qc_v4_effective_weight"] = effective_w
    out["z_pred_hybrid_qc_v4"] = out["z_pred_pure_angle"] + effective_w * (
        out["z_pred_hybrid_angle_length_qc"] - out["z_pred_pure_angle"]
    )
    return out


def add_stack_offset_errors(df: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "pure_angle": "z_pred_pure_angle",
        "current_hard_qc": "z_pred_hybrid_angle_length_qc",
        "qc_v2_credibility_shrinkage": "z_pred_hybrid_qc_v2",
        "qc_v4_spot_triggered_shrinkage": "z_pred_hybrid_qc_v4",
        "pure_length": "z_pred_pure_length",
    }
    return qc_v2.add_stack_offset_errors(df, methods)


def offset_aligned_error_array(stack_codes: np.ndarray, z_true: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    offsets = np.zeros(int(np.nanmax(stack_codes)) + 1, dtype=float)
    for code in range(len(offsets)):
        idx = stack_codes == code
        offsets[code] = float(np.nanmedian(z_pred[idx] - z_true[idx]))
    return z_pred - offsets[stack_codes] - z_true


def metric_row_from_error(
    err: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    split: str,
    params_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finite = mask & np.isfinite(err)
    vals = err[finite]
    abs_vals = np.abs(vals)
    row: dict[str, Any] = {
        "params_id": params_id,
        "split": split,
        "method": method,
        "n_frames": int(len(vals)),
        "rmse_um": float(np.sqrt(np.mean(vals**2))) if len(vals) else np.nan,
        "mae_um": float(np.mean(abs_vals)) if len(vals) else np.nan,
        "p95_abs_error_um": float(np.percentile(abs_vals, 95)) if len(vals) else np.nan,
        "p99_abs_error_um": float(np.percentile(abs_vals, 99)) if len(vals) else np.nan,
    }
    if params:
        row.update(params)
    return row


def fixed_metric_template_rows(base: pd.DataFrame) -> list[dict[str, Any]]:
    stack_codes = pd.Categorical(base["stack_id"].astype(str)).codes.astype(int)
    z_true = pd.to_numeric(base["z_true_um"], errors="coerce").to_numpy(dtype=float)
    seed = pd.to_numeric(base["seed"], errors="coerce").to_numpy(dtype=int)
    split_masks = [
        ("train_seed_0_49", seed < 50),
        ("test_seed_50_99", seed >= 50),
        ("all", np.ones(len(base), dtype=bool)),
    ]
    cols = {
        "pure_angle": "z_pred_pure_angle",
        "current_hard_qc": "z_pred_hybrid_angle_length_qc",
        "qc_v2_credibility_shrinkage": "z_pred_hybrid_qc_v2",
        "pure_length": "z_pred_pure_length",
    }
    rows: list[dict[str, Any]] = []
    for method, col in cols.items():
        z_pred = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        err = offset_aligned_error_array(stack_codes, z_true, z_pred)
        for split, mask in split_masks:
            rows.append(metric_row_from_error(err, mask, method=method, split=split, params_id="fixed_template"))
    return rows


def fast_v4_metric_rows(base: pd.DataFrame, params: dict[str, Any], params_id: str) -> tuple[pd.DataFrame, int]:
    stack_codes = pd.Categorical(base["stack_id"].astype(str)).codes.astype(int)
    z_true = pd.to_numeric(base["z_true_um"], errors="coerce").to_numpy(dtype=float)
    seed = pd.to_numeric(base["seed"], errors="coerce").to_numpy(dtype=int)
    split_masks = [
        ("train_seed_0_49", seed < 50),
        ("test_seed_50_99", seed >= 50),
        ("all", np.ones(len(base), dtype=bool)),
    ]
    score_col = str(params["score_col"])
    score = pd.to_numeric(base[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    low = float(params["score_low"])
    high = float(params["score_high"])
    power = float(params["score_power"])
    spot_confidence = np.clip((score - low) / max(high - low, 1e-9), 0.0, 1.0) ** power
    candidate = base["hybrid_source"].astype(str).eq("angle_length_trend_interpolated").to_numpy(dtype=bool)
    w = pd.to_numeric(base["qc_v2_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    accepted = candidate & (w >= float(params["w_min"])) & (spot_confidence > 0.0)
    effective_w = np.where(accepted, w * spot_confidence, 0.0)
    z_angle = pd.to_numeric(base["z_pred_pure_angle"], errors="coerce").to_numpy(dtype=float)
    z_hard = pd.to_numeric(base["z_pred_hybrid_angle_length_qc"], errors="coerce").to_numpy(dtype=float)
    z_v4 = z_angle + effective_w * (z_hard - z_angle)
    err = offset_aligned_error_array(stack_codes, z_true, z_v4)
    rows = [
        metric_row_from_error(
            err,
            mask,
            method="qc_v4_spot_triggered_shrinkage",
            split=split,
            params_id=params_id,
            params=params,
        )
        for split, mask in split_masks
    ]
    train_accepted = int(np.sum(accepted & (seed < 50)))
    return pd.DataFrame(rows), train_accepted


def metric_row(df: pd.DataFrame, method: str, split: str, params_id: str) -> dict[str, Any]:
    err = pd.to_numeric(df[f"error_{method}_offset_aligned_um"], errors="coerce").to_numpy(dtype=float)
    err = err[np.isfinite(err)]
    abs_err = np.abs(err)
    return {
        "params_id": params_id,
        "split": split,
        "method": method,
        "n_frames": int(len(err)),
        "rmse_um": float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "mae_um": float(np.mean(abs_err)) if len(err) else np.nan,
        "p95_abs_error_um": float(np.percentile(abs_err, 95)) if len(err) else np.nan,
        "p99_abs_error_um": float(np.percentile(abs_err, 99)) if len(err) else np.nan,
    }


def score_params(base: pd.DataFrame, params: dict[str, Any], params_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = add_stack_offset_errors(apply_trigger_gate(base, params))
    methods = [
        "pure_angle",
        "current_hard_qc",
        "qc_v2_credibility_shrinkage",
        "qc_v4_spot_triggered_shrinkage",
        "pure_length",
    ]
    rows = []
    for split, part in [
        ("train_seed_0_49", pred[pred["seed"] < 50]),
        ("test_seed_50_99", pred[pred["seed"] >= 50]),
        ("all", pred),
    ]:
        for method in methods:
            row = metric_row(part, method, split, params_id)
            row.update(params)
            rows.append(row)
    return pred, pd.DataFrame(rows)


def parameter_grid() -> list[dict[str, Any]]:
    values = {
        "score_col": [
            "spot_radial_residual_score",
            "spot_square_residual_score",
            "spot_blur_difference_score",
            "spot_quality_combined_score",
            "spot_quality_weighted_score",
        ],
        "score_low": [0.0, 0.30, 0.65, 0.90],
        "score_high": [0.65, 1.20, 1.60],
        "score_power": [0.5, 1.0, 2.0],
        "w_min": [0.0, 0.15, 0.25],
    }
    keys = list(values)
    out = []
    for combo in itertools.product(*(values[k] for k in keys)):
        params = dict(zip(keys, combo))
        if float(params["score_high"]) > float(params["score_low"]):
            out.append(params)
    return out


def qc_event_summary(pred: pd.DataFrame, params_id: str) -> pd.DataFrame:
    q = pred[pred["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")].copy()
    geom = bool_series(q["geometry_disturbance_applied"])
    q["current_gain_um"] = q["abs_error_pure_angle_offset_aligned_um"] - q[
        "abs_error_current_hard_qc_offset_aligned_um"
    ]
    q["v2_gain_um"] = q["abs_error_pure_angle_offset_aligned_um"] - q[
        "abs_error_qc_v2_credibility_shrinkage_offset_aligned_um"
    ]
    q["v4_gain_um"] = q["abs_error_pure_angle_offset_aligned_um"] - q[
        "abs_error_qc_v4_spot_triggered_shrinkage_offset_aligned_um"
    ]
    subsets = [
        ("all_qc_candidates", q),
        ("candidates_with_geometry", q[geom]),
        ("candidates_without_geometry", q[~geom]),
        ("v4_accepted_candidates", q[bool_series(q["qc_v4_trigger_accepted"])]),
        ("v4_rejected_candidates", q[~bool_series(q["qc_v4_trigger_accepted"])]),
    ]
    rows = []
    for subset, part in subsets:
        rows.append(
            {
                "params_id": params_id,
                "subset": subset,
                "n": int(len(part)),
                "v4_accepted": int(bool_series(part["qc_v4_trigger_accepted"]).sum()) if len(part) else 0,
                "current_positive": int((part["current_gain_um"] > 0).sum()),
                "current_negative": int((part["current_gain_um"] <= 0).sum()),
                "v2_positive": int((part["v2_gain_um"] > 0).sum()),
                "v2_negative": int((part["v2_gain_um"] <= 0).sum()),
                "v4_positive": int((part["v4_gain_um"] > 0).sum()),
                "v4_negative": int((part["v4_gain_um"] <= 0).sum()),
                "current_gain_mean_um": float(part["current_gain_um"].mean()) if len(part) else np.nan,
                "v2_gain_mean_um": float(part["v2_gain_um"].mean()) if len(part) else np.nan,
                "v4_gain_mean_um": float(part["v4_gain_um"].mean()) if len(part) else np.nan,
                "v4_effective_weight_median": float(part["qc_v4_effective_weight"].median()) if len(part) else np.nan,
                "v4_spot_confidence_median": float(part["qc_v4_spot_confidence"].median()) if len(part) else np.nan,
                "v4_trigger_score_median": float(part["qc_v4_trigger_score"].median()) if len(part) else np.nan,
                "geometry_fraction": float(bool_series(part["geometry_disturbance_applied"]).mean()) if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, params_id: str) -> None:
    data = metrics[(metrics["params_id"].eq(params_id)) & (metrics["split"].eq("test_seed_50_99"))].copy()
    order = [
        "pure_angle",
        "current_hard_qc",
        "qc_v2_credibility_shrinkage",
        "qc_v4_spot_triggered_shrinkage",
        "pure_length",
    ]
    data["method"] = pd.Categorical(data["method"], categories=order, ordered=True)
    data = data.sort_values("method")
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    colors = ["#1f77b4", "#dc2626", "#15803d", "#7c3aed", "#e67e00"]
    ax.bar(data["method"].astype(str), data["rmse_um"], color=colors)
    ax.set_ylabel("Offset-aligned RMSE (um)")
    ax.set_title("QC v4 spot-quality trigger, held-out seeds 50-99")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(data["rmse_um"]):
        ax.text(idx, float(value) + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    fig.savefig(FIGURES / "qc_v4_test_rmse_by_method.png", dpi=220)
    plt.close(fig)


def plot_feature_distributions(pred: pd.DataFrame) -> None:
    q = pred[pred["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")].copy()
    geom = bool_series(q["geometry_disturbance_applied"])
    features = [
        ("spot_radial_residual_score", "Radial-center residual"),
        ("spot_square_residual_score", "Square/diagonal residual"),
        ("spot_blur_difference_score", "Blur vs no-blur difference"),
        ("spot_quality_combined_score", "Combined spot-quality score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), features):
        ax.hist(q.loc[~geom, col].dropna(), bins=16, alpha=0.65, label="No geometry", color="#64748b")
        ax.hist(q.loc[geom, col].dropna(), bins=16, alpha=0.65, label="Geometry", color="#dc2626")
        ax.set_title(title)
        ax.set_xlabel("Normalized score")
        ax.set_ylabel("QC candidates")
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)
    fig.savefig(FIGURES / "qc_v4_spot_feature_distributions.png", dpi=220)
    plt.close(fig)


def write_report(best_params: dict[str, Any], metrics: pd.DataFrame, event_summary: pd.DataFrame) -> None:
    lines = [
        "# QC v4: spot-quality weighted trigger plus credibility shrinkage",
        "",
        "## Method",
        "",
        "This experiment modifies QC trigger logic using spot-level quality features. The original production QC trigger is treated as a candidate detector. The selected spot-quality score is converted to a continuous confidence between `score_low` and `score_high`; the final correction weight is `qc_v2_weight * spot_confidence`. Candidates with zero spot confidence remain pure angle.",
        "",
        "No maximum correction-amplitude cap is used. Four-spot intensity balance is intentionally not used.",
        "",
        "Spot-quality features used here are: single-spot radial-center residuals, four-spot square residual, diagonal length consistency, diagonal midpoint/orthogonality residuals, and blur-vs-no-blur localization differences.",
        "",
        "## Selected Trigger Parameters",
        "",
    ]
    for key, value in best_params.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Held-out Metrics", ""])
    show = metrics[metrics["split"].isin(["train_seed_0_49", "test_seed_50_99", "all"])].copy()
    show = show.sort_values(["split", "method"])
    lines.append("| split | method | RMSE (um) | MAE (um) | P95 (um) | P99 (um) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in show.iterrows():
        lines.append(
            f"| {row['split']} | {row['method']} | {row['rmse_um']:.4f} | {row['mae_um']:.4f} | {row['p95_abs_error_um']:.4f} | {row['p99_abs_error_um']:.4f} |"
        )
    lines.extend(["", "## QC Candidate Summary", ""])
    lines.append("| subset | n | accepted | current +/- | v2 +/- | v4 +/- | current mean gain | v2 mean gain | v4 mean gain | median spot confidence | geometry fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in event_summary.iterrows():
        lines.append(
            f"| {row['subset']} | {int(row['n'])} | {int(row['v4_accepted'])} | {int(row['current_positive'])}/{int(row['current_negative'])} | {int(row['v2_positive'])}/{int(row['v2_negative'])} | {int(row['v4_positive'])}/{int(row['v4_negative'])} | {row['current_gain_mean_um']:.4f} | {row['v2_gain_mean_um']:.4f} | {row['v4_gain_mean_um']:.4f} | {row['v4_spot_confidence_median']:.3f} | {row['geometry_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This tests whether the trigger can be improved by using the specific spot-level evidence discussed in the manuscript workflow. Because the spot score is used as a continuous confidence term rather than a hard switch, this version is less sensitive to a single threshold than the previous gate-only test.",
            "",
            "## Files",
            "",
            f"- Per-frame v4 predictions: `{DATA / 'per_frame_qc_v4_predictions.csv'}`",
            f"- Candidate spot features: `{DATA / 'candidate_spot_quality_features.csv'}`",
            f"- Trigger sweep metrics: `{DATA / 'spot_trigger_parameter_sweep_metrics.csv'}`",
            f"- Event summary: `{DATA / 'qc_v4_event_summary.csv'}`",
            f"- Held-out RMSE figure: `{FIGURES / 'qc_v4_test_rmse_by_method.png'}`",
        ]
    )
    (REPORTS / "qc_v4_spot_quality_trigger_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    v2_pred_path = V2 / "data" / "per_frame_qc_v2_predictions.csv"
    if not v2_pred_path.exists():
        raise FileNotFoundError(f"Run QC v2 first: {v2_pred_path}")
    pred = pd.read_csv(v2_pred_path, low_memory=False)
    manifest = pd.read_csv(RUN / "results" / "sweep_manifest.csv")
    cfg = rb.load_config(BASE / "config.json")
    candidate_features = compute_candidate_features(pred, manifest, cfg, DATA / "candidate_spot_quality_features.csv")
    base = pred.merge(candidate_features, on=["stack_id", "frame_index"], how="left")
    base = add_spot_quality_scores(base)

    all_metrics = []
    fixed_rows = fixed_metric_template_rows(base)
    best_score = np.inf
    best_metrics = None
    best_params = None
    best_id = ""
    for i, params in enumerate(parameter_grid()):
        params_id = f"spot_trigger_{i:03d}"
        v4_metrics, train_accepted = fast_v4_metric_rows(base, params, params_id)
        fixed_metrics = []
        for row in fixed_rows:
            item = row.copy()
            item["params_id"] = params_id
            item.update(params)
            fixed_metrics.append(item)
        metrics = pd.concat([pd.DataFrame(fixed_metrics), v4_metrics], ignore_index=True)
        all_metrics.append(metrics)
        train_rmse = v4_metrics[
            (v4_metrics["split"] == "train_seed_0_49")
            & (v4_metrics["method"] == "qc_v4_spot_triggered_shrinkage")
        ]["rmse_um"].iloc[0]
        score = float(train_rmse + (0.002 if train_accepted < 3 else 0.0))
        if score < best_score:
            best_score = score
            best_metrics = metrics
            best_params = params
            best_id = params_id

    if best_metrics is None or best_params is None:
        raise RuntimeError("No spot trigger parameter set evaluated.")
    sweep = pd.concat(all_metrics, ignore_index=True)
    sweep.to_csv(DATA / "spot_trigger_parameter_sweep_metrics.csv", index=False)

    best_pred, selected_metrics = score_params(base, best_params, "best_train_rmse")
    best_pred = best_pred.copy()
    best_pred["qc_v4_params_id"] = "best_train_rmse"
    best_pred.to_csv(DATA / "per_frame_qc_v4_predictions.csv", index=False)
    selected_metrics.to_csv(DATA / "selected_qc_v4_metrics.csv", index=False)
    pd.DataFrame([{**{"params_id": "best_train_rmse", "source_grid_id": best_id}, **best_params}]).to_csv(
        DATA / "selected_qc_v4_parameters.csv", index=False
    )
    event_summary = qc_event_summary(best_pred, "best_train_rmse")
    event_summary.to_csv(DATA / "qc_v4_event_summary.csv", index=False)
    plot_metrics(selected_metrics, "best_train_rmse")
    plot_feature_distributions(best_pred)
    write_report(best_params, selected_metrics, event_summary)

    print(f"Wrote QC v4 spot-quality trigger experiment to {OUT}")
    print(pd.DataFrame([{**{'source_grid_id': best_id}, **best_params}]).to_string(index=False))
    print(selected_metrics[selected_metrics["split"].eq("test_seed_50_99")].to_string(index=False))
    print(event_summary.to_string(index=False))


if __name__ == "__main__":
    main()
