from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from skimage.feature import peak_local_max


METHODS = ["hybrid_angle_length_qc", "pure_angle", "pure_length"]
NOISE_ORDER = ["clean", "low", "medium", "high", "very_high", "extreme", "ec10ms_extreme"]
ABERRATION_ORDER = ["none", "very_mild", "mild", "moderate", "strong", "roi_check_extreme"]
MOTION_ORDER = [
    "none",
    "z_exposure_medium",
    "z_exposure_strong",
    "ec10ms_tail_moderate",
    "ec10ms_tail_extreme",
    "persistence_mild",
    "medium",
    "strong",
    "extreme",
]
ARTIFACT_ORDER = [
    "none",
    "balanced_artifact_mild",
    "balanced_artifact_medium",
    "balanced_artifact_strong",
    "adversarial_radial_control",
    "adversarial_tangential_control",
    "spot_ghost_mild",
    "spot_ghost_strong",
]
GEOMETRY_ORDER = ["none", "ec10ms_boundary_mild", "ec10ms_boundary_medium", "ec10ms_boundary_strong"]

_PRODUCTION_Z_MODULE = None


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(cfg["_root"]) / path).resolve()


def resolve_model_path(cfg: dict[str, Any], key: str, default_name: str) -> Path:
    configured = str(cfg.get(key, ""))
    if configured:
        try:
            path = resolve_path(cfg, configured)
            if path.exists():
                return path
        except (OSError, ValueError):
            pass
    return Path(__file__).resolve().parents[2] / "models" / default_name


def load_production_z_module(cfg: dict[str, Any]):
    global _PRODUCTION_Z_MODULE
    if _PRODUCTION_Z_MODULE is not None:
        return _PRODUCTION_Z_MODULE
    import sys
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))
    from v4psf import z_fit as module
    _PRODUCTION_Z_MODULE = module
    return module


@contextmanager
def production_z_environment():
    values = {
        "Z_SPLINE_OUT_OF_DOMAIN": "clip",
        "Z_ANGLE_SPLINE_OUT_OF_DOMAIN": "clip",
        "Z_ANGLE_LENGTH_D2_QC_ENABLED": "1",
        "Z_ALIGN_LENGTH_BACKUP_TO_ANGLE": "1",
    }
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def output_dirs(cfg: dict[str, Any], mode: str) -> dict[str, Path]:
    root = Path(cfg["_root"])
    if mode == "smoke":
        base = root / "smoke"
        dirs = {
            "base": base,
            "stacks": base / "generated_stacks",
            "results": base / "results",
            "figures": base / "figures",
            "reports": base / "reports",
        }
    elif mode == "full":
        dirs = {
            "base": root,
            "stacks": root / "generated_stacks",
            "results": root / "results",
            "figures": root / "figures",
            "reports": root / "reports",
        }
    else:
        base = root / safe_name(mode)
        dirs = {
            "base": base,
            "stacks": base / "generated_stacks",
            "results": base / "results",
            "figures": base / "figures",
            "reports": base / "reports",
        }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def reflect_bounds(value: float, low: float, high: float) -> float:
    if high <= low:
        return low
    span = high - low
    value = float(value)
    while value < low or value > high:
        if value < low:
            value = low + (low - value)
        if value > high:
            value = high - (value - high)
        if value < low - 10 * span or value > high + 10 * span:
            value = low + abs((value - low) % (2 * span))
            if value > high:
                value = high - (value - high)
    return float(np.clip(value, low, high))


def load_template_stack(
    cfg: dict[str, Any],
    template_dir_value: str,
    *,
    filename_prefix: str = "",
    filename_suffix: str = ".tif",
    source_label: str = "primary",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    template_dir = resolve_path(cfg, template_dir_value)
    offset = int(cfg["template_file_offset"])
    start = int(cfg["template_frame_start"])
    end = int(cfg["template_frame_end"])
    templates: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    shape = None
    for frame_number in range(start, end + 1):
        path = template_dir / f"{filename_prefix}{offset + frame_number}{filename_suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Missing template file: {path}")
        with Image.open(path) as img:
            image = np.asarray(img, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a single 2D image in {path}, got shape {image.shape}")
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f"Template shape mismatch: {path} has {image.shape}, expected {shape}")
        z_true = float(frame_number - start)
        templates.append(image)
        rows.append(
            {
                "template_source": source_label,
                "template_frame_number": frame_number,
                "template_file": str(path),
                "z_true_um": z_true,
                "height_px": image.shape[0],
                "width_px": image.shape[1],
                "dtype_loaded": str(image.dtype),
                "min_count": float(np.min(image)),
                "max_count": float(np.max(image)),
                "mean_count": float(np.mean(image)),
                "std_count": float(np.std(image)),
            }
        )
    z_values = np.arange(0, end - start + 1, dtype=float)
    return np.stack(templates, axis=0), z_values, pd.DataFrame(rows)


def load_templates(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return load_template_stack(cfg, cfg["template_dir"], source_label="primary")


def fit_template_stack_to_shape(templates: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    stack = np.asarray(templates, dtype=np.float32)
    target_h, target_w = map(int, target_shape)
    out = np.empty((stack.shape[0], target_h, target_w), dtype=np.float32)
    for idx, image in enumerate(stack):
        bg = float(np.percentile(image, 15.0))
        canvas = np.full((target_h, target_w), bg, dtype=np.float32)
        src_h, src_w = image.shape
        copy_h = min(src_h, target_h)
        copy_w = min(src_w, target_w)
        src_y0 = max((src_h - copy_h) // 2, 0)
        src_x0 = max((src_w - copy_w) // 2, 0)
        dst_y0 = max((target_h - copy_h) // 2, 0)
        dst_x0 = max((target_w - copy_w) // 2, 0)
        canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = image[
            src_y0 : src_y0 + copy_h,
            src_x0 : src_x0 + copy_w,
        ]
        out[idx] = canvas
    return out


def interpolate_template(templates: np.ndarray, z_um: float) -> np.ndarray:
    z = float(np.clip(z_um, 0.0, templates.shape[0] - 1))
    lo = int(math.floor(z))
    hi = int(math.ceil(z))
    if lo == hi:
        return templates[lo].astype(np.float32, copy=True)
    alpha = z - lo
    return ((1.0 - alpha) * templates[lo] + alpha * templates[hi]).astype(np.float32, copy=False)


def generate_brownian_z(cfg: dict[str, Any], seed: int, n_frames: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    brownian = cfg["brownian"]
    z_min = float(cfg["z_min_um"])
    z_max = float(cfg["z_max_um"])
    z = np.empty(n_frames, dtype=float)
    z[0] = float(cfg["z_start_um"])
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    step_sigma = float(brownian["step_sigma_um"])
    drift_amp = float(brownian["drift_step_amplitude_um"])
    period = max(float(brownian["drift_period_frames"]), 1.0)
    for idx in range(1, n_frames):
        drift = drift_amp * math.sin(2.0 * math.pi * idx / period + phase)
        dz = float(rng.normal(0.0, step_sigma) + drift)
        z[idx] = reflect_bounds(z[idx - 1] + dz, z_min, z_max)
    return z


def generate_z_for_run(cfg: dict[str, Any], run_cfg: dict[str, Any], seed: int, n_frames: int) -> np.ndarray:
    profile = str(run_cfg.get("trajectory_profile", "brownian")).strip().lower()
    if profile != "ec10ms_boundary_brownian":
        return generate_brownian_z(cfg, seed, n_frames)

    rng = np.random.default_rng(seed)
    z_min = float(run_cfg.get("z_min_um", cfg["z_min_um"]))
    z_max = float(run_cfg.get("z_max_um", 12.0))
    z = np.empty(n_frames, dtype=float)
    z[0] = float(run_cfg.get("z_start_um", min(max(float(cfg["z_start_um"]), z_min), z_max)))
    step_sigma = float(run_cfg.get("step_sigma_um", 1.15))
    drift_amp = float(run_cfg.get("drift_step_amplitude_um", 0.18))
    period = max(float(run_cfg.get("drift_period_frames", 28.0)), 1.0)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    for idx in range(1, n_frames):
        drift = drift_amp * math.sin(2.0 * math.pi * idx / period + phase)
        z[idx] = reflect_bounds(z[idx - 1] + float(rng.normal(0.0, step_sigma) + drift), z_min, z_max)

    segment_count = int(run_cfg.get("boundary_dwell_segments", 1))
    min_len = int(run_cfg.get("boundary_dwell_min_frames", 6))
    max_len = int(run_cfg.get("boundary_dwell_max_frames", 12))
    target_max = float(run_cfg.get("boundary_dwell_target_max_um", 0.9))
    if n_frames >= 8 and segment_count > 0:
        for _ in range(segment_count):
            seg_len = int(rng.integers(max(2, min_len), max(max_len, min_len) + 1))
            start_hi = max(1, n_frames - seg_len - 1)
            start = int(rng.integers(1, start_hi + 1))
            stop = min(n_frames, start + seg_len)
            target = float(rng.uniform(z_min, min(z_min + target_max, z_max)))
            for offset, idx in enumerate(range(start, stop)):
                taper = 0.5 - 0.5 * math.cos(2.0 * math.pi * offset / max(stop - start - 1, 1))
                local = target + abs(float(rng.normal(0.0, 0.28)))
                z[idx] = (1.0 - taper) * z[idx] + taper * reflect_bounds(local, z_min, z_max)
    return np.clip(z, z_min, z_max)


def make_aberration_params(level: str, preset: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    if preset.get("template_source"):
        return {
            "level": level,
            "severity": 0.0,
            "mode": "reference_templates",
            "template_source": str(preset.get("template_source")),
            "template_z_offset_um": float(preset.get("template_z_offset_um", 0.0)),
            "rotation_deg": 0.0,
            "shear_x": 0.0,
            "shear_y": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "sigma_y_px": 0.0,
            "sigma_x_px": 0.0,
            "quad_x_px": 0.0,
            "quad_y_px": 0.0,
        }
    severity = float(preset.get("severity", 0.0))
    if severity <= 0:
        return {
            "level": level,
            "severity": severity,
            "rotation_deg": 0.0,
            "shear_x": 0.0,
            "shear_y": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shift_x_px": 0.0,
            "shift_y_px": 0.0,
            "sigma_y_px": 0.0,
            "sigma_x_px": 0.0,
            "quad_x_px": 0.0,
            "quad_y_px": 0.0,
        }
    return {
        "level": level,
        "severity": severity,
        "rotation_deg": float(rng.normal(0.0, 0.06 * severity)),
        "shear_x": float(rng.normal(0.0, 0.0018 * severity)),
        "shear_y": float(rng.normal(0.0, 0.0018 * severity)),
        "scale_x": float(1.0 + rng.normal(0.0, 0.0018 * severity)),
        "scale_y": float(1.0 + rng.normal(0.0, 0.0018 * severity)),
        "shift_x_px": float(rng.normal(0.0, 0.06 * severity)),
        "shift_y_px": float(rng.normal(0.0, 0.06 * severity)),
        "sigma_y_px": float(max(0.0, rng.normal(0.08 * severity, 0.02 * severity))),
        "sigma_x_px": float(max(0.0, rng.normal(0.02 * severity, 0.015 * severity))),
        "quad_x_px": float(rng.normal(0.0, 0.16 * severity)),
        "quad_y_px": float(rng.normal(0.0, 0.16 * severity)),
    }


def affine_matrix(params: dict[str, float]) -> np.ndarray:
    theta = math.radians(float(params["rotation_deg"]))
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=float)
    shear = np.array([[1.0, float(params["shear_x"])], [float(params["shear_y"]), 1.0]], dtype=float)
    scale = np.array([[float(params["scale_x"]), 0.0], [0.0, float(params["scale_y"])]], dtype=float)
    return rot @ shear @ scale


def apply_aberration(image: np.ndarray, params: dict[str, float]) -> np.ndarray:
    severity = float(params.get("severity", 0.0))
    if severity <= 0:
        return image.astype(np.float32, copy=True)
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=float)
    # ndimage coordinates are row/col, so use y/x order.
    a_xy = affine_matrix(params)
    a_yx = np.array([[a_xy[1, 1], a_xy[1, 0]], [a_xy[0, 1], a_xy[0, 0]]], dtype=float)
    shift_yx = np.array([float(params["shift_y_px"]), float(params["shift_x_px"])], dtype=float)
    offset = center - a_yx @ center - shift_yx
    warped = ndimage.affine_transform(arr, a_yx, offset=offset, order=1, mode="nearest")

    quad_x = float(params.get("quad_x_px", 0.0))
    quad_y = float(params.get("quad_y_px", 0.0))
    if abs(quad_x) > 1e-9 or abs(quad_y) > 1e-9:
        yy, xx = np.indices(arr.shape, dtype=float)
        xn = (xx - (w - 1) / 2.0) / max((w - 1) / 2.0, 1.0)
        yn = (yy - (h - 1) / 2.0) / max((h - 1) / 2.0, 1.0)
        map_x = xx + quad_x * (xn * xn - yn * yn)
        map_y = yy + quad_y * (2.0 * xn * yn)
        warped = ndimage.map_coordinates(warped, [map_y, map_x], order=1, mode="nearest")

    sigma_y = float(params.get("sigma_y_px", 0.0))
    sigma_x = float(params.get("sigma_x_px", 0.0))
    if sigma_x > 0 or sigma_y > 0:
        warped = ndimage.gaussian_filter(warped, sigma=(sigma_y, sigma_x))
    return warped.astype(np.float32, copy=False)


def line_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    length = int(max(1, round(float(length_px))))
    if length <= 1:
        return np.ones((1, 1), dtype=np.float32)
    size = int(max(3, 2 * length + 1))
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    theta = math.radians(float(angle_deg))
    dx = math.cos(theta)
    dy = math.sin(theta)
    for t in np.linspace(-length / 2.0, length / 2.0, max(2, length * 2 + 1)):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total <= 0:
        kernel[int(center), int(center)] = 1.0
        total = 1.0
    return kernel / total


def apply_directional_blur(image: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    if float(length_px) <= 0:
        return np.asarray(image, dtype=np.float32)
    return ndimage.convolve(np.asarray(image, dtype=np.float32), line_kernel(length_px, angle_deg), mode="nearest")


def empty_artifact_meta(preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_applied": False,
        "artifact_spot_index": np.nan,
        "artifact_shift_px": 0.0,
        "artifact_ghost_fraction": 0.0,
        "artifact_dropout_fraction": 0.0,
        "artifact_angle_deg": np.nan,
        "artifact_direction_mode_config": str(preset.get("direction_mode", "none")),
        "artifact_direction_mode_applied": "",
        "artifact_energy_conserving": bool(preset.get("energy_conserving", False)),
        "artifact_persistent": bool(preset.get("artifact_persistent", False)),
        "artifact_moved_signal_counts": 0.0,
        "artifact_added_signal_counts": 0.0,
        "artifact_lost_signal_counts": 0.0,
    }


def artifact_direction(
    radial: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    radial_norm = float(np.linalg.norm(radial))
    if mode == "balanced":
        mode = str(rng.choice(["radial", "tangential", "random"]))
    if mode == "random" or radial_norm <= 1e-6:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array([math.cos(angle), math.sin(angle)], dtype=float), "random"

    radial_unit = np.asarray(radial, dtype=float) / radial_norm
    if mode == "radial":
        direction = radial_unit
    elif mode == "tangential":
        direction = np.array([-radial_unit[1], radial_unit[0]], dtype=float)
    else:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array([math.cos(angle), math.sin(angle)], dtype=float), "random"
    if rng.random() < 0.5:
        direction *= -1.0
    return direction, mode


def apply_spot_artifact(image: np.ndarray, preset: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    probability = float(preset.get("probability", 0.0))
    if probability <= 0 or rng.random() > probability:
        return image.astype(np.float32, copy=True), empty_artifact_meta(preset)

    arr = np.asarray(image, dtype=np.float32).copy()
    peaks = find_four_peaks(arr, min_distance=20)
    if len(peaks) != 4:
        return arr, empty_artifact_meta(preset)

    pts = np.array([[p[0], p[1]] for p in peaks], dtype=float)
    center = np.mean(pts, axis=0)
    spot_idx = int(rng.integers(0, 4))
    spot = pts[spot_idx]
    radial = spot - center
    direction, direction_mode = artifact_direction(radial, str(preset.get("direction_mode", "random")), rng)
    shift_px = float(preset.get("shift_px", 0.0)) * float(rng.uniform(0.75, 1.25))
    ghost_fraction = float(preset.get("ghost_fraction", 0.0)) * float(rng.uniform(0.7, 1.25))
    dropout_fraction = float(preset.get("dropout_fraction", 0.0)) * float(rng.uniform(0.7, 1.2))
    sigma = max(float(preset.get("sigma_px", 2.0)) * float(rng.uniform(0.85, 1.25)), 0.8)
    energy_conserving = bool(preset.get("energy_conserving", False))

    yy, xx = np.indices(arr.shape, dtype=float)
    bg = float(np.percentile(arr, 15.0))
    signal_peak = max(float(arr[int(round(spot[1])), int(round(spot[0]))] - bg), 1.0)
    ghost_center = spot + direction * shift_px
    ghost = np.exp(-((xx - ghost_center[0]) ** 2 + (yy - ghost_center[1]) ** 2) / (2.0 * sigma**2))
    source = np.exp(-((xx - spot[0]) ** 2 + (yy - spot[1]) ** 2) / (2.0 * (sigma * 0.9) ** 2))
    signal = np.maximum(arr - bg, 0.0)
    moved_signal = 0.0
    added_signal = 0.0
    lost_signal = 0.0

    if energy_conserving:
        move_map = signal * source * ghost_fraction
        dropout_map = signal * source * dropout_fraction
        moved_signal = float(np.sum(move_map))
        lost_signal = float(np.sum(dropout_map))
        arr -= move_map + dropout_map
        ghost_sum = float(np.sum(ghost))
        if ghost_sum > 1e-12 and moved_signal > 0:
            added = moved_signal * ghost / ghost_sum
            arr += added
            added_signal = float(np.sum(added))
    else:
        arr += signal_peak * ghost_fraction * ghost
        added_signal = float(np.sum(signal_peak * ghost_fraction * ghost))
        dropout_map = signal * dropout_fraction * source
        arr -= dropout_map
        lost_signal = float(np.sum(dropout_map))

    arr = np.clip(arr, 0.0, 65535.0)
    return arr.astype(np.float32, copy=False), {
        "artifact_applied": True,
        "artifact_spot_index": spot_idx,
        "artifact_shift_px": shift_px,
        "artifact_ghost_fraction": ghost_fraction,
        "artifact_dropout_fraction": dropout_fraction,
        "artifact_angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0),
        "artifact_direction_mode_config": str(preset.get("direction_mode", "random")),
        "artifact_direction_mode_applied": direction_mode,
        "artifact_energy_conserving": energy_conserving,
        "artifact_persistent": bool(preset.get("artifact_persistent", False)),
        "artifact_moved_signal_counts": moved_signal,
        "artifact_added_signal_counts": added_signal,
        "artifact_lost_signal_counts": lost_signal,
    }


def empty_geometry_meta(preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry_disturbance_applied": False,
        "geometry_disturbance_spot_index": np.nan,
        "geometry_disturbance_shift_px": 0.0,
        "geometry_disturbance_angle_deg": np.nan,
        "geometry_disturbance_direction_mode": str(preset.get("direction_mode", "balanced")),
        "geometry_disturbance_moved_signal_counts": 0.0,
        "domain_pressure_applied": False,
        "domain_pressure_theta_bias_deg": 0.0,
    }


def rotate_image_center(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) <= 1e-9:
        return np.asarray(image, dtype=np.float32)
    return ndimage.rotate(
        np.asarray(image, dtype=np.float32),
        angle=float(angle_deg),
        reshape=False,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


def apply_geometry_disturbance(
    image: np.ndarray,
    preset: dict[str, Any],
    z_um: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not preset or str(preset.get("mode", "none")).lower() == "none":
        return image.astype(np.float32, copy=True), empty_geometry_meta(preset)

    arr = np.asarray(image, dtype=np.float32).copy()
    meta = empty_geometry_meta(preset)

    theta_bias = float(preset.get("theta_bias_near_low_z_deg", 0.0))
    if theta_bias > 0:
        low_window = max(float(preset.get("theta_bias_low_z_window_um", 2.0)), 1e-6)
        strength = float(np.clip((low_window - float(z_um)) / low_window, 0.0, 1.0))
        probability = float(preset.get("theta_bias_probability", 1.0))
        if strength > 0 and rng.random() <= probability:
            jitter = float(preset.get("theta_bias_jitter_deg", 0.4))
            bias_sign = float(preset.get("theta_bias_sign", -1.0))
            bias = bias_sign * strength * max(0.0, float(rng.normal(theta_bias, jitter)))
            arr = rotate_image_center(arr, bias)
            meta["domain_pressure_applied"] = True
            meta["domain_pressure_theta_bias_deg"] = float(bias)

    probability = float(preset.get("local_probability", 0.0))
    if probability <= 0 or rng.random() > probability:
        return arr.astype(np.float32, copy=False), meta

    peaks = find_four_peaks(arr, min_distance=int(preset.get("min_peak_distance_px", 20)))
    if len(peaks) != 4:
        return arr.astype(np.float32, copy=False), meta

    pts = np.array([[p[0], p[1]] for p in peaks], dtype=float)
    center = np.mean(pts, axis=0)
    spot_idx = int(rng.integers(0, 4))
    spot = pts[spot_idx]
    radial = spot - center
    direction, direction_mode = artifact_direction(radial, str(preset.get("direction_mode", "balanced")), rng)
    shift_px = float(preset.get("local_shift_px", 2.0)) * float(rng.uniform(0.75, 1.25))
    dx, dy = direction * shift_px
    sigma = max(float(preset.get("local_sigma_px", 4.0)) * float(rng.uniform(0.85, 1.2)), 1.0)

    yy, xx = np.indices(arr.shape, dtype=float)
    bg = float(np.percentile(arr, 15.0))
    signal = np.clip(arr - bg, 0.0, None)
    mask = np.exp(-((xx - spot[0]) ** 2 + (yy - spot[1]) ** 2) / (2.0 * sigma**2))
    local_signal = signal * mask
    moved_signal = float(np.sum(local_signal))
    if moved_signal <= 1e-12:
        return arr.astype(np.float32, copy=False), meta

    shifted = ndimage.shift(local_signal, shift=(dy, dx), order=1, mode="constant", cval=0.0, prefilter=False)
    arr = arr - local_signal + shifted
    arr = np.clip(arr, 0.0, 65535.0)
    meta.update(
        {
            "geometry_disturbance_applied": True,
            "geometry_disturbance_spot_index": spot_idx,
            "geometry_disturbance_shift_px": float(shift_px),
            "geometry_disturbance_angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0),
            "geometry_disturbance_direction_mode": direction_mode,
            "geometry_disturbance_moved_signal_counts": moved_signal,
        }
    )
    return arr.astype(np.float32, copy=False), meta


def add_noise(image: np.ndarray, preset: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32).copy()
    subtract_percentile = preset.get("subtract_background_percentile", None)
    if subtract_percentile is not None:
        arr = np.clip(arr - float(np.percentile(arr, float(subtract_percentile))), 0.0, None)
    intensity_scale = float(preset.get("intensity_scale_fraction", 1.0))
    if intensity_scale != 1.0:
        arr *= max(intensity_scale, 0.0)
    background_offset = float(preset.get("background_offset_counts", 0.0))
    if background_offset != 0.0:
        arr += background_offset
    jitter = float(preset.get("intensity_jitter_fraction", 0.0))
    if jitter > 0:
        arr *= float(max(0.0, rng.normal(1.0, jitter)))
    bg_jitter = float(preset.get("background_jitter_counts", 0.0))
    if bg_jitter > 0:
        arr += float(rng.normal(0.0, bg_jitter))
    poisson_scale = float(preset.get("poisson_scale", 0.0))
    if poisson_scale > 0:
        scaled = np.clip(arr * poisson_scale, 0.0, 65535.0 * poisson_scale)
        arr = rng.poisson(scaled).astype(np.float32) / poisson_scale
    read_sigma = float(preset.get("read_noise_sigma_counts", 0.0))
    if read_sigma > 0:
        arr += rng.normal(0.0, read_sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 65535.0).astype(np.float32, copy=False)


def render_exposure_image(
    templates: np.ndarray,
    z_prev: float,
    z_curr: float,
    motion_preset: dict[str, Any],
    template_z_offset_um: float = 0.0,
) -> np.ndarray:
    exposure_fraction = float(motion_preset.get("exposure_fraction", 0.0))
    subframes = int(max(1, motion_preset.get("subframes", 1)))
    if exposure_fraction <= 0 or subframes <= 1:
        return interpolate_template(templates, float(z_curr) + float(template_z_offset_um))
    z_start = z_curr - exposure_fraction * (z_curr - z_prev)
    samples = np.linspace(z_start, z_curr, subframes)
    accum = np.zeros_like(templates[0], dtype=np.float32)
    for z in samples:
        z_sample = reflect_bounds(float(z), 0.0, templates.shape[0] - 1) + float(template_z_offset_um)
        accum += interpolate_template(templates, z_sample)
    return accum / float(subframes)


def exposure_z_summary(
    z_prev: float,
    z_curr: float,
    motion_preset: dict[str, Any],
    z_low: float,
    z_high: float,
) -> dict[str, float]:
    exposure_fraction = float(motion_preset.get("exposure_fraction", 0.0))
    subframes = int(max(1, motion_preset.get("subframes", 1)))
    if exposure_fraction <= 0 or subframes <= 1:
        z_sample = reflect_bounds(float(z_curr), z_low, z_high)
        return {
            "z_frame_end_um": float(z_curr),
            "z_prev_frame_end_um": float(z_prev),
            "z_exposure_start_um": float(z_curr),
            "z_exposure_mean_um": float(z_sample),
            "z_exposure_min_um": float(z_sample),
            "z_exposure_max_um": float(z_sample),
        }
    z_start = float(z_curr) - exposure_fraction * (float(z_curr) - float(z_prev))
    samples = np.array([reflect_bounds(float(z), z_low, z_high) for z in np.linspace(z_start, z_curr, subframes)], dtype=float)
    return {
        "z_frame_end_um": float(z_curr),
        "z_prev_frame_end_um": float(z_prev),
        "z_exposure_start_um": float(z_start),
        "z_exposure_mean_um": float(np.mean(samples)),
        "z_exposure_min_um": float(np.min(samples)),
        "z_exposure_max_um": float(np.max(samples)),
    }


def synthesize_stack(
    templates: np.ndarray,
    z_true: np.ndarray,
    noise_preset: dict[str, Any],
    aberration_params: dict[str, float],
    motion_preset: dict[str, Any],
    artifact_preset: dict[str, Any],
    geometry_preset: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    motion_angle = float(rng.uniform(0.0, 180.0))
    persistence_alpha = float(motion_preset.get("persistence_alpha", 0.0))
    line_blur_px = float(motion_preset.get("line_blur_px", 0.0))
    line_blur_angle_mode = str(motion_preset.get("line_blur_angle_mode", "stack"))
    artifact_persistent = bool(artifact_preset.get("artifact_persistent", False))
    template_z_offset_um = float(aberration_params.get("template_z_offset_um", 0.0))
    frames = []
    meta_rows = []
    prev_clean = None
    z_low = 0.0
    z_high = float(templates.shape[0] - 1)
    for frame_index, z in enumerate(z_true):
        z_prev = float(z_true[frame_index - 1]) if frame_index > 0 else float(z)
        z_meta = exposure_z_summary(z_prev, float(z), motion_preset, z_low, z_high)
        image = render_exposure_image(templates, z_prev, float(z), motion_preset, template_z_offset_um)
        image = apply_aberration(image, aberration_params)
        if prev_clean is not None and persistence_alpha > 0:
            image = (1.0 - persistence_alpha) * image + persistence_alpha * prev_clean
        if line_blur_px > 0:
            dz = abs(float(z) - z_prev)
            frame_motion_angle = float(rng.uniform(0.0, 180.0)) if line_blur_angle_mode == "per_frame" else motion_angle
            line_blur_effective_px = line_blur_px * (0.6 + 0.4 * min(dz, 2.5) / 2.5)
            image = apply_directional_blur(image, line_blur_effective_px, frame_motion_angle)
        else:
            frame_motion_angle = motion_angle
            line_blur_effective_px = 0.0
        persistence_base = image.astype(np.float32, copy=True)
        image, artifact_meta = apply_spot_artifact(image, artifact_preset, rng)
        image, geometry_meta = apply_geometry_disturbance(image, geometry_preset, float(z), rng)
        prev_clean = image.astype(np.float32, copy=True) if artifact_persistent else persistence_base
        image = add_noise(image, noise_preset, rng)
        frames.append(np.rint(image).astype(np.uint16))
        meta_rows.append(
            {
                "frame_index": frame_index,
                "motion_angle_deg": frame_motion_angle,
                "motion_line_blur_px": line_blur_effective_px,
                "motion_line_blur_nominal_px": line_blur_px,
                "motion_line_blur_angle_mode": line_blur_angle_mode,
                "motion_persistence_alpha": persistence_alpha,
                "motion_exposure_fraction": float(motion_preset.get("exposure_fraction", 0.0)),
                **z_meta,
                "template_z_offset_um": template_z_offset_um,
                **artifact_meta,
                **geometry_meta,
            }
        )
    return np.stack(frames, axis=0), pd.DataFrame(meta_rows)


def stack_hash(stack: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(stack).view(np.uint8)).hexdigest()


def intensity_centroid(roi: np.ndarray, background_percentile: float) -> tuple[float, float, bool, float, float]:
    arr = np.asarray(roi, dtype=np.float32)
    if arr.size == 0:
        return np.nan, np.nan, False, np.nan, np.nan
    bg = float(np.percentile(arr, background_percentile))
    signal = np.clip(arr - bg, 0.0, None)
    total = float(np.sum(signal))
    if not np.isfinite(total) or total <= 1e-12:
        return (arr.shape[1] - 1) / 2.0, (arr.shape[0] - 1) / 2.0, False, np.nan, bg
    yy, xx = np.indices(arr.shape)
    cx = float(np.sum(xx * signal) / total)
    cy = float(np.sum(yy * signal) / total)
    moment2 = float(np.sum(signal * ((xx - cx) ** 2 + (yy - cy) ** 2)) / total)
    return cx, cy, True, moment2, bg


def phasor_center(roi: np.ndarray, background_percentile: float) -> tuple[float, float, bool, float, float]:
    arr = np.asarray(roi, dtype=np.float32)
    if arr.size == 0:
        return np.nan, np.nan, False, np.nan, np.nan
    bg = float(np.percentile(arr, background_percentile))
    signal = np.clip(arr - bg, 0.0, None)
    total = float(np.sum(signal))
    if not np.isfinite(total) or total <= 1e-12:
        cx, cy, _, residual, _ = intensity_centroid(arr, background_percentile)
        return cx, cy, False, residual, np.nan
    h, w = signal.shape
    yy, xx = np.indices(signal.shape)
    fx = np.sum(signal * np.exp(-2j * np.pi * xx / max(w, 1)))
    fy = np.sum(signal * np.exp(-2j * np.pi * yy / max(h, 1)))
    px = float((-np.angle(fx) * w / (2 * np.pi)) % w)
    py = float((-np.angle(fy) * h / (2 * np.pi)) % h)
    cx, cy, _, residual, _ = intensity_centroid(arr, background_percentile)
    candidates_x = np.array([px - w, px, px + w], dtype=float)
    candidates_y = np.array([py - h, py, py + h], dtype=float)
    px = float(candidates_x[np.argmin(np.abs(candidates_x - cx))])
    py = float(candidates_y[np.argmin(np.abs(candidates_y - cy))])
    success = -1 <= px <= w and -1 <= py <= h and np.isfinite(px + py)
    quality = float(np.hypot(np.abs(fx), np.abs(fy)) / (total + 1e-12))
    return px, py, bool(success), residual, quality


def crop(image: np.ndarray, x: float, y: float, radius: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    return image[y0:y1, x0:x1], x0, y0


def find_four_peaks(image: np.ndarray, min_distance: int) -> list[tuple[float, float, float]]:
    arr = np.asarray(image, dtype=np.float32)
    signal = np.clip(arr - np.percentile(arr, 10.0), 0.0, None)
    score = ndimage.gaussian_filter(signal, sigma=1.4)
    coords = peak_local_max(
        score,
        min_distance=max(4, int(min_distance)),
        threshold_abs=float(np.percentile(score, 99.2)),
        num_peaks=16,
        exclude_border=False,
    )
    if len(coords) < 4:
        coords = peak_local_max(
            score,
            min_distance=max(4, int(min_distance * 0.6)),
            threshold_abs=float(np.percentile(score, 98.5)),
            num_peaks=24,
            exclude_border=False,
        )
    candidates = [(float(x), float(y), float(score[int(y), int(x)])) for y, x in coords]
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[float, float, float]] = []
    min_sep = max(5.0, 0.65 * float(min_distance))
    for cand in candidates:
        if all(np.hypot(cand[0] - prev[0], cand[1] - prev[1]) >= min_sep for prev in selected):
            selected.append(cand)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        return []
    pts = np.array([[p[0], p[1]] for p in selected], dtype=float)
    center = np.mean(pts, axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    return [selected[int(i)] for i in order]


def line_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    vector = p1 - p0
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
        return np.nan
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)


def geometry_from_coords(coords: np.ndarray) -> dict[str, Any]:
    if coords.shape != (4, 2) or not np.isfinite(coords).all():
        return {
            "success": False,
            "center_x": np.nan,
            "center_y": np.nan,
            "mean_diag_px": np.nan,
            "diag0_px": np.nan,
            "diag1_px": np.nan,
            "diag0_angle_deg": np.nan,
            "diag1_angle_deg": np.nan,
            "ordered_coords": np.full((4, 2), np.nan),
        }
    center = np.mean(coords, axis=0)
    order = np.argsort(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))
    ordered = coords[order]
    diag0 = float(np.linalg.norm(ordered[0] - ordered[2]))
    diag1 = float(np.linalg.norm(ordered[1] - ordered[3]))
    return {
        "success": True,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "mean_diag_px": float((diag0 + diag1) / 2.0),
        "diag0_px": diag0,
        "diag1_px": diag1,
        "diag0_angle_deg": line_angle_deg(ordered[0], ordered[2]),
        "diag1_angle_deg": line_angle_deg(ordered[1], ordered[3]),
        "ordered_coords": ordered,
    }


def localize_four_spots(
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
    rough = find_four_peaks(work, min_peak_distance)
    if len(rough) != 4:
        geom = geometry_from_coords(np.full((4, 2), np.nan))
        geom.update({"message": "peak_detection_failed", "localization_success_count": 0})
        return geom
    coords = []
    success_count = 0
    qualities = []
    for x, y, _score in rough:
        roi, x0, y0 = crop(work, x, y, roi_radius)
        cx, cy, ok, _residual, quality = phasor_center(roi, background_percentile)
        gx = float(x0 + cx)
        gy = float(y0 + cy)
        if ok and np.hypot(gx - x, gy - y) <= max(float(roi_radius), 1.0):
            success_count += 1
        coords.append((gx, gy))
        qualities.append(quality)
    geom = geometry_from_coords(np.asarray(coords, dtype=float))
    geom.update(
        {
            "message": "ok" if geom["success"] and success_count == 4 else "partial_phasor_success",
            "localization_success_count": success_count,
            "mean_quality": float(np.nanmean(qualities)) if len(qualities) else np.nan,
        }
    )
    return geom


def periodic_unwrap_near(value: float, reference: float, period: float = 180.0) -> float:
    if not np.isfinite(value):
        return np.nan
    if not np.isfinite(reference):
        return float(value)
    return float(value + period * np.round((reference - value) / period))


def track_two_diagonal_angles(diag0: np.ndarray, diag1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw0 = np.asarray(diag0, dtype=float)
    raw1 = np.asarray(diag1, dtype=float)
    tracked0 = np.full(len(raw0), np.nan, dtype=float)
    tracked1 = np.full(len(raw0), np.nan, dtype=float)
    avg = np.full(len(raw0), np.nan, dtype=float)
    source = np.array(["unavailable"] * len(raw0), dtype=object)
    prev0 = np.nan
    prev1 = np.nan
    prev_avg = np.nan

    def cost(values: list[float], refs: list[float], avg_value: float) -> float:
        costs = [abs(v - r) for v, r in zip(values, refs) if np.isfinite(v) and np.isfinite(r)]
        if np.isfinite(avg_value) and np.isfinite(prev_avg):
            costs.append(abs(avg_value - prev_avg))
        return float(np.sum(costs)) if costs else 0.0

    for idx in range(len(raw0)):
        a_ok = np.isfinite(raw0[idx])
        b_ok = np.isfinite(raw1[idx])
        if a_ok and b_ok:
            if np.isfinite(prev0) or np.isfinite(prev1):
                same0 = periodic_unwrap_near(raw0[idx], prev0)
                same1 = periodic_unwrap_near(raw1[idx], prev1)
                same_avg = periodic_unwrap_near((same0 + same1) / 2.0, prev_avg)
                swap0 = periodic_unwrap_near(raw1[idx], prev0)
                swap1 = periodic_unwrap_near(raw0[idx], prev1)
                swap_avg = periodic_unwrap_near((swap0 + swap1) / 2.0, prev_avg)
                if cost([swap0, swap1], [prev0, prev1], swap_avg) < cost([same0, same1], [prev0, prev1], same_avg):
                    tracked0[idx], tracked1[idx], avg[idx] = swap0, swap1, swap_avg
                else:
                    tracked0[idx], tracked1[idx], avg[idx] = same0, same1, same_avg
            else:
                tracked0[idx] = raw0[idx]
                tracked1[idx] = raw1[idx]
                avg[idx] = (tracked0[idx] + tracked1[idx]) / 2.0
            source[idx] = "two_diagonal"
        elif a_ok or b_ok:
            raw_value = raw0[idx] if a_ok else raw1[idx]
            candidates = []
            for branch_idx, previous in [(0, prev0), (1, prev1)]:
                unwrapped = periodic_unwrap_near(raw_value, previous)
                avg_value = unwrapped + 45.0 if branch_idx == 0 else unwrapped - 45.0
                avg_value = periodic_unwrap_near(avg_value, prev_avg)
                default_penalty = 0.0 if ((a_ok and branch_idx == 0) or (b_ok and branch_idx == 1)) else 1.0
                candidates.append((cost([unwrapped], [previous], avg_value) + default_penalty, branch_idx, unwrapped, avg_value))
            _score, branch_idx, unwrapped, avg_value = min(candidates, key=lambda item: item[0])
            if branch_idx == 0:
                tracked0[idx] = unwrapped
                source[idx] = "single_diag0"
            else:
                tracked1[idx] = unwrapped
                source[idx] = "single_diag1"
            avg[idx] = avg_value
        if np.isfinite(tracked0[idx]):
            prev0 = tracked0[idx]
        if np.isfinite(tracked1[idx]):
            prev1 = tracked1[idx]
        if np.isfinite(avg[idx]):
            prev_avg = avg[idx]
    return tracked0, tracked1, avg, source


def load_models(cfg: dict[str, Any]) -> dict[str, Any]:
    objective = str(cfg["objective_key"])
    angle_model_path = resolve_model_path(cfg, "angle_model_file", "phasor_angle_tracked_models.json")
    length_model_path = resolve_model_path(cfg, "length_model_file", "phasor_length_backup_models.json")
    with angle_model_path.open("r", encoding="utf-8") as fh:
        angle_payload = json.load(fh)
    with length_model_path.open("r", encoding="utf-8") as fh:
        length_payload = json.load(fh)
    angle_key = angle_payload.get("objective_aliases", {}).get(objective, objective)
    length_key = length_payload.get("objective_aliases", {}).get(objective, objective)
    if angle_key not in angle_payload["models"]:
        raise KeyError(f"No angle model for objective {objective!r}")
    if length_key not in length_payload["models"]:
        raise KeyError(f"No length model for objective {objective!r}")
    angle_model = angle_payload["models"][angle_key]
    length_model = length_payload["models"][length_key]
    angle_spline = angle_model["angle_spline"]
    length_correction = length_model["real_correction"]
    angle_order = np.argsort(angle_spline["knots_x_theta_deg"])
    length_order = np.argsort(length_correction["knots_x_z_uncorrected_um"])
    return {
        "angle_payload_name": angle_payload.get("model_name", ""),
        "length_payload_name": length_payload.get("model_name", ""),
        "angle_model_file": str(angle_model_path),
        "length_model_file": str(length_model_path),
        "angle_key": angle_key,
        "length_key": length_key,
        "angle_model": angle_model,
        "length_model": length_model,
        "angle_pchip": PchipInterpolator(
            np.asarray(angle_spline["knots_x_theta_deg"], dtype=float)[angle_order],
            np.asarray(angle_spline["knots_y_z_corrected_relative_um"], dtype=float)[angle_order],
            extrapolate=True,
        ),
        "angle_domain": tuple(map(float, angle_spline["domain_theta_deg"])),
        "length_pchip": PchipInterpolator(
            np.asarray(length_correction["knots_x_z_uncorrected_um"], dtype=float)[length_order],
            np.asarray(length_correction["knots_y_z_corrected_relative_um"], dtype=float)[length_order],
            extrapolate=True,
        ),
        "length_domain": tuple(map(float, length_correction["domain_z_uncorrected_um"])),
        "length_poly_coeff": np.asarray(length_model["simulation_poly2_coefficients"], dtype=float),
    }


def predict_angle_raw(theta_deg: np.ndarray, models: dict[str, Any]) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    out = np.full(len(theta), np.nan, dtype=float)
    mask = np.isfinite(theta)
    out[mask] = np.asarray(models["angle_pchip"](theta[mask]), dtype=float)
    return out


def predict_length_raw(mean_diag_px: np.ndarray, models: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean_diag = np.asarray(mean_diag_px, dtype=float)
    z_uncorrected = np.full(len(mean_diag), np.nan, dtype=float)
    z_raw = np.full(len(mean_diag), np.nan, dtype=float)
    mask = np.isfinite(mean_diag)
    z_uncorrected[mask] = np.polyval(models["length_poly_coeff"], mean_diag[mask])
    z_raw[mask] = np.asarray(models["length_pchip"](z_uncorrected[mask]), dtype=float)
    return z_raw, z_uncorrected


def fit_linear_alignment(raw_pred: np.ndarray, z_true: np.ndarray, method: str) -> dict[str, Any]:
    x = np.asarray(raw_pred, dtype=float)
    y = np.asarray(z_true, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        raise ValueError(f"Not enough finite points to fit {method} alignment")
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    aligned = slope * x[mask] + intercept
    residual = aligned - y[mask]
    return {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": int(np.sum(mask)),
        "alignment_rmse_um": float(np.sqrt(np.mean(residual**2))),
        "alignment_mae_um": float(np.mean(np.abs(residual))),
        "raw_min": float(np.nanmin(x[mask])),
        "raw_max": float(np.nanmax(x[mask])),
    }


def apply_alignment(raw_pred: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_pred, dtype=float)
    return float(alignment["slope"]) * raw + float(alignment["intercept"])


def localize_sequence(stack: np.ndarray, cfg: dict[str, Any], blur_sigma: float) -> pd.DataFrame:
    loc_cfg = cfg["localization"]
    rows = []
    for frame_index, image in enumerate(stack):
        geom = localize_four_spots(
            image,
            blur_sigma=blur_sigma,
            roi_radius=int(loc_cfg["roi_radius_px"]),
            min_peak_distance=int(loc_cfg["min_peak_distance_px"]),
            background_percentile=float(loc_cfg["background_percentile"]),
        )
        row = {
            "frame_index": int(frame_index),
            "success": bool(geom["success"]),
            "message": geom.get("message", ""),
            "localization_success_count": int(geom.get("localization_success_count", 0)),
            "center_x_px": geom["center_x"],
            "center_y_px": geom["center_y"],
            "mean_diag_px": geom["mean_diag_px"],
            "diag0_px": geom["diag0_px"],
            "diag1_px": geom["diag1_px"],
            "diag0_angle_deg": geom["diag0_angle_deg"],
            "diag1_angle_deg": geom["diag1_angle_deg"],
            "mean_quality": geom.get("mean_quality", np.nan),
        }
        ordered = np.asarray(geom.get("ordered_coords", np.full((4, 2), np.nan)), dtype=float)
        for idx in range(4):
            row[f"spot{idx}_x_px"] = float(ordered[idx, 0]) if ordered.shape == (4, 2) else np.nan
            row[f"spot{idx}_y_px"] = float(ordered[idx, 1]) if ordered.shape == (4, 2) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    tracked0, tracked1, theta_avg, source = track_two_diagonal_angles(
        out["diag0_angle_deg"].to_numpy(dtype=float),
        out["diag1_angle_deg"].to_numpy(dtype=float),
    )
    out["theta_tracked_diag0_unwrapped_deg"] = tracked0
    out["theta_tracked_diag1_unwrapped_deg"] = tracked1
    out["theta_tracked_simple_average_unwrapped_deg"] = theta_avg
    out["theta_source"] = source
    return out


def production_geometry_input(angle_loc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, loc in angle_loc.iterrows():
        coords = np.array(
            [[loc[f"spot{idx}_x_px"], loc[f"spot{idx}_y_px"]] for idx in range(4)],
            dtype=float,
        )
        finite = np.isfinite(coords).all(axis=1)
        available = int(np.sum(finite))
        row = {
            "Cluster_ID": 0,
            "FRAME": int(loc["frame_index"]),
            "Calculated_Size": float(loc["mean_diag_px"]),
            "Diag1": float(loc["diag0_px"]),
            "Diag2": float(loc["diag1_px"]),
            "Diag1_Angle_Deg": float(loc["diag0_angle_deg"]),
            "Diag2_Angle_Deg": float(loc["diag1_angle_deg"]),
            "Z_Input_X_Column": "sim_phasor_x",
            "Z_Input_Y_Column": "sim_phasor_y",
            "Available_Spot_Count": available,
            "Geometry_Mode": "four_spot" if bool(loc["success"]) else "failed",
            "Geometry_Success": bool(loc["success"]),
            "Calculated_Size_Source": "angle_blurred_phasor_geometry",
            "Single_Diagonal_Name": "",
            "Single_Diagonal_Length_Px": np.nan,
            "Single_Diagonal_Angle_Deg": np.nan,
            "Single_Diagonal_Track_ID_A": np.nan,
            "Single_Diagonal_Track_ID_B": np.nan,
            "Single_Diagonal_Source": "",
        }
        for idx in range(4):
            row[f"Spot{idx}_X_Px"] = float(coords[idx, 0]) if finite[idx] else np.nan
            row[f"Spot{idx}_Y_Px"] = float(coords[idx, 1]) if finite[idx] else np.nan
            row[f"Spot{idx}_Track_ID"] = idx if finite[idx] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def predict_stack_production_raw(
    stack: np.ndarray,
    z_true: np.ndarray,
    cfg: dict[str, Any],
    models: dict[str, Any],
) -> pd.DataFrame:
    angle_loc = localize_sequence(stack, cfg, blur_sigma=float(cfg["angle_blur_sigma_px"]))
    length_loc = localize_sequence(stack, cfg, blur_sigma=0.0)
    prod_z = load_production_z_module(cfg)
    prod_df = production_geometry_input(angle_loc)

    with production_z_environment():
        prod_df = prod_z.apply_angle_primary_model(prod_df, models["angle_model"])
        length_uncorrected = np.full(len(prod_df), np.nan, dtype=float)
        length_raw = np.full(len(prod_df), np.nan, dtype=float)
        length_domain_ok = np.zeros(len(prod_df), dtype=bool)
        length_eval = np.full(len(prod_df), np.nan, dtype=float)
        length_mean = length_loc["mean_diag_px"].to_numpy(dtype=float)
        finite = np.isfinite(length_mean)
        length_uncorrected[finite] = np.polyval(models["length_poly_coeff"], length_mean[finite])
        z_length, domain_ok, eval_x = prod_z.apply_spline(length_uncorrected, models["length_model"])
        length_raw[:] = z_length
        length_domain_ok[:] = domain_ok
        length_eval[:] = eval_x
        prod_df["Z_Length_Uncorrected_SimulationPoly2"] = length_uncorrected
        prod_df["Z_Length_Spline_Input"] = length_eval
        prod_df["Z_Length_Model_Domain_OK"] = length_domain_ok
        prod_df["Z_Length_Backup_Raw"] = length_raw
        prod_df["Z_Length_Backup"] = length_raw
        prod_df["Z_Calibration_Objective"] = models["angle_key"]
        prod_df["Z_Calibration_Method"] = "production_equivalent_angle_primary_length_backup"
        prod_df["Z_Localization_Method"] = "phasor_angle_blur_length_no_blur"
        prod_df["Z_Length_Backup_Method"] = "production_length_backup_on_unblurred_geometry"
        prod_df = prod_z.apply_angle_length_d2_qc(prod_df)

    with production_z_environment():
        raw_angle, raw_angle_domain_ok, raw_angle_eval = prod_z.apply_angle_spline(
            prod_df["Theta_Tracked_Simple_Average_Unwrapped_Deg"].to_numpy(dtype=float),
            models["angle_model"],
        )
    out = pd.DataFrame(
        {
            "frame_index": np.arange(len(z_true), dtype=int),
            "z_true_um": z_true,
            "production_frame": prod_df["FRAME"].to_numpy(dtype=int),
            "production_angle_primary_raw_z": prod_df["Z_Angle_Primary"].to_numpy(dtype=float),
            "production_raw_angle_clip_no_soft_z": raw_angle,
            "production_length_backup_raw_z": prod_df["Z_Length_Backup_Raw"].to_numpy(dtype=float),
            "production_length_backup_aligned_z": prod_df["Z_Length_Backup_Aligned"].to_numpy(dtype=float),
            "production_hybrid_final_raw_z": prod_df["Z_Final"].to_numpy(dtype=float),
            "production_final_source": prod_df["Z_Final_Source"].astype(str).to_numpy(),
            "production_confidence": prod_df["Z_Confidence"].astype(str).to_numpy(),
            "production_angle_qc_message": prod_df["Z_Angle_QC_Message"].astype(str).to_numpy(),
            "production_angle_soft_domain_corrected": prod_df["Z_Angle_Soft_Domain_Corrected"].to_numpy(dtype=bool),
            "production_angle_soft_domain_method": prod_df["Z_Angle_Soft_Domain_Method"].astype(str).to_numpy(),
            "production_angle_domain_distance_deg": prod_df["Z_Angle_Domain_Distance_Deg"].to_numpy(dtype=float),
            "production_angle_model_domain_ok": prod_df["Z_Angle_Model_Domain_OK"].to_numpy(dtype=bool),
            "production_length_model_domain_ok": prod_df["Z_Length_Model_Domain_OK"].to_numpy(dtype=bool),
            "production_angle_second_diff_abs": prod_df["Z_Angle_Second_Diff_Abs"].to_numpy(dtype=float),
            "production_length_second_diff_abs": prod_df["Z_Length_Second_Diff_Abs"].to_numpy(dtype=float),
            "production_angle_length_d2_relative_percent": prod_df["Angle_Length_D2_Relative_Percent"].to_numpy(dtype=float),
            "production_angle_jump_flag": prod_df["Z_Angle_Jump_Flag"].to_numpy(dtype=bool),
            "production_length_jump_flag": prod_df["Z_Length_Jump_Flag"].to_numpy(dtype=bool),
            "production_angle_only_jump_flag": prod_df["Angle_Only_Jump_Flag"].to_numpy(dtype=bool),
            "theta_tracked_simple_average_deg": prod_df["Theta_Tracked_Simple_Average_Unwrapped_Deg"].to_numpy(dtype=float),
            "theta_source": prod_df["Theta_Tracked_Simple_Average_Source"].astype(str).to_numpy(),
            "mean_diag_px": length_loc["mean_diag_px"].to_numpy(dtype=float),
            "angle_success": angle_loc["success"].to_numpy(dtype=bool),
            "length_success": length_loc["success"].to_numpy(dtype=bool),
            "angle_message": angle_loc["message"].astype(str).to_numpy(),
            "length_message": length_loc["message"].astype(str).to_numpy(),
            "angle_localization_success_count": angle_loc["localization_success_count"].to_numpy(dtype=int),
            "length_localization_success_count": length_loc["localization_success_count"].to_numpy(dtype=int),
            "length_uncorrected_z": prod_df["Z_Length_Uncorrected_SimulationPoly2"].to_numpy(dtype=float),
            "raw_angle_clip_eval_theta_deg": raw_angle_eval,
            "raw_angle_clip_domain_ok": raw_angle_domain_ok,
        }
    )
    return out


def apply_prediction_alignment(raw: pd.DataFrame, alignment: dict[str, dict[str, Any]]) -> pd.DataFrame:
    df = raw.copy()
    df["z_pred_pure_angle"] = apply_alignment(
        df["production_angle_primary_raw_z"].to_numpy(dtype=float),
        alignment["pure_angle"],
    )
    df["z_pred_pure_length"] = apply_alignment(
        df["production_length_backup_raw_z"].to_numpy(dtype=float),
        alignment["pure_length"],
    )
    df["z_pred_hybrid_angle_length_qc"] = apply_alignment(
        df["production_hybrid_final_raw_z"].to_numpy(dtype=float),
        alignment["hybrid_angle_length_qc"],
    )
    df["z_pred_raw_angle_clip_no_soft"] = apply_alignment(
        df["production_raw_angle_clip_no_soft_z"].to_numpy(dtype=float),
        alignment.get("raw_angle_clip_no_soft", alignment["pure_angle"]),
    )
    df["angle_raw_model_z"] = df["production_angle_primary_raw_z"]
    df["length_raw_model_z"] = df["production_length_backup_raw_z"]
    df["hybrid_source"] = df["production_final_source"]
    df["hybrid_confidence"] = df["production_confidence"]
    df["hybrid_qc_reason"] = df["production_angle_qc_message"]
    df["angle_second_diff_abs"] = df["production_angle_second_diff_abs"]
    df["length_second_diff_abs"] = df["production_length_second_diff_abs"]
    df["angle_second_diff"] = np.nan
    df["length_second_diff"] = np.nan
    df["angle_second_diff_normalized"] = np.nan
    df["length_second_diff_normalized"] = np.nan
    df["angle_length_d2_relative_percent"] = df["production_angle_length_d2_relative_percent"]
    df["angle_jump_flag"] = df["production_angle_jump_flag"]
    df["length_jump_flag"] = df["production_length_jump_flag"]
    df["angle_only_jump_flag"] = df["production_angle_only_jump_flag"]
    df["trend_interp_left_frame"] = np.nan
    df["trend_interp_right_frame"] = np.nan
    df["trend_interp_mode"] = ""
    df["trend_interp_ratio"] = np.nan
    df["angle_d2_norm_low"] = np.nan
    df["angle_d2_norm_high"] = np.nan
    df["length_d2_norm_low"] = np.nan
    df["length_d2_norm_high"] = np.nan
    df["angle_model_domain_ok"] = df["production_angle_model_domain_ok"]
    df["length_model_domain_ok"] = df["production_length_model_domain_ok"]
    for method, pred_col in [
        ("pure_angle", "z_pred_pure_angle"),
        ("pure_length", "z_pred_pure_length"),
        ("hybrid_angle_length_qc", "z_pred_hybrid_angle_length_qc"),
    ]:
        df[f"error_{method}_um"] = df[pred_col] - df["z_true_um"]
        df[f"abs_error_{method}_um"] = np.abs(df[f"error_{method}_um"])
    return df


def predict_stack_production_equivalent(
    stack: np.ndarray,
    z_true: np.ndarray,
    cfg: dict[str, Any],
    models: dict[str, Any],
    alignment: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    raw = predict_stack_production_raw(stack, z_true, cfg, models)
    return apply_prediction_alignment(raw, alignment)


def compute_clean_alignment(
    templates: np.ndarray,
    template_z: np.ndarray,
    cfg: dict[str, Any],
    models: dict[str, Any],
    dirs: dict[str, Path],
    *,
    production_equivalent: bool = False,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    clean_stack = np.rint(templates).astype(np.uint16)
    if production_equivalent:
        raw = predict_stack_production_raw(clean_stack, template_z, cfg, models)
        angle_alignment = fit_linear_alignment(raw["production_angle_primary_raw_z"], template_z, "pure_angle")
        raw_angle_alignment = fit_linear_alignment(raw["production_raw_angle_clip_no_soft_z"], template_z, "raw_angle_clip_no_soft")
        length_alignment = fit_linear_alignment(raw["production_length_backup_raw_z"], template_z, "pure_length")
        hybrid_alignment = dict(angle_alignment)
        hybrid_alignment["method"] = "hybrid_angle_length_qc"
        alignment = {
            "pure_angle": angle_alignment,
            "raw_angle_clip_no_soft": raw_angle_alignment,
            "pure_length": length_alignment,
            "hybrid_angle_length_qc": hybrid_alignment,
        }
        clean = apply_prediction_alignment(raw, alignment)
        pd.DataFrame(alignment.values()).to_csv(dirs["results"] / "alignment_parameters.csv", index=False)
        clean.to_csv(dirs["results"] / "clean_template_alignment_predictions.csv", index=False)
        return alignment, clean

    angle_loc = localize_sequence(clean_stack, cfg, blur_sigma=float(cfg["angle_blur_sigma_px"]))
    length_loc = localize_sequence(clean_stack, cfg, blur_sigma=0.0)
    angle_raw = predict_angle_raw(angle_loc["theta_tracked_simple_average_unwrapped_deg"].to_numpy(dtype=float), models)
    length_raw, length_uncorrected = predict_length_raw(length_loc["mean_diag_px"].to_numpy(dtype=float), models)
    angle_alignment = fit_linear_alignment(angle_raw, template_z, "pure_angle")
    length_alignment = fit_linear_alignment(length_raw, template_z, "pure_length")
    hybrid_alignment = dict(angle_alignment)
    hybrid_alignment["method"] = "hybrid_angle_length_qc"
    alignment = {
        "pure_angle": angle_alignment,
        "pure_length": length_alignment,
        "hybrid_angle_length_qc": hybrid_alignment,
    }
    clean = pd.DataFrame(
        {
            "frame_index": np.arange(len(template_z)),
            "z_true_um": template_z,
            "angle_theta_deg": angle_loc["theta_tracked_simple_average_unwrapped_deg"],
            "angle_raw_model_z": angle_raw,
            "angle_aligned_z_um": apply_alignment(angle_raw, angle_alignment),
            "length_mean_diag_px": length_loc["mean_diag_px"],
            "length_uncorrected_z": length_uncorrected,
            "length_raw_model_z": length_raw,
            "length_aligned_z_um": apply_alignment(length_raw, length_alignment),
            "angle_success": angle_loc["success"],
            "length_success": length_loc["success"],
        }
    )
    pd.DataFrame(alignment.values()).to_csv(dirs["results"] / "alignment_parameters.csv", index=False)
    clean.to_csv(dirs["results"] / "clean_template_alignment_predictions.csv", index=False)
    return alignment, clean


def robust_quantile_normalize(values: np.ndarray, low_percent: float, high_percent: float) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return out, np.nan, np.nan
    lo = float(np.nanpercentile(finite, low_percent))
    hi = float(np.nanpercentile(finite, high_percent))
    denom = hi - lo
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = max(float(np.nanmedian(np.abs(finite))) if len(finite) else 1.0, 1.0)
        lo = 0.0
    out[np.isfinite(arr)] = (arr[np.isfinite(arr)] - lo) / denom
    return out, lo, hi


def local_second_difference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    d2 = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= 3:
        d2[1:-1] = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
    return d2, np.abs(d2)


def signed_symmetric_relative_percent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    denom = np.abs(a) + np.abs(b)
    mask = np.isfinite(a) & np.isfinite(b) & (denom > 1e-12)
    out[mask] = 200.0 * (a[mask] - b[mask]) / denom[mask]
    return out


def nonmax_suppress(candidate: np.ndarray, score: np.ndarray, radius: int) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    score = np.asarray(score, dtype=float)
    keep = np.zeros(len(candidate), dtype=bool)
    for idx in np.where(candidate)[0]:
        lo = max(0, idx - radius)
        hi = min(len(candidate), idx + radius + 1)
        local = score[lo:hi]
        if np.isfinite(local).any() and score[idx] >= np.nanmax(local) - 1e-12:
            keep[idx] = True
    return keep


def interpolate_by_length_trend(
    z_angle: np.ndarray,
    z_length: np.ndarray,
    idx: int,
    left: int,
    right: int,
) -> tuple[float, str, float]:
    denom = float(z_length[right] - z_length[left])
    if np.isfinite(denom) and abs(denom) >= 0.2:
        ratio = float((z_length[idx] - z_length[left]) / denom)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return float(z_angle[left] + ratio * (z_angle[right] - z_angle[left])), "length_trend", ratio
    ratio = float((idx - left) / max(right - left, 1))
    return float(z_angle[left] + ratio * (z_angle[right] - z_angle[left])), "frame_linear_length_flat", ratio


def hybrid_qc(z_angle: np.ndarray, z_length: np.ndarray, angle_ok: np.ndarray, length_ok: np.ndarray, cfg: dict[str, Any]) -> pd.DataFrame:
    qc = cfg["qc"]
    z_angle = np.asarray(z_angle, dtype=float)
    z_length = np.asarray(z_length, dtype=float)
    angle_ok = np.asarray(angle_ok, dtype=bool) & np.isfinite(z_angle)
    length_ok = np.asarray(length_ok, dtype=bool) & np.isfinite(z_length)
    final = z_angle.copy()
    source = np.array(["angle"] * len(z_angle), dtype=object)
    confidence = np.array(["high"] * len(z_angle), dtype=object)
    reason = np.array(["ok"] * len(z_angle), dtype=object)
    fallback = (~angle_ok) & length_ok
    final[fallback] = z_length[fallback]
    source[fallback] = "length_fallback_angle_unavailable"
    confidence[fallback] = "low"
    reason[fallback] = "angle_unavailable"
    invalid = (~angle_ok) & (~length_ok)
    source[invalid] = "invalid"
    confidence[invalid] = "low"
    reason[invalid] = "angle_and_length_unavailable"

    angle_d2, angle_d2_abs = local_second_difference(z_angle)
    length_d2, length_d2_abs = local_second_difference(z_length)
    angle_norm, angle_lo, angle_hi = robust_quantile_normalize(
        angle_d2_abs,
        float(qc["trim_low_percent"]),
        float(qc["trim_high_percent"]),
    )
    length_norm, length_lo, length_hi = robust_quantile_normalize(
        length_d2_abs,
        float(qc["trim_low_percent"]),
        float(qc["trim_high_percent"]),
    )
    rel = signed_symmetric_relative_percent(angle_norm, length_norm)
    enabled = bool(qc.get("enabled", True))
    angle_candidate = (
        enabled
        & np.isfinite(rel)
        & np.isfinite(angle_norm)
        & (angle_norm >= float(qc["angle_d2_normalized_min"]))
        & (rel >= float(qc["angle_length_relative_threshold_percent"]))
    )
    length_candidate = (
        enabled
        & np.isfinite(rel)
        & np.isfinite(length_norm)
        & (length_norm >= float(qc["angle_d2_normalized_min"]))
        & (rel <= -float(qc["angle_length_relative_threshold_percent"]))
    )
    angle_jump = nonmax_suppress(angle_candidate, angle_norm, int(qc["jump_nms_radius"]))
    length_jump = nonmax_suppress(length_candidate, length_norm, int(qc["jump_nms_radius"]))
    angle_only = angle_jump & (~length_jump) & angle_ok & length_ok
    max_search = int(qc["trend_interp_max_search_frames"])
    left_frames = np.full(len(z_angle), np.nan)
    right_frames = np.full(len(z_angle), np.nan)
    interp_mode = np.array([""] * len(z_angle), dtype=object)
    interp_ratio = np.full(len(z_angle), np.nan)

    for idx in np.where(angle_only)[0]:
        left = None
        right = None
        for step in range(1, max_search + 1):
            cand = idx - step
            if cand >= 0 and not angle_jump[cand] and angle_ok[cand] and length_ok[cand]:
                left = cand
                break
        for step in range(1, max_search + 1):
            cand = idx + step
            if cand < len(z_angle) and not angle_jump[cand] and angle_ok[cand] and length_ok[cand]:
                right = cand
                break
        if left is None or right is None:
            confidence[idx] = "low"
            reason[idx] = "angle_only_jump_no_bracketing_frames"
            continue
        z_value, mode, ratio = interpolate_by_length_trend(z_angle, z_length, idx, left, right)
        final[idx] = z_value
        source[idx] = "angle_length_trend_interpolated"
        confidence[idx] = "medium_high"
        reason[idx] = f"angle_only_jump_interpolated:{mode}"
        left_frames[idx] = left
        right_frames[idx] = right
        interp_mode[idx] = mode
        interp_ratio[idx] = ratio

    both_jump = angle_jump & length_jump
    confidence[both_jump] = "low"
    reason[both_jump] = "angle_and_length_both_jump"
    return pd.DataFrame(
        {
            "z_pred_hybrid_angle_length_qc": final,
            "hybrid_source": source,
            "hybrid_confidence": confidence,
            "hybrid_qc_reason": reason,
            "angle_second_diff": angle_d2,
            "length_second_diff": length_d2,
            "angle_second_diff_abs": angle_d2_abs,
            "length_second_diff_abs": length_d2_abs,
            "angle_second_diff_normalized": angle_norm,
            "length_second_diff_normalized": length_norm,
            "angle_length_d2_relative_percent": rel,
            "angle_jump_flag": angle_jump,
            "length_jump_flag": length_jump,
            "angle_only_jump_flag": angle_only,
            "trend_interp_left_frame": left_frames,
            "trend_interp_right_frame": right_frames,
            "trend_interp_mode": interp_mode,
            "trend_interp_ratio": interp_ratio,
            "angle_d2_norm_low": angle_lo,
            "angle_d2_norm_high": angle_hi,
            "length_d2_norm_low": length_lo,
            "length_d2_norm_high": length_hi,
        }
    )


def predict_stack(
    stack: np.ndarray,
    z_true: np.ndarray,
    cfg: dict[str, Any],
    models: dict[str, Any],
    alignment: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    angle_loc = localize_sequence(stack, cfg, blur_sigma=float(cfg["angle_blur_sigma_px"]))
    length_loc = localize_sequence(stack, cfg, blur_sigma=0.0)
    angle_raw = predict_angle_raw(angle_loc["theta_tracked_simple_average_unwrapped_deg"].to_numpy(dtype=float), models)
    length_raw, length_uncorrected = predict_length_raw(length_loc["mean_diag_px"].to_numpy(dtype=float), models)
    angle_pred = apply_alignment(angle_raw, alignment["pure_angle"])
    length_pred = apply_alignment(length_raw, alignment["pure_length"])
    qc_df = hybrid_qc(
        angle_pred,
        length_pred,
        angle_loc["success"].to_numpy(dtype=bool),
        length_loc["success"].to_numpy(dtype=bool),
        cfg,
    )
    df = pd.DataFrame(
        {
            "frame_index": np.arange(len(z_true), dtype=int),
            "z_true_um": z_true,
            "z_pred_pure_angle": angle_pred,
            "z_pred_pure_length": length_pred,
            "angle_raw_model_z": angle_raw,
            "length_raw_model_z": length_raw,
            "length_uncorrected_z": length_uncorrected,
            "theta_tracked_simple_average_deg": angle_loc["theta_tracked_simple_average_unwrapped_deg"],
            "theta_source": angle_loc["theta_source"],
            "mean_diag_px": length_loc["mean_diag_px"],
            "angle_success": angle_loc["success"],
            "length_success": length_loc["success"],
            "angle_message": angle_loc["message"],
            "length_message": length_loc["message"],
            "angle_localization_success_count": angle_loc["localization_success_count"],
            "length_localization_success_count": length_loc["localization_success_count"],
        }
    )
    df = pd.concat([df, qc_df], axis=1)
    for method, pred_col in [
        ("pure_angle", "z_pred_pure_angle"),
        ("pure_length", "z_pred_pure_length"),
        ("hybrid_angle_length_qc", "z_pred_hybrid_angle_length_qc"),
    ]:
        df[f"error_{method}_um"] = df[pred_col] - df["z_true_um"]
        df[f"abs_error_{method}_um"] = np.abs(df[f"error_{method}_um"])
    lo, hi = map(float, models["angle_domain"])
    df["angle_model_domain_ok"] = df["theta_tracked_simple_average_deg"].between(lo, hi)
    llo, lhi = map(float, models["length_domain"])
    df["length_model_domain_ok"] = df["length_uncorrected_z"].between(llo, lhi)
    return df


def method_metrics(df: pd.DataFrame, method: str, pred_col: str) -> dict[str, Any]:
    pred = df[pred_col].to_numpy(dtype=float)
    true = df["z_true_um"].to_numpy(dtype=float)
    finite = np.isfinite(pred) & np.isfinite(true)
    err = pred[finite] - true[finite]
    abs_err = np.abs(err)
    return {
        "method": method,
        "n_frames": int(len(df)),
        "n_valid": int(np.sum(finite)),
        "failure_rate": float(1.0 - np.mean(finite)) if len(finite) else np.nan,
        "rmse_um": float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "mae_um": float(np.mean(abs_err)) if len(err) else np.nan,
        "bias_um": float(np.mean(err)) if len(err) else np.nan,
        "p95_abs_error_um": float(np.percentile(abs_err, 95)) if len(err) else np.nan,
        "max_abs_error_um": float(np.max(abs_err)) if len(err) else np.nan,
    }


def compute_condition_metrics(per_frame: pd.DataFrame) -> pd.DataFrame:
    pred_cols = {
        "hybrid_angle_length_qc": "z_pred_hybrid_angle_length_qc",
        "pure_angle": "z_pred_pure_angle",
        "pure_length": "z_pred_pure_length",
    }
    rows = []
    group_cols = ["stack_id", "noise_level", "aberration_level", "motion_level"]
    if "geometry_level" in per_frame.columns:
        group_cols.append("geometry_level")
    group_cols.extend(["artifact_level", "seed"])
    for keys, group in per_frame.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for method, pred_col in pred_cols.items():
            row = base | method_metrics(group, method, pred_col)
            if method == "hybrid_angle_length_qc":
                row["qc_interpolated_fraction"] = float(np.mean(group["hybrid_source"] == "angle_length_trend_interpolated"))
                row["length_fallback_fraction"] = float(np.mean(group["hybrid_source"] == "length_fallback_angle_unavailable"))
                row["low_confidence_fraction"] = float(np.mean(group["hybrid_confidence"] == "low"))
            else:
                row["qc_interpolated_fraction"] = np.nan
                row["length_fallback_fraction"] = np.nan
                row["low_confidence_fraction"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(condition_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["rmse_um", "mae_um", "bias_um", "p95_abs_error_um", "failure_rate"]
    agg_cols = ["noise_level", "aberration_level", "motion_level"]
    if "geometry_level" in condition_metrics.columns:
        agg_cols.append("geometry_level")
    agg_cols.extend(["artifact_level", "method"])
    rows = []
    for keys, group in condition_metrics.groupby(agg_cols, sort=False):
        row = dict(zip(agg_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            row[f"{col}_ci95"] = float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        rows.append(row)
    agg = pd.DataFrame(rows)

    piv = condition_metrics.pivot_table(
        index=[
            col
            for col in ["noise_level", "aberration_level", "motion_level", "geometry_level", "artifact_level", "seed"]
            if col in condition_metrics.columns
        ],
        columns="method",
        values="rmse_um",
        aggfunc="first",
    ).reset_index()
    rel_rows = []
    for row in piv.to_dict("records"):
        hybrid = row.get("hybrid_angle_length_qc", np.nan)
        angle = row.get("pure_angle", np.nan)
        length = row.get("pure_length", np.nan)
        rel_rows.append(
            {
                "noise_level": row["noise_level"],
                "aberration_level": row["aberration_level"],
                "motion_level": row["motion_level"],
                **({"geometry_level": row["geometry_level"]} if "geometry_level" in row else {}),
                "artifact_level": row["artifact_level"],
                "seed": row["seed"],
                "hybrid_rmse_um": hybrid,
                "pure_angle_rmse_um": angle,
                "pure_length_rmse_um": length,
                "hybrid_improvement_vs_pure_angle_percent": float((angle - hybrid) / angle * 100.0) if np.isfinite(angle) and angle > 0 and np.isfinite(hybrid) else np.nan,
                "hybrid_improvement_vs_pure_length_percent": float((length - hybrid) / length * 100.0) if np.isfinite(length) and length > 0 and np.isfinite(hybrid) else np.nan,
            }
        )
    rel = pd.DataFrame(rel_rows)
    rel_summary = (
        rel.groupby(
            [
                col
                for col in ["noise_level", "aberration_level", "motion_level", "geometry_level", "artifact_level"]
                if col in rel.columns
            ],
            as_index=False,
        )
        .agg(
            hybrid_improvement_vs_pure_angle_percent_mean=("hybrid_improvement_vs_pure_angle_percent", "mean"),
            hybrid_improvement_vs_pure_length_percent_mean=("hybrid_improvement_vs_pure_length_percent", "mean"),
        )
    )
    merge_cols = [
        col
        for col in ["noise_level", "aberration_level", "motion_level", "geometry_level", "artifact_level"]
        if col in agg.columns and col in rel_summary.columns
    ]
    agg = agg.merge(rel_summary, on=merge_cols, how="left")
    mask = agg["method"] != "hybrid_angle_length_qc"
    agg.loc[mask, ["hybrid_improvement_vs_pure_angle_percent_mean", "hybrid_improvement_vs_pure_length_percent_mean"]] = np.nan
    return agg, rel


def sort_levels(values: list[str], order: list[str]) -> list[str]:
    return sorted(values, key=lambda x: order.index(x) if x in order else len(order))


def plot_rmse_heatmap(aggregate: pd.DataFrame, dirs: dict[str, Path]) -> None:
    noise_levels = sort_levels(list(aggregate["noise_level"].dropna().unique()), NOISE_ORDER)
    aberration_levels = sort_levels(list(aggregate["aberration_level"].dropna().unique()), ABERRATION_ORDER)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.2 * len(METHODS), 4.5), constrained_layout=True)
    if len(METHODS) == 1:
        axes = [axes]
    vmax = float(np.nanmax(aggregate["rmse_um_mean"].to_numpy(dtype=float))) if len(aggregate) else 1.0
    for ax, method in zip(axes, METHODS):
        data = np.full((len(noise_levels), len(aberration_levels)), np.nan)
        sub = aggregate[aggregate["method"] == method]
        for i, noise in enumerate(noise_levels):
            for j, aberr in enumerate(aberration_levels):
                vals = sub[(sub["noise_level"] == noise) & (sub["aberration_level"] == aberr)]["rmse_um_mean"]
                if len(vals):
                    data[i, j] = float(np.nanmean(vals.to_numpy(dtype=float)))
        im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(method)
        ax.set_xticks(range(len(aberration_levels)), aberration_levels, rotation=30, ha="right")
        ax.set_yticks(range(len(noise_levels)), noise_levels)
        ax.set_xlabel("Aberration")
        ax.set_ylabel("Noise")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean RMSE (um)")
    fig.savefig(dirs["figures"] / "rmse_heatmap_by_method.png", dpi=180)
    plt.close(fig)


def plot_example_tracks(per_frame: pd.DataFrame, dirs: dict[str, Path]) -> None:
    preferred = per_frame[(per_frame["noise_level"] == "high") & (per_frame["aberration_level"] == "moderate") & (per_frame["seed"] == 0)]
    if preferred.empty:
        first_stack = per_frame["stack_id"].iloc[0]
        preferred = per_frame[per_frame["stack_id"] == first_stack]
    else:
        first_stack = preferred["stack_id"].iloc[0]
        preferred = per_frame[per_frame["stack_id"] == first_stack]
    stack_id = str(preferred["stack_id"].iloc[0])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(preferred["frame_index"], preferred["z_true_um"], "k-", lw=2.0, label="true exposure-mean Z")
    ax.plot(preferred["frame_index"], preferred["z_pred_hybrid_angle_length_qc"], "o-", ms=3, label="hybrid angle+length QC")
    ax.plot(preferred["frame_index"], preferred["z_pred_pure_angle"], "s--", ms=3, label="pure angle")
    ax.plot(preferred["frame_index"], preferred["z_pred_pure_length"], "^--", ms=3, label="pure length")
    ax.set_title(f"Example Brownian-Z reconstruction: {stack_id}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z (um)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(dirs["figures"] / "example_brownian_tracks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_qc_examples(per_frame: pd.DataFrame, dirs: dict[str, Path]) -> None:
    corrections = per_frame[per_frame["hybrid_source"].astype(str) != "angle"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not corrections.empty:
        row = corrections.sort_values("abs_error_pure_angle_um", ascending=False).iloc[0]
        stack = per_frame[per_frame["stack_id"] == row["stack_id"]].copy()
        frame = int(row["frame_index"])
        window = stack[(stack["frame_index"] >= frame - 10) & (stack["frame_index"] <= frame + 10)]
        ax.plot(window["frame_index"], window["z_true_um"], "k-", lw=2.0, label="true exposure-mean Z")
        ax.plot(window["frame_index"], window["z_pred_pure_angle"], "s--", ms=4, label="pure angle")
        ax.plot(window["frame_index"], window["z_pred_pure_length"], "^--", ms=4, label="pure length")
        ax.plot(window["frame_index"], window["z_pred_hybrid_angle_length_qc"], "o-", ms=4, label="hybrid")
        ax.axvline(frame, color="tab:red", ls=":", label=str(row["hybrid_source"]))
        ax.set_title(f"Hybrid intervention example: {row['stack_id']}, frame {frame}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Z (um)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        by_frame = per_frame.copy()
        ax.scatter(by_frame["abs_error_pure_angle_um"], by_frame["abs_error_hybrid_angle_length_qc_um"], s=8, alpha=0.35)
        lim = float(np.nanmax([by_frame["abs_error_pure_angle_um"].max(), by_frame["abs_error_hybrid_angle_length_qc_um"].max()]))
        lim = max(lim, 1.0)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title("No hybrid intervention was triggered; hybrid equals angle except fallbacks")
        ax.set_xlabel("Pure angle abs error (um)")
        ax.set_ylabel("Hybrid abs error (um)")
        ax.grid(True, alpha=0.3)
    fig.savefig(dirs["figures"] / "qc_correction_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    cfg: dict[str, Any],
    mode: str,
    manifest: pd.DataFrame,
    condition: pd.DataFrame,
    aggregate: pd.DataFrame,
    alignment: dict[str, dict[str, Any]],
    dirs: dict[str, Path],
) -> None:
    report = dirs["reports"] / "benchmark_report.md"
    hybrid = aggregate[aggregate["method"] == "hybrid_angle_length_qc"].copy()
    best_rows = hybrid.sort_values("rmse_um_mean").head(5)
    worst_rows = hybrid.sort_values("rmse_um_mean", ascending=False).head(5)
    overall = (
        condition.groupby("method", as_index=False)
        .agg(rmse_um_mean=("rmse_um", "mean"), mae_um_mean=("mae_um", "mean"), failure_rate_mean=("failure_rate", "mean"))
        .sort_values("rmse_um_mean")
    )
    qc_summary_lines = []
    per_frame_path = dirs["results"] / "per_frame_predictions.csv"
    if per_frame_path.exists():
        pf = pd.read_csv(
            per_frame_path,
            usecols=lambda c: c
            in {
                "hybrid_source",
                "hybrid_confidence",
                "angle_jump_flag",
                "length_jump_flag",
                "artifact_applied",
                "artifact_direction_mode_applied",
                "artifact_energy_conserving",
                "artifact_persistent",
                "geometry_disturbance_applied",
                "domain_pressure_applied",
                "production_angle_soft_domain_corrected",
                "production_final_source",
            },
        )
        artifact_frames = int(pf["artifact_applied"].sum()) if "artifact_applied" in pf.columns else 0
        geometry_frames = int(pf["geometry_disturbance_applied"].sum()) if "geometry_disturbance_applied" in pf.columns else 0
        domain_pressure_frames = int(pf["domain_pressure_applied"].sum()) if "domain_pressure_applied" in pf.columns else 0
        soft_domain_frames = (
            int(pf["production_angle_soft_domain_corrected"].sum())
            if "production_angle_soft_domain_corrected" in pf.columns
            else int((pf["hybrid_source"] == "angle_soft_domain").sum())
        )
        direction_summary = ""
        if "artifact_direction_mode_applied" in pf.columns and artifact_frames > 0:
            counts = (
                pf.loc[pf["artifact_applied"].astype(bool), "artifact_direction_mode_applied"]
                .fillna("")
                .replace("", "unspecified")
                .value_counts()
            )
            direction_summary = ", ".join(f"{idx}: {int(val)}" for idx, val in counts.items())
        energy_conserving_frames = (
            int((pf["artifact_applied"].astype(bool) & pf["artifact_energy_conserving"].astype(bool)).sum())
            if {"artifact_applied", "artifact_energy_conserving"}.issubset(pf.columns)
            else 0
        )
        persistent_artifact_frames = (
            int((pf["artifact_applied"].astype(bool) & pf["artifact_persistent"].astype(bool)).sum())
            if {"artifact_applied", "artifact_persistent"}.issubset(pf.columns)
            else 0
        )
        qc_summary_lines = [
            "## QC trigger summary",
            "",
            f"- Frames: `{len(pf)}`",
            f"- Hybrid angle-length trend interpolations: `{int((pf['hybrid_source'] == 'angle_length_trend_interpolated').sum())}`",
            f"- Hybrid non-angle source frames: `{int((pf['hybrid_source'].astype(str) != 'angle').sum())}`",
            f"- Production soft-domain frames: `{soft_domain_frames}`",
            f"- Angle jump flags: `{int(pf['angle_jump_flag'].sum())}`",
            f"- Length jump flags: `{int(pf['length_jump_flag'].sum())}`",
            f"- Low-confidence hybrid frames: `{int((pf['hybrid_confidence'] == 'low').sum())}`",
            f"- Frames with local geometry disturbance: `{geometry_frames}`",
            f"- Frames with low-Z domain pressure: `{domain_pressure_frames}`",
            f"- Frames with legacy local spot ghost/dropout artifact: `{artifact_frames}`",
            f"- Energy-conserving artifact frames: `{energy_conserving_frames}`",
            f"- Artifact-persistent frames: `{persistent_artifact_frames}`",
            f"- Applied artifact direction modes: `{direction_summary or 'none'}`",
            "",
        ]
    lines = [
        "# 20X Brownian-Z Method Benchmark",
        "",
        f"- Mode: `{mode}`",
        f"- Stacks generated: `{len(manifest)}`",
        f"- Frames per stack: `{int(manifest['n_frames'].iloc[0]) if len(manifest) else 'n/a'}`",
        f"- Angle blur sigma: `{cfg['angle_blur_sigma_px']}` px",
        f"- Objective: `{cfg['objective_key']}`",
        "- Ground-truth Z for error metrics: exposure-mean Z (`z_exposure_mean_um`), with frame-end Z retained as `z_frame_end_um`.",
        "",
        "## Linear alignment parameters",
        "",
        pd.DataFrame(alignment.values()).to_markdown(index=False),
        "",
        "## Overall method ranking",
        "",
        overall.to_markdown(index=False),
        "",
        *qc_summary_lines,
        "## Best hybrid conditions",
        "",
        best_rows[
            [
                col
                for col in [
                    "noise_level",
                    "aberration_level",
                    "motion_level",
                    "geometry_level",
                    "artifact_level",
                    "rmse_um_mean",
                    "mae_um_mean",
                    "hybrid_improvement_vs_pure_angle_percent_mean",
                    "hybrid_improvement_vs_pure_length_percent_mean",
                ]
                if col in best_rows.columns
            ]
        ].to_markdown(index=False),
        "",
        "## Worst hybrid conditions",
        "",
        worst_rows[
            [
                col
                for col in [
                    "noise_level",
                    "aberration_level",
                    "motion_level",
                    "geometry_level",
                    "artifact_level",
                    "rmse_um_mean",
                    "mae_um_mean",
                    "hybrid_improvement_vs_pure_angle_percent_mean",
                    "hybrid_improvement_vs_pure_length_percent_mean",
                ]
                if col in worst_rows.columns
            ]
        ].to_markdown(index=False),
        "",
        "## Output files",
        "",
        "- `results/sweep_manifest.csv`",
        "- `results/per_frame_predictions.csv`",
        "- `results/condition_metrics.csv`",
        "- `results/aggregate_metrics.csv`",
        "- `results/relative_improvements.csv`",
        "- `results/alignment_parameters.csv`",
        "- `results/clean_template_alignment_predictions.csv`",
        "- `figures/rmse_heatmap_by_method.png`",
        "- `figures/example_brownian_tracks.png`",
        "- `figures/qc_correction_examples.png`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def run(mode: str, config_path: Path) -> None:
    cfg = load_config(config_path)
    if mode not in cfg or not isinstance(cfg.get(mode), dict) or "frames_per_stack" not in cfg[mode]:
        available_modes = sorted(
            key
            for key, value in cfg.items()
            if isinstance(value, dict) and "frames_per_stack" in value
        )
        raise KeyError(f"Unknown benchmark mode {mode!r}. Available modes: {', '.join(available_modes)}")
    dirs = output_dirs(cfg, mode)
    run_cfg = cfg[mode]
    templates, template_z, template_manifest = load_templates(cfg)
    template_manifest.to_csv(dirs["results"] / "template_manifest.csv", index=False)
    aberration_template_sets: dict[str, np.ndarray] = {}
    for level, preset in cfg.get("aberration_presets", {}).items():
        if preset.get("template_source") == "roi_for_check":
            ref_templates, ref_z, ref_manifest = load_template_stack(
                cfg,
                cfg["reference_inputs"]["roi_check_template_dir"],
                filename_prefix=str(cfg["reference_inputs"].get("roi_check_file_prefix", "original-")),
                source_label=f"aberration:{level}",
            )
            if ref_templates.shape[1:] != templates.shape[1:]:
                ref_manifest["source_height_px"] = ref_templates.shape[1]
                ref_manifest["source_width_px"] = ref_templates.shape[2]
                ref_manifest["shape_adapter"] = "center_pad_or_crop"
                ref_templates = fit_template_stack_to_shape(ref_templates, templates.shape[1:])
            if ref_templates.shape != templates.shape:
                raise ValueError(f"Reference template stack for {level} could not be adapted to {templates.shape}")
            if not np.allclose(ref_z, template_z):
                raise ValueError(f"Reference template z grid for {level} does not match the primary template z grid")
            aberration_template_sets[level] = ref_templates
            ref_manifest.to_csv(dirs["results"] / f"template_manifest_{safe_name(level)}.csv", index=False)
    models = load_models(cfg)
    use_production_equivalent = bool(run_cfg.get("use_production_equivalent_z", cfg.get("use_production_equivalent_z", False)))
    alignment, _clean = compute_clean_alignment(
        templates,
        template_z,
        cfg,
        models,
        dirs,
        production_equivalent=use_production_equivalent,
    )

    frames_per_stack = int(run_cfg["frames_per_stack"])
    seeds = [int(s) for s in run_cfg["seeds"]]
    noise_levels = list(run_cfg["noise_levels"])
    aberration_levels = list(run_cfg["aberration_levels"])
    motion_levels = list(run_cfg.get("motion_levels", ["none"]))
    geometry_levels = list(run_cfg.get("geometry_levels", ["none"]))
    artifact_levels = list(run_cfg.get("artifact_levels", ["none"]))
    manifest_rows: list[dict[str, Any]] = []
    per_frame_parts: list[pd.DataFrame] = []
    total = len(noise_levels) * len(aberration_levels) * len(motion_levels) * len(geometry_levels) * len(artifact_levels) * len(seeds)
    stack_counter = 0
    print(f"Running {mode} benchmark: {total} stacks x {frames_per_stack} frames")

    for noise_level in noise_levels:
        noise_preset = cfg["noise_presets"][noise_level]
        for aberration_level in aberration_levels:
            aberr_preset = cfg["aberration_presets"][aberration_level]
            for motion_level in motion_levels:
                motion_preset = cfg.get("motion_presets", {"none": {}})[motion_level]
                for geometry_level in geometry_levels:
                    geometry_preset = cfg.get("geometry_disturbance_presets", {"none": {}})[geometry_level]
                    for artifact_level in artifact_levels:
                        artifact_preset = cfg.get("artifact_presets", {"none": {}})[artifact_level]
                        for seed in seeds:
                            stack_counter += 1
                            stack_id = (
                                f"noise-{safe_name(noise_level)}_aberr-{safe_name(aberration_level)}"
                                f"_motion-{safe_name(motion_level)}_geom-{safe_name(geometry_level)}"
                                f"_artifact-{safe_name(artifact_level)}_seed-{seed:02d}"
                            )
                            stack_seed = int(
                                seed
                                + 1000 * noise_levels.index(noise_level)
                                + 10000 * aberration_levels.index(aberration_level)
                                + 100000 * motion_levels.index(motion_level)
                                + 1000000 * geometry_levels.index(geometry_level)
                                + 10000000 * artifact_levels.index(artifact_level)
                            )
                            z_frame_end = generate_z_for_run(cfg, run_cfg, stack_seed, frames_per_stack)
                            aberr_rng = np.random.default_rng(stack_seed + 123456)
                            aberr_params = make_aberration_params(aberration_level, aberr_preset, aberr_rng)
                            source_templates = aberration_template_sets.get(aberration_level, templates)
                            stack, frame_meta = synthesize_stack(
                                source_templates,
                                z_frame_end,
                                noise_preset,
                                aberr_params,
                                motion_preset,
                                artifact_preset,
                                geometry_preset,
                                stack_seed + 654321,
                            )
                            stack_path = dirs["stacks"] / f"{stack_id}.tif"
                            tifffile.imwrite(stack_path, stack, photometric="minisblack")
                            sha = stack_hash(stack)
                            z_eval = frame_meta["z_exposure_mean_um"].to_numpy(dtype=float)
                            if use_production_equivalent:
                                pred = predict_stack_production_equivalent(stack, z_eval, cfg, models, alignment)
                            else:
                                pred = predict_stack(stack, z_eval, cfg, models, alignment)
                            pred = pred.merge(frame_meta, on="frame_index", how="left")
                            pred.insert(0, "stack_id", stack_id)
                            pred.insert(1, "noise_level", noise_level)
                            pred.insert(2, "aberration_level", aberration_level)
                            pred.insert(3, "motion_level", motion_level)
                            pred.insert(4, "geometry_level", geometry_level)
                            pred.insert(5, "artifact_level", artifact_level)
                            pred.insert(6, "seed", seed)
                            per_frame_parts.append(pred)
                            manifest_rows.append(
                                {
                                    "stack_id": stack_id,
                                    "stack_path": str(stack_path),
                                    "noise_level": noise_level,
                                    "aberration_level": aberration_level,
                                    "motion_level": motion_level,
                                    "geometry_level": geometry_level,
                                    "artifact_level": artifact_level,
                                    "seed": seed,
                                    "stack_seed": stack_seed,
                                    "n_frames": frames_per_stack,
                                    "z_min_um": float(np.min(z_eval)),
                                    "z_max_um": float(np.max(z_eval)),
                                    "z_mean_um": float(np.mean(z_eval)),
                                    "z_truth_basis": "exposure_mean",
                                    "z_frame_end_min_um": float(np.min(z_frame_end)),
                                    "z_frame_end_max_um": float(np.max(z_frame_end)),
                                    "z_frame_end_mean_um": float(np.mean(z_frame_end)),
                                    "stack_sha256": sha,
                                    "artifact_applied_frames": int(frame_meta["artifact_applied"].sum()) if "artifact_applied" in frame_meta else 0,
                                    "geometry_disturbance_applied_frames": int(frame_meta["geometry_disturbance_applied"].sum()) if "geometry_disturbance_applied" in frame_meta else 0,
                                    "domain_pressure_applied_frames": int(frame_meta["domain_pressure_applied"].sum()) if "domain_pressure_applied" in frame_meta else 0,
                                    "use_production_equivalent_z": use_production_equivalent,
                                    **{f"aberration_{k}": v for k, v in aberr_params.items() if k != "level"},
                                    **{f"motion_{k}": v for k, v in motion_preset.items()},
                                    **{f"geometry_{k}": v for k, v in geometry_preset.items()},
                                    **{f"artifact_{k}": v for k, v in artifact_preset.items()},
                                }
                            )
                            print(f"[{stack_counter:03d}/{total:03d}] {stack_id} written")

    manifest = pd.DataFrame(manifest_rows)
    per_frame = pd.concat(per_frame_parts, ignore_index=True) if per_frame_parts else pd.DataFrame()
    condition = compute_condition_metrics(per_frame)
    aggregate, relative = aggregate_metrics(condition)

    manifest.to_csv(dirs["results"] / "sweep_manifest.csv", index=False)
    per_frame.to_csv(dirs["results"] / "per_frame_predictions.csv", index=False)
    condition.to_csv(dirs["results"] / "condition_metrics.csv", index=False)
    aggregate.to_csv(dirs["results"] / "aggregate_metrics.csv", index=False)
    relative.to_csv(dirs["results"] / "relative_improvements.csv", index=False)

    plot_rmse_heatmap(aggregate, dirs)
    plot_example_tracks(per_frame, dirs)
    plot_qc_examples(per_frame, dirs)
    write_report(cfg, mode, manifest, condition, aggregate, alignment, dirs)
    print(f"Done. Results: {dirs['results']}")
    print(f"Figures: {dirs['figures']}")
    print(f"Report: {dirs['reports'] / 'benchmark_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="20X Brownian-Z simulation benchmark for angle, length, and hybrid Z methods.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    run(args.mode, args.config)


if __name__ == "__main__":
    main()
