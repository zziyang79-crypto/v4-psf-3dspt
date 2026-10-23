from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation


METHODS = [
    "trackmate_log",
    "centroid",
    "radial_symmetry",
    "gaussian2d",
    "elliptical_gaussian",
    "phasor",
    "template_registration",
    "psf_mle",
]


OBJECTIVE_PARAMS: dict[str, dict[str, float]] = {
    "20x_0.45NA_air": {
        "log_sigma": 3.0,
        "min_distance": 20.0,
        "roi_radius": 12.0,
        "template_radius": 14.0,
    },
    "40x_0.6NA_air": {
        "log_sigma": 4.0,
        "min_distance": 20.0,
        "roi_radius": 16.0,
        "template_radius": 18.0,
    },
    "60x_1.0NA_water": {
        "log_sigma": 4.0,
        "min_distance": 20.0,
        "roi_radius": 16.0,
        "template_radius": 18.0,
    },
}


@dataclass
class SpotLocalization:
    x: float
    y: float
    success: bool
    method: str
    quality: float = np.nan
    residual: float = np.nan
    background: float = np.nan
    intensity: float = np.nan
    runtime_ms: float = np.nan
    message: str = ""
    direct_z: float = np.nan

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FourSpotLocalization:
    objective: str
    method: str
    spots: list[SpotLocalization]
    center_x: float
    center_y: float
    mean_diag_px: float
    success: bool
    quality: float
    residual: float
    runtime_ms: float
    message: str = ""
    direct_z: float = np.nan


def load_tiff_stack(path: Path):
    with Image.open(path) as img:
        frame = 0
        while True:
            try:
                img.seek(frame)
            except EOFError:
                break
            yield frame, np.asarray(img, dtype=np.float32)
            frame += 1


def robust_8bit(image: np.ndarray, p_low: float = 0.5, p_high: float = 99.8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(np.round(out * 255), dtype=np.uint8)


def preprocess_signal(image: np.ndarray, bg_percentile: float = 10.0) -> tuple[np.ndarray, float]:
    arr = np.asarray(image, dtype=np.float32)
    bg = float(np.percentile(arr, bg_percentile))
    signal = np.clip(arr - bg, 0.0, None)
    return signal, bg


def image_center(image: np.ndarray) -> tuple[float, float]:
    h, w = image.shape[:2]
    return float(w // 2), float(h // 2)


def crop_with_bounds(
    image: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    return image[y0:y1, x0:x1], x0, y0


def quadratic_peak_offset(values: np.ndarray, y: int, x: int) -> tuple[float, float]:
    h, w = values.shape
    if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
        return 0.0, 0.0
    center = float(values[y, x])
    denom_x = float(values[y, x - 1] - 2 * center + values[y, x + 1])
    denom_y = float(values[y - 1, x] - 2 * center + values[y + 1, x])
    dx = 0.0 if abs(denom_x) < 1e-12 else 0.5 * float(values[y, x - 1] - values[y, x + 1]) / denom_x
    dy = 0.0 if abs(denom_y) < 1e-12 else 0.5 * float(values[y - 1, x] - values[y + 1, x]) / denom_y
    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def sort_spots_by_angle(spots: list[SpotLocalization], center: tuple[float, float] | None = None) -> list[SpotLocalization]:
    if len(spots) <= 1:
        return spots
    if center is None:
        cx = float(np.nanmean([s.x for s in spots]))
        cy = float(np.nanmean([s.y for s in spots]))
    else:
        cx, cy = center
    return sorted(spots, key=lambda s: np.arctan2(s.y - cy, s.x - cx))


def mean_diagonal(spots: list[SpotLocalization]) -> float:
    if len(spots) != 4 or any(not np.isfinite(s.x + s.y) for s in spots):
        return np.nan
    ordered = sort_spots_by_angle(spots)
    pts = np.array([[s.x, s.y] for s in ordered], dtype=float)
    return float((np.linalg.norm(pts[0] - pts[2]) + np.linalg.norm(pts[1] - pts[3])) / 2.0)


def find_four_peaks(
    image: np.ndarray,
    objective: str,
    *,
    use_log: bool = False,
    exclude_center_radius: float = 3.0,
) -> list[tuple[float, float, float]]:
    params = OBJECTIVE_PARAMS.get(objective, OBJECTIVE_PARAMS["40x_0.6NA_air"])
    min_distance = int(params["min_distance"])
    sigma = float(params["log_sigma"])
    signal, _ = preprocess_signal(image)
    if use_log:
        response = -ndimage.gaussian_laplace(signal, sigma=sigma) * sigma * sigma
        score_img = response
    else:
        score_img = ndimage.gaussian_filter(signal, sigma=max(1.0, sigma / 2.0))

    threshold = float(np.percentile(score_img, 99.7))
    coords = peak_local_max(
        score_img,
        min_distance=min_distance,
        num_peaks=16,
        threshold_abs=threshold,
        exclude_border=False,
    )
    if len(coords) < 4:
        coords = peak_local_max(
            score_img,
            min_distance=max(5, min_distance // 2),
            num_peaks=16,
            threshold_abs=float(np.percentile(score_img, 99.0)),
            exclude_border=False,
        )
    cx, cy = image_center(image)
    candidates: list[tuple[float, float, float]] = []
    for y, x in coords:
        if np.hypot(float(x) - cx, float(y) - cy) < exclude_center_radius:
            continue
        dx, dy = quadratic_peak_offset(score_img, int(y), int(x)) if use_log else (0.0, 0.0)
        candidates.append((float(x) + dx, float(y) + dy, float(score_img[int(y), int(x)])))
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[float, float, float]] = []
    for cand in candidates:
        if all(np.hypot(cand[0] - prev[0], cand[1] - prev[1]) >= min_distance * 0.7 for prev in selected):
            selected.append(cand)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        for cand in candidates:
            if cand not in selected:
                selected.append(cand)
            if len(selected) == 4:
                break
    rough_spots = [SpotLocalization(x=c[0], y=c[1], success=True, method="rough", quality=c[2]) for c in selected[:4]]
    rough_spots = sort_spots_by_angle(rough_spots, center=(cx, cy))
    return [(s.x, s.y, s.quality) for s in rough_spots]


def localize_centroid(image: np.ndarray, rough_x: float, rough_y: float, radius: int) -> SpotLocalization:
    start = perf_counter()
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    signal, bg = preprocess_signal(roi)
    yy, xx = np.indices(signal.shape)
    total = float(signal.sum())
    if total <= 0 or not np.isfinite(total):
        return SpotLocalization(rough_x, rough_y, False, "centroid", background=bg, runtime_ms=(perf_counter() - start) * 1000, message="empty signal")
    x = float(x0 + (xx * signal).sum() / total)
    y = float(y0 + (yy * signal).sum() / total)
    return SpotLocalization(x, y, True, "centroid", quality=float(signal.max()), background=bg, intensity=total, runtime_ms=(perf_counter() - start) * 1000)


def localize_radial_symmetry(image: np.ndarray, rough_x: float, rough_y: float, radius: int) -> SpotLocalization:
    start = perf_counter()
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    signal, bg = preprocess_signal(roi)
    smooth = ndimage.gaussian_filter(signal.astype(float), 1.0)
    gy, gx = np.gradient(smooth)
    mag2 = gx * gx + gy * gy
    if not np.isfinite(mag2).any() or float(mag2.max()) <= 0:
        return SpotLocalization(rough_x, rough_y, False, "radial_symmetry", background=bg, runtime_ms=(perf_counter() - start) * 1000, message="no gradient")
    threshold = np.percentile(mag2, 70)
    mask = mag2 > threshold
    if int(mask.sum()) < 6:
        mask = mag2 > 0
    yy, xx = np.indices(smooth.shape)
    a1 = -gy[mask].ravel()
    a2 = gx[mask].ravel()
    b = a1 * xx[mask].ravel() + a2 * yy[mask].ravel()
    w = mag2[mask].ravel()
    A = np.column_stack([a1, a2])
    Aw = A * np.sqrt(w[:, None])
    bw = b * np.sqrt(w)
    try:
        sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        lx, ly = float(sol[0]), float(sol[1])
        if lx < -radius or ly < -radius or lx > roi.shape[1] + radius or ly > roi.shape[0] + radius:
            raise ValueError("radial solution outside bounds")
        residual = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
        return SpotLocalization(x0 + lx, y0 + ly, True, "radial_symmetry", quality=float(mag2.max()), residual=residual, background=bg, intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000)
    except Exception as exc:
        return SpotLocalization(rough_x, rough_y, False, "radial_symmetry", background=bg, runtime_ms=(perf_counter() - start) * 1000, message=str(exc))


def _gaussian_circular(params: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    bg, amp, x0, y0, sigma = params
    sigma = max(float(sigma), 1e-6)
    return bg + amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma * sigma))


def _gaussian_elliptical(params: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    bg, amp, x0, y0, sx, sy, theta = params
    sx = max(float(sx), 1e-6)
    sy = max(float(sy), 1e-6)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xx - x0) * ct + (yy - y0) * st
    yr = -(xx - x0) * st + (yy - y0) * ct
    return bg + amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def localize_gaussian(
    image: np.ndarray,
    rough_x: float,
    rough_y: float,
    radius: int,
    *,
    elliptical: bool,
) -> SpotLocalization:
    method = "elliptical_gaussian" if elliptical else "gaussian2d"
    start = perf_counter()
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    arr = np.asarray(roi, dtype=float)
    signal, bg = preprocess_signal(arr)
    cent = localize_centroid(image, rough_x, rough_y, radius)
    lx0 = float(np.clip(cent.x - x0, 0, roi.shape[1] - 1))
    ly0 = float(np.clip(cent.y - y0, 0, roi.shape[0] - 1))
    amp0 = float(max(np.percentile(arr, 99) - bg, 1e-6))
    sigma0 = max(1.2, min(radius / 2.0, 4.0))
    yy, xx = np.indices(arr.shape, dtype=float)
    scale = max(float(np.std(arr)), 1e-6)

    try:
        if elliptical:
            p0 = np.array([bg, amp0, lx0, ly0, sigma0, sigma0 * 1.2, 0.0], dtype=float)
            lower = [float(np.min(arr)) - abs(amp0), 0.0, -1.0, -1.0, 0.5, 0.5, -np.pi / 2]
            upper = [float(np.max(arr)), max(float(np.ptp(arr)) * 3, amp0 * 5), roi.shape[1], roi.shape[0], radius * 1.5, radius * 1.5, np.pi / 2]

            def residuals(p):
                return ((_gaussian_elliptical(p, xx, yy) - arr) / scale).ravel()
        else:
            p0 = np.array([bg, amp0, lx0, ly0, sigma0], dtype=float)
            lower = [float(np.min(arr)) - abs(amp0), 0.0, -1.0, -1.0, 0.5]
            upper = [float(np.max(arr)), max(float(np.ptp(arr)) * 3, amp0 * 5), roi.shape[1], roi.shape[0], radius * 1.5]

            def residuals(p):
                return ((_gaussian_circular(p, xx, yy) - arr) / scale).ravel()

        fit = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=250, loss="soft_l1")
        if not fit.success:
            raise ValueError(fit.message)
        lx, ly = float(fit.x[2]), float(fit.x[3])
        residual = float(np.sqrt(np.mean(residuals(fit.x) ** 2)))
        return SpotLocalization(x0 + lx, y0 + ly, True, method, quality=float(fit.x[1]), residual=residual, background=float(fit.x[0]), intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000)
    except Exception as exc:
        return SpotLocalization(rough_x, rough_y, False, method, background=bg, intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000, message=str(exc))


def localize_trackmate_log(image: np.ndarray, rough_x: float, rough_y: float, radius: int, sigma: float) -> SpotLocalization:
    start = perf_counter()
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    signal, bg = preprocess_signal(roi)
    response = -ndimage.gaussian_laplace(signal, sigma=sigma) * sigma * sigma
    if response.size == 0 or not np.isfinite(response).any():
        return SpotLocalization(rough_x, rough_y, False, "trackmate_log", background=bg, runtime_ms=(perf_counter() - start) * 1000, message="empty response")
    y, x = np.unravel_index(int(np.nanargmax(response)), response.shape)
    dx, dy = quadratic_peak_offset(response, int(y), int(x))
    return SpotLocalization(x0 + float(x) + dx, y0 + float(y) + dy, True, "trackmate_log", quality=float(response[y, x]), background=bg, intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000)


def localize_phasor(image: np.ndarray, rough_x: float, rough_y: float, radius: int) -> SpotLocalization:
    start = perf_counter()
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    signal, bg = preprocess_signal(roi)
    if signal.sum() <= 0:
        return SpotLocalization(rough_x, rough_y, False, "phasor", background=bg, runtime_ms=(perf_counter() - start) * 1000, message="empty signal")
    h, w = signal.shape
    yy, xx = np.indices(signal.shape)
    fx = np.sum(signal * np.exp(-2j * np.pi * xx / max(w, 1)))
    fy = np.sum(signal * np.exp(-2j * np.pi * yy / max(h, 1)))
    lx = (-np.angle(fx) * w / (2 * np.pi)) % w
    ly = (-np.angle(fy) * h / (2 * np.pi)) % h
    centroid = localize_centroid(image, rough_x, rough_y, radius)
    # Resolve periodic ambiguity by selecting the equivalent position nearest the centroid.
    c_lx, c_ly = centroid.x - x0, centroid.y - y0
    candidates_x = np.array([lx - w, lx, lx + w])
    candidates_y = np.array([ly - h, ly, ly + h])
    lx = float(candidates_x[np.argmin(np.abs(candidates_x - c_lx))])
    ly = float(candidates_y[np.argmin(np.abs(candidates_y - c_ly))])
    if lx < -1 or ly < -1 or lx > w or ly > h:
        return SpotLocalization(centroid.x, centroid.y, False, "phasor", background=bg, intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000, message="phasor outside ROI")
    quality = float(np.hypot(np.abs(fx), np.abs(fy)) / (signal.sum() + 1e-12))
    return SpotLocalization(x0 + lx, y0 + ly, True, "phasor", quality=quality, background=bg, intensity=float(signal.sum()), runtime_ms=(perf_counter() - start) * 1000)


def _linear_template_residual(crop: np.ndarray, template: np.ndarray) -> float:
    y = crop.astype(float).ravel()
    x = template.astype(float).ravel()
    A = np.column_stack([x, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        denom = np.std(y) + 1e-12
        return float(np.sqrt(np.mean(((pred - y) / denom) ** 2)))
    except Exception:
        return np.inf


def localize_template_registration(
    image: np.ndarray,
    rough_x: float,
    rough_y: float,
    radius: int,
    template: np.ndarray | None,
    method: str,
    z_value: float = np.nan,
) -> SpotLocalization:
    start = perf_counter()
    if template is None:
        return SpotLocalization(rough_x, rough_y, False, method, runtime_ms=0.0, message="missing template")
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    if roi.shape != template.shape:
        # Center-crop/pad through a new crop with the template half-size.
        radius = max(template.shape) // 2
        roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    if roi.shape != template.shape:
        return SpotLocalization(rough_x, rough_y, False, method, runtime_ms=(perf_counter() - start) * 1000, message="template shape mismatch")
    roi_sig, bg = preprocess_signal(roi)
    tmpl_sig, _ = preprocess_signal(template)
    try:
        shift, error, _ = phase_cross_correlation(tmpl_sig, roi_sig, upsample_factor=20, normalization=None)
        # shift maps moving image (roi) onto reference (template), so target center is rough center + shift.
        cy, cx = (np.array(roi.shape) - 1) / 2.0
        y = y0 + cy + float(shift[0])
        x = x0 + cx + float(shift[1])
        residual = _linear_template_residual(roi_sig, tmpl_sig)
        return SpotLocalization(x, y, True, method, quality=float(1.0 / (error + 1e-9)), residual=residual, background=bg, intensity=float(roi_sig.sum()), runtime_ms=(perf_counter() - start) * 1000, direct_z=z_value)
    except Exception as exc:
        return SpotLocalization(rough_x, rough_y, False, method, background=bg, runtime_ms=(perf_counter() - start) * 1000, message=str(exc), direct_z=z_value)


def localize_spot_roi(
    roi: np.ndarray,
    method: str,
    *,
    objective: str = "40x_0.6NA_air",
    rough_x: float | None = None,
    rough_y: float | None = None,
    params: dict[str, Any] | None = None,
) -> SpotLocalization:
    params = params or {}
    objective_params = OBJECTIVE_PARAMS.get(objective, OBJECTIVE_PARAMS["40x_0.6NA_air"])
    radius = int(params.get("roi_radius", objective_params["roi_radius"]))
    if rough_x is None or rough_y is None:
        rough_y_i, rough_x_i = np.unravel_index(int(np.nanargmax(roi)), roi.shape)
        rough_x = float(rough_x_i)
        rough_y = float(rough_y_i)
    if method == "legacy":
        return SpotLocalization(float(rough_x), float(rough_y), True, "legacy", quality=float(np.nanmax(roi)), intensity=float(np.nansum(roi)))
    if method == "centroid":
        return localize_centroid(roi, rough_x, rough_y, radius)
    if method == "radial_symmetry":
        return localize_radial_symmetry(roi, rough_x, rough_y, radius)
    if method == "gaussian2d":
        return localize_gaussian(roi, rough_x, rough_y, radius, elliptical=False)
    if method == "elliptical_gaussian":
        return localize_gaussian(roi, rough_x, rough_y, radius, elliptical=True)
    if method == "trackmate_log":
        return localize_trackmate_log(roi, rough_x, rough_y, radius, sigma=float(params.get("log_sigma", objective_params["log_sigma"])))
    if method == "phasor":
        return localize_phasor(roi, rough_x, rough_y, radius)
    return SpotLocalization(float(rough_x), float(rough_y), False, method, message=f"unsupported single-spot method: {method}")


class PSFTemplateModel:
    def __init__(self, objective: str, z_values: np.ndarray, templates_by_z: dict[float, list[np.ndarray]], reference_frame_templates: list[np.ndarray]):
        self.objective = objective
        self.z_values = np.asarray(z_values, dtype=float)
        self.templates_by_z = templates_by_z
        self.reference_frame_templates = reference_frame_templates

    @classmethod
    def from_simulation_stack(cls, objective: str, stack_path: Path, z_values: np.ndarray) -> "PSFTemplateModel":
        params = OBJECTIVE_PARAMS.get(objective, OBJECTIVE_PARAMS["40x_0.6NA_air"])
        radius = int(params["template_radius"])
        templates_by_z: dict[float, list[np.ndarray]] = {}
        reference_templates: list[np.ndarray] | None = None
        for frame_idx, image in load_tiff_stack(stack_path):
            z = float(z_values[frame_idx])
            rough = find_four_peaks(image, objective, use_log=False)
            if len(rough) != 4:
                continue
            templates = []
            for x, y, _ in rough:
                crop, _, _ = crop_with_bounds(image, x, y, radius)
                templates.append(crop.astype(np.float32))
            templates_by_z[z] = templates
            if frame_idx == len(z_values) // 2:
                reference_templates = templates
        if reference_templates is None and templates_by_z:
            reference_templates = templates_by_z[sorted(templates_by_z.keys(), key=lambda v: abs(v))[0]]
        if reference_templates is None:
            raise RuntimeError(f"Could not build templates for {objective}")
        return cls(objective, z_values, templates_by_z, reference_templates)


def localize_four_spots(
    image: np.ndarray,
    objective: str,
    method: str,
    *,
    template_model: PSFTemplateModel | None = None,
    params: dict[str, Any] | None = None,
    rough_peaks: list[tuple[float, float, float]] | None = None,
) -> FourSpotLocalization:
    start = perf_counter()
    params = params or {}
    objective_params = OBJECTIVE_PARAMS.get(objective, OBJECTIVE_PARAMS["40x_0.6NA_air"])
    radius = int(params.get("roi_radius", objective_params["roi_radius"]))
    sigma = float(params.get("log_sigma", objective_params["log_sigma"]))
    use_log_detector = method == "trackmate_log"
    rough = rough_peaks if rough_peaks is not None and not use_log_detector else find_four_peaks(image, objective, use_log=use_log_detector)
    if len(rough) != 4:
        runtime_ms = (perf_counter() - start) * 1000
        return FourSpotLocalization(objective, method, [], np.nan, np.nan, np.nan, False, np.nan, np.nan, runtime_ms, f"found {len(rough)} peaks")

    rough_spots = [SpotLocalization(x=x, y=y, success=True, method="rough", quality=q) for x, y, q in rough]
    rough_spots = sort_spots_by_angle(rough_spots, center=image_center(image))
    spots: list[SpotLocalization] = []
    direct_z_values = []

    for idx, rough_spot in enumerate(rough_spots):
        if method == "trackmate_log":
            spot = localize_trackmate_log(image, rough_spot.x, rough_spot.y, radius, sigma=sigma)
        elif method == "centroid":
            spot = localize_centroid(image, rough_spot.x, rough_spot.y, radius)
        elif method == "radial_symmetry":
            spot = localize_radial_symmetry(image, rough_spot.x, rough_spot.y, radius)
        elif method == "gaussian2d":
            spot = localize_gaussian(image, rough_spot.x, rough_spot.y, radius, elliptical=False)
        elif method == "elliptical_gaussian":
            spot = localize_gaussian(image, rough_spot.x, rough_spot.y, radius, elliptical=True)
        elif method == "phasor":
            spot = localize_phasor(image, rough_spot.x, rough_spot.y, radius)
        elif method == "template_registration":
            template = template_model.reference_frame_templates[idx] if template_model else None
            spot = localize_template_registration(image, rough_spot.x, rough_spot.y, int(objective_params["template_radius"]), template, method)
        elif method == "psf_mle":
            spot = _localize_psf_mle(image, rough_spot.x, rough_spot.y, idx, template_model, int(objective_params["template_radius"]))
        else:
            spot = SpotLocalization(rough_spot.x, rough_spot.y, False, method, message=f"unknown method {method}")
        if not spot.success or not np.isfinite(spot.x + spot.y):
            spot.x = rough_spot.x
            spot.y = rough_spot.y
        if np.isfinite(spot.direct_z):
            direct_z_values.append(spot.direct_z)
        spots.append(spot)

    spots = sort_spots_by_angle(spots, center=image_center(image))
    center_x = float(np.nanmean([s.x for s in spots])) if spots else np.nan
    center_y = float(np.nanmean([s.y for s in spots])) if spots else np.nan
    diag = mean_diagonal(spots)
    success = len(spots) == 4 and all(s.success for s in spots) and np.isfinite(diag)
    residuals = [s.residual for s in spots if np.isfinite(s.residual)]
    qualities = [s.quality for s in spots if np.isfinite(s.quality)]
    runtime_ms = (perf_counter() - start) * 1000
    direct_z = float(np.nanmedian(direct_z_values)) if direct_z_values else np.nan
    message = "; ".join([s.message for s in spots if s.message])
    return FourSpotLocalization(
        objective=objective,
        method=method,
        spots=spots,
        center_x=center_x,
        center_y=center_y,
        mean_diag_px=diag,
        success=success,
        quality=float(np.nanmean(qualities)) if qualities else np.nan,
        residual=float(np.nanmean(residuals)) if residuals else np.nan,
        runtime_ms=runtime_ms,
        message=message,
        direct_z=direct_z,
    )


def _localize_psf_mle(
    image: np.ndarray,
    rough_x: float,
    rough_y: float,
    arm_index: int,
    template_model: PSFTemplateModel | None,
    radius: int,
) -> SpotLocalization:
    start = perf_counter()
    if template_model is None:
        return SpotLocalization(rough_x, rough_y, False, "psf_mle", runtime_ms=0.0, message="missing template model")
    roi, x0, y0 = crop_with_bounds(image, rough_x, rough_y, radius)
    roi_sig, bg = preprocess_signal(roi)
    best: tuple[float, SpotLocalization] | None = None
    for z, templates in template_model.templates_by_z.items():
        if arm_index >= len(templates):
            continue
        template = templates[arm_index]
        if template.shape != roi.shape:
            continue
        tmpl_sig, _ = preprocess_signal(template)
        try:
            shift, error, _ = phase_cross_correlation(tmpl_sig, roi_sig, upsample_factor=20, normalization=None)
            shifted_roi = ndimage.shift(roi_sig, shift=shift, order=1, mode="nearest")
            residual = _linear_template_residual(shifted_roi, tmpl_sig)
            score = residual + 0.01 * float(error)
            cy, cx = (np.array(roi.shape) - 1) / 2.0
            loc = SpotLocalization(
                x0 + cx + float(shift[1]),
                y0 + cy + float(shift[0]),
                True,
                "psf_mle",
                quality=float(1.0 / (score + 1e-9)),
                residual=float(residual),
                background=bg,
                intensity=float(roi_sig.sum()),
                runtime_ms=(perf_counter() - start) * 1000,
                direct_z=float(z),
            )
            if best is None or score < best[0]:
                best = (score, loc)
        except Exception:
            continue
    if best is None:
        return SpotLocalization(rough_x, rough_y, False, "psf_mle", background=bg, intensity=float(roi_sig.sum()), runtime_ms=(perf_counter() - start) * 1000, message="no template fit")
    best[1].runtime_ms = (perf_counter() - start) * 1000
    return best[1]


def match_spots_to_reference(spots: list[SpotLocalization], reference_xy: np.ndarray) -> list[float]:
    if len(spots) != len(reference_xy):
        return [np.nan] * len(spots)
    pts = np.array([[s.x, s.y] for s in spots], dtype=float)
    distances = cdist(pts, reference_xy)
    # Four spots are small enough for a greedy assignment after sorting by best edge.
    errors = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _ in range(len(spots)):
        best = None
        for i in range(distances.shape[0]):
            if i in used_rows:
                continue
            for j in range(distances.shape[1]):
                if j in used_cols:
                    continue
                val = float(distances[i, j])
                if best is None or val < best[0]:
                    best = (val, i, j)
        if best is None:
            break
        errors.append(best[0])
        used_rows.add(best[1])
        used_cols.add(best[2])
    while len(errors) < len(spots):
        errors.append(np.nan)
    return errors
