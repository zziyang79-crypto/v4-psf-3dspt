"""Shared scale-space spot detection and lazy TIFF access for auto-detect and GUI preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.special import ndtr
from sklearn.mixture import GaussianMixture


OBJECTIVE_NA = {
    "20x_NA0.45_n1.0": 0.45,
    "40x_NA0.6_n1.0": 0.60,
    "60x_NA1.0_n1.333": 1.00,
    "60x_NA1.0_n1.515": 1.00,
}


@dataclass(frozen=True)
class DetectionSettings:
    objective_config: str = "20x_NA0.45_n1.0"
    camera_pixel_um: float = 4.5
    emission_nm: float = 650.0
    spot_radius_px: float = 0.0
    quality_threshold: float = 0.0
    signal_posterior: float = 0.99


class LazyTiffStack:
    """Expose one TIFF frame at a time, preferring virtual-memory backed arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tiff = tifffile.TiffFile(str(self.path))
        self._series = self._tiff.series[0]
        self._pages = self._tiff.pages
        self._memmap: np.ndarray | None = None
        self.storage_mode = "page_decode"
        self._lock = RLock()
        self._cached_index: int | None = None
        self._cached_frame: np.ndarray | None = None

        shape = tuple(int(value) for value in self._series.shape)
        if len(shape) >= 3:
            self.frame_count = shape[0]
            self.frame_shape = shape[-2:]
        elif len(self._pages) > 1:
            # Some writers expose every page as a separate 2D series.
            self.frame_count = len(self._pages)
            self.frame_shape = tuple(int(value) for value in self._pages[0].shape[-2:])
        elif len(shape) == 2:
            self.frame_count = 1
            self.frame_shape = shape
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")

        # Direct memmap is zero-copy for contiguous, uncompressed TIFF data.  For
        # compressed TIFFs, tifffile's ``out='memmap'`` decodes into a temporary
        # disk-backed array instead of materialising the complete stack in RAM.
        try:
            mapped = tifffile.memmap(str(self.path), series=0, mode="r")
            if self.frame_count > 1 and (mapped.ndim < 3 or mapped.shape[0] < self.frame_count):
                raise ValueError("series memmap does not contain all TIFF pages")
            self._memmap = mapped
            self.storage_mode = "direct_memmap"
        except Exception:
            try:
                mapped = self._series.asarray(out="memmap")
                if self.frame_count > 1 and (mapped.ndim < 3 or mapped.shape[0] < self.frame_count):
                    raise ValueError("decoded series memmap does not contain all TIFF pages")
                self._memmap = mapped
                self.storage_mode = "decoded_memmap"
            except Exception:
                self._memmap = None
                self.storage_mode = "page_decode"

    def get_frame(self, frame_index: int) -> np.ndarray:
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"Frame {index} outside [0, {self.frame_count - 1}]")
        with self._lock:
            if self._cached_index == index and self._cached_frame is not None:
                return self._cached_frame

            if self._memmap is not None:
                frame = self._memmap if self.frame_count == 1 else self._memmap[index]
            elif self.frame_count == 1:
                frame = self._series.asarray()
            elif len(self._pages) == self.frame_count:
                frame = self._pages[index].asarray()
            else:
                raise ValueError(
                    "TIFF stack cannot be memory-mapped and has no frame pages; "
                    "random frame access is unavailable."
                )
            frame = np.asarray(frame)
            while frame.ndim > 2:
                frame = frame[0]
            if frame.ndim != 2:
                raise ValueError(f"Frame {index} is not 2D after selection: {frame.shape}")
            self._cached_index = index
            self._cached_frame = frame
            return frame

    def close(self) -> None:
        with self._lock:
            self._cached_frame = None
            self._memmap = None
            self._tiff.close()

    def __enter__(self) -> "LazyTiffStack":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def magnification_from_objective(objective_config: str) -> float:
    value = str(objective_config).lower()
    for prefix, magnification in (("20x", 20.0), ("40x", 40.0), ("60x", 60.0)):
        if value.startswith(prefix):
            return magnification
    return 20.0


def diffraction_limited_radius_px(settings: DetectionSettings) -> float:
    """Estimate the lateral Airy radius in camera pixels from the acquisition optics."""
    na = OBJECTIVE_NA.get(settings.objective_config, 0.45)
    sample_pixel_um = float(settings.camera_pixel_um) / magnification_from_objective(settings.objective_config)
    wavelength_um = float(settings.emission_nm) / 1000.0
    airy_radius_um = 0.61 * wavelength_um / max(na, 1e-9)
    return max(airy_radius_um / max(sample_pixel_um, 1e-9), 0.5)


def resolve_log_sigmas(settings: DetectionSettings) -> np.ndarray:
    """Use a physics-derived center scale unless the user supplies a spot radius."""
    radius = float(settings.spot_radius_px)
    if not np.isfinite(radius) or radius <= 0:
        radius = diffraction_limited_radius_px(settings)
    center_sigma = max(radius / np.sqrt(2.0), 0.35)
    return center_sigma * np.asarray([1.0 / np.sqrt(2.0), 1.0, np.sqrt(2.0)], dtype=float)


def _robust_scale(values: np.ndarray, fallback: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return max(float(fallback), 1e-9)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    return max(scale, float(fallback), 1e-9)


def _automatic_quality_threshold(positive_quality: np.ndarray, signal_posterior: float) -> float:
    values = np.asarray(positive_quality, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 8:
        return np.inf
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=1,
        max_iter=100,
        reg_covar=1e-6,
        random_state=0,
    ).fit(values.reshape(-1, 1))
    means = model.means_.reshape(-1)
    low_component = int(np.argmin(means))
    high_component = int(np.argmax(means))
    grid = np.linspace(float(means[low_component]), float(np.max(values)), 2048)
    probabilities = model.predict_proba(grid.reshape(-1, 1))
    posterior = float(np.clip(signal_posterior, 0.5, 0.9999))
    passing = np.where(probabilities[:, high_component] >= posterior)[0]
    if len(passing) == 0:
        return float(np.max(values) + np.finfo(float).eps)
    return float(grid[int(passing[0])])


def _centroid_and_features(image: np.ndarray, row: int, col: int, sigma: float) -> dict[str, float]:
    patch_radius = max(2, int(np.ceil(3.0 * sigma)))
    r0, r1 = max(0, row - patch_radius), min(image.shape[0], row + patch_radius + 1)
    c0, c1 = max(0, col - patch_radius), min(image.shape[1], col + patch_radius + 1)
    patch = image[r0:r1, c0:c1]
    border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
    background = float(np.median(border)) if len(border) else float(np.median(patch))
    local_noise = _robust_scale(border, fallback=1.0)
    signal = np.clip(patch - background, 0, None)
    signal_sum = float(signal.sum())
    yy, xx = np.indices(signal.shape)
    if signal_sum > 1e-12:
        local_y = float((yy * signal).sum() / signal_sum)
        local_x = float((xx * signal).sum() / signal_sum)
        sigma_x = float(np.sqrt(np.sum(signal * (xx - local_x) ** 2) / signal_sum))
        sigma_y = float(np.sqrt(np.sum(signal * (yy - local_y) ** 2) / signal_sum))
        area = int(np.sum(signal > max(1.0, 0.25 * float(signal.max()))))
    else:
        local_y = (patch.shape[0] - 1) / 2.0
        local_x = (patch.shape[1] - 1) / 2.0
        sigma_x = 0.0
        sigma_y = 0.0
        area = 0
    peak = float(image[row, col])
    return {
        "X": float(c0 + local_x),
        "Y": float(r0 + local_y),
        "intensity": peak,
        "integrated_intensity": signal_sum,
        "local_bg": background,
        "local_contrast": float((peak - background) / max(local_noise, 1.0)),
        "spot_area": area,
        "spot_sigma_x": sigma_x,
        "spot_sigma_y": sigma_y,
        "spot_shape_ratio": float(max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-6)),
        "log_peak_intensity": float(np.log1p(max(peak - background, 0.0))),
        "log_integrated_intensity": float(np.log1p(max(signal_sum, 0.0))),
    }


def detect_spots_in_frame(
    image: np.ndarray,
    settings: DetectionSettings,
    *,
    frame: int = 0,
) -> tuple[pd.DataFrame, dict[str, float | np.ndarray]]:
    """Detect all significant LoG extrema without a target count or peak cap."""
    raw = np.asarray(image, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D image frame, got {raw.shape}")

    sigmas = resolve_log_sigmas(settings)
    context_sigma = float(4.0 * np.max(sigmas))
    residual = ndi.gaussian_filter(raw, sigma=context_sigma, mode="nearest", output=np.float32)
    np.subtract(raw, residual, out=residual)
    local_noise = ndi.gaussian_filter(np.abs(residual), sigma=context_sigma, mode="nearest", output=np.float32)
    local_noise *= np.float32(1.4826)
    global_noise = _robust_scale(residual[residual <= 0], fallback=1.0)
    np.maximum(local_noise, np.float32(global_noise), out=local_noise)

    response = np.full(raw.shape, -np.inf, dtype=np.float32)
    scale_index = np.zeros(raw.shape, dtype=np.uint8)
    for index, sigma in enumerate(sigmas):
        scale_response = ndi.gaussian_laplace(residual, sigma=float(sigma), mode="nearest", output=np.float32)
        scale_response *= np.float32(-float(sigma) ** 2)
        better = scale_response > response
        response[better] = scale_response[better]
        scale_index[better] = index

    np.maximum(local_noise, np.float32(1e-9), out=local_noise)
    np.divide(response, local_noise, out=local_noise)

    null_values = local_noise[residual <= 0]
    null_center = float(np.median(null_values)) if len(null_values) else 0.0
    null_scale = _robust_scale(null_values, fallback=1.0)
    quality_image = local_noise
    quality_image -= np.float32(null_center)
    quality_image /= np.float32(null_scale)

    radius_for_separation = float(settings.spot_radius_px)
    if not np.isfinite(radius_for_separation) or radius_for_separation <= 0:
        radius_for_separation = diffraction_limited_radius_px(settings)
    border = max(1, int(np.ceil(radius_for_separation)))
    maximum = ndi.maximum_filter(response, size=2 * border + 1, mode="nearest", output=np.float32)
    peak_mask = (response > 0.0) & (response == maximum)
    peak_mask[:border, :] = False
    peak_mask[-border:, :] = False
    peak_mask[:, :border] = False
    peak_mask[:, -border:] = False
    peaks = np.argwhere(peak_mask)
    if len(peaks) == 0:
        return pd.DataFrame(), {
            "sigmas": sigmas,
            "auto_quality_threshold": float("nan"),
            "active_quality_threshold": float("nan"),
            "selection_method": "no_local_maxima",
            "candidate_count": 0.0,
            "signal_posterior": float(settings.signal_posterior),
            "accepted_count": 0.0,
        }

    peak_rows = peaks[:, 0].astype(int)
    peak_cols = peaks[:, 1].astype(int)
    quality = quality_image[peak_rows, peak_cols]
    p_values = ndtr(-quality)
    positive_quality = quality[np.isfinite(quality) & (quality > 0.0)]
    auto_threshold = _automatic_quality_threshold(positive_quality, settings.signal_posterior)
    manual_threshold = float(settings.quality_threshold)
    active_threshold = manual_threshold if manual_threshold > 0.0 else auto_threshold
    keep = quality >= active_threshold
    selection_method = "manual_quality" if manual_threshold > 0.0 else "auto_gaussian_mixture_signal_posterior"

    records: list[dict[str, float | int | bool]] = []
    for detection_index, peak_index in enumerate(np.where(keep)[0]):
        row = int(peak_rows[peak_index])
        col = int(peak_cols[peak_index])
        sigma = float(sigmas[int(scale_index[row, col])])
        record = _centroid_and_features(raw, row, col, sigma)
        record.update(
            {
                "FRAME": int(frame),
                "detection_id": f"img_f{int(frame)}_d{detection_index}",
                "Detection_LoG_Sigma_Px": sigma,
                "Detection_Quality": float(quality[peak_index]),
                "Detection_P_Value": float(p_values[peak_index]),
                "Detection_Manual_Quality_Threshold": float(settings.quality_threshold),
                "Detection_Auto_Quality_Threshold": auto_threshold,
                "Detection_Active_Quality_Threshold": active_threshold,
                "Detection_Selection_Method": selection_method,
                "Detection_Significance_Pass": True,
            }
        )
        records.append(record)

    detected = pd.DataFrame(records)
    return detected, {
        "sigmas": sigmas,
        "auto_quality_threshold": auto_threshold,
        "active_quality_threshold": active_threshold,
        "selection_method": selection_method,
        "candidate_count": float(len(peaks)),
        "signal_posterior": float(settings.signal_posterior),
        "accepted_count": float(len(detected)),
    }


def detection_summary(settings: DetectionSettings) -> dict[str, float | list[float] | str]:
    sigmas = resolve_log_sigmas(settings)
    radius = float(settings.spot_radius_px)
    source = "manual" if np.isfinite(radius) and radius > 0 else "optics"
    if source == "optics":
        radius = diffraction_limited_radius_px(settings)
    return {
        "scale_source": source,
        "spot_radius_px": radius,
        "log_sigmas_px": [float(value) for value in sigmas],
        "quality_threshold": float(settings.quality_threshold),
        "signal_posterior": float(settings.signal_posterior),
        "quality_selection": "manual" if float(settings.quality_threshold) > 0.0 else "auto_gaussian_mixture_signal_posterior",
    }
