import os
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage import io

from .csv_io import find_header_row, read_csv as read_pipeline_csv

warnings.filterwarnings("ignore")


DEFAULT_PARAMS = {
    "PHYSICAL_SCALE": 17.0,
    "MODEL_REFERENCE_CAMERA_PIXEL_UM": 4.5,
    "ROI_RADIUS": 7,
    "INTEGRATION_RADIUS": 10.0,
    "USE_SUBPIXEL_REFINE": True,
    "XY_LOCALIZATION_METHOD": "phasor",
    "Z_LOCALIZATION_METHOD": "phasor",
    "LOCALIZATION_METHOD": "phasor",
    "LOCALIZATION_FALLBACK": "trackmate_position",
    "CENTROID_BACKGROUND_PERCENTILE": 15.0,
    "MAX_SUBPIXEL_SHIFT_PX": 6.0,
    "OBJECTIVE_CONFIG": "",
    "Z_PHASOR_BLUR_SIGMA_PX": "auto",
}

OBJECTIVE_Z_PHASOR_BLUR_SIGMA = {
    "20x": 2.4,
    "40x": 2.4,
    "60x": 2.4,
}


def auto_find_files():
    current_dir = os.getcwd()
    files = os.listdir(current_dir)
    csv_file, tif_file = None, None
    for name in files:
        lower = name.lower()
        if "spots" in lower and lower.endswith(".csv"):
            csv_file = name
        if lower.endswith((".tif", ".tiff")):
            tif_file = name
    if not csv_file or not tif_file:
        return None, None
    print(f"Matched input files: {csv_file} and {tif_file}")
    return csv_file, tif_file


def find_col_fuzzy(df, keywords):
    for col in df.columns:
        c_str = str(col).strip()
        if c_str in keywords:
            return col
        c_lower = c_str.lower()
        for keyword in keywords:
            if keyword.lower() == c_lower:
                return col
    return None


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def objective_family(objective_config):
    text = str(objective_config or "").strip().lower()
    if text.startswith("20x") or "20x" in text:
        return "20x"
    if text.startswith("40x") or "40x" in text:
        return "40x"
    if text.startswith("60x") or "60x" in text:
        return "60x"
    return ""


def resolve_z_phasor_blur_sigma(params):
    """Convert the model's reference-pixel blur to current acquisition pixels."""
    raw = str(params.get("Z_PHASOR_BLUR_SIGMA_PX", "auto")).strip().lower()
    if raw in {"", "auto", "objective"}:
        family = objective_family(params.get("OBJECTIVE_CONFIG", ""))
        reference_sigma = float(OBJECTIVE_Z_PHASOR_BLUR_SIGMA.get(family, 0.0))
        source = family or "unknown"
    else:
        try:
            reference_sigma = max(0.0, float(raw))
            source = "manual_reference_px"
        except ValueError:
            return 0.0, "invalid"
    current_pixel_um = float(params.get("PHYSICAL_SCALE", 4.5))
    reference_pixel_um = float(params.get("MODEL_REFERENCE_CAMERA_PIXEL_UM", 4.5))
    if current_pixel_um <= 0 or reference_pixel_um <= 0:
        raise ValueError("Current and model-reference camera pixel sizes must both be positive.")
    current_sigma = reference_sigma * reference_pixel_um / current_pixel_um
    return current_sigma, (
        f"{source}; {reference_sigma:g} reference px at {reference_pixel_um:g} um/px "
        f"-> {current_sigma:g} current px at {current_pixel_um:g} um/px"
    )


def auto_load_csv(path):
    if path is None:
        return None
    try:
        header = find_header_row(path, ["POSITION_X", "x", "Spot position X"])
        df = read_pipeline_csv(path, header=header)
        if find_col_fuzzy(df, ["POSITION_X", "x", "Spot position X"]):
            return df
    except Exception:
        pass
    return None


def get_radial_center(roi):
    """Legacy radial-symmetry center; kept for compatibility with other scripts."""
    img = ndimage.gaussian_filter(roi.astype(float), 1.0)
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
            np.array([[sum_gx2, sum_gxgy], [sum_gxgy, sum_gy2]]),
            np.array([sum_b1, sum_b2]),
        )
        return offset[0] + xmid, offset[1] + ymid
    except Exception:
        return xmid, ymid


def _border_background(roi):
    if roi.size == 0:
        return 0.0
    border = np.zeros_like(roi, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float(np.median(roi[border])) if border.any() else float(np.median(roi))


def get_intensity_centroid(roi, background_percentile=15.0):
    """Background-subtracted intensity centroid in local ROI coordinates."""
    roi = roi.astype(float)
    if roi.size == 0:
        return np.nan, np.nan, False, np.nan, np.nan

    percentile_bg = float(np.percentile(roi, background_percentile))
    bg = min(percentile_bg, _border_background(roi))
    signal = np.clip(roi - bg, 0, None)
    total = float(signal.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return (roi.shape[1] - 1) / 2.0, (roi.shape[0] - 1) / 2.0, False, np.nan, bg

    yy, xx = np.indices(signal.shape)
    cx = float((xx * signal).sum() / total)
    cy = float((yy * signal).sum() / total)
    moment2 = float((((xx - cx) ** 2 + (yy - cy) ** 2) * signal).sum() / total)
    peak = float(np.max(signal))
    quality = total / (np.sqrt(max(moment2, 1e-12)) + 1e-12)
    success = 0 <= cx <= roi.shape[1] - 1 and 0 <= cy <= roi.shape[0] - 1 and peak > 0
    return cx, cy, bool(success), float(moment2), float(quality)


def get_phasor_center(roi, background_percentile=10.0):
    """First Fourier harmonic phase localization in local ROI coordinates."""
    roi = roi.astype(float)
    if roi.size == 0:
        return np.nan, np.nan, False, np.nan, np.nan
    bg = float(np.percentile(roi, background_percentile))
    signal = np.clip(roi - bg, 0, None)
    total = float(signal.sum())
    if not np.isfinite(total) or total <= 1e-12:
        cx, cy, success, residual, quality = get_intensity_centroid(roi, background_percentile)
        return cx, cy, False, residual, quality

    h, w = signal.shape
    yy, xx = np.indices(signal.shape)
    fx = np.sum(signal * np.exp(-2j * np.pi * xx / max(w, 1)))
    fy = np.sum(signal * np.exp(-2j * np.pi * yy / max(h, 1)))
    px = (-np.angle(fx) * w / (2 * np.pi)) % w
    py = (-np.angle(fy) * h / (2 * np.pi)) % h

    cx, cy, _, residual, _ = get_intensity_centroid(roi, background_percentile)
    candidates_x = np.array([px - w, px, px + w], dtype=float)
    candidates_y = np.array([py - h, py, py + h], dtype=float)
    px = float(candidates_x[np.argmin(np.abs(candidates_x - cx))])
    py = float(candidates_y[np.argmin(np.abs(candidates_y - cy))])
    success = -1 <= px <= w and -1 <= py <= h and np.isfinite(px + py)
    quality = float(np.hypot(np.abs(fx), np.abs(fy)) / (total + 1e-12))
    return px, py, bool(success), residual, quality


def _local_bounds(cx, cy, radius, width, height):
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(width, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(height, int(np.ceil(cy + radius)) + 1)
    return x0, x1, y0, y1


def subpixel_refine_center(large_roi, init_x, init_y, method, params):
    if method in {"none", "legacy_position", "trackmate_position"}:
        return init_x, init_y, True, 0.0, np.nan, "trackmate_position"

    radius = int(params["ROI_RADIUS"])
    max_shift = float(params.get("MAX_SUBPIXEL_SHIFT_PX", radius))
    x0, x1, y0, y1 = _local_bounds(init_x, init_y, radius, large_roi.shape[1], large_roi.shape[0])
    local_roi = large_roi[y0:y1, x0:x1]
    if local_roi.size == 0:
        return init_x, init_y, False, np.nan, np.nan, "empty_roi"

    if method == "centroid":
        cx, cy, success, residual, quality = get_intensity_centroid(
            local_roi,
            background_percentile=float(params.get("CENTROID_BACKGROUND_PERCENTILE", 15.0)),
        )
    elif method == "phasor":
        cx, cy, success, residual, quality = get_phasor_center(
            local_roi,
            background_percentile=float(params.get("CENTROID_BACKGROUND_PERCENTILE", 15.0)),
        )
    elif method == "radial_symmetry":
        cx, cy = get_radial_center(local_roi)
        success = np.isfinite(cx + cy)
        residual = np.nan
        quality = np.nan
    else:
        return init_x, init_y, False, np.nan, np.nan, f"unsupported_method:{method}"

    refined_x = x0 + float(cx)
    refined_y = y0 + float(cy)
    shift = float(np.hypot(refined_x - init_x, refined_y - init_y))
    if not success or not np.isfinite(refined_x + refined_y) or shift > max_shift:
        return init_x, init_y, False, residual, quality, f"fallback_shift={shift:.3f}"
    return refined_x, refined_y, True, residual, quality, method


def _estimate_annulus_background(roi, center_x, center_y, inner_r, outer_r):
    yy, xx = np.mgrid[: roi.shape[0], : roi.shape[1]]
    dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    bg_mask = (dist2 > inner_r**2) & (dist2 <= outer_r**2)
    if bg_mask.any():
        return float(np.mean(roi[bg_mask]))
    return _border_background(roi)


def main(csv_path=None, image_path=None, output_csv_path="spots_refined.csv", params=None, show_plot=False):
    csv_path = csv_path
    image_path = image_path
    if csv_path is None or image_path is None:
        auto_csv_path, auto_image_path = auto_find_files()
        csv_path = csv_path or auto_csv_path
        image_path = image_path or auto_image_path
    if not csv_path or not image_path:
        print("ERROR: missing spots CSV or image stack.")
        return

    run_params = DEFAULT_PARAMS.copy()
    if params:
        run_params.update(params)

    physical_scale = float(run_params["PHYSICAL_SCALE"])
    roi_radius = int(run_params["ROI_RADIUS"])
    integration_radius = float(run_params["INTEGRATION_RADIUS"])
    use_subpixel = as_bool(run_params.get("USE_SUBPIXEL_REFINE", True))
    xy_method = str(run_params.get("XY_LOCALIZATION_METHOD", run_params.get("LOCALIZATION_METHOD", "phasor"))).lower()
    z_method = str(run_params.get("Z_LOCALIZATION_METHOD", "phasor")).lower()
    if z_method == "phasor_blur":
        z_method = "phasor"
    z_blur_sigma, z_blur_source = resolve_z_phasor_blur_sigma(run_params)
    if not use_subpixel:
        xy_method = "trackmate_position"
        z_method = "trackmate_position"
        z_blur_sigma = 0.0

    bg_inner_r = integration_radius + 2.0
    bg_outer_r = bg_inner_r + 5.0

    df = auto_load_csv(csv_path)
    if df is None:
        print(f"ERROR: could not parse spots CSV: {csv_path}")
        return

    col_x = find_col_fuzzy(df, ["POSITION_X", "x", "Spot position X"])
    col_y = find_col_fuzzy(df, ["POSITION_Y", "y", "Spot position Y"])
    col_frame = find_col_fuzzy(df, ["FRAME", "t", "frame"])
    if col_x is None or col_y is None or col_frame is None:
        print("ERROR: spots CSV is missing X/Y/FRAME columns.")
        return

    for col in [col_x, col_y, col_frame]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col_x, col_y, col_frame]).copy()
    df[col_frame] = df[col_frame].astype(int)

    img_stack = io.imread(image_path)
    if img_stack.ndim == 2:
        img_stack = img_stack[np.newaxis, :, :]

    refined_x, refined_y, refined_intensity = [], [], []
    z_localization_x, z_localization_y = [], []
    xy_method_used, xy_success, xy_residual, xy_quality, xy_message = [], [], [], [], []
    z_method_used, z_success, z_residual, z_quality, z_message = [], [], [], [], []
    vis_samples = []

    crop_r = max(roi_radius, int(np.ceil(max(integration_radius, bg_outer_r)))) + 2
    max_f, max_h, max_w = img_stack.shape

    print("Running spot refinement and intensity integration...")
    print(f"  USE_SUBPIXEL_REFINE = {use_subpixel}")
    print(f"  XY_LOCALIZATION_METHOD = {xy_method}")
    print(f"  Z_LOCALIZATION_METHOD = {z_method}")
    print(f"  Z_PHASOR_BLUR_SIGMA_PX = {z_blur_sigma:.3f} ({z_blur_source})")
    print(f"  PHYSICAL_SCALE = {physical_scale} um/pixel")
    print(f"  INTEGRATION_RADIUS = {integration_radius} px")

    for idx, row in df.iterrows():
        tx = ty = np.nan
        try:
            frame = int(row[col_frame])
            tx = float(row[col_x])
            ty = float(row[col_y])
            ix, iy = int(round(tx)), int(round(ty))

            r_start = max(0, iy - crop_r)
            r_end = min(max_h, iy + crop_r + 1)
            c_start = max(0, ix - crop_r)
            c_end = min(max_w, ix + crop_r + 1)

            if frame >= max_f or r_end <= r_start or c_end <= c_start:
                refined_x.append(tx * physical_scale)
                refined_y.append(ty * physical_scale)
                z_localization_x.append(tx * physical_scale)
                z_localization_y.append(ty * physical_scale)
                refined_intensity.append(np.nan)
                xy_method_used.append("fallback_trackmate")
                xy_success.append(False)
                xy_residual.append(np.nan)
                xy_quality.append(np.nan)
                xy_message.append("out_of_bounds")
                z_method_used.append("fallback_trackmate")
                z_success.append(False)
                z_residual.append(np.nan)
                z_quality.append(np.nan)
                z_message.append("out_of_bounds")
                continue

            large_roi = img_stack[frame, r_start:r_end, c_start:c_end]
            init_x = tx - c_start
            init_y = ty - r_start
            z_large_roi = large_roi
            z_blur_applied = z_method == "phasor" and z_blur_sigma > 0
            if z_blur_applied:
                z_large_roi = ndimage.gaussian_filter(large_roi.astype(float), z_blur_sigma)

            xy_xc, xy_yc, xy_ok, xy_res, xy_qual, xy_msg = subpixel_refine_center(
                large_roi,
                init_x,
                init_y,
                xy_method,
                run_params,
            )
            z_xc, z_yc, z_ok, z_res, z_qual, z_msg = subpixel_refine_center(
                z_large_roi,
                init_x,
                init_y,
                z_method,
                run_params,
            )

            final_xy_px = c_start + xy_xc
            final_xy_py = r_start + xy_yc
            final_z_px = c_start + z_xc
            final_z_py = r_start + z_yc
            refined_x.append(final_xy_px * physical_scale)
            refined_y.append(final_xy_py * physical_scale)
            z_localization_x.append(final_z_px * physical_scale)
            z_localization_y.append(final_z_py * physical_scale)
            xy_method_used.append(xy_method if xy_ok else "fallback_trackmate")
            xy_success.append(bool(xy_ok))
            xy_residual.append(xy_res)
            xy_quality.append(xy_qual)
            xy_message.append(xy_msg)
            z_method_label = "phasor_blur" if z_ok and z_blur_applied else z_method
            z_method_used.append(z_method_label if z_ok else "fallback_trackmate")
            z_success.append(bool(z_ok))
            z_residual.append(z_res)
            z_quality.append(z_qual)
            z_message.append(z_msg)

            bg = _estimate_annulus_background(large_roi, z_xc, z_yc, bg_inner_r, bg_outer_r)
            signal = np.maximum(large_roi.astype(float) - bg, 0)
            yy, xx = np.mgrid[: large_roi.shape[0], : large_roi.shape[1]]
            dist2 = (xx - z_xc) ** 2 + (yy - z_yc) ** 2
            intensity = float(np.sum(signal[dist2 <= integration_radius**2]))
            refined_intensity.append(intensity)

            if len(vis_samples) < 4:
                vis_samples.append(
                    {
                        "roi": large_roi,
                        "old": (init_x, init_y),
                        "xy": (xy_xc, xy_yc),
                        "z": (z_xc, z_yc),
                        "z_label": z_method_label,
                    }
                )
        except Exception as exc:
            print(f"WARNING: failed to process row {idx}: {exc}")
            refined_x.append(tx * physical_scale if np.isfinite(tx) else np.nan)
            refined_y.append(ty * physical_scale if np.isfinite(ty) else np.nan)
            z_localization_x.append(tx * physical_scale if np.isfinite(tx) else np.nan)
            z_localization_y.append(ty * physical_scale if np.isfinite(ty) else np.nan)
            refined_intensity.append(np.nan)
            xy_method_used.append("fallback_trackmate")
            xy_success.append(False)
            xy_residual.append(np.nan)
            xy_quality.append(np.nan)
            xy_message.append(str(exc))
            z_method_used.append("fallback_trackmate")
            z_success.append(False)
            z_residual.append(np.nan)
            z_quality.append(np.nan)
            z_message.append(str(exc))

    df["Refined_X"] = refined_x
    df["Refined_Y"] = refined_y
    df["Z_Localization_X"] = z_localization_x
    df["Z_Localization_Y"] = z_localization_y
    df["Refined_Intensity"] = refined_intensity
    df["Localization_Method"] = xy_method_used
    df["Localization_Success"] = xy_success
    df["Localization_Residual"] = xy_residual
    df["Localization_Quality"] = xy_quality
    df["Localization_Message"] = xy_message
    df["XY_Localization_Method"] = xy_method_used
    df["XY_Localization_Success"] = xy_success
    df["XY_Localization_Residual"] = xy_residual
    df["XY_Localization_Quality"] = xy_quality
    df["XY_Localization_Message"] = xy_message
    df["Z_Localization_Method"] = z_method_used
    df["Z_Localization_Success"] = z_success
    df["Z_Localization_Residual"] = z_residual
    df["Z_Localization_Quality"] = z_quality
    df["Z_Localization_Message"] = z_message

    df.to_csv(output_csv_path, index=False)
    xy_success_rate = float(np.mean(xy_success)) if xy_success else 0.0
    z_success_rate = float(np.mean(z_success)) if z_success else 0.0
    print(f"Saved refined spots to: {output_csv_path}")
    print(f"XY localization success rate: {xy_success_rate:.3f}")
    print(f"Z localization success rate: {z_success_rate:.3f}")

    if vis_samples:
        fig1, axes1 = plt.subplots(1, min(4, len(vis_samples)), figsize=(12, 3))
        if len(vis_samples) == 1:
            axes1 = [axes1]
        for ax, sample in zip(axes1, vis_samples):
            ax.imshow(sample["roi"], cmap="gray")
            ax.plot(sample["old"][0], sample["old"][1], "rx", ms=8, label="TrackMate")
            ax.plot(sample["xy"][0], sample["xy"][1], "c+", ms=10, label="XY phasor")
            ax.plot(sample["z"][0], sample["z"][1], "g+", ms=10, label=f"Z {sample.get('z_label', 'phasor')}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig("check_position.png", dpi=150)
        print("Saved check_position.png")

    first_frame = int(df[col_frame].min())
    df_first = df[df[col_frame] == first_frame]
    if not df_first.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(img_stack[first_frame], cmap="gray")
        for _, row in df_first.iterrows():
            rx = row["Z_Localization_X"] / physical_scale
            ry = row["Z_Localization_Y"] / physical_scale
            ax2.add_patch(
                patches.Circle(
                    (rx, ry),
                    integration_radius,
                    edgecolor="yellow",
                    facecolor="none",
                    lw=0.7,
                )
            )
        plt.tight_layout()
        plt.savefig("check_overlap.png", dpi=300)
        print("Saved check_overlap.png")

    if show_plot:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
