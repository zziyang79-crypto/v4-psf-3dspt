"""
Auto-detect sub-signals from a TIF image stack, then process them with the
trajectory-level LAP association workflow ported from the benchmark project.

This replaces the old projection/static-geometry auto-detect branch. It writes
the normal downstream files used by refine, XY, Z, angle, and viscosity steps.

Environment variables (optional):
    AUTODETECT_PHYSICAL_SCALE       um per camera pixel (default 4.5)
    AUTODETECT_OBJECTIVE_CONFIG     objective key for refine (default 20x_NA0.45_n1.0)
    AUTODETECT_EMISSION_NM          emission wavelength for refine metadata (default 650)
    AUTODETECT_MANUAL_SPOT_DETECT   records that LoG settings came from the manual GUI
    AUTODETECT_SPOT_RADIUS_PX       estimated spot radius for manual LoG mode; 0=optics-derived
    AUTODETECT_LOG_QUALITY_THRESHOLD minimum standardized LoG quality; 0=automatic mixture threshold
    AUTODETECT_LOG_SIGNAL_POSTERIOR confidence required for automatic mixture selection
    AUTODETECT_DETECTION_WORKERS    bounded worker count for frame detection (default 2)
    AUTODETECT_MATCH_RADIUS_SENSOR_UM fixed LAP link distance on camera sensor (default 60.75 um)
    TRAJ_ASSOC_MAX_GAP              max frame gap during first linking (default 1)
    TRAJ_ASSOC_MAX_RELINK_GAP       max fragment relink/interp gap (default 4)
    TRAJ_ASSOC_MAX_RELINK_DIST_SENSOR_UM fixed fragment relink distance (default 63 um)
    TRAJ_ASSOC_MAX_RELINK_COST      max fragment relink custom cost (default 0.95)
    TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM fixed reconstructed group diagonal minimum (default 202.5 um)
    TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM fixed reconstructed group diagonal maximum (default 585 um)
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from .csv_io import read_csv as read_pipeline_csv
from .spot_detection import DetectionSettings, LazyTiffStack, detection_summary


PHYSICAL_SCALE = float(os.getenv("AUTODETECT_PHYSICAL_SCALE", "4.5"))
OBJECTIVE_CONFIG = os.getenv("AUTODETECT_OBJECTIVE_CONFIG", "20x_NA0.45_n1.0")
EMISSION_NM = float(os.getenv("AUTODETECT_EMISSION_NM", "650.0"))

REPO_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: str = "true") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "auto"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    return int(round(_env_float(name, float(default))))


def _sensor_distance_px(sensor_name: str, legacy_pixel_name: str, sensor_default_um: float, legacy_default_px: float) -> float:
    value = os.getenv(sensor_name)
    if PHYSICAL_SCALE <= 0:
        raise ValueError(f"AUTODETECT_PHYSICAL_SCALE must be positive, got {PHYSICAL_SCALE!r}")
    if value not in (None, ""):
        return float(value) / PHYSICAL_SCALE
    legacy = os.getenv(legacy_pixel_name)
    if legacy not in (None, ""):
        return _env_float(legacy_pixel_name, legacy_default_px)
    return float(sensor_default_um) / PHYSICAL_SCALE


def build_detect_params() -> dict:
    return {
        "objective_config": OBJECTIVE_CONFIG,
        "camera_pixel_um": PHYSICAL_SCALE,
        "emission_nm": EMISSION_NM,
        "manual_spot_detect": _env_bool("AUTODETECT_MANUAL_SPOT_DETECT", "false"),
        "spot_radius_px": _env_float("AUTODETECT_SPOT_RADIUS_PX", 0.0),
        "log_quality_threshold": _env_float("AUTODETECT_LOG_QUALITY_THRESHOLD", 0.0),
        "log_signal_posterior": _env_float("AUTODETECT_LOG_SIGNAL_POSTERIOR", 0.99),
        "max_link_dist": _sensor_distance_px("AUTODETECT_MATCH_RADIUS_SENSOR_UM", "AUTODETECT_MATCH_RADIUS", 60.75, 13.5),
        "max_gap": _env_int("TRAJ_ASSOC_MAX_GAP", 1),
        "max_relink_gap": _env_int("TRAJ_ASSOC_MAX_RELINK_GAP", 4),
        "max_relink_dist": _sensor_distance_px("TRAJ_ASSOC_MAX_RELINK_DIST_SENSOR_UM", "TRAJ_ASSOC_MAX_RELINK_DIST", 63.0, 14.0),
        "max_relink_cost": _env_float("TRAJ_ASSOC_MAX_RELINK_COST", 0.95),
        "group_min_diagonal": _sensor_distance_px("TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM", "TRAJ_ASSOC_MIN_DIAGONAL", 202.5, 45.0),
        "group_max_diagonal": _sensor_distance_px("TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM", "TRAJ_ASSOC_MAX_DIAGONAL", 585.0, 130.0),
    }


def _apply_detect_params_to_env(params: dict) -> None:
    os.environ["AUTODETECT_MANUAL_SPOT_DETECT"] = str(params["manual_spot_detect"]).lower()
    os.environ["AUTODETECT_SPOT_RADIUS_PX"] = str(params["spot_radius_px"])
    os.environ["AUTODETECT_LOG_QUALITY_THRESHOLD"] = str(params["log_quality_threshold"])
    os.environ["AUTODETECT_LOG_SIGNAL_POSTERIOR"] = str(params["log_signal_posterior"])
    os.environ["AUTODETECT_MATCH_RADIUS"] = str(params["max_link_dist"])
    os.environ["TRAJ_ASSOC_MAX_GAP"] = str(params["max_gap"])
    os.environ["TRAJ_ASSOC_MAX_RELINK_GAP"] = str(params["max_relink_gap"])
    os.environ["TRAJ_ASSOC_MAX_RELINK_DIST"] = str(params["max_relink_dist"])
    os.environ["TRAJ_ASSOC_MAX_RELINK_COST"] = str(params["max_relink_cost"])
    os.environ["TRAJ_ASSOC_MIN_DIAGONAL"] = str(params["group_min_diagonal"])
    os.environ["TRAJ_ASSOC_MAX_DIAGONAL"] = str(params["group_max_diagonal"])


def estimate_detection_params(image_path: Path, base_params: dict) -> tuple[dict, dict]:
    """Derive LoG scales from optics rather than an expected number of detections."""
    if not _env_bool("AUTODETECT_AUTO_PARAMS", "true"):
        return {}, {"enabled": False, "reason": "AUTODETECT_AUTO_PARAMS=false"}
    manual = str(base_params["manual_spot_detect"]).strip().lower() in {"1", "true", "yes", "y", "on"}
    settings = DetectionSettings(
        objective_config=str(base_params["objective_config"]),
        camera_pixel_um=float(base_params["camera_pixel_um"]),
        emission_nm=float(base_params["emission_nm"]),
        spot_radius_px=float(base_params["spot_radius_px"]) if manual else 0.0,
        quality_threshold=float(base_params["log_quality_threshold"]) if manual else 0.0,
        signal_posterior=float(base_params["log_signal_posterior"]),
    )
    summary = detection_summary(settings)
    with LazyTiffStack(image_path) as stack:
        image_shape = [stack.frame_count, *stack.frame_shape]

    scale_rows = pd.DataFrame(
        {
            "log_sigma_px": summary["log_sigmas_px"],
            "scale_source": summary["scale_source"],
            "spot_radius_px": summary["spot_radius_px"],
            "quality_threshold": summary["quality_threshold"],
            "signal_posterior": summary["signal_posterior"],
            "quality_selection": summary["quality_selection"],
        }
    )
    scale_rows.to_csv("auto_detect_parameter_sweep.csv", index=False)
    # The automatic mode keeps ``spot_radius_px=0`` so the detector derives
    # its scale from acquisition optics at run time rather than freezing an
    # image-count-driven number into the pipeline parameters.
    estimated: dict[str, float] = {}
    report = {
        "enabled": True,
        "image_path": str(image_path.resolve()),
        "image_shape": image_shape,
        "method": "multi_scale_log_local_background_auto_quality",
        "count_prior_used": False,
        "peak_cap_used": False,
        "manual_spot_detect": manual,
        "selected": estimated,
        "detection_settings": summary,
        "scale_plan_csv": "auto_detect_parameter_sweep.csv",
    }
    Path("auto_detect_parameter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return estimated, report


def _load_module(script_name: str):
    script_path = REPO_DIR / script_name
    safe_name = (
        script_name
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
    )
    spec = importlib.util.spec_from_file_location(safe_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[safe_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_image() -> Path | None:
    for path in Path(".").iterdir():
        name_lower = path.name.lower()
        if "selected_image_stack" in name_lower and path.suffix.lower() in (".tif", ".tiff"):
            return path
    return None


def _copy_if_exists(src: str, dst: str) -> None:
    src_path = Path(src)
    if src_path.exists():
        shutil.copyfile(src_path, dst)
        print(f"Saved compatibility copy: {dst}")


def main() -> None:
    print("=== Auto-Detect: trajectory-level sub-signal association ===")

    repo_str = str(REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    tif_path = _find_image()
    if tif_path is None:
        print("ERROR: No image stack found. Expected selected_image_stack.tif/.tiff in the workspace.")
        sys.exit(1)

    print(f"Image:          {tif_path.resolve()}")
    print(f"Physical scale: {PHYSICAL_SCALE} um/pixel")

    detect_params = build_detect_params()
    auto_params, auto_report = estimate_detection_params(tif_path, detect_params)
    detect_params.update(auto_params)
    _apply_detect_params_to_env(detect_params)
    print(f"Detect params:  {detect_params}")
    if auto_report.get("enabled"):
        print(
            "Auto LoG scale: "
            f"source={auto_report['detection_settings']['scale_source']}, "
            f"radius={auto_report['detection_settings']['spot_radius_px']:.2f}px, "
            f"sigmas={auto_report['detection_settings']['log_sigmas_px']}"
        )

    trajectory_mod = _load_module("trajectory_association_adapter.py")
    refine_mod = _load_module("refine.py")
    pos_mod = _load_module("position_xy.py")

    groups_path, spots_path, membership, repaired = trajectory_mod.process_image_stack(
        tif_path,
        groups_csv="classified_groups_pca.csv",
        repaired_spots_csv="trajectory_association_spots.csv",
        plot_path="classified_groups_pca.png",
    )
    print(
        f"Trajectory association found {membership['Cluster_ID'].nunique()} group(s), "
        f"{membership['TRACK_ID'].nunique()} member track(s), {len(repaired)} repaired spot row(s)."
    )

    _copy_if_exists(str(groups_path), "classified_groups_multi.csv")
    _copy_if_exists(str(spots_path), "spots_multi_group_interpolate.csv")

    refine_params = {
        "PHYSICAL_SCALE": PHYSICAL_SCALE,
        "MODEL_REFERENCE_CAMERA_PIXEL_UM": float(os.getenv("Z_MODEL_REFERENCE_CAMERA_PIXEL_UM", "4.5")),
        "OBJECTIVE_CONFIG": OBJECTIVE_CONFIG,
        "EMISSION_NM": EMISSION_NM,
        "ROI_RADIUS": int(os.getenv("AUTODETECT_REFINE_ROI_RADIUS", "7")),
        "INTEGRATION_RADIUS": float(os.getenv("AUTODETECT_INTEGRATION_RADIUS", "10.0")),
        "USE_SUBPIXEL_REFINE": True,
        "XY_LOCALIZATION_METHOD": "phasor",
        "Z_LOCALIZATION_METHOD": "phasor",
        "LOCALIZATION_METHOD": "phasor",
        "Z_PHASOR_BLUR_SIGMA_PX": os.getenv("AUTODETECT_Z_PHASOR_BLUR_SIGMA_PX", "2.4"),
    }
    refine_mod.main(
        csv_path=str(spots_path),
        image_path=str(tif_path),
        output_csv_path="spots_refined.csv",
        params=refine_params,
        show_plot=False,
    )
    spots_refined = read_pipeline_csv("spots_refined.csv")
    print(f"Saved spots_refined.csv ({len(spots_refined)} rows).")

    pos_mod.calculate_hybrid_centers(
        "classified_groups_pca.csv",
        "spots_refined.csv",
        "Lateral positions.csv",
        image_path=str(tif_path),
        params={
            "PHYSICAL_SCALE": PHYSICAL_SCALE,
            "INTEGRATION_RADIUS": refine_params["INTEGRATION_RADIUS"],
            "REINTEGRATE_INTENSITY": True,
        },
    )

    print("\n=== Auto-detect complete. Steps 0/1-3 outputs are ready. ===")


if __name__ == "__main__":
    main()
