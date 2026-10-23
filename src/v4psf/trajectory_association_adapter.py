from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .csv_io import read_csv as read_pipeline_csv
from .spot_detection import DetectionSettings, LazyTiffStack, detect_spots_in_frame
from . import trajectory_subsignal_core as core


TRACKMATE_SKIPROWS = [1, 2, 3]


def _env_float(names: str | tuple[str, ...], default: float) -> float:
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


def _env_int(names: str | tuple[str, ...], default: int) -> int:
    return int(round(_env_float(names, float(default))))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    attempts = [
        {"encoding": "utf-8-sig"},
        {"encoding": "latin1"},
        {"encoding": "gbk"},
        {"header": 0, "skiprows": TRACKMATE_SKIPROWS, "encoding": "utf-8-sig"},
        {"header": 0, "skiprows": TRACKMATE_SKIPROWS, "encoding": "latin1"},
        {"header": 0, "skiprows": TRACKMATE_SKIPROWS, "encoding": "gbk"},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            df = read_pipeline_csv(path, **kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            if len(df.columns):
                return df
        except Exception as exc:
            last_exc = exc
    raise ValueError(f"Could not parse CSV {path}: {last_exc}")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = cand.strip().lower()
        for norm, original in normalized.items():
            if key in norm:
                return original
    return None


def _series_or_default(df: pd.DataFrame, col: str | None, default: float = 0.0) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _normalize_spot_table(spots: pd.DataFrame, track_ids: set[int] | None = None) -> pd.DataFrame:
    frame_col = _find_col(spots, ["FRAME", "POSITION_T", "t", "frame"])
    track_col = _find_col(spots, ["TRACK_ID", "TRACK_INDEX", "track id"])
    x_col = _find_col(spots, ["X", "POSITION_X", "Spot position X", "POSITION_X [pixel]", "POSITION_X [px]"])
    y_col = _find_col(spots, ["Y", "POSITION_Y", "Spot position Y", "POSITION_Y [pixel]", "POSITION_Y [px]"])
    intensity_col = _find_col(
        spots,
        [
            "Refined_Intensity",
            "MEAN_INTENSITY_CH1",
            "TOTAL_INTENSITY_CH1",
            "MAX_INTENSITY_CH1",
            "QUALITY",
            "intensity",
        ],
    )
    if frame_col is None or x_col is None or y_col is None:
        raise ValueError(
            "Spots CSV must contain frame and X/Y columns. "
            f"Found columns: {spots.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["FRAME"] = pd.to_numeric(spots[frame_col], errors="coerce").astype("Int64")
    if track_col is not None:
        out["RAW_TRACK_ID"] = pd.to_numeric(spots[track_col], errors="coerce").astype("Int64")
    else:
        out["RAW_TRACK_ID"] = pd.Series(range(1, len(spots) + 1), index=spots.index, dtype="Int64")
    out["TRACK_ID"] = out["RAW_TRACK_ID"]
    out["X"] = pd.to_numeric(spots[x_col], errors="coerce")
    out["Y"] = pd.to_numeric(spots[y_col], errors="coerce")
    out["intensity"] = _series_or_default(spots, intensity_col, 0.0).fillna(0.0)
    out["integrated_intensity"] = out["intensity"]
    out["detection_id"] = [
        f"tm_f{int(frame) if pd.notna(frame) else -1}_t{int(tid) if pd.notna(tid) else -1}_r{i}"
        for i, (frame, tid) in enumerate(zip(out["FRAME"], out["TRACK_ID"]))
    ]
    out["source_id"] = ""
    out["possible_source_ids"] = ""
    out["possible_particle_ids"] = ""
    out["particle_id"] = 0
    out["spot_index"] = 0
    out["truth_track_id"] = 0
    out["is_noise"] = True
    out["is_interpolated"] = False
    out["lap_link_cost"] = 0.0
    out["lap_link_status"] = "trackmate_input"
    out = out.dropna(subset=["FRAME", "TRACK_ID", "X", "Y"]).copy()
    out["FRAME"] = out["FRAME"].astype(int)
    out["RAW_TRACK_ID"] = out["RAW_TRACK_ID"].astype(int)
    out["TRACK_ID"] = out["TRACK_ID"].astype(int)
    if track_ids:
        out = out[out["TRACK_ID"].isin(track_ids)].copy()
    return core.ensure_spot_appearance_columns(out)


def _valid_track_ids(track_csv: str | Path | None) -> set[int] | None:
    if not track_csv:
        return None
    path = Path(track_csv)
    if not path.exists():
        return None
    try:
        tracks = _read_csv_flexible(path)
    except Exception:
        return None
    track_col = _find_col(tracks, ["TRACK_ID", "TRACK_INDEX", "track id"])
    if track_col is None:
        return None
    vals = pd.to_numeric(tracks[track_col], errors="coerce").dropna().astype(int)
    return set(vals.tolist()) if len(vals) else None


def repair_trackmate_spots(track_csv: str | Path | None, spots_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spots = _read_csv_flexible(spots_csv)
    detections = _normalize_spot_table(spots, _valid_track_ids(track_csv))
    n_frames = int(detections["FRAME"].max()) + 1 if len(detections) else 0
    repaired = core.repair_and_interpolate_tracks_lap_cost(
        detections,
        n_frames=n_frames,
        max_relink_gap=_env_int("TRAJ_ASSOC_MAX_RELINK_GAP", 4),
        max_relink_dist=core.runtime_sensor_distance_px(
            "TRAJ_ASSOC_MAX_RELINK_DIST_SENSOR_UM",
            "TRAJ_ASSOC_MAX_RELINK_DIST",
            63.0,
        ),
        max_relink_cost=_env_float("TRAJ_ASSOC_MAX_RELINK_COST", 0.95),
    )
    groups = core.trajectory_lap_cost_group(repaired)
    return repaired, groups


def _detection_settings_from_env() -> DetectionSettings:
    manual = _env_bool("AUTODETECT_MANUAL_SPOT_DETECT", False)
    return DetectionSettings(
        objective_config=os.getenv("AUTODETECT_OBJECTIVE_CONFIG", "20x_NA0.45_n1.0"),
        camera_pixel_um=_env_float("AUTODETECT_PHYSICAL_SCALE", 4.5),
        emission_nm=_env_float("AUTODETECT_EMISSION_NM", 650.0),
        spot_radius_px=_env_float("AUTODETECT_SPOT_RADIUS_PX", 0.0) if manual else 0.0,
        quality_threshold=_env_float("AUTODETECT_LOG_QUALITY_THRESHOLD", 0.0) if manual else 0.0,
        signal_posterior=_env_float("AUTODETECT_LOG_SIGNAL_POSTERIOR", 0.99),
    )


def _detect_spots_from_stack(stack: LazyTiffStack) -> pd.DataFrame:
    detected_frames: list[pd.DataFrame] = []
    settings = _detection_settings_from_env()
    workers = max(1, _env_int("AUTODETECT_DETECTION_WORKERS", 2))

    def detect_one(item: tuple[int, np.ndarray]) -> pd.DataFrame:
        frame, image = item
        detected, _ = detect_spots_in_frame(image, settings, frame=frame)
        return detected

    batch_size = workers * 2
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="log-spot") as executor:
        for start in range(0, stack.frame_count, batch_size):
            frame_indices = range(start, min(start + batch_size, stack.frame_count))
            batch = [(frame, stack.get_frame(frame)) for frame in frame_indices]
            for detected in executor.map(detect_one, batch):
                if detected.empty:
                    continue
                detected["TRACK_ID"] = -1
                detected["particle_id"] = 0
                detected["source_id"] = ""
                detected["possible_source_ids"] = ""
                detected["possible_particle_ids"] = ""
                detected["spot_index"] = 0
                detected["truth_track_id"] = 0
                detected["is_noise"] = False
                detected["is_interpolated"] = False
                detected_frames.append(detected)
            completed = min(start + batch_size, stack.frame_count)
            if completed == stack.frame_count or completed % max(20, batch_size * 5) == 0:
                print(f"LoG detection progress: {completed}/{stack.frame_count} frames", flush=True)

    detected = pd.concat(detected_frames, ignore_index=True) if detected_frames else pd.DataFrame()
    if not detected.empty:
        detected["image_detection_index"] = np.arange(len(detected), dtype=int)
        detected["TRACK_ID"] = detected["image_detection_index"].astype(int)
    return core.ensure_spot_appearance_columns(detected)


def detect_and_group_image_stack(image_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with LazyTiffStack(image_path) as stack:
        detected = _detect_spots_from_stack(stack)
        n_frames = stack.frame_count
    save_intermediates = _env_bool("TRAJ_ASSOC_SAVE_INTERMEDIATES", False)
    if save_intermediates:
        detected.to_csv("trajectory_detected_spots.csv", index=False)
    raw_tracks = core.link_detections_to_tracks_lap_cost(
        detected,
        max_link_dist=core.runtime_sensor_distance_px(
            ("AUTODETECT_MATCH_RADIUS_SENSOR_UM", "TRAJ_ASSOC_MAX_LINK_DIST_SENSOR_UM"),
            ("TRAJ_ASSOC_MAX_LINK_DIST", "AUTODETECT_MATCH_RADIUS"),
            60.75,
        ),
        max_gap=_env_int("TRAJ_ASSOC_MAX_GAP", 1),
        max_total_cost=_env_float("TRAJ_ASSOC_MAX_TOTAL_COST", 1.45),
    )
    if save_intermediates:
        raw_tracks.to_csv("trajectory_raw_tracks.csv", index=False)
    repaired = core.repair_and_interpolate_tracks_lap_cost(
        raw_tracks,
        n_frames=n_frames,
        max_relink_gap=_env_int("TRAJ_ASSOC_MAX_RELINK_GAP", 4),
        max_relink_dist=core.runtime_sensor_distance_px(
            "TRAJ_ASSOC_MAX_RELINK_DIST_SENSOR_UM",
            "TRAJ_ASSOC_MAX_RELINK_DIST",
            63.0,
        ),
        max_relink_cost=_env_float("TRAJ_ASSOC_MAX_RELINK_COST", 0.95),
    )
    if save_intermediates:
        repaired.to_csv("trajectory_repaired_tracks.csv", index=False)
    groups = core.trajectory_lap_cost_group(repaired)
    if _env_bool("TRAJ_ASSOC_STABILIZE_TRACK_IDS", True) and not groups.empty:
        repaired, stabilization_audit = core.stabilize_track_ids_from_parent_groups(
            repaired,
            groups,
        )
        groups = core.trajectory_lap_cost_group(repaired)
        groups.attrs["track_id_stabilization_audit"] = stabilization_audit
    return detected, repaired, groups


def _split_track_ids(value: Any) -> list[int]:
    out: list[int] = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except Exception:
            continue
    return out


def groups_to_membership_rows(groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if groups.empty:
        return pd.DataFrame(columns=["TRACK_ID", "Cluster_ID", "Cluster_Type"])
    for cluster_id, (_, group) in enumerate(groups.iterrows(), start=1):
        tids = _split_track_ids(group.get("member_track_ids", ""))
        cluster_type = "4-Square" if len(tids) >= 4 else "3-Triangle"
        for tid in tids:
            row = {
                "TRACK_ID": int(tid),
                "Cluster_ID": int(cluster_id),
                "Cluster_Type": cluster_type,
                "Trajectory_Group_ID": int(group.get("trajectory_group_id", cluster_id)),
                "Trajectory_Parent_Group_ID": group.get("trajectory_parent_group_id", np.nan),
                "Trajectory_Method": str(group.get("method", "trajectory_lap_cost")),
                "Trajectory_Parent_Link_Method": str(group.get("trajectory_parent_link_method", "unknown")),
                "Trajectory_Score": float(group.get("score", np.nan)),
                "Trajectory_Geometry_Score": float(group.get("geometry_score", np.nan)),
                "Trajectory_Mean_Diagonal_Px": float(group.get("mean_diag_px", np.nan)),
                "Trajectory_Mean_Diagonal_Sensor_Um": float(group.get("mean_diag_sensor_um", np.nan)),
                "Trajectory_N_Tracks": int(len(tids)),
                "Trajectory_Recoverable_Frames": int(group.get("recoverable_track_frames", 0)),
                "segment_start_frame": group.get("segment_start_frame", np.nan),
                "segment_end_frame": group.get("segment_end_frame", np.nan),
                "segment_index": group.get("segment_index", np.nan),
                "segment_count": group.get("segment_count", np.nan),
                "local_segment_group_index": group.get("local_segment_group_index", np.nan),
                "Assigned_Source_ID": str(group.get("assigned_source_id", "")),
                "Association_Status": str(group.get("association_status", "")),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def repaired_tracks_to_spots_csv(repaired: pd.DataFrame) -> pd.DataFrame:
    out = repaired.copy()
    if "Refined_Intensity" not in out.columns:
        if "integrated_intensity" in out.columns:
            out["Refined_Intensity"] = pd.to_numeric(out["integrated_intensity"], errors="coerce")
        elif "intensity" in out.columns:
            out["Refined_Intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
        else:
            out["Refined_Intensity"] = np.nan
    keep = [
        "TRACK_ID",
        "RAW_TRACK_ID",
        "TRACK_SEGMENT_ID",
        "PRE_STABILIZATION_TRACK_ID",
        "TRACK_ID_STABILIZATION_METHOD",
        "FRAME",
        "X",
        "Y",
        "Refined_Intensity",
        "intensity",
        "integrated_intensity",
        "detection_id",
        "is_interpolated",
        "repair_model",
        "lap_link_status",
        "lap_link_cost",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)


def write_association_outputs(
    repaired: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    groups_csv: str | Path = "classified_groups_pca.csv",
    spots_csv: str | Path = "trajectory_association_spots.csv",
    groups_debug_csv: str | Path = "trajectory_lap_cost_groups.csv",
    plot_path: str | Path = "classified_groups_pca.png",
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    if groups.empty:
        raise ValueError("trajectory_lap_cost produced no valid 3/4-track groups.")
    groups_path = Path(groups_csv)
    spots_path = Path(spots_csv)
    groups_debug_path = Path(groups_debug_csv)
    membership = groups_to_membership_rows(groups)
    spots_out = repaired_tracks_to_spots_csv(repaired)
    membership.to_csv(groups_path, index=False)
    spots_out.to_csv(spots_path, index=False)
    groups.to_csv(groups_debug_path, index=False)
    stabilization_audit = groups.attrs.get("track_id_stabilization_audit")
    if isinstance(stabilization_audit, pd.DataFrame) and not stabilization_audit.empty:
        audit_path = spots_path.with_name("track_id_stabilization_audit.csv")
        stabilization_audit.to_csv(audit_path, index=False)
    plot_groups(membership, spots_out, plot_path)
    return groups_path, spots_path, membership, spots_out


def plot_groups(membership: pd.DataFrame, spots: pd.DataFrame, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if membership.empty or spots.empty:
        ax.text(0.5, 0.5, "No trajectory groups", ha="center", va="center", transform=ax.transAxes)
    else:
        merged = spots.merge(membership, on="TRACK_ID", how="inner")
        if "segment_start_frame" in merged.columns:
            start = pd.to_numeric(merged["segment_start_frame"], errors="coerce")
            frame_values = pd.to_numeric(merged["FRAME"], errors="coerce")
            merged = merged[start.isna() | (frame_values >= start)]
        if "segment_end_frame" in merged.columns:
            end = pd.to_numeric(merged["segment_end_frame"], errors="coerce")
            frame_values = pd.to_numeric(merged["FRAME"], errors="coerce")
            merged = merged[end.isna() | (frame_values <= end)]
        frames = sorted(pd.to_numeric(merged["FRAME"], errors="coerce").dropna().astype(int).unique())
        frame = frames[0] if frames else 0
        frame_data = merged[merged["FRAME"].astype(int) == frame]
        cmap = plt.cm.tab20
        for i, (cid, sub) in enumerate(frame_data.groupby("Cluster_ID")):
            color = cmap(i % 20)
            ax.scatter(sub["X"], sub["Y"], color=color, s=55, label=f"C{int(cid)}")
            ax.text(float(sub["X"].mean()), float(sub["Y"].mean()), str(int(cid)), color="white", ha="center", va="center")
        ax.set_title(f"Trajectory-level groups, frame {frame}")
        ax.invert_yaxis()
        ax.set_aspect("equal", "datalim")
        if frame_data["Cluster_ID"].nunique() <= 16:
            ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def process_trackmate_inputs(
    track_csv: str | Path | None,
    spots_csv: str | Path,
    *,
    groups_csv: str | Path = "classified_groups_pca.csv",
    repaired_spots_csv: str | Path = "trajectory_association_spots.csv",
    plot_path: str | Path = "classified_groups_pca.png",
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    repaired, groups = repair_trackmate_spots(track_csv, spots_csv)
    return write_association_outputs(
        repaired,
        groups,
        groups_csv=groups_csv,
        spots_csv=repaired_spots_csv,
        plot_path=plot_path,
    )


def process_image_stack(
    image_path: str | Path,
    *,
    groups_csv: str | Path = "classified_groups_pca.csv",
    repaired_spots_csv: str | Path = "trajectory_association_spots.csv",
    plot_path: str | Path = "classified_groups_pca.png",
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    _detected, repaired, groups = detect_and_group_image_stack(image_path)
    return write_association_outputs(
        repaired,
        groups,
        groups_csv=groups_csv,
        spots_csv=repaired_spots_csv,
        plot_path=plot_path,
    )
