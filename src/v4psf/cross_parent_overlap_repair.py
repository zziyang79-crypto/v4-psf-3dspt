"""Cross-parent subspot overlap detection and repair.

When two parent trajectories pass close to one another near a segment
boundary, a localized subspot can be pulled toward the combined intensity
blob.  The repair is deliberately performed on both coordinate channels:
``X/Y`` drive XY and angle-intensity sampling, while
``Z_Localization_X/Y`` drive the Z geometry.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _enabled(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _finite_xy(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y))


def detect_and_repair_cross_parent_overlap(
    df: pd.DataFrame,
    overlap_distance_sensor_um: float = 63.0,
    boundary_margin_frames: int = 3,
) -> pd.DataFrame:
    """Repair subspot overlap near parent-segment boundaries.

    The detector considers same-frame and neighboring-frame candidates.  A
    bracket interpolation is applied only when the same ``(parent, track)``
    has clean observations on both sides.  The caller should reintegrate
    intensity *after* this function returns.
    """
    required = {"Cluster_ID", "TRACK_ID", "FRAME", "X", "Y"}
    if df.empty or not required.issubset(df.columns):
        return df
    if not _enabled("POSITION_XY_CROSS_PARENT_OVERLAP_ENABLED"):
        return df

    overlap_text = os.getenv("POSITION_XY_CROSS_PARENT_OVERLAP_DISTANCE_SENSOR_UM")
    if overlap_text in (None, ""):
        legacy_px = os.getenv("POSITION_XY_CROSS_PARENT_OVERLAP_DISTANCE_PX")
        overlap_sensor_um = (
            float(legacy_px) * float(os.getenv("POSITION_XY_PHYSICAL_SCALE", "4.5"))
            if legacy_px not in (None, "") else float(overlap_distance_sensor_um)
        )
    else:
        overlap_sensor_um = float(overlap_text)
    margin = int(float(os.getenv("POSITION_XY_CROSS_PARENT_OVERLAP_MARGIN_FRAMES", str(boundary_margin_frames))))
    max_bracket_gap = int(float(os.getenv("POSITION_XY_CROSS_PARENT_OVERLAP_MAX_BRACKET_GAP", "8")))

    out = df.copy()
    for column, default in [
        ("Cross_Parent_Overlap_Detected", False),
        ("Cross_Parent_Overlap_Neighbor_Parent", np.nan),
        ("Cross_Parent_Overlap_Neighbor_TRACK_ID", np.nan),
        ("Cross_Parent_Overlap_Distance_Sensor_Um", np.nan),
        ("Cross_Parent_Overlap_Repaired", False),
        ("Cross_Parent_Overlap_Repair_Source", ""),
    ]:
        if column not in out.columns:
            out[column] = default

    coordinate_columns = [
        column
        for column in ["X", "Y", "Z_Localization_X", "Z_Localization_Y"]
        if column in out.columns
    ]
    for column in coordinate_columns:
        original_column = f"Cross_Parent_Overlap_Original_{column}"
        if original_column not in out.columns:
            out[original_column] = np.nan

    if "Trajectory_Parent_Group_ID" in out.columns:
        out["_parent_group"] = pd.to_numeric(out["Trajectory_Parent_Group_ID"], errors="coerce")
    else:
        out["_parent_group"] = pd.to_numeric(out["Cluster_ID"], errors="coerce")
    out["_parent_group"] = out["_parent_group"].fillna(pd.to_numeric(out["Cluster_ID"], errors="coerce"))

    row_labels = out.index.to_numpy()
    x_values = pd.to_numeric(out["X"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(out["Y"], errors="coerce").to_numpy(dtype=float)
    parent_values = out["_parent_group"].to_numpy(dtype=float)
    frame_values = pd.to_numeric(out["FRAME"], errors="coerce").to_numpy(dtype=float)
    track_values = pd.to_numeric(out["TRACK_ID"], errors="coerce").to_numpy(dtype=float)

    parent_frames: dict[int, set[int]] = {}
    boundary_frames: dict[int, set[int]] = {}
    for parent_id, group in out.groupby("_parent_group", sort=False):
        if pd.isna(parent_id):
            continue
        pid = int(parent_id)
        frames = set(pd.to_numeric(group["FRAME"], errors="coerce").dropna().astype(int).tolist())
        parent_frames[pid] = frames
        if {"segment_start_frame", "segment_end_frame"}.issubset(group.columns):
            marked: set[int] = set()
            for row in group[["segment_start_frame", "segment_end_frame"]].drop_duplicates().itertuples(index=False):
                try:
                    start = int(float(row.segment_start_frame))
                    end = int(float(row.segment_end_frame))
                except (TypeError, ValueError):
                    continue
                marked.update(range(start, start + margin))
                marked.update(range(max(start, end - margin + 1), end + 1))
            if marked:
                boundary_frames[pid] = marked

    overlap_rows: set[object] = set()
    overlap_details: dict[object, dict[str, float]] = {}

    def add_overlap(label: object, neighbor_parent: int, neighbor_track: float, distance: float) -> None:
        previous = overlap_details.get(label)
        if previous is not None and float(previous["distance_sensor_um"]) <= distance:
            return
        overlap_rows.add(label)
        overlap_details[label] = {
            "neighbor_parent": float(neighbor_parent),
            "neighbor_track": neighbor_track,
            "distance_sensor_um": distance,
        }

    parent_ids = sorted(parent_frames)
    for left_pos, parent_a in enumerate(parent_ids):
        for parent_b in parent_ids[left_pos + 1 :]:
            candidate_frames = boundary_frames.get(parent_a, parent_frames[parent_a]) | boundary_frames.get(
                parent_b, parent_frames[parent_b]
            )
            if not candidate_frames:
                continue
            mask_a = (np.abs(parent_values - parent_a) < 0.1) & np.isin(frame_values, list(candidate_frames))
            mask_b = (np.abs(parent_values - parent_b) < 0.1) & np.isin(frame_values, list(candidate_frames))
            positions_a = np.flatnonzero(mask_a)
            positions_b = np.flatnonzero(mask_b)
            for pos_a in positions_a:
                if not _finite_xy(x_values[pos_a], y_values[pos_a]):
                    continue
                for pos_b in positions_b:
                    if abs(frame_values[pos_a] - frame_values[pos_b]) > 1:
                        continue
                    if not _finite_xy(x_values[pos_b], y_values[pos_b]):
                        continue
                    distance = float(np.hypot(x_values[pos_a] - x_values[pos_b], y_values[pos_a] - y_values[pos_b]))
                    if distance > overlap_sensor_um:
                        continue
                    add_overlap(row_labels[pos_a], parent_b, track_values[pos_b], distance)
                    add_overlap(row_labels[pos_b], parent_a, track_values[pos_a], distance)

    if not overlap_rows:
        return out.drop(columns=["_parent_group"], errors="ignore")

    for row_label in overlap_rows:
        detail = overlap_details[row_label]
        out.at[row_label, "Cross_Parent_Overlap_Detected"] = True
        out.at[row_label, "Cross_Parent_Overlap_Neighbor_Parent"] = detail["neighbor_parent"]
        out.at[row_label, "Cross_Parent_Overlap_Neighbor_TRACK_ID"] = detail["neighbor_track"]
        out.at[row_label, "Cross_Parent_Overlap_Distance_Sensor_Um"] = detail["distance_sensor_um"]

    repaired_count = 0
    for _, track_group in out.groupby(["_parent_group", "TRACK_ID"], sort=False):
        track_group = track_group.sort_values("FRAME")
        if len(track_group) < 3 or not track_group["Cross_Parent_Overlap_Detected"].fillna(False).astype(bool).any():
            continue

        labels = track_group.index.to_numpy()
        frames = pd.to_numeric(track_group["FRAME"], errors="coerce").to_numpy(dtype=float)
        x_track = pd.to_numeric(track_group["X"], errors="coerce").to_numpy(dtype=float)
        y_track = pd.to_numeric(track_group["Y"], errors="coerce").to_numpy(dtype=float)
        usable = np.asarray([_finite_xy(x, y) for x, y in zip(x_track, y_track)], dtype=bool)
        overlap_mask = track_group["Cross_Parent_Overlap_Detected"].fillna(False).astype(bool).to_numpy()

        for pos in np.flatnonzero(overlap_mask):
            left = next(
                (
                    candidate
                    for candidate in range(pos - 1, -1, -1)
                    if usable[candidate] and not overlap_mask[candidate] and frames[pos] - frames[candidate] <= max_bracket_gap
                ),
                None,
            )
            right = next(
                (
                    candidate
                    for candidate in range(pos + 1, len(track_group))
                    if usable[candidate] and not overlap_mask[candidate] and frames[candidate] - frames[pos] <= max_bracket_gap
                ),
                None,
            )
            if left is None or right is None or frames[right] == frames[left]:
                continue

            ratio = float((frames[pos] - frames[left]) / (frames[right] - frames[left]))
            row_label = labels[pos]
            for column in coordinate_columns:
                values = pd.to_numeric(track_group[column], errors="coerce").to_numpy(dtype=float)
                if not (np.isfinite(values[left]) and np.isfinite(values[right])):
                    continue
                original_column = f"Cross_Parent_Overlap_Original_{column}"
                if pd.isna(out.at[row_label, original_column]):
                    out.at[row_label, original_column] = out.at[row_label, column]
                out.at[row_label, column] = float(values[left] + ratio * (values[right] - values[left]))

            out.at[row_label, "Cross_Parent_Overlap_Repaired"] = True
            out.at[row_label, "Cross_Parent_Overlap_Repair_Source"] = "same_track_bracket_interpolation"
            if "Geometry_Subspot_Repaired" in out.columns:
                out.at[row_label, "Geometry_Subspot_Repaired"] = True
            if "Geometry_Subspot_Repair_Source" in out.columns:
                out.at[row_label, "Geometry_Subspot_Repair_Source"] = "cross_parent_overlap_bracket_interpolation"
            if "Subspot_Position_Source" in out.columns:
                out.at[row_label, "Subspot_Position_Source"] = "cross_parent_overlap_bracket_interpolated"
            if "Subspot_Completion_Note" in out.columns:
                out.at[row_label, "Subspot_Completion_Note"] = "cross_parent_overlap_replaced_by_bracketing_track_interp"
            if "Z_Use_For_Geometry" in out.columns:
                out.at[row_label, "Z_Use_For_Geometry"] = True
            repaired_count += 1

    if repaired_count:
        print(f"   Cross-parent overlap: {len(overlap_rows)} subspots flagged, {repaired_count} repaired by same-track bracket interpolation.")
    else:
        print(f"   Cross-parent overlap: {len(overlap_rows)} subspots flagged, none repairable (insufficient bracket frames).")
    return out.drop(columns=["_parent_group"], errors="ignore")
