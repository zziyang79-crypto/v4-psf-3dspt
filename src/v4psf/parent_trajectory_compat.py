"""Version-stable parent-frame selection and conservative 3D fragment stitching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_3D_COLUMNS = {
    "Parent_Group_ID",
    "FRAME",
    "Center_X",
    "Center_Y",
    "Predicted_Z",
}


def select_parent_frame_sources(base: pd.DataFrame, selector) -> pd.DataFrame:
    """Run a one-row selector without depending on ``GroupBy.apply`` shape rules.

    pandas changed whether grouping columns are included in ``apply`` inputs and
    how their keys are added to the result index.  Building the rows explicitly
    keeps the output contract identical on pandas 1.x and 2.x.
    """

    rows: list[dict] = []
    for (_parent_id, _frame), group in base.groupby(
        ["Parent_Group_ID", "FRAME"], sort=True, dropna=False
    ):
        selected = selector(group)
        if isinstance(selected, pd.Series):
            rows.append(selected.to_dict())
        elif isinstance(selected, dict):
            rows.append(dict(selected))
        else:
            raise TypeError(f"Parent-frame selector returned {type(selected)!r}, expected Series or dict")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows).sort_values(
        ["Parent_Group_ID", "FRAME"], kind="stable"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class StitchLink:
    source_parent: int
    target_parent: int
    source_end_frame: int
    target_start_frame: int
    observed_missing_frames: int
    source_boundary_end_frame: int
    target_boundary_start_frame: int
    boundary_gap_frames: int
    xy_endpoint_distance_sensor_um: float
    z_endpoint_distance_um: float
    spatial_normalized_cost: float
    normalized_cost: float
    link_method: str = "cross_parent_endpoint"


AUDIT_COLUMNS = list(StitchLink.__dataclass_fields__)


def _endpoint_state(part: pd.DataFrame, *, at_end: bool) -> tuple[np.ndarray, int, int]:
    ordered = part.sort_values("FRAME", kind="stable")
    edge = ordered.iloc[-1] if at_end else ordered.iloc[0]
    frame = int(round(float(edge["FRAME"])))
    position = edge[["Center_X", "Center_Y", "Predicted_Z"]].to_numpy(dtype=float)
    boundary_column = "Segment_End_Max" if at_end else "Segment_Start_Min"
    boundary = frame
    if boundary_column in ordered.columns:
        values = pd.to_numeric(ordered[boundary_column], errors="coerce").dropna()
        if not values.empty:
            boundary = int(round(float(values.max() if at_end else values.min())))
    return position, frame, boundary


def stitch_parent_fragments(
    parent_df: pd.DataFrame,
    *,
    magnification: float,
    max_gap_frames: int = 12,
    max_xy_sensor_um: float = 400.0,
    max_z_um: float = 4.0,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join non-overlapping parent fragments and interpolate only the accepted gaps.

    ``Center_X/Y`` are sample-plane micrometres at this stage. Endpoint XY
    distance is evaluated after multiplying by objective magnification, so the
    gate is a fixed camera-sensor distance independent of pixel pitch or binning.
    The gap gate uses original segment boundaries when available because Z source
    selection can remove valid edge frames from the 3D table. No calibration or
    model file is modified.
    """

    missing = REQUIRED_3D_COLUMNS.difference(parent_df.columns)
    if missing:
        raise ValueError(f"Parent stitching missing columns: {sorted(missing)}")
    out = parent_df.copy()
    if out.empty:
        return out, pd.DataFrame(columns=AUDIT_COLUMNS)

    for column in ["Parent_Group_ID", "FRAME", "Center_X", "Center_Y", "Predicted_Z"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_3D_COLUMNS)).copy()
    out["Parent_Group_ID"] = out["Parent_Group_ID"].astype(int)
    out["FRAME"] = out["FRAME"].round().astype(int)
    out["Original_Parent_Group_ID"] = out["Parent_Group_ID"]
    out["Parent_Track_Observed"] = True
    out["Parent_Track_Interpolated"] = False
    out["Parent_Track_Stitch_Source"] = "observed"

    if (
        not enabled
        or max_gap_frames < 0
        or max_xy_sensor_um <= 0
        or max_z_um <= 0
    ):
        return (
            out.sort_values(["Parent_Group_ID", "FRAME"], kind="stable").reset_index(drop=True),
            pd.DataFrame(columns=AUDIT_COLUMNS),
        )

    parts = {int(pid): group.copy() for pid, group in out.groupby("Parent_Group_ID", sort=True)}
    endpoints = {
        pid: {
            "start": _endpoint_state(part, at_end=False),
            "end": _endpoint_state(part, at_end=True),
        }
        for pid, part in parts.items()
    }

    candidates: list[StitchLink] = []
    for source_id, source_state in endpoints.items():
        source_pos, source_frame, source_boundary_end = source_state["end"]
        for target_id, target_state in endpoints.items():
            if source_id == target_id:
                continue
            target_pos, target_frame, target_boundary_start = target_state["start"]
            observed_missing_frames = target_frame - source_frame - 1
            boundary_gap_frames = target_boundary_start - source_boundary_end - 1
            if (
                observed_missing_frames < 0
                or boundary_gap_frames < 0
                or boundary_gap_frames > int(max_gap_frames)
            ):
                continue
            residual = target_pos - source_pos
            xy_sensor_um = float(np.linalg.norm(residual[:2]) * float(magnification))
            z_um = float(abs(residual[2]))
            if not np.isfinite(xy_sensor_um + z_um):
                continue
            if xy_sensor_um <= max_xy_sensor_um and z_um <= max_z_um:
                spatial_cost = float(np.hypot(xy_sensor_um / max_xy_sensor_um, z_um / max_z_um))
                if spatial_cost > 1.0:
                    continue
                gap_cost = 0.1 * boundary_gap_frames / float(max(int(max_gap_frames), 1))
                cost = spatial_cost + gap_cost
                candidates.append(
                    StitchLink(
                        source_parent=source_id,
                        target_parent=target_id,
                        source_end_frame=source_frame,
                        target_start_frame=target_frame,
                        observed_missing_frames=observed_missing_frames,
                        source_boundary_end_frame=source_boundary_end,
                        target_boundary_start_frame=target_boundary_start,
                        boundary_gap_frames=boundary_gap_frames,
                        xy_endpoint_distance_sensor_um=xy_sensor_um,
                        z_endpoint_distance_um=z_um,
                        spatial_normalized_cost=spatial_cost,
                        normalized_cost=cost,
                        link_method="cross_parent_endpoint",
                    )
                )

    parent = {pid: pid for pid in parts}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    accepted: list[StitchLink] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for link in sorted(
        candidates,
        key=lambda item: (item.normalized_cost, item.boundary_gap_frames, item.target_start_frame),
    ):
        if link.source_parent in used_sources or link.target_parent in used_targets:
            continue
        if find(link.source_parent) == find(link.target_parent):
            continue
        accepted.append(link)
        used_sources.add(link.source_parent)
        used_targets.add(link.target_parent)
        union(link.source_parent, link.target_parent)

    canonical = {pid: find(pid) for pid in parts}
    out["Parent_Group_ID"] = out["Parent_Group_ID"].map(canonical).astype(int)
    link_by_pair = {(link.source_parent, link.target_parent): link for link in accepted}
    link_text_by_parent: dict[int, list[str]] = {}
    for link in accepted:
        text = f"{link.source_parent}->{link.target_parent}"
        link_text_by_parent.setdefault(link.source_parent, []).append(text)
        link_text_by_parent.setdefault(link.target_parent, []).append(text)
    out["Parent_Track_Stitch_Source"] = out["Original_Parent_Group_ID"].map(
        lambda value: ";".join(link_text_by_parent.get(int(value), [])) or "observed"
    )

    interpolated: list[dict] = []
    for (source_id, target_id), link in link_by_pair.items():
        if link.observed_missing_frames <= 0:
            continue
        source_row = parts[source_id].sort_values("FRAME", kind="stable").iloc[-1]
        target_row = parts[target_id].sort_values("FRAME", kind="stable").iloc[0]
        total = link.target_start_frame - link.source_end_frame
        for offset in range(1, total):
            fraction = offset / float(total)
            row = {column: np.nan for column in out.columns}
            row.update(source_row.to_dict())
            row["Parent_Group_ID"] = canonical[source_id]
            row["Original_Parent_Group_ID"] = np.nan
            row["FRAME"] = link.source_end_frame + offset
            for column in ["Center_X", "Center_Y", "Predicted_Z"]:
                row[column] = (1.0 - fraction) * float(source_row[column]) + fraction * float(target_row[column])
            for column in [
                "Selected_Source_Cluster_ID",
                "Source_Cluster_IDs",
                "Candidate_Source_Cluster_IDs",
                "Z_Confidence_Selected",
                "Z_Final_Source",
            ]:
                if column in row:
                    row[column] = np.nan
            row["Parent_Frame_Source_Selection"] = "linear_parent_gap_interpolation"
            row["Parent_Track_Observed"] = False
            row["Parent_Track_Interpolated"] = True
            row["Parent_Track_Stitch_Source"] = f"{source_id}->{target_id}"
            interpolated.append(row)
    # Stable TRACK_ID association can already assign discontinuous source
    # clusters to one physical parent. Those boundaries no longer appear as
    # cross-parent candidates, but their accepted short gaps still require the
    # same interpolation. Gaps inside one source segment are left untouched as
    # possible Z/localization failures.
    same_parent_links: list[StitchLink] = []
    observed_by_parent = out.sort_values(
        ["Parent_Group_ID", "FRAME"], kind="stable"
    ).drop_duplicates(["Parent_Group_ID", "FRAME"], keep="first")
    for parent_id, part in observed_by_parent.groupby("Parent_Group_ID", sort=True):
        ordered = part.sort_values("FRAME", kind="stable").reset_index(drop=True)
        for row_index in range(len(ordered) - 1):
            source_row = ordered.iloc[row_index]
            target_row = ordered.iloc[row_index + 1]
            source_frame = int(source_row["FRAME"])
            target_frame = int(target_row["FRAME"])
            observed_missing_frames = target_frame - source_frame - 1
            if observed_missing_frames <= 0:
                continue

            source_original = int(source_row["Original_Parent_Group_ID"])
            target_original = int(target_row["Original_Parent_Group_ID"])
            if source_original != target_original:
                # Accepted cross-parent boundaries were handled above.
                continue

            source_boundary_end = source_frame
            target_boundary_start = target_frame
            if "Segment_End_Max" in ordered.columns and pd.notna(source_row.get("Segment_End_Max")):
                source_boundary_end = int(round(float(source_row["Segment_End_Max"])))
            if "Segment_Start_Min" in ordered.columns and pd.notna(target_row.get("Segment_Start_Min")):
                target_boundary_start = int(round(float(target_row["Segment_Start_Min"])))
            boundary_gap_frames = target_boundary_start - source_boundary_end - 1
            if boundary_gap_frames < 0 or boundary_gap_frames > int(max_gap_frames):
                continue

            source_pos = source_row[["Center_X", "Center_Y", "Predicted_Z"]].to_numpy(dtype=float)
            target_pos = target_row[["Center_X", "Center_Y", "Predicted_Z"]].to_numpy(dtype=float)
            residual = target_pos - source_pos
            xy_sensor_um = float(np.linalg.norm(residual[:2]) * float(magnification))
            z_um = float(abs(residual[2]))
            if not np.isfinite(xy_sensor_um + z_um):
                continue
            spatial_cost = float(np.hypot(xy_sensor_um / max_xy_sensor_um, z_um / max_z_um))
            if xy_sensor_um > max_xy_sensor_um or z_um > max_z_um or spatial_cost > 1.0:
                continue
            gap_cost = 0.1 * boundary_gap_frames / float(max(int(max_gap_frames), 1))
            link = StitchLink(
                source_parent=source_original,
                target_parent=target_original,
                source_end_frame=source_frame,
                target_start_frame=target_frame,
                observed_missing_frames=observed_missing_frames,
                source_boundary_end_frame=source_boundary_end,
                target_boundary_start_frame=target_boundary_start,
                boundary_gap_frames=boundary_gap_frames,
                xy_endpoint_distance_sensor_um=xy_sensor_um,
                z_endpoint_distance_um=z_um,
                spatial_normalized_cost=spatial_cost,
                normalized_cost=spatial_cost + gap_cost,
                link_method="within_parent_segment_gap",
            )
            same_parent_links.append(link)

            total = target_frame - source_frame
            for offset in range(1, total):
                fraction = offset / float(total)
                row = {column: np.nan for column in out.columns}
                row.update(source_row.to_dict())
                row["Parent_Group_ID"] = int(parent_id)
                row["Original_Parent_Group_ID"] = np.nan
                row["FRAME"] = source_frame + offset
                for column in ["Center_X", "Center_Y", "Predicted_Z"]:
                    row[column] = (1.0 - fraction) * float(source_row[column]) + fraction * float(target_row[column])
                for column in [
                    "Selected_Source_Cluster_ID",
                    "Source_Cluster_IDs",
                    "Candidate_Source_Cluster_IDs",
                    "Z_Confidence_Selected",
                    "Z_Final_Source",
                ]:
                    if column in row:
                        row[column] = np.nan
                row["Parent_Frame_Source_Selection"] = "linear_within_parent_segment_gap_interpolation"
                row["Parent_Track_Observed"] = False
                row["Parent_Track_Interpolated"] = True
                row["Parent_Track_Stitch_Source"] = f"{source_original}->{target_original}:within_parent"
                interpolated.append(row)

    if interpolated:
        out = pd.concat([out, pd.DataFrame.from_records(interpolated)], ignore_index=True, sort=False)
    out = out.drop_duplicates(["Parent_Group_ID", "FRAME"], keep="first")
    out = out.sort_values(["Parent_Group_ID", "FRAME"], kind="stable").reset_index(drop=True)
    accepted_all = accepted + same_parent_links
    audit = pd.DataFrame([link.__dict__ for link in accepted_all], columns=AUDIT_COLUMNS)
    audit.attrs["eligible_candidate_count"] = len(candidates) + len(same_parent_links)
    return out, audit
