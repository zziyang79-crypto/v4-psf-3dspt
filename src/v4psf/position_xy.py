import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings
import os
from pathlib import Path

from .csv_io import find_header_row, read_csv as read_pipeline_csv

warnings.filterwarnings("ignore")


def longest_edge_midpoint(coords):
    if len(coords) < 2:
        return np.array([np.nan, np.nan], dtype=float)
    dist_matrix = squareform(pdist(coords))
    max_idx = np.argmax(dist_matrix)
    p1_idx, p2_idx = np.unravel_index(max_idx, dist_matrix.shape)
    return (coords[p1_idx] + coords[p2_idx]) / 2.0


def _finite_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default

def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}



def _frame_bounds_for_track_segment(group):
    observed_min = int(pd.to_numeric(group['FRAME'], errors='coerce').min())
    observed_max = int(pd.to_numeric(group['FRAME'], errors='coerce').max())
    start = _finite_float(group['segment_start_frame'].iloc[0]) if 'segment_start_frame' in group.columns else np.nan
    end = _finite_float(group['segment_end_frame'].iloc[0]) if 'segment_end_frame' in group.columns else np.nan
    min_frame = int(start) if np.isfinite(start) else observed_min
    max_frame = int(end) if np.isfinite(end) else observed_max
    return min_frame, max_frame


def infer_virtual_fourth_point(coords):
    """Infer the missing fourth corner from three rectangle/square corners."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) != 3 or not np.isfinite(coords).all():
        return np.array([np.nan, np.nan], dtype=float)
    dist_matrix = squareform(pdist(coords))
    max_idx = np.argmax(dist_matrix)
    i, j = np.unravel_index(max_idx, dist_matrix.shape)
    k = [idx for idx in range(3) if idx not in {i, j}][0]
    return coords[i] + coords[j] - coords[k]


def complete_three_track_groups_with_virtual_fourth(df):
    """Complete every three-row group-frame to four angle apertures.

    A four-member cluster keeps the missing member's real TRACK_ID, preserving
    arm identity. A genuinely three-track cluster receives one stable synthetic
    TRACK_ID. The inferred point is used only for raw-image angle-aperture
    integration and is explicitly excluded from Z geometry.
    """
    if df.empty or 'Cluster_ID' not in df.columns or 'FRAME' not in df.columns:
        return df

    out_parts = [df]
    virtual_rows = []
    cluster_track_map = {
        cluster_id: set(pd.to_numeric(group['TRACK_ID'], errors='coerce').dropna().astype(int).tolist())
        for cluster_id, group in df.groupby('Cluster_ID', sort=False)
    }
    for (cluster_id, frame), group in df.groupby(['Cluster_ID', 'FRAME'], sort=False):
        finite = np.isfinite(group[['X', 'Y']].to_numpy(dtype=float)).all(axis=1)
        clean = group.loc[finite].copy()
        if len(clean) != 3:
            continue

        existing_tracks = set(pd.to_numeric(clean['TRACK_ID'], errors='coerce').dropna().astype(int).tolist())
        all_cluster_tracks = cluster_track_map.get(cluster_id, set())
        missing_tracks = sorted(all_cluster_tracks - existing_tracks)
        if len(all_cluster_tracks) == 4 and len(missing_tracks) == 1:
            virtual_track_id = int(missing_tracks[0])
            completion_source = 'four_track_missing_member_parallelogram'
            completion_note = 'missing_member_expected_position_from_three_corners'
        elif len(all_cluster_tracks) <= 3:
            virtual_track_id = -int(abs(int(cluster_id)) * 100000 + 1)
            while virtual_track_id in existing_tracks:
                virtual_track_id -= 1
            completion_source = 'virtual_fourth_from_three_spot_parallelogram'
            completion_note = 'synthetic_fourth_expected_position_from_three_corners'
        else:
            continue

        coords = clean[['X', 'Y']].to_numpy(dtype=float)
        virtual_xy = infer_virtual_fourth_point(coords)
        if not np.isfinite(virtual_xy).all():
            continue

        virtual_row = clean.iloc[0].copy()
        virtual_row['TRACK_ID'] = virtual_track_id
        virtual_row['FRAME'] = frame
        virtual_row['X'] = float(virtual_xy[0])
        virtual_row['Y'] = float(virtual_xy[1])

        if {'Z_Localization_X', 'Z_Localization_Y'}.issubset(clean.columns):
            z_coords = clean[['Z_Localization_X', 'Z_Localization_Y']].to_numpy(dtype=float)
            virtual_z = infer_virtual_fourth_point(z_coords) if np.isfinite(z_coords).all() else virtual_xy
            virtual_row['Z_Localization_X'] = float(virtual_z[0])
            virtual_row['Z_Localization_Y'] = float(virtual_z[1])

        if 'Refined_Intensity' in virtual_row.index:
            virtual_row['Refined_Intensity'] = np.nan
        if 'is_interpolated' in virtual_row.index:
            virtual_row['is_interpolated'] = True
        virtual_row['Subspot_Observed'] = False
        virtual_row['Subspot_Position_Source'] = completion_source
        virtual_row['Subspot_Virtual'] = True
        virtual_row['Subspot_Expected_TRACK_ID'] = virtual_track_id
        virtual_row['Subspot_Completion_Note'] = completion_note
        virtual_row['Z_Use_For_Geometry'] = False
        if 'Geometry_Subspot_Repaired' in virtual_row.index:
            virtual_row['Geometry_Subspot_Repaired'] = False
        if 'Geometry_Subspot_Repair_Residual_Sensor_Um' in virtual_row.index:
            virtual_row['Geometry_Subspot_Repair_Residual_Sensor_Um'] = np.nan
        if 'Geometry_Subspot_Repair_Source' in virtual_row.index:
            virtual_row['Geometry_Subspot_Repair_Source'] = ''
        for col in ['Cross_Parent_Overlap_Detected', 'Cross_Parent_Overlap_Repaired']:
            if col in virtual_row.index:
                virtual_row[col] = False
        for col in [
            'Cross_Parent_Overlap_Neighbor_Parent',
            'Cross_Parent_Overlap_Neighbor_TRACK_ID',
            'Cross_Parent_Overlap_Distance_Sensor_Um',
        ]:
            if col in virtual_row.index:
                virtual_row[col] = np.nan
        if 'Cross_Parent_Overlap_Repair_Source' in virtual_row.index:
            virtual_row['Cross_Parent_Overlap_Repair_Source'] = ''
        for col in ['X', 'Y', 'Z_Localization_X', 'Z_Localization_Y']:
            original_col = f'Geometry_Repair_Original_{col}'
            if original_col in virtual_row.index:
                virtual_row[original_col] = np.nan
            overlap_original_col = f'Cross_Parent_Overlap_Original_{col}'
            if overlap_original_col in virtual_row.index:
                virtual_row[overlap_original_col] = np.nan
        virtual_rows.append(virtual_row)

    if virtual_rows:
        out_parts.append(pd.DataFrame(virtual_rows))
        print(f"   Added {len(virtual_rows)} expected fourth-subspot rows for raw-image angle integration.")
    return pd.concat(out_parts, ignore_index=True).sort_values(['Cluster_ID', 'FRAME', 'TRACK_ID']).reset_index(drop=True)

def complete_member_tracks_to_cluster_frames(df, target_tracks):
    """Fill missing member-track rows across each 3/4-track cluster segment."""
    if df.empty or target_tracks.empty or 'Cluster_ID' not in df.columns:
        return df

    completed_parts = []
    completed_cluster_ids: set[int] = set()
    numeric_cols = [
        col for col in ['X', 'Y', 'Z_Localization_X', 'Z_Localization_Y', 'Refined_Intensity']
        if col in df.columns
    ]

    for cid, members in target_tracks.groupby('Cluster_ID', sort=False):
        track_ids = sorted(pd.to_numeric(members['TRACK_ID'], errors='coerce').dropna().astype(int).unique().tolist())
        if len(track_ids) not in {3, 4}:
            continue

        cluster_df = df[df['Cluster_ID'] == cid].copy()
        if cluster_df.empty:
            continue
        frame_min, frame_max = _frame_bounds_for_track_segment(cluster_df)
        starts = pd.to_numeric(members.get('segment_start_frame', pd.Series(dtype=float)), errors='coerce').dropna()
        ends = pd.to_numeric(members.get('segment_end_frame', pd.Series(dtype=float)), errors='coerce').dropna()
        if len(starts):
            frame_min = min(frame_min, int(starts.min()))
        if len(ends):
            frame_max = max(frame_max, int(ends.max()))
        full_index = np.arange(frame_min, frame_max + 1)

        member_parts = []
        for tid in track_ids:
            sub = cluster_df[pd.to_numeric(cluster_df['TRACK_ID'], errors='coerce') == tid].copy()
            if sub.empty:
                continue
            sub = sub.sort_values('FRAME').drop_duplicates('FRAME', keep='last')
            indexed = sub.set_index('FRAME').reindex(full_index)
            generated_mask = indexed['TRACK_ID'].isna()

            if numeric_cols:
                indexed[numeric_cols] = indexed[numeric_cols].interpolate(method='linear', limit_area='inside')

            fill_cols = [col for col in indexed.columns if col not in numeric_cols]
            if fill_cols:
                indexed[fill_cols] = indexed[fill_cols].ffill().bfill()
            indexed['TRACK_ID'] = tid
            indexed['Cluster_ID'] = cid
            if 'Subspot_Observed' in indexed.columns:
                indexed['Subspot_Observed'] = indexed['Subspot_Observed'].fillna(False).astype(bool)
                indexed.loc[generated_mask, 'Subspot_Observed'] = False
            if 'Subspot_Virtual' in indexed.columns:
                indexed['Subspot_Virtual'] = indexed['Subspot_Virtual'].fillna(False).astype(bool)
            if 'Subspot_Position_Source' in indexed.columns:
                indexed['Subspot_Position_Source'] = indexed['Subspot_Position_Source'].fillna(
                    'member_track_expected_position_interpolated'
                )
                indexed.loc[generated_mask, 'Subspot_Position_Source'] = 'member_track_expected_position_interpolated'
            if 'Subspot_Completion_Note' in indexed.columns:
                indexed['Subspot_Completion_Note'] = indexed['Subspot_Completion_Note'].fillna(
                    'completed_to_cluster_frame_range'
                )
                indexed.loc[generated_mask, 'Subspot_Completion_Note'] = 'completed_to_cluster_frame_range'
            if 'Z_Use_For_Geometry' in indexed.columns:
                indexed['Z_Use_For_Geometry'] = indexed['Z_Use_For_Geometry'].fillna(True).astype(bool)

            restored = indexed.reset_index().rename(columns={'index': 'FRAME'})
            restored = restored[np.isfinite(restored[['X', 'Y']].to_numpy(dtype=float)).all(axis=1)].copy()
            member_parts.append(restored)

        if member_parts:
            completed_parts.append(pd.concat(member_parts, ignore_index=True))
            completed_cluster_ids.add(int(cid))

    untouched = df[~pd.to_numeric(df['Cluster_ID'], errors='coerce').isin(completed_cluster_ids)].copy()
    out_parts = [untouched] + completed_parts
    out = pd.concat(out_parts, ignore_index=True)
    before = len(df)
    out = out.drop_duplicates(['Cluster_ID', 'FRAME', 'TRACK_ID'], keep='last')
    added = len(out) - before
    if added > 0:
        print(f"   Added {added} member-track expected-position rows across cluster frame ranges.")
    return out.sort_values(['Cluster_ID', 'FRAME', 'TRACK_ID']).reset_index(drop=True)


def repair_geometry_outlier_subspots(df):
    """Replace local member-track position outliers by bracketing-frame interpolation."""
    required = {'Cluster_ID', 'TRACK_ID', 'FRAME', 'X', 'Y'}
    if df.empty or not required.issubset(df.columns):
        return df

    threshold_text = os.getenv('POSITION_XY_GEOMETRY_REPAIR_MIN_RESIDUAL_SENSOR_UM')
    if threshold_text in (None, ''):
        legacy_px = os.getenv('POSITION_XY_GEOMETRY_REPAIR_MIN_RESIDUAL_PX')
        threshold_sensor_um = (
            float(legacy_px) * float(os.getenv('POSITION_XY_PHYSICAL_SCALE', '4.5'))
            if legacy_px not in (None, '') else 36.0
        )
    else:
        threshold_sensor_um = float(threshold_text)
    mad_scale = float(os.getenv('POSITION_XY_GEOMETRY_REPAIR_MAD_SCALE', '6.0'))
    max_gap = int(float(os.getenv('POSITION_XY_GEOMETRY_REPAIR_MAX_BRACKET_GAP_FRAMES', '12')))
    enabled = os.getenv('POSITION_XY_GEOMETRY_REPAIR_ENABLED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
    if not enabled:
        return df

    out = df.copy()
    out['Geometry_Subspot_Repaired'] = out.get('Geometry_Subspot_Repaired', False)
    out['Geometry_Subspot_Repair_Residual_Sensor_Um'] = out.get('Geometry_Subspot_Repair_Residual_Sensor_Um', np.nan)
    out['Geometry_Subspot_Repair_Source'] = out.get('Geometry_Subspot_Repair_Source', '')
    for col in ['X', 'Y', 'Z_Localization_X', 'Z_Localization_Y']:
        if col in out.columns and f'Geometry_Repair_Original_{col}' not in out.columns:
            out[f'Geometry_Repair_Original_{col}'] = np.nan

    repaired_count = 0
    for _, group in out.groupby(['Cluster_ID', 'TRACK_ID'], sort=False):
        group = group.sort_values('FRAME')
        if len(group) < 5:
            continue
        if 'Subspot_Virtual' in group.columns and group['Subspot_Virtual'].fillna(False).astype(bool).all():
            continue

        frames = pd.to_numeric(group['FRAME'], errors='coerce').to_numpy(dtype=float)
        xy = group[['X', 'Y']].to_numpy(dtype=float)
        finite = np.isfinite(frames) & np.isfinite(xy).all(axis=1)
        residual = np.full(len(group), np.nan, dtype=float)
        for i in range(1, len(group) - 1):
            if not (finite[i - 1] and finite[i] and finite[i + 1]):
                continue
            if frames[i + 1] == frames[i - 1]:
                continue
            pred = xy[i - 1] + (xy[i + 1] - xy[i - 1]) * ((frames[i] - frames[i - 1]) / (frames[i + 1] - frames[i - 1]))
            residual[i] = float(np.linalg.norm(xy[i] - pred))

        valid_res = residual[np.isfinite(residual)]
        if len(valid_res) < 3:
            continue
        med = float(np.median(valid_res))
        mad = float(np.median(np.abs(valid_res - med)))
        local_threshold = max(threshold_sensor_um, med + mad_scale * max(mad, 1e-6))
        bad = np.isfinite(residual) & (residual >= local_threshold)
        if not np.any(bad):
            continue

        bad_indices = np.where(bad)[0]
        runs = []
        start = int(bad_indices[0])
        prev = start
        for idx in bad_indices[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
            else:
                runs.append((start, prev))
                start = prev = idx
        runs.append((start, prev))

        for start, end in runs:
            left_candidates = [i for i in range(0, start) if finite[i] and not bad[i]]
            right_candidates = [i for i in range(end + 1, len(group)) if finite[i] and not bad[i]]
            if not left_candidates or not right_candidates:
                continue
            left = left_candidates[-1]
            right = right_candidates[0]
            if frames[right] == frames[left]:
                continue
            if frames[start] - frames[left] > max_gap or frames[right] - frames[end] > max_gap:
                continue

            index_values = group.index.to_numpy()
            cols_to_interp = [c for c in ['X', 'Y', 'Z_Localization_X', 'Z_Localization_Y'] if c in out.columns]
            for i in range(start, end + 1):
                if not finite[i]:
                    continue
                ratio = float((frames[i] - frames[left]) / (frames[right] - frames[left]))
                row_index = index_values[i]
                for col in cols_to_interp:
                    left_val = pd.to_numeric(out.at[index_values[left], col], errors='coerce')
                    right_val = pd.to_numeric(out.at[index_values[right], col], errors='coerce')
                    if not (np.isfinite(left_val) and np.isfinite(right_val)):
                        continue
                    out.at[row_index, f'Geometry_Repair_Original_{col}'] = out.at[row_index, col]
                    out.at[row_index, col] = float(left_val + ratio * (right_val - left_val))
                if 'Refined_Intensity' in out.columns:
                    out.at[row_index, 'Refined_Intensity'] = np.nan
                out.at[row_index, 'Geometry_Subspot_Repaired'] = True
                out.at[row_index, 'Geometry_Subspot_Repair_Residual_Sensor_Um'] = float(residual[i])
                out.at[row_index, 'Geometry_Subspot_Repair_Source'] = 'track_second_difference_bracket_interp'
                out.at[row_index, 'Subspot_Position_Source'] = 'geometry_outlier_track_interpolated'
                out.at[row_index, 'Subspot_Completion_Note'] = 'geometry_outlier_replaced_by_bracketing_track_interp'
                repaired_count += 1

    if repaired_count:
        print(f"   Repaired {repaired_count} geometry-outlier subspot positions by bracketing interpolation.")
    return out.sort_values(['Cluster_ID', 'FRAME', 'TRACK_ID']).reset_index(drop=True)


def _auto_find_image_stack():
    configured = os.getenv('POSITION_XY_IMAGE_PATH', '').strip()
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        raise FileNotFoundError(f"POSITION_XY_IMAGE_PATH does not exist: {path}")

    candidates = sorted(
        path for path in Path('.').iterdir()
        if path.is_file() and path.suffix.lower() in {'.tif', '.tiff'}
    )
    preferred = [
        path for path in candidates
        if path.stem.lower() in {'selected_image_stack', 'image_stack'}
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    names = ', '.join(path.name for path in candidates)
    raise RuntimeError(
        f"Multiple TIFF files found ({names}). Set POSITION_XY_IMAGE_PATH or pass image_path explicitly."
    )


def _border_background(roi):
    if roi.size == 0:
        return 0.0
    border = np.zeros_like(roi, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float(np.median(roi[border])) if border.any() else float(np.median(roi))


def _estimate_annulus_background(roi, center_x, center_y, inner_r, outer_r):
    yy, xx = np.mgrid[: roi.shape[0], : roi.shape[1]]
    dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    bg_mask = (dist2 > inner_r**2) & (dist2 <= outer_r**2)
    if bg_mask.any():
        return float(np.mean(roi[bg_mask]))
    return _border_background(roi)


def reintegrate_intensity_from_image(df, image_path=None, physical_scale=None, integration_radius=None):
    """Recompute angle-aperture intensity at every completed/interpolated XY position."""
    if df.empty or 'FRAME' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
        return df

    completion_mask = pd.Series(False, index=df.index, dtype=bool)
    if 'Subspot_Observed' in df.columns:
        completion_mask |= ~df['Subspot_Observed'].fillna(True).astype(bool)
    if 'Subspot_Virtual' in df.columns:
        completion_mask |= df['Subspot_Virtual'].fillna(False).astype(bool)

    image_path = Path(image_path) if image_path else _auto_find_image_stack()
    if image_path is None or not image_path.exists():
        if bool(completion_mask.any()):
            raise FileNotFoundError(
                "Completed/interpolated angle apertures require the source image stack for intensity integration."
            )
        print("   No image stack found for intensity reintegration; keeping existing observed intensities.")
        df['Angle_Intensity_Source'] = df.get('Angle_Intensity_Source', 'refined_existing')
        return df

    try:
        from .spot_detection import LazyTiffStack
    except Exception as exc:
        if bool(completion_mask.any()):
            raise RuntimeError(
                "Could not import LazyTiffStack required for completed-aperture intensity integration"
            ) from exc
        print(f"   Could not import TIFF reader for intensity reintegration ({exc}); keeping existing intensities.")
        df['Angle_Intensity_Source'] = df.get('Angle_Intensity_Source', 'refined_existing')
        return df

    scale = float(physical_scale if physical_scale is not None else os.getenv('POSITION_XY_PHYSICAL_SCALE', '4.5'))
    radius = float(integration_radius if integration_radius is not None else os.getenv('POSITION_XY_INTEGRATION_RADIUS', '10.0'))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"POSITION_XY_PHYSICAL_SCALE must be positive, got {scale!r}")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"POSITION_XY_INTEGRATION_RADIUS must be positive, got {radius!r}")
    bg_inner_r = radius + 2.0
    bg_outer_r = bg_inner_r + 5.0
    crop_r = int(np.ceil(max(radius, bg_outer_r))) + 2

    stack = LazyTiffStack(image_path)
    max_f = int(stack.frame_count)
    max_h, max_w = (int(value) for value in stack.frame_shape)

    intensities = []
    messages = []
    try:
        for _, row in df.iterrows():
            frame = int(_finite_float(row.get('FRAME'), -1))
            # Angle apertures must be sampled at the expected XY positions passed
            # downstream. Z_Localization_X/Y are a separate blurred-geometry channel.
            x_um = _finite_float(row.get('X'), np.nan)
            y_um = _finite_float(row.get('Y'), np.nan)
            if not np.isfinite(x_um + y_um):
                x_um = _finite_float(row.get('Z_Localization_X'), np.nan)
                y_um = _finite_float(row.get('Z_Localization_Y'), np.nan)
            if frame < 0 or frame >= max_f or not np.isfinite(x_um + y_um) or scale == 0:
                intensities.append(np.nan)
                messages.append('out_of_bounds_or_nan')
                continue

            x_px = x_um / scale
            y_px = y_um / scale
            ix = int(round(x_px))
            iy = int(round(y_px))
            r_start = max(0, iy - crop_r)
            r_end = min(max_h, iy + crop_r + 1)
            c_start = max(0, ix - crop_r)
            c_end = min(max_w, ix + crop_r + 1)

            if r_end <= r_start or c_end <= c_start:
                intensities.append(np.nan)
                messages.append('out_of_bounds')
                continue

            roi = stack.get_frame(frame)[r_start:r_end, c_start:c_end]
            cx = x_px - c_start
            cy = y_px - r_start
            bg = _estimate_annulus_background(roi, cx, cy, bg_inner_r, bg_outer_r)
            signal = np.maximum(roi.astype(float) - bg, 0)
            yy, xx = np.mgrid[: roi.shape[0], : roi.shape[1]]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            intensities.append(float(np.sum(signal[dist2 <= radius**2])))
            messages.append('ok')

    finally:
        stack.close()
    out = df.copy()
    out['Refined_Intensity'] = intensities
    out['Angle_Intensity_Source'] = 'image_reintegrated_after_spot_completion'
    out['Angle_Intensity_Message'] = messages
    out['Angle_Aperture_Expected_Position'] = completion_mask.to_numpy(dtype=bool)
    intensity_values = np.asarray(intensities, dtype=float)
    failed_required = completion_mask.to_numpy(dtype=bool) & ~np.isfinite(intensity_values)
    if np.any(failed_required):
        failed_rows = out.loc[failed_required, ['Cluster_ID', 'FRAME', 'TRACK_ID']].to_dict('records')
        raise ValueError(
            "Could not integrate all completed/interpolated angle apertures from the image; "
            f"failed rows={failed_rows[:10]}"
        )
    finite_count = int(np.isfinite(intensity_values).sum())
    expected_count = int(completion_mask.sum())
    print(
        f"   Reintegrated intensity from {image_path.name}: {finite_count}/{len(out)} rows; "
        f"expected/completed apertures verified={expected_count}."
    )
    return out


def build_center_interpolation_groups(target_tracks):
    if 'Trajectory_Parent_Group_ID' in target_tracks.columns:
        parent = pd.to_numeric(target_tracks['Trajectory_Parent_Group_ID'], errors='coerce')
        if parent.notna().any():
            mapping = {}
            for cid, group in target_tracks.assign(_parent=parent).groupby('Cluster_ID'):
                values = group['_parent'].dropna()
                mapping[int(cid)] = int(values.iloc[0]) if len(values) else int(cid)
            return mapping

    meta = []
    for cid, group in target_tracks.groupby('Cluster_ID'):
        tracks = set(pd.to_numeric(group['TRACK_ID'], errors='coerce').dropna().astype(int).tolist())
        start = pd.to_numeric(group.get('segment_start_frame', pd.Series([np.nan])), errors='coerce').min()
        end = pd.to_numeric(group.get('segment_end_frame', pd.Series([np.nan])), errors='coerce').max()
        meta.append({
            'cid': int(cid),
            'tracks': tracks,
            'start': float(start) if pd.notna(start) else np.nan,
            'end': float(end) if pd.notna(end) else np.nan,
        })

    parent = {row['cid']: row['cid'] for row in meta}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(meta):
        for b in meta[i + 1:]:
            if len(a['tracks'] & b['tracks']) < 2:
                continue
            if np.isfinite(a['end']) and np.isfinite(b['start']) and b['start'] - a['end'] <= 2:
                union(a['cid'], b['cid'])
            elif np.isfinite(b['end']) and np.isfinite(a['start']) and a['start'] - b['end'] <= 2:
                union(a['cid'], b['cid'])
    return {cid: find(cid) for cid in parent}


def interpolate_two_point_centers(result_df):
    if result_df.empty or 'Center_Interpolation_Group' not in result_df.columns:
        return result_df

    result_df = result_df.sort_values(['Center_Interpolation_Group', 'FRAME', 'Cluster_ID']).copy()
    result_df['Center_Interpolated'] = False
    result_df['Center_Interpolation_Note'] = ''

    for _, sub in result_df.groupby('Center_Interpolation_Group', sort=False):
        sub = sub.sort_values('FRAME')
        trusted = sub[
            sub['Center_Trusted'].astype(bool)
            & np.isfinite(pd.to_numeric(sub['Center_X'], errors='coerce'))
            & np.isfinite(pd.to_numeric(sub['Center_Y'], errors='coerce'))
        ]
        if trusted.empty:
            continue

        for idx, row in sub.iterrows():
            if bool(row.get('Center_Trusted', False)) or int(row.get('N_Subspots', 0)) != 2:
                continue
            frame = float(row['FRAME'])
            prev = trusted[trusted['FRAME'].astype(float) < frame].tail(1)
            nxt = trusted[trusted['FRAME'].astype(float) > frame].head(1)
            if len(prev) and len(nxt):
                p = prev.iloc[0]
                n = nxt.iloc[0]
                denom = float(n['FRAME']) - float(p['FRAME'])
                alpha = 0.0 if denom == 0 else (frame - float(p['FRAME'])) / denom
                result_df.loc[idx, 'Center_X'] = (1 - alpha) * float(p['Center_X']) + alpha * float(n['Center_X'])
                result_df.loc[idx, 'Center_Y'] = (1 - alpha) * float(p['Center_Y']) + alpha * float(n['Center_Y'])
                result_df.loc[idx, 'Center_Method'] = 'two_point_interpolated_from_trusted_frames'
                result_df.loc[idx, 'Center_Trusted'] = True
                result_df.loc[idx, 'Center_Interpolated'] = True
                result_df.loc[idx, 'Center_Interpolation_Note'] = (
                    f"bracketed_by_frames_{int(p['FRAME'])}_{int(n['FRAME'])}"
                )

    unresolved = (
        (result_df['N_Subspots'].astype(int) == 2)
        & ~np.isfinite(pd.to_numeric(result_df['Center_X'], errors='coerce'))
        & np.isfinite(pd.to_numeric(result_df['Raw_Two_Point_Mid_X'], errors='coerce'))
    )
    if unresolved.any():
        result_df.loc[unresolved, 'Center_X'] = result_df.loc[unresolved, 'Raw_Two_Point_Mid_X']
        result_df.loc[unresolved, 'Center_Y'] = result_df.loc[unresolved, 'Raw_Two_Point_Mid_Y']
        result_df.loc[unresolved, 'Center_Method'] = 'two_point_midpoint_fallback_unbracketed'
        result_df.loc[unresolved, 'Center_Trusted'] = False
        result_df.loc[unresolved, 'Center_Interpolation_Note'] = 'no_bracketing_trusted_frames'

    return result_df


def interpolate_missing_frames(df):
    """
    对每个 TRACK_ID 进行处理：
    1. 补全中间缺失的 FRAME (时间轴重采样)
    2. 对 X, Y 坐标 以及 Refined_Intensity (咱们新算的强度) 进行线性插值
    3. 填充其他固定列 (Cluster_ID, Cluster_Type)
    """
    segment_cols = [
        'Cluster_ID',
        'Cluster_Type',
        'Trajectory_Group_ID',
        'Trajectory_Parent_Group_ID',
        'Trajectory_Method',
        'segment_start_frame',
        'segment_end_frame',
        'segment_index',
        'segment_count',
        'local_segment_group_index',
    ]
    grouping_cols = ['TRACK_ID'] + [c for c in segment_cols if c in df.columns]

    print(f"-> 正在对 {df[grouping_cols].drop_duplicates().shape[0]} 条轨迹/片段进行插值补全...")

    df = df.copy()
    upstream_interpolated = (
        df['is_interpolated'].fillna(False).astype(bool)
        if 'is_interpolated' in df.columns
        else pd.Series(False, index=df.index)
    )
    df['Subspot_Observed'] = ~upstream_interpolated
    df['Subspot_Position_Source'] = np.where(
        upstream_interpolated,
        'trajectory_expected_position_interpolated',
        'observed_refined',
    )
    df['Subspot_Virtual'] = False
    df['Z_Use_For_Geometry'] = True
    df['Subspot_Completion_Note'] = ''
    interpolated_list = []

    for _, group in df.groupby(grouping_cols, dropna=False):
        group = group.sort_values('FRAME').drop_duplicates('FRAME', keep='last')
        min_frame, max_frame = _frame_bounds_for_track_segment(group)
        full_index = np.arange(min_frame, max_frame + 1)
        group_indexed = group.set_index('FRAME').reindex(full_index)
        generated_mask = group_indexed['Subspot_Observed'].isna()

        cols_to_interp = ['X', 'Y']
        for optional_col in ['Z_Localization_X', 'Z_Localization_Y']:
            if optional_col in group_indexed.columns:
                cols_to_interp.append(optional_col)
        if 'Refined_Intensity' in group_indexed.columns:
            cols_to_interp.append('Refined_Intensity')

        group_indexed[cols_to_interp] = group_indexed[cols_to_interp].interpolate(method='linear', limit_area='inside')

        cols_fixed = [
            'TRACK_ID',
            'Cluster_ID',
            'Cluster_Type',
            'Trajectory_Group_ID',
            'Trajectory_Parent_Group_ID',
            'Trajectory_Method',
            'Trajectory_Score',
            'Trajectory_Geometry_Score',
            'Trajectory_N_Tracks',
            'Trajectory_Recoverable_Frames',
            'segment_start_frame',
            'segment_end_frame',
            'segment_index',
            'segment_count',
            'local_segment_group_index',
            'Assigned_Source_ID',
            'Association_Status',
            'Localization_Method',
            'Localization_Success',
            'Localization_Message',
            'XY_Localization_Method',
            'XY_Localization_Success',
            'XY_Localization_Message',
            'Z_Localization_Method',
            'Z_Localization_Success',
            'Z_Localization_Message',
        ]
        cols_to_fill = [c for c in cols_fixed if c in group_indexed.columns]

        if cols_to_fill:
            group_indexed[cols_to_fill] = group_indexed[cols_to_fill].ffill().bfill()

        group_indexed['Subspot_Observed'] = group_indexed['Subspot_Observed'].fillna(False).astype(bool)
        group_indexed['Subspot_Virtual'] = group_indexed['Subspot_Virtual'].fillna(False).astype(bool)
        group_indexed['Z_Use_For_Geometry'] = group_indexed['Z_Use_For_Geometry'].fillna(True).astype(bool)
        group_indexed['Subspot_Position_Source'] = group_indexed['Subspot_Position_Source'].fillna(
            'track_expected_position_interpolated'
        )
        group_indexed.loc[generated_mask, 'Subspot_Position_Source'] = 'track_expected_position_interpolated'
        group_indexed['Subspot_Completion_Note'] = group_indexed['Subspot_Completion_Note'].fillna(
            'completed_to_segment_frame_range'
        )
        group_indexed.loc[generated_mask, 'Subspot_Completion_Note'] = 'completed_to_segment_frame_range'

        group_restored = group_indexed.reset_index().rename(columns={'index': 'FRAME'})
        group_restored = group_restored[np.isfinite(group_restored[['X', 'Y']].to_numpy(dtype=float)).all(axis=1)].copy()
        interpolated_list.append(group_restored)

    if not interpolated_list:
        return df

    df_interpolated = pd.concat(interpolated_list, ignore_index=True)
    added_points = len(df_interpolated) - len(df)
    if added_points > 0:
        print(f"   已补全 {added_points} 个缺失的数据点 (包含坐标和强度)。")
    else:
        print("   数据完整，无需补全。")

    return df_interpolated


def calculate_hybrid_centers(groups_csv_path, spots_csv_path, output_csv_path, image_path=None, params=None):
    print(f"--- 开始计算轨迹中心 (混合模式 + 线性插值) ---")

    run_params = params or {}
    legacy_trackmate_mode = _as_bool(
        run_params.get(
            'LEGACY_TRACKMATE_COMPATIBILITY_MODE',
            os.getenv('POSITION_XY_LEGACY_TRACKMATE_MODE', 'false'),
        )
    )
    if legacy_trackmate_mode:
        print("-> Legacy TrackMate compatibility mode: preserving old centers/intensities and linear interpolation only.")

    # ===============================
    # 2. 读取分类信息
    # ===============================
    try:
        df_groups = read_pipeline_csv(groups_csv_path)
        if 'Cluster_ID' not in df_groups.columns:
            print("错误: 分类文件缺少 'Cluster_ID' 列")
            return
        df_groups = df_groups[df_groups['Cluster_ID'] > 0]
        membership_cols = ['TRACK_ID', 'Cluster_ID', 'Cluster_Type']
        optional_membership_cols = [
            'Trajectory_Group_ID',
            'Trajectory_Parent_Group_ID',
            'Trajectory_Method',
            'Trajectory_Score',
            'Trajectory_Geometry_Score',
            'Trajectory_N_Tracks',
            'Trajectory_Recoverable_Frames',
            'segment_start_frame',
            'segment_end_frame',
            'segment_index',
            'segment_count',
            'local_segment_group_index',
            'Assigned_Source_ID',
            'Association_Status',
        ]
        if not legacy_trackmate_mode:
            membership_cols += [c for c in optional_membership_cols if c in df_groups.columns]
        target_tracks = df_groups[membership_cols].drop_duplicates()
        center_group_map = build_center_interpolation_groups(target_tracks)
        print(f"-> 已加载分类信息，共 {len(target_tracks)} 条轨迹/片段。")
    except Exception as e:
        print(f"分类文件读取失败: {e}")
        return

    # ===============================
    # 3. 读取 Spots Refined 数据 (修改版)
    # ===============================
    try:
        header = find_header_row(spots_csv_path, ['Refined_X', 'POSITION_X', 'x'], encoding='latin1')
        df_spots = read_pipeline_csv(spots_csv_path, header=header, encoding='latin1')
        target_x = 'Refined_X'
        target_y = 'Refined_Y'
        target_intensity = 'Refined_Intensity'

        if target_x not in df_spots.columns or target_y not in df_spots.columns:
            print(f" 错误: 未找到坐标列 ({target_x}, {target_y})。")
            return

        read_cols = ['TRACK_ID', 'FRAME', target_x, target_y]

        has_intensity = False
        if target_intensity in df_spots.columns:
            read_cols.append(target_intensity)
            has_intensity = True
        else:
            print(f" 警告: 未在 spots 文件中找到 '{target_intensity}' 列，将不进行该列的插值。")

        optional_cols = [
            'Z_Localization_X',
            'Z_Localization_Y',
            'Localization_Method',
            'Localization_Success',
            'Localization_Message',
            'XY_Localization_Method',
            'XY_Localization_Success',
            'XY_Localization_Message',
            'Z_Localization_Method',
            'Z_Localization_Success',
            'Z_Localization_Message',
            'is_interpolated',
            'detection_id',
        ]
        if not legacy_trackmate_mode:
            for optional_col in optional_cols:
                if optional_col in df_spots.columns and optional_col not in read_cols:
                    read_cols.append(optional_col)

        spots_data = df_spots[read_cols].copy()
        spots_data.rename(columns={target_x: 'X', target_y: 'Y'}, inplace=True)

    except Exception as e:
        print(f"Spots 文件读取失败: {e}")
        return

    # ===============================
    # 4. 合并数据 & 执行插值
    # ===============================
    merged_df = pd.merge(spots_data, target_tracks, on='TRACK_ID', how='inner')
    if 'segment_start_frame' in merged_df.columns:
        merged_df['segment_start_frame'] = pd.to_numeric(merged_df['segment_start_frame'], errors='coerce')
        merged_df = merged_df[
            merged_df['segment_start_frame'].isna()
            | (pd.to_numeric(merged_df['FRAME'], errors='coerce') >= merged_df['segment_start_frame'])
        ]
    if 'segment_end_frame' in merged_df.columns:
        merged_df['segment_end_frame'] = pd.to_numeric(merged_df['segment_end_frame'], errors='coerce')
        merged_df = merged_df[
            merged_df['segment_end_frame'].isna()
            | (pd.to_numeric(merged_df['FRAME'], errors='coerce') <= merged_df['segment_end_frame'])
        ]

    if len(merged_df) == 0:
        print(" 错误: 合并后数据为空。")
        return

    merged_df = interpolate_missing_frames(merged_df)
    if not legacy_trackmate_mode:
        merged_df = complete_member_tracks_to_cluster_frames(merged_df, target_tracks)
        merged_df = repair_geometry_outlier_subspots(merged_df)

    # Repair real overlapping subspots before virtual completion and intensity
    # integration, so XY, Z geometry, and angle apertures share one position.
    if not legacy_trackmate_mode:
        try:
            from .cross_parent_overlap_repair import detect_and_repair_cross_parent_overlap
            merged_df = detect_and_repair_cross_parent_overlap(merged_df)
        except ImportError as exc:
            print(f"   Cross-parent overlap repair unavailable: {exc}")

    if not legacy_trackmate_mode:
        merged_df = complete_three_track_groups_with_virtual_fourth(merged_df)

    physical_scale = run_params.get('PHYSICAL_SCALE', os.getenv('POSITION_XY_PHYSICAL_SCALE', '4.5'))
    integration_radius = run_params.get('INTEGRATION_RADIUS', os.getenv('POSITION_XY_INTEGRATION_RADIUS', '10.0'))
    reintegrate_enabled = not legacy_trackmate_mode and str(
        run_params.get('REINTEGRATE_INTENSITY', os.getenv('POSITION_XY_REINTEGRATE_INTENSITY', 'true'))
    ).strip().lower() in {'1', 'true', 'yes', 'y'}
    if reintegrate_enabled:
        merged_df = reintegrate_intensity_from_image(
            merged_df,
            image_path=image_path,
            physical_scale=physical_scale,
            integration_radius=integration_radius,
        )

    interpolate_save_path = 'spots_refined_interpolate.csv'
    merged_df.to_csv(interpolate_save_path, index=False)
    print(f"-> 插值后的 Spots 数据已保存至: {interpolate_save_path}")

    print(f"正在处理 {merged_df['Cluster_ID'].nunique()} 个组的数据...")

    # ===============================
    # 5. 混合计算逻辑
    # ===============================
    results = []
    grouped = merged_df.groupby(['Cluster_ID', 'Cluster_Type', 'FRAME'])

    for (gid, gtype, frame), group_data in grouped:
        coords = group_data[['X', 'Y']].values
        n_points = len(coords)

        if np.isnan(coords).any():
            coords = coords[~np.isnan(coords).any(axis=1)]
            n_points = len(coords)

        if n_points == 0:
            continue

        center_x, center_y = np.nan, np.nan
        raw_two_mid_x, raw_two_mid_y = np.nan, np.nan
        center_method = 'unresolved'
        center_trusted = False

        if gtype == '4-Square':
            if n_points >= 4:
                center_x = np.mean(coords[:, 0])
                center_y = np.mean(coords[:, 1])
                center_method = 'four_or_more_mean'
                center_trusted = True
            elif n_points == 3:
                center = longest_edge_midpoint(coords)
                center_x = float(center[0])
                center_y = float(center[1])
                center_method = 'three_point_longest_edge_midpoint'
                center_trusted = True
            elif n_points == 2:
                center = longest_edge_midpoint(coords)
                raw_two_mid_x = float(center[0])
                raw_two_mid_y = float(center[1])
                center_method = 'two_point_pending_interpolation'

        elif gtype == '3-Triangle':
            if n_points >= 3:
                center = longest_edge_midpoint(coords)
                center_x = float(center[0])
                center_y = float(center[1])
                center_method = 'three_point_longest_edge_midpoint'
                center_trusted = True
            elif n_points == 2:
                center = longest_edge_midpoint(coords)
                raw_two_mid_x = float(center[0])
                raw_two_mid_y = float(center[1])
                center_method = 'two_point_pending_interpolation'
        else:
            if n_points >= 4:
                center_x = np.mean(coords[:, 0])
                center_y = np.mean(coords[:, 1])
                center_method = 'four_or_more_mean'
                center_trusted = True
            elif n_points == 3:
                center = longest_edge_midpoint(coords)
                center_x = float(center[0])
                center_y = float(center[1])
                center_method = 'three_point_longest_edge_midpoint'
                center_trusted = True
            elif n_points == 2:
                center = longest_edge_midpoint(coords)
                raw_two_mid_x = float(center[0])
                raw_two_mid_y = float(center[1])
                center_method = 'two_point_pending_interpolation'

        results.append({
            'Cluster_ID': gid,
            'Cluster_Type': gtype,
            'FRAME': frame,
            'Center_X': center_x,
            'Center_Y': center_y,
            'N_Subspots': n_points,
            'Center_Method': center_method,
            'Center_Trusted': center_trusted,
            'Raw_Two_Point_Mid_X': raw_two_mid_x,
            'Raw_Two_Point_Mid_Y': raw_two_mid_y,
            'Center_Interpolation_Group': center_group_map.get(int(gid), int(gid)),
        })

    # ===============================
    # 6. 保存最终结果
    # ===============================
    result_df = pd.DataFrame(results).sort_values(by=['Cluster_ID', 'FRAME'])
    result_df = interpolate_two_point_centers(result_df)
    n_interp = int(result_df.get('Center_Interpolated', pd.Series(dtype=bool)).astype(bool).sum())
    n_fallback = int((result_df.get('Center_Method', pd.Series(dtype=str)) == 'two_point_midpoint_fallback_unbracketed').sum())
    print(f"-> 2 点帧中心处理: 插值 {n_interp} 帧，未包围 fallback {n_fallback} 帧。")
    result_df.to_csv(output_csv_path, index=False)
    print(f" 计算完成，中心坐标结果已保存: {output_csv_path}")

    # ===============================
    # 7. 绘图验证
    # ===============================
    visualize_results(result_df, output_csv_path.replace('.csv', '_plot.png'))


def visualize_results(df, save_path):
    print("正在生成轨迹校验图...")
    plt.figure(figsize=(10, 8), dpi=150)
    unique_ids = df['Cluster_ID'].unique()
    import matplotlib.cm as cm
    colors = [cm.tab20(i % 20) for i in range(len(unique_ids))]

    for i, gid in enumerate(unique_ids):
        subset = df[df['Cluster_ID'] == gid]
        if len(subset) == 0:
            continue
        gtype = subset['Cluster_Type'].iloc[0]
        color = colors[i]
        ls = '-' if gtype == '4-Square' else '--'
        plt.plot(subset['Center_X'], subset['Center_Y'], linestyle=ls, marker='.',
                 markersize=4, color=color, label=f"ID {gid}", alpha=0.8)
        plt.text(subset.iloc[0]['Center_X'], subset.iloc[0]['Center_Y'],
                 str(gid), color=color, fontweight='bold', fontsize=9)

    plt.title('Trajectory Centers (With Interpolation)\nSolid = 4-Square,  Dashed = 3-Triangle')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.xlabel('X (um)')
    plt.ylabel('Y (um)')
    plt.legend(fontsize=7, ncol=2, loc='best')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图像已保存: {save_path}")


if __name__ == "__main__":
    calculate_hybrid_centers(
        "classified_groups_pca.csv",
        "spots_refined.csv",
        "Lateral positions.csv",
    )
