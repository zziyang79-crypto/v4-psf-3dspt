import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import sys

from v4psf.csv_io import read_csv as read_pipeline_csv
from v4psf.parent_trajectory_compat import select_parent_frame_sources, stitch_parent_fragments

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================= 1. 配置区域 =================
file_lateral = 'Lateral positions.csv'
file_axial   = 'Axial positions.csv'

# 物镜放大倍率：将 Center_X/Y（相机面 µm）换算到样品面 µm
magnification = float(os.getenv("POS3D_MAGNIFICATION", "60"))
camera_pixel_um = float(os.getenv("POS3D_CAMERA_PIXEL_UM", "1.0"))

output_static_img  = os.getenv('POS3D_OUTPUT_STATIC_IMG', '3d_trajectory_static.png')
output_dynamic_gif = os.getenv('POS3D_OUTPUT_DYNAMIC_GIF', '3d_trajectory_dynamic.gif')
output_merged_csv  = os.getenv('POS3D_OUTPUT_CSV', '3d_positions.csv')
output_segment_csv = os.getenv('POS3D_SEGMENT_OUTPUT_CSV', '3d_positions_segments.csv')
parent_groups_csv  = os.getenv('POS3D_PARENT_GROUPS_CSV', 'classified_groups_pca.csv')
write_parent_diagnostics = os.getenv('POS3D_WRITE_PARENT_DIAGNOSTICS', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
parent_stitch_enabled = os.getenv('POS3D_PARENT_STITCH_ENABLED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
parent_stitch_max_gap = int(float(os.getenv('POS3D_PARENT_STITCH_MAX_GAP_FRAMES', '12')))
parent_stitch_max_xy_sensor_um = float(os.getenv('POS3D_PARENT_STITCH_MAX_XY_SENSOR_UM', '400.0'))
parent_stitch_max_z_um = float(os.getenv('POS3D_PARENT_STITCH_MAX_Z_UM', '4.0'))


def _as_int_frame(value):
    try:
        return int(round(float(value)))
    except Exception:
        return np.nan


def _join_unique(values):
    out = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str) and ';' in value:
            parts = value.split(';')
        else:
            parts = [value]
        for part in parts:
            try:
                text = str(int(float(part)))
            except Exception:
                text = str(part).strip()
            if text and text not in out:
                out.append(text)
    return ';'.join(out)


def _confidence_rank(value):
    text = str(value).strip().lower()
    ranks = {
        'high': 4,
        'medium_high': 3,
        'medium': 2,
        'low': 1,
    }
    return ranks.get(text, 0)


def _worst_confidence(values):
    labels = [str(v).strip().lower() for v in values if not pd.isna(v)]
    if 'low' in labels:
        return 'low'
    if 'medium' in labels:
        return 'medium'
    if 'medium_high' in labels:
        return 'medium_high'
    if 'high' in labels:
        return 'high'
    return ''


def _format_cluster_id(value):
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def _select_parent_frame_source(group):
    group = group.copy()
    candidate_ids = _join_unique(group['Cluster_ID'])
    candidate_count = int(group['Cluster_ID'].nunique())

    ranking = pd.DataFrame(index=group.index)
    if 'Z_Confidence' in group.columns:
        ranking['confidence_rank'] = group['Z_Confidence'].map(_confidence_rank).fillna(0).astype(float)
    else:
        ranking['confidence_rank'] = 0.0
    if 'Z_Angle_Model_Domain_OK' in group.columns:
        ranking['domain_rank'] = group['Z_Angle_Model_Domain_OK'].fillna(False).astype(bool).astype(int)
    else:
        ranking['domain_rank'] = 0
    if 'Z_Angle_Severe_Out_Of_Domain' in group.columns:
        ranking['not_severe_rank'] = (~group['Z_Angle_Severe_Out_Of_Domain'].fillna(False).astype(bool)).astype(int)
    else:
        ranking['not_severe_rank'] = 1
    if 'Z_Final_Source' in group.columns:
        ranking['not_extrapolated_rank'] = (~group['Z_Final_Source'].astype(str).str.contains('extrapolated', na=False)).astype(int)
    else:
        ranking['not_extrapolated_rank'] = 1
    if 'Trajectory_Score_Median' in group.columns:
        ranking['trajectory_score'] = pd.to_numeric(group['Trajectory_Score_Median'], errors='coerce').fillna(np.inf)
    else:
        ranking['trajectory_score'] = np.inf
    if 'Trajectory_Geometry_Score_Median' in group.columns:
        ranking['geometry_score'] = pd.to_numeric(group['Trajectory_Geometry_Score_Median'], errors='coerce').fillna(np.inf)
    else:
        ranking['geometry_score'] = np.inf

    best_idx = ranking.sort_values(
        ['confidence_rank', 'domain_rank', 'not_severe_rank', 'not_extrapolated_rank', 'trajectory_score', 'geometry_score'],
        ascending=[False, False, False, False, True, True],
    ).index[0]
    best = group.loc[best_idx]

    out = {
        'Parent_Group_ID': best['Parent_Group_ID'],
        'FRAME': best['FRAME'],
        'Center_X': best['Center_X'],
        'Center_Y': best['Center_Y'],
        'Predicted_Z': best['Predicted_Z'],
        'Source_Cluster_IDs': _format_cluster_id(best['Cluster_ID']),
        'Source_Cluster_Count': 1,
        'Selected_Source_Cluster_ID': _format_cluster_id(best['Cluster_ID']),
        'Candidate_Source_Cluster_IDs': candidate_ids,
        'Candidate_Source_Cluster_Count': candidate_count,
        'Parent_Frame_Source_Selection': 'single_source' if candidate_count <= 1 else 'best_confidence_domain',
    }
    if 'Z_Confidence' in group.columns:
        out['Z_Confidence_Worst'] = _worst_confidence(group['Z_Confidence'])
        out['Z_Confidence_Selected'] = best.get('Z_Confidence', '')
    if 'Z_Angle_Model_Domain_OK' in group.columns:
        out['Z_Angle_Model_Domain_OK_All'] = bool(group['Z_Angle_Model_Domain_OK'].fillna(False).all())
        out['Z_Angle_Model_Domain_OK_Selected'] = bool(best.get('Z_Angle_Model_Domain_OK', False))
    if 'Z_Final_Source' in group.columns:
        out['Z_Final_Source'] = best.get('Z_Final_Source', '')
        out['Candidate_Z_Final_Source'] = _join_unique(group['Z_Final_Source'].dropna())
    if 'Z_Angle_Out_Of_Domain_Level' in group.columns:
        out['Z_Angle_Out_Of_Domain_Level'] = best.get('Z_Angle_Out_Of_Domain_Level', '')
        out['Candidate_Z_Angle_Out_Of_Domain_Level'] = _join_unique(group['Z_Angle_Out_Of_Domain_Level'].dropna())
    if 'Z_Angle_Severe_Out_Of_Domain' in group.columns:
        out['Z_Angle_Severe_Out_Of_Domain_Any'] = bool(group['Z_Angle_Severe_Out_Of_Domain'].fillna(False).any())
        out['Z_Angle_Severe_Out_Of_Domain_Selected'] = bool(best.get('Z_Angle_Severe_Out_Of_Domain', False))
    for column, use_min in [('Segment_Start_Min', True), ('Segment_End_Max', False)]:
        if column in group.columns:
            values = pd.to_numeric(group[column], errors='coerce').dropna()
            if not values.empty:
                out[column] = float(values.min() if use_min else values.max())
    return pd.Series(out)


def _load_first_plane(path):
    try:
        import tifffile
        arr = tifffile.imread(path, key=0)
    except Exception:
        try:
            from skimage import io
            arr = io.imread(path)
        except Exception as exc:
            print(f"Parent XY overlay skipped: cannot read image {path}: {exc}")
            return None
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _write_parent_xy_overlay(parent_df, parent_ids, image_path, output_path):
    if not image_path or not os.path.exists(image_path):
        return
    img = _load_first_plane(image_path)
    if img is None:
        return

    lo, hi = np.percentile(img, [1, 99.8])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(parent_ids), 1)))

    for color, pid in zip(colors, parent_ids):
        part = parent_df[parent_df['Parent_Group_ID'].astype(int) == int(pid)].sort_values('FRAME')
        if part.empty:
            continue
        x = part['Center_X'].to_numpy(dtype=float) * magnification / camera_pixel_um
        y = part['Center_Y'].to_numpy(dtype=float) * magnification / camera_pixel_um
        ax.plot(x, y, color=color, lw=1.8, alpha=0.95, label=f'P{pid}')
        ax.scatter([x[0]], [y[0]], s=22, color=color, edgecolors='white', linewidths=0.6)
        ax.scatter([x[-1]], [y[-1]], s=30, color=color, marker='x', linewidths=1.3)
        ax.text(x[0] + 4, y[0] + 4, f'P{pid}', color=color, fontsize=8, weight='bold')

    ax.set_xlim(0, img.shape[1] - 1)
    ax.set_ylim(img.shape[0] - 1, 0)
    ax.set_title(f'XY projection on first frame ({len(parent_ids)} parent trajectories)')
    ax.set_xlabel('X (camera px)')
    ax.set_ylabel('Y (camera px)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Parent XY overlay saved: {output_path}")


def write_parent_trajectory_outputs(merged_df, df_ax):
    """Emit parent-level 3D trajectories for auto-detect segment stitching."""
    if not os.path.exists(parent_groups_csv):
        return None, None

    try:
        groups = read_pipeline_csv(parent_groups_csv)
    except Exception as exc:
        print(f"Parent trajectory output skipped: cannot read {parent_groups_csv}: {exc}")
        return None, None

    groups.columns = [c.strip() for c in groups.columns]
    if 'Cluster_ID' not in groups.columns or 'Trajectory_Parent_Group_ID' not in groups.columns:
        print("Parent trajectory output skipped: no Trajectory_Parent_Group_ID column.")
        return None, None

    work_groups = groups.copy()
    work_groups['Cluster_ID'] = pd.to_numeric(work_groups['Cluster_ID'], errors='coerce').astype('Int64')
    work_groups['Parent_Group_ID'] = pd.to_numeric(
        work_groups['Trajectory_Parent_Group_ID'], errors='coerce'
    ).astype('Int64')
    work_groups = work_groups.dropna(subset=['Cluster_ID', 'Parent_Group_ID'])
    if work_groups.empty:
        return None, None

    score_col = 'Trajectory_Score' if 'Trajectory_Score' in work_groups.columns else None
    geom_col = 'Trajectory_Geometry_Score' if 'Trajectory_Geometry_Score' in work_groups.columns else None

    agg_spec = {'Parent_Group_ID': ('Parent_Group_ID', 'first')}
    if score_col:
        agg_spec['Trajectory_Score_Median'] = (score_col, 'median')
    if geom_col:
        agg_spec['Trajectory_Geometry_Score_Median'] = (geom_col, 'median')
    if 'TRACK_ID' in work_groups.columns:
        agg_spec['Member_Track_Count'] = ('TRACK_ID', 'nunique')
        agg_spec['Member_Track_IDs'] = ('TRACK_ID', _join_unique)
    if 'segment_start_frame' in work_groups.columns:
        agg_spec['Segment_Start_Min'] = ('segment_start_frame', 'min')
    if 'segment_end_frame' in work_groups.columns:
        agg_spec['Segment_End_Max'] = ('segment_end_frame', 'max')

    cluster_meta = work_groups.groupby('Cluster_ID', as_index=False).agg(**agg_spec)
    parent_meta = work_groups.groupby('Parent_Group_ID', as_index=False).agg(
        Source_Clusters=('Cluster_ID', _join_unique),
        Source_Cluster_Count=('Cluster_ID', 'nunique'),
        Member_Track_IDs=('TRACK_ID', _join_unique) if 'TRACK_ID' in work_groups.columns else ('Cluster_ID', _join_unique),
        Member_Track_Count=('TRACK_ID', 'nunique') if 'TRACK_ID' in work_groups.columns else ('Cluster_ID', 'nunique'),
        Trajectory_Score_Median=(score_col, 'median') if score_col else ('Cluster_ID', 'size'),
        Trajectory_Geometry_Score_Median=(geom_col, 'median') if geom_col else ('Cluster_ID', 'size'),
    )

    base = merged_df.copy()
    base['Cluster_ID'] = pd.to_numeric(base['Cluster_ID'], errors='coerce').astype('Int64')
    base = base.merge(cluster_meta, on='Cluster_ID', how='left')
    base = base.dropna(subset=['Parent_Group_ID'])
    if base.empty:
        return None, None

    qc_cols = [
        'Z_Confidence',
        'Z_Angle_Model_Domain_OK',
        'Z_Final_Source',
        'Z_Angle_Out_Of_Domain_Level',
        'Z_Angle_Severe_Out_Of_Domain',
    ]
    available_qc = [c for c in qc_cols if c in df_ax.columns]
    if available_qc:
        qc = df_ax[['Cluster_ID', 'FRAME'] + available_qc].copy()
        qc['Cluster_ID'] = pd.to_numeric(qc['Cluster_ID'], errors='coerce').astype('Int64')
        base = base.merge(qc, on=['Cluster_ID', 'FRAME'], how='left')

    parent_observed = select_parent_frame_sources(base, _select_parent_frame_source)
    if parent_observed.empty:
        return None, None
    parent_observed['Parent_Group_ID'] = parent_observed['Parent_Group_ID'].astype(int)
    if write_parent_diagnostics:
        parent_observed.to_csv('3d_parent_positions_observed.csv', index=False)

    parent_observed, stitch_audit = stitch_parent_fragments(
        parent_observed,
        magnification=magnification,
        max_gap_frames=parent_stitch_max_gap,
        max_xy_sensor_um=parent_stitch_max_xy_sensor_um,
        max_z_um=parent_stitch_max_z_um,
        enabled=parent_stitch_enabled,
    )
    eligible_candidate_count = int(stitch_audit.attrs.get('eligible_candidate_count', len(stitch_audit)))
    stitch_audit.to_csv('3d_parent_stitch_links.csv', index=False)
    print(
        "Parent stitching: "
        f"enabled={parent_stitch_enabled}, eligible_candidates={eligible_candidate_count}, accepted_links={len(stitch_audit)}, "
        f"interpolated_rows={int(parent_observed['Parent_Track_Interpolated'].sum())}, "
        f"XY_gate={parent_stitch_max_xy_sensor_um:g} sensor um, Z_gate={parent_stitch_max_z_um:g} um"
    )

    frame_min = _as_int_frame(parent_observed['FRAME'].min())
    frame_max = _as_int_frame(parent_observed['FRAME'].max())
    full_frame_count = frame_max - frame_min + 1 if pd.notna(frame_min) and pd.notna(frame_max) else np.nan

    summary_rows = []
    for pid, part in parent_observed.groupby('Parent_Group_ID'):
        original_ids = pd.to_numeric(part.get('Original_Parent_Group_ID'), errors='coerce').dropna().astype(int).unique()
        meta = parent_meta[parent_meta['Parent_Group_ID'].astype(int).isin(original_ids)]
        meta_row = {
            'Source_Clusters': _join_unique(meta.get('Source_Clusters', pd.Series(dtype=object))),
            'Source_Cluster_Count': pd.to_numeric(meta.get('Source_Cluster_Count'), errors='coerce').sum(),
            'Trajectory_Score_Median': pd.to_numeric(meta.get('Trajectory_Score_Median'), errors='coerce').median(),
            'Trajectory_Geometry_Score_Median': pd.to_numeric(meta.get('Trajectory_Geometry_Score_Median'), errors='coerce').median(),
            'Member_Track_Count': pd.to_numeric(meta.get('Member_Track_Count'), errors='coerce').sum(),
            'Member_Track_IDs': _join_unique(meta.get('Member_Track_IDs', pd.Series(dtype=object))),
        } if not meta.empty else {}
        observed_mask = part['Parent_Track_Observed'].fillna(True).astype(bool)
        z_low = int((part.loc[observed_mask].get('Z_Confidence_Worst', pd.Series([], dtype=object)) == 'low').sum()) if 'Z_Confidence_Worst' in part.columns else 0
        domain_bad = int((part.loc[observed_mask].get('Z_Angle_Model_Domain_OK_All', pd.Series([], dtype=bool)) == False).sum()) if 'Z_Angle_Model_Domain_OK_All' in part.columns else 0
        observed_frames = int(part.loc[observed_mask, 'FRAME'].nunique())
        interpolated_frames = int(part.loc[~observed_mask, 'FRAME'].nunique())
        summary_rows.append({
            'Parent_Group_ID': int(pid),
            'frame_min': int(part['FRAME'].min()),
            'frame_max': int(part['FRAME'].max()),
            'observed_frames': observed_frames,
            'interpolated_frames': interpolated_frames,
            'connected_original_parent_ids': _join_unique(original_ids),
            'full_frame_min': frame_min,
            'full_frame_max': frame_max,
            'full_frame_count': full_frame_count,
            'observed_fraction': observed_frames / full_frame_count if full_frame_count else np.nan,
            'missing_frames_to_full_range': int(max(full_frame_count - observed_frames, 0)) if full_frame_count else 0,
            'z_min': float(part['Predicted_Z'].min()),
            'z_max': float(part['Predicted_Z'].max()),
            'z_median': float(part['Predicted_Z'].median()),
            'z_low_confidence_frames': z_low,
            'z_domain_bad_frames': domain_bad,
            'source_clusters': meta_row.get('Source_Clusters', ''),
            'source_cluster_count': meta_row.get('Source_Cluster_Count', np.nan),
            'score_median': meta_row.get('Trajectory_Score_Median', np.nan),
            'geometry_score_median': meta_row.get('Trajectory_Geometry_Score_Median', np.nan),
            'member_track_count': meta_row.get('Member_Track_Count', np.nan),
            'member_track_ids': meta_row.get('Member_Track_IDs', ''),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ['observed_frames', 'score_median', 'geometry_score_median'],
        ascending=[False, True, True],
        na_position='last',
    )
    summary.to_csv('3d_parent_trajectory_summary.csv', index=False)

    parent_output = (
        parent_observed
        .drop_duplicates(['Parent_Group_ID', 'FRAME'], keep='first')
        .sort_values(['Parent_Group_ID', 'FRAME'])
        .copy()
    )
    if parent_output.empty:
        return parent_observed, summary

    parent_output['Cluster_ID'] = parent_output['Parent_Group_ID'].astype(int)
    preferred_cols = [
        'Cluster_ID', 'Parent_Group_ID', 'FRAME',
        'Center_X', 'Center_Y', 'Predicted_Z',
    ]
    trailing_cols = [c for c in parent_output.columns if c not in preferred_cols]
    parent_output = parent_output[preferred_cols + trailing_cols]
    parent_output.to_csv(output_merged_csv, index=False)

    parent_ids = sorted(parent_output['Parent_Group_ID'].astype(int).unique().tolist())
    print(
        f"Parent-level 3D output: {len(parent_ids)} trajectories, "
        f"{len(parent_output)} rows (observed + accepted-gap interpolation) -> {output_merged_csv}"
    )

    image_for_overlay = os.getenv('POS3D_XY_OVERLAY_IMAGE', 'selected_image_stack.tif')
    _write_parent_xy_overlay(
        parent_output,
        parent_ids,
        image_for_overlay,
        'xy_projection_parent_trajectories.png',
    )

    return parent_output, summary

# ================= 2. 数据读取与合并 =================
print("--- 步骤 1: 读取并合并数据 ---")

try:
    df_lat = read_pipeline_csv(file_lateral)
    df_ax  = read_pipeline_csv(file_axial)
    df_lat.columns = [c.strip() for c in df_lat.columns]
    df_ax.columns  = [c.strip() for c in df_ax.columns]

    print(f"Lateral 数据行数: {len(df_lat)}")
    print(f"Axial   数据行数: {len(df_ax)}")

    req_cols_lat = ['Cluster_ID', 'FRAME', 'Center_X', 'Center_Y']
    req_cols_ax  = ['Cluster_ID', 'FRAME', 'Predicted_Z']
    if not all(col in df_lat.columns for col in req_cols_lat):
        raise ValueError(f"Lateral 文件列名不匹配，需要: {req_cols_lat}，实际: {df_lat.columns.tolist()}")
    if not all(col in df_ax.columns for col in req_cols_ax):
        raise ValueError(f"Axial 文件列名不匹配，需要: {req_cols_ax}，实际: {df_ax.columns.tolist()}")

    merged_df = pd.merge(df_lat[req_cols_lat], df_ax[req_cols_ax],
                         on=['Cluster_ID', 'FRAME'], how='inner')
    merged_df = merged_df.sort_values(by=['Cluster_ID', 'FRAME'])

    merged_df['Center_X'] = merged_df['Center_X'] / magnification
    merged_df['Center_Y'] = merged_df['Center_Y'] / magnification
    print(f"XY 坐标已除以放大倍率 {magnification}x，换算至样品面 µm")

    merged_df.to_csv(output_segment_csv, index=False)
    print(f"Segment-level 3D rows: {len(merged_df)} -> {output_segment_csv}")
    parent_df, parent_summary = write_parent_trajectory_outputs(merged_df, df_ax)
    if parent_df is not None and not parent_df.empty:
        plot_df = parent_df.copy()
        plot_id_label = 'Parent'
    else:
        merged_df.to_csv(output_merged_csv, index=False)
        plot_df = merged_df.copy()
        plot_id_label = 'Segment'
        print(f"Parent metadata not available; fallback 3D output -> {output_merged_csv}")

except Exception as e:
    print(f"读取或合并数据时出错: {e}")
    exit()

plot_df = plot_df[
    np.isfinite(plot_df[['Center_X', 'Center_Y', 'Predicted_Z']].to_numpy(dtype=float)).all(axis=1)
].copy()
if plot_df.empty:
    print("No finite 3D points available for plotting.")
    print("\n--- Step 3: GIF generation disabled ---")
    if os.path.exists(output_dynamic_gif):
        try:
            os.remove(output_dynamic_gif)
            print(f"Removed stale GIF: {output_dynamic_gif}")
        except Exception as exc:
            print(f"Could not remove stale GIF {output_dynamic_gif}: {exc}")
    print("\nProgram finished.")
    sys.exit(0)

cluster_ids  = plot_df['Cluster_ID'].unique()
tab_colors   = [plt.cm.tab10(i % 10) for i in range(len(cluster_ids))]
CMAP_TRAIL   = 'plasma'   # 时序色带：早=深紫 → 晚=亮黄

# 全局轴范围（静态图共用）
x_all, y_all, z_all = plot_df['Center_X'], plot_df['Center_Y'], plot_df['Predicted_Z']
pad_xy = max((x_all.max() - x_all.min()) * 0.08, 0.05)
pad_z  = max((z_all.max() - z_all.min()) * 0.08, 0.05)
x_lim  = (x_all.min() - pad_xy, x_all.max() + pad_xy)
y_lim  = (y_all.min() - pad_xy, y_all.max() + pad_xy)
z_lim  = (z_all.min() - pad_z,  z_all.max() + pad_z)


def _draw_colormap_track(ax, xs, ys, zs, lw=1.5, alpha=0.9):
    """将一条轨迹按时间（plasma）着色绘制为逐段折线。"""
    n = len(xs)
    if n < 2:
        return
    t_vals = np.linspace(0, 1, n)
    for k in range(n - 1):
        ax.plot(xs[k:k+2], ys[k:k+2], zs[k:k+2],
                color=plt.cm.plasma(t_vals[k]), lw=lw, alpha=alpha,
                solid_capstyle='round')


# ================= 3. 静态图 =================
print("\n--- 步骤 2: 生成静态 3D 轨迹图 ---")

fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(111, projection='3d')

for i, cid in enumerate(cluster_ids):
    track = plot_df[plot_df['Cluster_ID'] == cid]
    xs, ys, zs = track['Center_X'].values, track['Center_Y'].values, track['Predicted_Z'].values

    _draw_colormap_track(ax, xs, ys, zs, lw=1.8)

    # 起点（圆点）和终点（叉号），用 tab 色标识所属粒子
    ax.scatter(xs[0],  ys[0],  zs[0],  color=tab_colors[i], marker='o', s=40, zorder=5)
    ax.scatter(xs[-1], ys[-1], zs[-1], color=tab_colors[i], marker='x', s=60, zorder=5)
    ax.text(xs[0], ys[0], zs[0], f'  {plot_id_label} {cid}', fontsize=8, color=tab_colors[i])

ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.set_zlim(*z_lim)
ax.set_xlabel('X (µm, sample plane)')
ax.set_ylabel('Y (µm, sample plane)')
ax.set_zlabel('Z (µm, sample plane)')
ax.set_title(f'Static 3D Trajectories ({len(cluster_ids)} {plot_id_label.lower()}-level particles)\n'
             f'Color: early=purple → late=yellow  |  Start: ●  End: ×')

# 颜色条：时间进程
sm = plt.cm.ScalarMappable(cmap=CMAP_TRAIL, norm=plt.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.1, label='Time (normalized)')

plt.savefig(output_static_img, dpi=150, bbox_inches='tight')
plt.close()
print(f"静态图已保存: {output_static_img}")

# ================= 4. GIF disabled =================
print("\n--- Step 3: GIF generation disabled ---")
if os.path.exists(output_dynamic_gif):
    try:
        os.remove(output_dynamic_gif)
        print(f"Removed stale GIF: {output_dynamic_gif}")
    except Exception as exc:
        print(f"Could not remove stale GIF {output_dynamic_gif}: {exc}")
print("\nProgram finished.")
sys.exit(0)

all_frames = sorted(merged_df['FRAME'].unique())

# 预计算每个 cluster 的帧→时间归一化映射
_track_cache: dict = {}
for cid in cluster_ids:
    t = merged_df[merged_df['Cluster_ID'] == cid].sort_values('FRAME')
    n = len(t)
    frame_arr = t['FRAME'].values
    _track_cache[cid] = {
        'frames': frame_arr,
        'xs': t['Center_X'].values,
        'ys': t['Center_Y'].values,
        'zs': t['Predicted_Z'].values,
        't_norm': np.linspace(0, 1, n) if n > 1 else np.array([0.0]),
    }

fig_g = plt.figure(figsize=(11, 9))
ax_g  = fig_g.add_subplot(111, projection='3d')


def update_anim(frame_idx):
    ax_g.cla()
    ax_g.set_xlim(*x_lim)
    ax_g.set_ylim(*y_lim)
    ax_g.set_zlim(*z_lim)
    ax_g.set_xlabel('X (µm)')
    ax_g.set_ylabel('Y (µm)')
    ax_g.set_zlabel('Z (µm)')

    cur_frame = all_frames[frame_idx]
    ax_g.set_title(f'3D Trajectory — Frame {cur_frame}', fontsize=11)

    info_lines = []

    for i, cid in enumerate(cluster_ids):
        c = _track_cache[cid]
        mask = c['frames'] <= cur_frame
        if not mask.any():
            continue

        xs_h = c['xs'][mask]
        ys_h = c['ys'][mask]
        zs_h = c['zs'][mask]
        t_h  = c['t_norm'][mask]

        # 彩色散点轨迹（plasma，时间色带）
        ax_g.scatter(xs_h, ys_h, zs_h,
                     c=t_h, cmap=CMAP_TRAIL, vmin=0, vmax=1,
                     s=10, alpha=0.85, zorder=4, depthshade=False)

        # 当前位置高亮（白心 + tab 色边框）
        ax_g.scatter([xs_h[-1]], [ys_h[-1]], [zs_h[-1]],
                     color='white', edgecolors=tab_colors[i],
                     s=80, linewidth=2, zorder=6, depthshade=False)

        # 记录当前坐标用于标注
        info_lines.append(
            f'ID {cid}: X={xs_h[-1]:.3f}  Y={ys_h[-1]:.3f}  Z={zs_h[-1]:.3f} µm'
        )

    if info_lines:
        ax_g.text2D(0.02, 0.97, '\n'.join(info_lines),
                    transform=ax_g.transAxes,
                    fontsize=7.5, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='white', alpha=0.78, edgecolor='gray'))

    return []


ani = animation.FuncAnimation(
    fig_g, update_anim, frames=len(all_frames), interval=100, blit=False
)

try:
    ani.save(output_dynamic_gif, writer=animation.PillowWriter(fps=10))
    print(f"动态 GIF 已保存: {output_dynamic_gif}")
except Exception as e:
    print(f"保存 GIF 时出错（请确保安装了 pillow）: {e}")

plt.close()
print("\n程序运行结束。")
