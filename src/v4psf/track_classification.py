import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
import argparse

from .csv_io import read_csv as read_pipeline_csv

try:
    from . import trajectory_association_adapter as trajectory_adapter
except Exception:
    trajectory_adapter = None

# --- 基础配置 ---
warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
plt.style.use('ggplot')

DEFAULT_PARAMS = {
    "MIN_DIAGONAL": 0.0,
    "MAX_DIAGONAL": 150.0,
    "SQUARE_TOLERANCE": 0.2,
    "NEIGHBORS_K": 20,
    "LIMIT_SQUARE": 4.0,
    "LIMIT_TRIANGLE": 0.3,
}


# ==========================================
# 1. 自动文件搜索函数
# ==========================================
def auto_find_track_file():
    """自动寻找包含 'track' 的 CSV 文件"""
    current_dir = os.getcwd()
    files = os.listdir(current_dir)

    target_csv = None
    for f in files:
        # 匹配包含 track 的 CSV 文件 (不区分大小写)
        if "track" in f.lower() and f.lower().endswith(".csv"):
            target_csv = f
            break  # 找到第一个就停止

    if not target_csv:
        print(f"❌ 错误：未能在当前目录找到包含 'track' 的 CSV 文件！")
        return None, None, None

    # 设置固定的输出文件名（如你所要求的）
    output_csv = "classified_groups_pca.csv"
    output_img = "classified_groups_pca.png"

    print(f"✅ 已自动匹配输入文件: {target_csv}")
    print(f"   输出结果将存为: {output_csv}")

    return target_csv, output_csv, output_img


def auto_find_spots_file():
    for name in ["selected_spots_input.csv", "spots.csv", "spots_refined.csv"]:
        if os.path.exists(name):
            return name
    for f in os.listdir(os.getcwd()):
        lower = f.lower()
        if "spot" in lower and lower.endswith(".csv"):
            return f
    return None


def classify_with_trajectory_association(track_csv, spots_csv, output_csv_path, output_img_path, params=None):
    if trajectory_adapter is None:
        return False
    if not spots_csv or not os.path.exists(spots_csv):
        return False
    try:
        print("--- trajectory-level subsignal association ---")
        print(f"Tracks CSV: {track_csv}")
        print(f"Spots CSV:  {spots_csv}")
        if params:
            if "CAMERA_PIXEL_UM" in params:
                os.environ["TRAJ_ASSOC_CAMERA_PIXEL_UM"] = str(params["CAMERA_PIXEL_UM"])
                os.environ["AUTODETECT_PHYSICAL_SCALE"] = str(params["CAMERA_PIXEL_UM"])
            if "MIN_DIAGONAL_SENSOR_UM" in params:
                os.environ["TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM"] = str(params["MIN_DIAGONAL_SENSOR_UM"])
            if "MAX_DIAGONAL_SENSOR_UM" in params:
                os.environ["TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM"] = str(params["MAX_DIAGONAL_SENSOR_UM"])
            if "MAX_DIAGONAL" in params:
                os.environ["TRAJ_ASSOC_MAX_DIAGONAL"] = str(params["MAX_DIAGONAL"])
            if "MIN_DIAGONAL" in params:
                os.environ["TRAJ_ASSOC_MIN_DIAGONAL"] = str(params["MIN_DIAGONAL"])
            if "PARENT_TURNOVER_MAX_DISTANCE_PX" in params:
                os.environ["TRAJ_ASSOC_PARENT_TRACK_TURNOVER_MAX_DISTANCE_PX"] = str(params["PARENT_TURNOVER_MAX_DISTANCE_PX"])
        groups_path, repaired_path, membership, repaired = trajectory_adapter.process_trackmate_inputs(
            track_csv,
            spots_csv,
            groups_csv=output_csv_path,
            repaired_spots_csv="trajectory_association_spots.csv",
            plot_path=output_img_path,
        )
        print(f"Saved trajectory groups: {groups_path} ({membership['Cluster_ID'].nunique()} groups)")
        print(f"Saved repaired/interpolated spots for refine: {repaired_path} ({len(repaired)} rows)")
        return True
    except Exception as exc:
        print(f"WARNING: trajectory-level association failed, falling back to legacy classifier: {exc}")
        return False


# ===============================
# 2. 核心分类与可视化逻辑
# ===============================
def classify_and_visualize(csv_path, output_csv_path, output_img_path, params=None):
    if csv_path is None:
        return

    print(f"--- 开始处理: {csv_path} ---")

    run_params = DEFAULT_PARAMS.copy()
    if params:
        run_params.update(params)

    # --- 核心参数设置 (PCA版) ---
    camera_pixel_um = float(
        run_params.get(
            "CAMERA_PIXEL_UM",
            os.getenv("TRAJ_ASSOC_CAMERA_PIXEL_UM", os.getenv("AUTODETECT_PHYSICAL_SCALE", "4.5")),
        )
    )
    if not np.isfinite(camera_pixel_um) or camera_pixel_um <= 0:
        raise ValueError(f"CAMERA_PIXEL_UM must be positive, got {camera_pixel_um!r}")
    explicit_min_sensor = (
        params is not None and "MIN_DIAGONAL_SENSOR_UM" in params
    ) or "TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM" in os.environ
    explicit_max_sensor = (
        params is not None and "MAX_DIAGONAL_SENSOR_UM" in params
    ) or "TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM" in os.environ
    if explicit_min_sensor:
        min_sensor_um = (
            params["MIN_DIAGONAL_SENSOR_UM"]
            if params is not None and "MIN_DIAGONAL_SENSOR_UM" in params
            else os.environ["TRAJ_ASSOC_MIN_DIAGONAL_SENSOR_UM"]
        )
        MIN_DIAGONAL = float(min_sensor_um) / camera_pixel_um
    elif params is None:
        MIN_DIAGONAL = 202.5 / camera_pixel_um
    else:
        MIN_DIAGONAL = float(run_params.get("MIN_DIAGONAL", 0.0))
    if explicit_max_sensor:
        max_sensor_um = (
            params["MAX_DIAGONAL_SENSOR_UM"]
            if params is not None and "MAX_DIAGONAL_SENSOR_UM" in params
            else os.environ["TRAJ_ASSOC_MAX_DIAGONAL_SENSOR_UM"]
        )
        MAX_DIAGONAL = float(max_sensor_um) / camera_pixel_um
    elif params is None:
        MAX_DIAGONAL = 675.0 / camera_pixel_um
    else:
        MAX_DIAGONAL = float(run_params["MAX_DIAGONAL"])
    SQUARE_TOLERANCE = float(run_params["SQUARE_TOLERANCE"])
    NEIGHBORS_K = int(run_params["NEIGHBORS_K"])
    LIMIT_SQUARE = float(run_params["LIMIT_SQUARE"])
    LIMIT_TRIANGLE = float(run_params["LIMIT_TRIANGLE"])

    # 读取数据
    try:
        df = read_pipeline_csv(csv_path, header=0, skiprows=[1, 2, 3], encoding='latin1')
    except:
        try:
            df = read_pipeline_csv(csv_path, header=0, skiprows=[1, 2, 3], encoding='gbk')
        except Exception as e:
            print(f"读取失败: {e}")
            return

    feat_cols_for_clustering = [
        'TRACK_MEAN_SPEED', 'TRACK_STD_SPEED', 'CONFINEMENT_RATIO',
        'LINEARITY_OF_FORWARD_PROGRESSION', 'MEAN_DIRECTIONAL_CHANGE_RATE'
    ]
    feat_cols_output = ['TRACK_DURATION'] + feat_cols_for_clustering
    pos_cols = ['TRACK_X_LOCATION', 'TRACK_Y_LOCATION']

    if 'TRACK_ID' not in df.columns:
        df['TRACK_ID'] = df.index

    target_metric = 'MAX_DISTANCE_TRAVELED'
    if target_metric not in df.columns:
        if 'TRACK_DISPLACEMENT' in df.columns:
            target_metric = 'TRACK_DISPLACEMENT'
        else:
            df['TOTAL_DISTANCE'] = df['TRACK_MEAN_SPEED'] * df['TRACK_DURATION']
            target_metric = 'TOTAL_DISTANCE'

    df_clean = df.dropna(subset=pos_cols + feat_cols_output + [target_metric, 'TRACK_ID']).reset_index(drop=False)
    if len(df_clean) < 3:
        print("有效数据点过少，无法分析。")
        return

    # 数据准备 (PCA)
    print("正在进行特征标准化与 PCA 降维...")
    X_pos = df_clean[pos_cols].values
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(df_clean[feat_cols_for_clustering].values)
    pca = PCA(n_components=0.9)
    X_feat_pca = pca.fit_transform(X_feat_scaled)
    X_feat_weighted = X_feat_pca

    nbrs = NearestNeighbors(n_neighbors=min(NEIGHBORS_K, len(df_clean)), algorithm='ball_tree').fit(X_pos)
    distances, indices = nbrs.kneighbors(X_pos)

    final_groups = []
    used_indices = set()
    df['Cluster_ID'] = 0
    df['Cluster_Type'] = 'None'

    # Pass 1: 寻找四边形
    print(f"正在寻找四边形...")
    candidates_4 = []
    seen_sets_4 = set()
    for i in range(len(df_clean)):
        for combo in combinations(indices[i], 4):
            combo = tuple(sorted(combo))
            if combo in seen_sets_4: continue
            seen_sets_4.add(combo)
            pts = X_pos[list(combo)]
            dists = sorted(pdist(pts))
            if dists[-1] < MIN_DIAGONAL or dists[-1] > MAX_DIAGONAL: continue
            sides, diags = dists[:4], dists[4:]
            mean_side = np.mean(sides)
            if mean_side == 0: continue
            sq_score = (np.std(sides) / mean_side) + (np.std(diags) / np.mean(diags)) + abs(
                (np.mean(diags) / mean_side) - 1.414)
            if sq_score > SQUARE_TOLERANCE: continue
            feat_dist = np.mean(pdist(X_feat_weighted[list(combo)]))
            if feat_dist > LIMIT_SQUARE: continue
            candidates_4.append({'indices': combo, 'score': feat_dist, 'pts': pts, 'type': '4-Square'})

    candidates_4.sort(key=lambda x: x['score'])
    group_id = 1
    count_pass1 = 0
    for cand in candidates_4:
        if set(cand['indices']).isdisjoint(used_indices):
            original_indices = df_clean.loc[list(cand['indices']), 'index'].values
            df.loc[original_indices, 'Cluster_ID'] = group_id
            df.loc[original_indices, 'Cluster_Type'] = '4-Square'
            final_groups.append(cand)
            used_indices.update(cand['indices'])
            group_id, count_pass1 = group_id + 1, count_pass1 + 1

    # Pass 2: 寻找三角形
    print(f"正在寻找三角形...")
    remaining = [i for i in range(len(df_clean)) if i not in used_indices]
    candidates_3 = []
    seen_sets_3 = set()
    for i in remaining:
        current_neighbors = [n for n in indices[i] if n not in used_indices]
        for combo in combinations(current_neighbors[:15], 3):
            combo = tuple(sorted(combo))
            if combo in seen_sets_3: continue
            seen_sets_3.add(combo)
            pts = X_pos[list(combo)]
            dists = sorted(pdist(pts))
            if dists[-1] < MIN_DIAGONAL or dists[-1] > MAX_DIAGONAL or (dists[0] + dists[1]) / dists[2] < 1.05: continue
            feat_dist = np.mean(pdist(X_feat_weighted[list(combo)]))
            if feat_dist > LIMIT_TRIANGLE: continue
            candidates_3.append({'indices': combo, 'score': feat_dist, 'pts': pts, 'type': '3-Triangle'})

    candidates_3.sort(key=lambda x: x['score'])
    count_pass2 = 0
    for cand in candidates_3:
        if set(cand['indices']).isdisjoint(used_indices):
            original_indices = df_clean.loc[list(cand['indices']), 'index'].values
            df.loc[original_indices, 'Cluster_ID'] = group_id
            df.loc[original_indices, 'Cluster_Type'] = '3-Triangle'
            final_groups.append(cand)
            used_indices.update(cand['indices'])
            group_id, count_pass2 = group_id + 1, count_pass2 + 1

    # 输出保存
    output_columns = ['TRACK_ID', 'LABEL'] + feat_cols_output + pos_cols + [target_metric, 'Cluster_ID', 'Cluster_Type']
    valid_cols = [c for c in output_columns if c in df.columns]
    try:
        df[valid_cols].to_csv(output_csv_path, index=False, encoding='gbk')
    except:
        df[valid_cols].to_csv(output_csv_path, index=False)

    # 绘图可视化
    plt.figure(figsize=(12, 10), dpi=150)
    plt.scatter(df_clean['TRACK_X_LOCATION'], df_clean['TRACK_Y_LOCATION'], c='lightgray', s=20, alpha=0.5,
                label='Background')
    colors = plt.cm.tab10(np.linspace(0, 1, len(final_groups))) if len(final_groups) > 0 else []

    for i, (group, color) in enumerate(zip(final_groups, colors)):
        pts = group['pts']
        marker_style = '-' if group['type'] == '4-Square' else '--'
        plt.scatter(pts[:, 0], pts[:, 1], color=color, s=80, edgecolors='black', zorder=5)
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                plt.plot(pts[simplex, 0], pts[simplex, 1], color=color, lw=2, alpha=0.8, linestyle=marker_style)
        centroid = np.mean(pts, axis=0)
        plt.text(centroid[0], centroid[1], str(i + 1), fontsize=8, color='white', fontweight='bold',
                 ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.2", fc=color, ec="white", alpha=0.9))

        # ... 前面代码保持不变 ...

        plt.gca().invert_yaxis()
        plt.title(
            f'Detected Groups\nInput: {os.path.basename(csv_path)}\n4-Squares: {count_pass1} | 3-Triangles: {count_pass2}')
        plt.tight_layout()
        plt.savefig(output_img_path)

        # --- 这里是增加的控制台输出 ---
        print("-" * 30)
        print(f"统计结果汇总:")
        print(f"找到四边形 (4-Squares): {count_pass1} 组")
        print(f"找到三角形 (3-Triangles): {count_pass2} 组")
        print("-" * 30)
        print(f"✅ 处理完成！数据已保存，图片见: {output_img_path}")



def main(input_csv=None, output_csv=None, output_img=None, show_plot=False, params=None, spots_csv=None):
    if input_csv is None:
        input_csv, auto_output_csv, auto_output_img = auto_find_track_file()
        output_csv = output_csv or auto_output_csv
        output_img = output_img or auto_output_img
    else:
        output_csv = output_csv or "classified_groups_pca.csv"
        output_img = output_img or "classified_groups_pca.png"

    if input_csv:
        for stale in ["trajectory_association_spots.csv", "trajectory_lap_cost_groups.csv", "track_id_stabilization_audit.csv"]:
            if os.path.exists(stale):
                os.remove(stale)
        spots_csv = spots_csv or os.getenv("TRACKMATE_SPOTS_CSV", "") or auto_find_spots_file()
        if not classify_with_trajectory_association(input_csv, spots_csv, output_csv, output_img, params=params):
            classify_and_visualize(input_csv, output_csv, output_img, params=params)
        if show_plot:
            plt.show()
        else:
            plt.close("all")


# ===============================
# 3. 运行入口
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify tracks into square/triangle groups.")
    parser.add_argument("--input-csv", dest="input_csv", help="Path to the track CSV file.")
    parser.add_argument("--output-csv", dest="output_csv", help="Path for the classified CSV output.")
    parser.add_argument("--output-img", dest="output_img", help="Path for the classification plot image.")
    parser.add_argument("--spots-csv", dest="spots_csv", help="Path to the TrackMate spots CSV file.")
    parser.add_argument("--show-plot", action="store_true", help="Display matplotlib windows after processing.")
    args = parser.parse_args()
    main(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_img=args.output_img,
        spots_csv=args.spots_csv,
        show_plot=args.show_plot,
    )
