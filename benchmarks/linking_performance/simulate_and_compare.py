from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from skimage import feature, filters, io
import tifffile


ROOT = Path(__file__).resolve().parent
PSF_DIR = ROOT / "data" / "psf_20x_roi16bit"
OUT_DIR = ROOT / "outputs"
logging.getLogger("tifffile").setLevel(logging.CRITICAL)


@dataclass
class PsfTemplate:
    file_index: int
    z_um: float
    image: np.ndarray
    offsets_rc: np.ndarray
    mean_diag_px: float


def intensity_centroid(roi: np.ndarray) -> tuple[float, float]:
    roi = roi.astype(float)
    bg = min(float(np.percentile(roi, 20)), float(np.median(roi)))
    signal = np.clip(roi - bg, 0, None)
    total = float(signal.sum())
    if total <= 1e-12:
        return (roi.shape[0] - 1) / 2.0, (roi.shape[1] - 1) / 2.0
    rr, cc = np.indices(signal.shape)
    return float((rr * signal).sum() / total), float((cc * signal).sum() / total)


def square_score(points_xy: np.ndarray) -> float:
    dists = np.sort(pdist(points_xy))
    if len(dists) != 6:
        return np.inf
    sides = dists[:4]
    diags = dists[4:]
    mean_side = float(np.mean(sides))
    mean_diag = float(np.mean(diags))
    if mean_side <= 0 or mean_diag <= 0:
        return np.inf
    return (
        float(np.std(sides) / mean_side)
        + float(np.std(diags) / mean_diag)
        + abs(mean_diag / mean_side - np.sqrt(2))
    )


def mean_diag(points_xy: np.ndarray) -> float:
    dists = np.sort(pdist(points_xy))
    if len(dists) != 6:
        return np.nan
    return float(np.mean(dists[4:]))


def local_four_combinations(points_xy: np.ndarray, neighbors_k: int = 14):
    """Yield unique local four-point combinations instead of all global C(n, 4)."""
    n = len(points_xy)
    if n < 4:
        return
    dist = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    seen: set[tuple[int, ...]] = set()
    for i in range(n):
        local_ids = np.argsort(dist[i])[: min(neighbors_k, n)]
        for combo in combinations(local_ids, 4):
            combo = tuple(sorted(int(x) for x in combo))
            if combo in seen:
                continue
            seen.add(combo)
            yield combo


def order_four_spots(points_rc: np.ndarray) -> np.ndarray:
    center = points_rc.mean(axis=0)
    angles = np.arctan2(points_rc[:, 0] - center[0], points_rc[:, 1] - center[1])
    return points_rc[np.argsort(angles)]


def load_psf_templates() -> list[PsfTemplate]:
    templates: list[PsfTemplate] = []
    for file_index in range(10015, 10036):
        path = PSF_DIR / f"{file_index}.tif"
        raw = tifffile.imread(str(path)).astype(float)
        bg = float(np.percentile(raw, 50))
        img = np.clip(raw - bg, 0, None)
        if img.max() > 0:
            img = img / img.max()

        smoothed = filters.gaussian(raw, sigma=1.0)
        peaks = feature.peak_local_max(
            smoothed,
            min_distance=20,
            threshold_abs=float(np.percentile(smoothed, 99.25)),
            num_peaks=8,
        )
        if len(peaks) < 4:
            raise RuntimeError(f"Could not detect four PSF spots in {path}")
        values = np.array([raw[int(r), int(c)] for r, c in peaks])
        peaks = peaks[np.argsort(values)[::-1][:4]].astype(float)

        refined = []
        for r, c in peaks:
            r_i, c_i = int(round(r)), int(round(c))
            r0, r1 = max(0, r_i - 7), min(raw.shape[0], r_i + 8)
            c0, c1 = max(0, c_i - 7), min(raw.shape[1], c_i + 8)
            dr, dc = intensity_centroid(raw[r0:r1, c0:c1])
            refined.append([r0 + dr, c0 + dc])
        spots_rc = order_four_spots(np.asarray(refined, dtype=float))
        center_rc = spots_rc.mean(axis=0)
        offsets_rc = spots_rc - center_rc
        templates.append(
            PsfTemplate(
                file_index=file_index,
                z_um=float(file_index - 10015),
                image=img,
                offsets_rc=offsets_rc,
                mean_diag_px=mean_diag(spots_rc[:, ::-1]),
            )
        )
    return templates


def template_for_z(templates: list[PsfTemplate], z_um: float) -> PsfTemplate:
    return min(templates, key=lambda t: abs(t.z_um - z_um))


def add_shifted_template(canvas: np.ndarray, template: np.ndarray, center_rc: tuple[float, float], amp: float) -> None:
    h, w = template.shape
    cr, cc = center_rc
    r0 = int(round(cr - h / 2))
    c0 = int(round(cc - w / 2))
    dst_r0, dst_c0 = max(0, r0), max(0, c0)
    dst_r1, dst_c1 = min(canvas.shape[0], r0 + h), min(canvas.shape[1], c0 + w)
    if dst_r1 <= dst_r0 or dst_c1 <= dst_c0:
        return
    src_r0, src_c0 = dst_r0 - r0, dst_c0 - c0
    src_r1, src_c1 = src_r0 + (dst_r1 - dst_r0), src_c0 + (dst_c1 - dst_c0)
    canvas[dst_r0:dst_r1, dst_c0:dst_c1] += amp * template[src_r0:src_r1, src_c0:src_c1]


def particle_state(pid: int, frame: int, n_frames: int, center_separation_px: float = 18.0) -> tuple[float, float, float]:
    t = frame / (n_frames - 1)
    if pid == 1:
        x = 88 + 94 * t
        y = 120 - center_separation_px / 2 + 6 * np.sin(2 * np.pi * t)
        z = 4 + 12 * t + 1.5 * np.sin(2 * np.pi * t)
    elif pid == 2:
        x = 182 - 94 * t
        y = 120 + center_separation_px / 2 - 6 * np.sin(2 * np.pi * t + 0.3)
        z = 16 - 12 * t + 1.2 * np.cos(2 * np.pi * t)
    else:
        x = 300 + 18 * np.sin(2 * np.pi * t)
        y = 112 + 68 * t
        z = 8 + 4 * np.sin(4 * np.pi * t)
    return float(x), float(y), float(np.clip(z, 0, 20))


def make_truth_and_detections(
    templates: list[PsfTemplate],
    seed: int,
    n_frames: int,
    dropout_prob: float = 0.045,
    center_separation_px: float = 18.0,
    clutter_points_per_frame: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    stack = []
    truth_rows = []
    det_rows = []
    h, w = 300, 390
    noise_sigma = 0.018

    for frame in range(n_frames):
        crossing_start = int(round(0.38 * n_frames))
        crossing_end = int(round(0.63 * n_frames))
        canvas = rng.normal(0.0, noise_sigma, size=(h, w))
        for pid in [1, 2, 3]:
            x, y, z = particle_state(pid, frame, n_frames, center_separation_px=center_separation_px)
            tmpl = template_for_z(templates, z)
            amp = {1: 0.95, 2: 0.92, 3: 0.85}[pid] * (0.95 + 0.1 * rng.random())
            add_shifted_template(canvas, tmpl.image, (y, x), amp)
            truth_rows.append({"FRAME": frame, "particle_id": pid, "X_true": x, "Y_true": y, "Z_true": z})

            for spot_idx, (dr, dc) in enumerate(tmpl.offsets_rc, start=1):
                sx = x + dc
                sy = y + dr
                missing = rng.random() < dropout_prob
                if pid in {1, 2} and crossing_start <= frame <= crossing_end and rng.random() < 0.09:
                    missing = True
                if missing:
                    continue
                det_rows.append(
                    {
                        "FRAME": frame,
                        "detection_id": f"f{frame}_p{pid}_s{spot_idx}",
                        "TRACK_ID": pid * 10 + spot_idx,
                        "particle_id": pid,
                        "spot_index": spot_idx,
                        "X": sx + rng.normal(0, 0.75),
                        "Y": sy + rng.normal(0, 0.75),
                        "is_noise": False,
                    }
                )

        if crossing_start <= frame <= crossing_end and clutter_points_per_frame > 0:
            # Deliberate clutter in the crossing region: it can form plausible
            # per-frame squares but has no coherent trajectory identity.
            cx = 135 + 12 * np.sin(frame * 0.45)
            cy = 120 + 10 * np.cos(frame * 0.35)
            base_offsets = [(-34, -28), (34, -28), (-34, 28), (34, 28)]
            random_offsets = [
                (rng.normal(0, 45), rng.normal(0, 38))
                for _ in range(max(0, clutter_points_per_frame - len(base_offsets)))
            ]
            for j, (dx, dy) in enumerate((base_offsets + random_offsets)[:clutter_points_per_frame]):
                if rng.random() < 0.72:
                    det_rows.append(
                        {
                            "FRAME": frame,
                            "detection_id": f"f{frame}_noise{j}",
                            "TRACK_ID": -1000 - frame * 10 - j,
                            "particle_id": 0,
                            "spot_index": 0,
                            "X": cx + dx + rng.normal(0, 1.2),
                            "Y": cy + dy + rng.normal(0, 1.2),
                            "is_noise": True,
                        }
                    )
                    rr, cc = int(round(cy + dy)), int(round(cx + dx))
                    if 0 <= rr < h and 0 <= cc < w:
                        canvas[max(0, rr - 2) : min(h, rr + 3), max(0, cc - 2) : min(w, cc + 3)] += 0.45

        stack.append(np.clip(canvas, 0, None))

    stack_arr = np.asarray(stack, dtype=float)
    stack_u16 = np.asarray(250 + 14000 * stack_arr / max(stack_arr.max(), 1e-12), dtype=np.uint16)
    return pd.DataFrame(truth_rows), pd.DataFrame(det_rows), stack_u16


def framewise_group(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 4:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            diag = mean_diag(group_pts)
            if not 55 <= diag <= 115:
                continue
            score = square_score(group_pts)
            if score > 0.28:
                continue
            center = group_pts.mean(axis=0)
            crossing_bias = 0.015 * np.linalg.norm(center - np.array([135.0, 120.0]))
            candidates.append((score + crossing_bias, combo))
        candidates.sort(key=lambda x: x[0])
        used = set()
        gid = 1
        for score, combo in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            pids = group["particle_id"].astype(int).tolist()
            majority_pid = max(set(pids), key=pids.count)
            rows.append(
                {
                    "FRAME": int(frame),
                    "group_id": gid,
                    "method": "framewise_geometry",
                    "score": float(score),
                    "member_detection_ids": ";".join(group["detection_id"].astype(str)),
                    "member_track_ids": ";".join(group["TRACK_ID"].astype(str)),
                    "member_particle_ids": ";".join(map(str, pids)),
                    "assigned_particle_id": int(majority_pid),
                    "group_purity": float(pids.count(majority_pid) / 4.0),
                    "is_correct": bool(pids.count(majority_pid) == 4 and majority_pid != 0),
                    "X_rec": float(group["X"].mean()),
                    "Y_rec": float(group["Y"].mean()),
                    "mean_diag_px": mean_diag(group[["X", "Y"]].to_numpy(float)),
                }
            )
            used.update(combo)
            gid += 1
    return pd.DataFrame(rows)


def rows_from_frame_group(
    frame: int,
    group_id: int,
    method: str,
    group: pd.DataFrame,
    score: float,
) -> dict:
    pids = group["particle_id"].astype(int).tolist()
    majority_pid = max(set(pids), key=pids.count)
    return {
        "FRAME": int(frame),
        "group_id": int(group_id),
        "method": method,
        "score": float(score),
        "member_detection_ids": ";".join(group["detection_id"].astype(str)),
        "member_track_ids": ";".join(group["TRACK_ID"].astype(str)),
        "member_particle_ids": ";".join(map(str, pids)),
        "assigned_particle_id": int(majority_pid),
        "group_purity": float(pids.count(majority_pid) / 4.0),
        "is_correct": bool(pids.count(majority_pid) == 4 and majority_pid != 0),
        "X_rec": float(group["X"].mean()),
        "Y_rec": float(group["Y"].mean()),
        "mean_diag_px": mean_diag(group[["X", "Y"]].to_numpy(float)),
    }


def nearest_neighbor_framewise_group(detections: pd.DataFrame) -> pd.DataFrame:
    """A simple per-frame baseline that favors compact nearest-neighbor quartets."""
    rows = []
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 4:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            dists = np.sort(pdist(group_pts))
            diag = float(np.mean(dists[4:]))
            if not 45 <= diag <= 125:
                continue
            compactness = float(np.mean(dists[:4]))
            diag_spread = float(np.std(dists[4:]))
            candidates.append((compactness + 0.25 * diag_spread, combo))
        candidates.sort(key=lambda x: x[0])
        used = set()
        gid = 1
        for score, combo in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            rows.append(rows_from_frame_group(int(frame), gid, "nearest_neighbor_framewise", group, score))
            used.update(combo)
            gid += 1
    return pd.DataFrame(rows)


def geometry_previous_frame_group(detections: pd.DataFrame) -> pd.DataFrame:
    """A stronger frame-wise baseline with a previous-frame center continuity term."""
    rows = []
    prev_centers: np.ndarray | None = None
    for frame, frame_df in detections.groupby("FRAME"):
        pts = frame_df[["X", "Y"]].to_numpy(float)
        if len(pts) < 4:
            continue
        candidates = []
        for combo in local_four_combinations(pts):
            group_pts = pts[list(combo)]
            diag = mean_diag(group_pts)
            if not 55 <= diag <= 115:
                continue
            geom = square_score(group_pts)
            if geom > 0.32:
                continue
            center = group_pts.mean(axis=0)
            temporal_penalty = 0.0
            if prev_centers is not None and len(prev_centers):
                temporal_penalty = 0.02 * float(np.min(np.linalg.norm(prev_centers - center[None, :], axis=1)))
            candidates.append((geom + temporal_penalty, combo, center))
        candidates.sort(key=lambda x: x[0])
        used = set()
        frame_rows = []
        centers = []
        gid = 1
        for score, combo, center in candidates:
            if set(combo) & used:
                continue
            group = frame_df.iloc[list(combo)]
            frame_rows.append(rows_from_frame_group(int(frame), gid, "geometry_previous_frame", group, score))
            centers.append(center)
            used.update(combo)
            gid += 1
        rows.extend(frame_rows)
        if centers:
            prev_centers = np.asarray(centers, dtype=float)
    return pd.DataFrame(rows)


def oracle_group(detections: pd.DataFrame) -> pd.DataFrame:
    """Upper bound baseline: uses true particle labels to group sub-spots."""
    rows = []
    for (frame, pid), group in detections[(detections["particle_id"] > 0)].groupby(["FRAME", "particle_id"]):
        if len(group) != 4:
            continue
        rows.append(rows_from_frame_group(int(frame), int(pid), "oracle_grouping", group, 0.0))
    return pd.DataFrame(rows)


def track_features(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, group in detections.groupby("TRACK_ID"):
        group = group.sort_values("FRAME")
        pts = group[["X", "Y"]].to_numpy(float)
        disp = np.diff(pts, axis=0)
        speed = np.linalg.norm(disp, axis=1)
        rows.append(
            {
                "TRACK_ID": int(tid),
                "mean_x": float(np.mean(pts[:, 0])),
                "mean_y": float(np.mean(pts[:, 1])),
                "vx": float(np.mean(disp[:, 0])) if len(disp) else 0.0,
                "vy": float(np.mean(disp[:, 1])) if len(disp) else 0.0,
                "speed_mean": float(np.mean(speed)) if len(speed) else 0.0,
                "speed_std": float(np.std(speed)) if len(speed) else 0.0,
                "duration": int(group["FRAME"].nunique()),
                "true_particle_id": int(group["particle_id"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def trajectory_level_group(detections: pd.DataFrame) -> pd.DataFrame:
    feats = track_features(detections)
    min_duration = max(12, int(0.35 * detections["FRAME"].nunique()))
    feats = feats[feats["duration"] >= min_duration].reset_index(drop=True)
    pts = feats[["mean_x", "mean_y"]].to_numpy(float)
    candidates = []
    for combo in combinations(range(len(feats)), 4):
        group_pts = pts[list(combo)]
        diag = mean_diag(group_pts)
        if not 55 <= diag <= 115:
            continue
        geom = square_score(group_pts)
        if geom > 0.35:
            continue
        motion = feats.iloc[list(combo)][["vx", "vy", "speed_mean", "speed_std"]].to_numpy(float)
        motion_dist = float(np.mean(pdist(motion))) if len(motion) > 1 else 0.0
        duration_penalty = 1.0 - float(feats.iloc[list(combo)]["duration"].mean()) / detections["FRAME"].nunique()
        candidates.append((geom + 0.35 * motion_dist + 0.5 * duration_penalty, combo))
    candidates.sort(key=lambda x: x[0])
    used = set()
    groups = []
    gid = 1
    for score, combo in candidates:
        if set(combo) & used:
            continue
        tids = feats.iloc[list(combo)]["TRACK_ID"].astype(int).tolist()
        pids = feats.iloc[list(combo)]["true_particle_id"].astype(int).tolist()
        majority_pid = max(set(pids), key=pids.count)
        groups.append(
            {
                "trajectory_group_id": gid,
                "method": "trajectory_level",
                "score": float(score),
                "member_track_ids": ";".join(map(str, tids)),
                "member_particle_ids": ";".join(map(str, pids)),
                "assigned_particle_id": int(majority_pid),
                "group_purity": float(pids.count(majority_pid) / 4.0),
                "is_correct": bool(pids.count(majority_pid) == 4 and majority_pid != 0),
            }
        )
        used.update(combo)
        gid += 1
    return pd.DataFrame(groups)


def recover_from_groups(groups: pd.DataFrame, detections: pd.DataFrame, z_cal: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    if method == "framewise":
        for _, group in groups.iterrows():
            ids = str(group["member_detection_ids"]).split(";")
            sub = detections[detections["detection_id"].astype(str).isin(ids)]
            if len(sub) != 4:
                continue
            rows.append(
                {
                    "FRAME": int(group["FRAME"]),
                    "assigned_particle_id": int(group["assigned_particle_id"]),
                    "X_rec": float(sub["X"].mean()),
                    "Y_rec": float(sub["Y"].mean()),
                    "Z_rec": diag_to_z(mean_diag(sub[["X", "Y"]].to_numpy(float)), z_cal),
                    "is_correct": bool(group["is_correct"]),
                }
            )
    else:
        group_map = []
        for _, group in groups.iterrows():
            tids = [int(x) for x in str(group["member_track_ids"]).split(";")]
            group_map.append((int(group["assigned_particle_id"]), tids, bool(group["is_correct"])))
        for pid, tids, is_correct in group_map:
            for frame, frame_df in detections[detections["TRACK_ID"].isin(tids)].groupby("FRAME"):
                if len(frame_df) < 3:
                    continue
                pts = frame_df[["X", "Y"]].to_numpy(float)
                rows.append(
                    {
                        "FRAME": int(frame),
                        "assigned_particle_id": pid,
                        "X_rec": float(np.mean(pts[:, 0])),
                        "Y_rec": float(np.mean(pts[:, 1])),
                        "Z_rec": diag_to_z(mean_diag(pts) if len(pts) == 4 else np.nan, z_cal),
                        "is_correct": is_correct,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["assigned_particle_id", "FRAME"]).reset_index(drop=True)


def diag_to_z(diag: float, z_cal: pd.DataFrame) -> float:
    if not np.isfinite(diag):
        return np.nan
    return float(np.interp(diag, z_cal["mean_diag_px"], z_cal["z_um"], left=np.nan, right=np.nan))


def attach_errors(recovered: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = recovered.merge(
        truth,
        left_on=["FRAME", "assigned_particle_id"],
        right_on=["FRAME", "particle_id"],
        how="left",
    )
    merged["xy_error"] = np.hypot(merged["X_rec"] - merged["X_true"], merged["Y_rec"] - merged["Y_true"])
    merged["z_error"] = merged["Z_rec"] - merged["Z_true"]
    merged["xyz_error"] = np.sqrt(merged["xy_error"] ** 2 + merged["z_error"] ** 2)
    return merged


def summarize(method: str, groups: pd.DataFrame, recovered: pd.DataFrame, n_frames: int, n_particles: int) -> dict:
    if groups.empty or recovered.empty:
        return {
            "method": method,
            "association_accuracy": 0.0 if groups.empty else float(groups["is_correct"].mean()),
            "mean_group_purity": 0.0 if groups.empty else float(groups["group_purity"].mean()),
            "missing_or_failed_frames": n_frames * n_particles,
            "xyz_rmse": np.nan,
            "z_rmse": np.nan,
        }
    total_slots = n_frames * n_particles
    valid = recovered[recovered.get("is_correct", False).astype(bool)].copy()
    if "particle_id" in valid.columns:
        valid = valid[valid["particle_id"].notna()]
    valid_slots = valid[["FRAME", "assigned_particle_id"]].drop_duplicates().shape[0] if not valid.empty else 0
    error_df = valid if not valid.empty else recovered
    return {
        "method": method,
        "association_accuracy": float(groups["is_correct"].mean()),
        "mean_group_purity": float(groups["group_purity"].mean()),
        "missing_or_failed_frames": int(total_slots - valid_slots),
        "xyz_rmse": float(np.sqrt(np.nanmean(error_df["xyz_error"] ** 2))),
        "z_rmse": float(np.sqrt(np.nanmean(error_df["z_error"] ** 2))),
    }


def plot_grouping(framewise: pd.DataFrame, traj_rows: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    frame_acc = framewise.groupby("FRAME")["is_correct"].mean()
    axes[0].plot(frame_acc.index, frame_acc.values, color="#d62728", lw=1.7, label="Frame-wise")
    axes[0].axvspan(45, 75, color="#f4a261", alpha=0.25, label="crossing/clutter")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Per-frame group correctness")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Correct group fraction")
    axes[0].legend(fontsize=8)

    counts = [
        float(framewise["group_purity"].mean()) if not framewise.empty else 0,
        float(traj_rows["group_purity"].mean()) if not traj_rows.empty else 0,
    ]
    axes[1].bar(["Frame-wise", "Trajectory-level"], counts, color=["#d62728", "#2ca02c"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Mean group purity")

    acc = [
        float(framewise["is_correct"].mean()) if not framewise.empty else 0,
        float(traj_rows["is_correct"].mean()) if not traj_rows.empty else 0,
    ]
    axes[2].bar(["Frame-wise", "Trajectory-level"], acc, color=["#d62728", "#2ca02c"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Association accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_3d(truth: pd.DataFrame, rec_frame: pd.DataFrame, rec_traj: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for pid, sub in truth.groupby("particle_id"):
        ax.plot(sub["X_true"], sub["Y_true"], sub["Z_true"], lw=2, label=f"truth P{pid}")
        ax2.plot(sub["X_true"], sub["Y_true"], sub["Z_true"], lw=2, label=f"truth P{pid}")
    for pid, sub in rec_frame.groupby("assigned_particle_id") if "assigned_particle_id" in rec_frame.columns else []:
        ax.scatter(sub["X_rec"], sub["Y_rec"], sub["Z_rec"], s=4, alpha=0.45)
    for pid, sub in rec_traj.groupby("assigned_particle_id") if "assigned_particle_id" in rec_traj.columns else []:
        ax2.scatter(sub["X_rec"], sub["Y_rec"], sub["Z_rec"], s=4, alpha=0.45)
    ax.set_title("Frame-wise geometry")
    ax2.set_title("Trajectory-level association")
    for a in [ax, ax2]:
        a.set_xlabel("X px")
        a.set_ylabel("Y px")
        a.set_zlabel("Z um")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_failure_examples(detections: pd.DataFrame, framewise: pd.DataFrame, traj_groups: pd.DataFrame, out: Path) -> None:
    bad = framewise[(~framewise["is_correct"]) & framewise["FRAME"].between(45, 75)]
    frames = list(bad["FRAME"].drop_duplicates().head(3))
    if not frames:
        frames = [50, 60, 70]
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]
    colors = {0: "#999999", 1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for ax, frame in zip(axes, frames):
        sub = detections[detections["FRAME"] == frame]
        ax.scatter(sub["X"], sub["Y"], c=[colors[int(pid)] for pid in sub["particle_id"]], s=26, alpha=0.85)
        for _, g in framewise[framewise["FRAME"] == frame].iterrows():
            ids = str(g["member_detection_ids"]).split(";")
            pts = sub[sub["detection_id"].isin(ids)][["X", "Y"]].to_numpy(float)
            if len(pts) == 4:
                order = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()))
                poly = np.vstack([pts[order], pts[order][0]])
                ax.plot(poly[:, 0], poly[:, 1], color="#d62728", lw=1.2, alpha=0.8)
        ax.set_title(f"Frame {frame}: red = frame-wise groups")
        ax.set_aspect("equal")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def z_calibration_from_templates(templates: list[PsfTemplate]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"file_index": t.file_index, "z_um": t.z_um, "mean_diag_px": t.mean_diag_px} for t in templates]
    ).sort_values("mean_diag_px")


def run_experiment(
    templates: list[PsfTemplate],
    seed: int,
    n_frames: int,
    dropout_prob: float,
    center_separation_px: float,
    clutter_points_per_frame: int,
) -> dict[str, object]:
    z_cal = z_calibration_from_templates(templates)
    truth, detections, stack = make_truth_and_detections(
        templates,
        seed,
        n_frames,
        dropout_prob=dropout_prob,
        center_separation_px=center_separation_px,
        clutter_points_per_frame=clutter_points_per_frame,
    )

    groupers = {
        "nearest_neighbor_framewise": nearest_neighbor_framewise_group,
        "framewise_geometry": framewise_group,
        "geometry_previous_frame": geometry_previous_frame_group,
        "trajectory_level": trajectory_level_group,
        "oracle_grouping": oracle_group,
    }
    groups: dict[str, pd.DataFrame] = {name: fn(detections) for name, fn in groupers.items()}
    recovered: dict[str, pd.DataFrame] = {}
    summaries = []
    for name, group_df in groups.items():
        mode = "trajectory" if name == "trajectory_level" else "framewise"
        rec = recover_from_groups(group_df, detections, z_cal, mode)
        rec = attach_errors(rec, truth) if not rec.empty else rec
        recovered[name] = rec
        summaries.append(summarize(name, group_df, rec, n_frames, 3))
    summary = pd.DataFrame(summaries)
    summary.insert(0, "seed", seed)
    summary.insert(1, "dropout_prob", dropout_prob)
    summary.insert(2, "center_separation_px", center_separation_px)
    summary.insert(3, "clutter_points_per_frame", clutter_points_per_frame)
    return {
        "z_cal": z_cal,
        "truth": truth,
        "detections": detections,
        "stack": stack,
        "groups": groups,
        "recovered": recovered,
        "summary": summary,
    }


def summarize_sweep(per_run: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sweep_name", "sweep_value", "method"]
    metric_cols = [
        "association_accuracy",
        "mean_group_purity",
        "missing_or_failed_frames",
        "xyz_rmse",
        "z_rmse",
    ]
    rows = []
    for keys, sub in per_run.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(sub))
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sd"] = float(vals.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def plot_sweep(summary: pd.DataFrame, out: Path) -> None:
    methods = [
        "oracle_grouping",
        "nearest_neighbor_framewise",
        "framewise_geometry",
        "geometry_previous_frame",
        "trajectory_level",
    ]
    colors = {
        "nearest_neighbor_framewise": "#9467bd",
        "framewise_geometry": "#d62728",
        "geometry_previous_frame": "#ff7f0e",
        "trajectory_level": "#2ca02c",
        "oracle_grouping": "#4d4d4d",
    }
    labels = {
        "nearest_neighbor_framewise": "Nearest-neighbor frame-wise",
        "framewise_geometry": "Frame-wise geometry",
        "geometry_previous_frame": "Geometry + previous frame",
        "trajectory_level": "Trajectory-level",
        "oracle_grouping": "Oracle upper bound",
    }
    sweeps = [
        ("dropout_prob", "Dropout probability"),
        ("center_separation_px", "Minimum center separation (px)"),
        ("clutter_points_per_frame", "Clutter detections per crossing frame"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row_idx, (sweep_name, xlabel) in enumerate(sweeps):
        sub_all = summary[summary["sweep_name"] == sweep_name]
        for method in methods:
            sub = sub_all[sub_all["method"] == method].sort_values("sweep_value")
            if sub.empty:
                continue
            x = sub["sweep_value"].astype(float).to_numpy()
            y = sub["association_accuracy_mean"].astype(float).to_numpy()
            e = sub["association_accuracy_sd"].astype(float).fillna(0).to_numpy()
            axes[row_idx, 0].errorbar(
                x,
                y,
                yerr=e,
                marker="o",
                lw=1.5,
                capsize=3,
                color=colors[method],
                label=labels[method],
                linestyle="--" if method == "oracle_grouping" else "-",
                alpha=0.65 if method == "oracle_grouping" else 1.0,
            )
            y2 = sub["z_rmse_mean"].astype(float).to_numpy()
            e2 = sub["z_rmse_sd"].astype(float).fillna(0).to_numpy()
            axes[row_idx, 1].errorbar(
                x,
                y2,
                yerr=e2,
                marker="o",
                lw=1.5,
                capsize=3,
                color=colors[method],
                label=labels[method],
                linestyle="--" if method == "oracle_grouping" else "-",
                alpha=0.65 if method == "oracle_grouping" else 1.0,
            )
        axes[row_idx, 0].set_xlabel(xlabel)
        axes[row_idx, 0].set_ylabel("Association accuracy")
        axes[row_idx, 0].set_ylim(-0.03, 1.05)
        axes[row_idx, 1].set_xlabel(xlabel)
        axes[row_idx, 1].set_ylabel("Z RMSE (um)")
        axes[row_idx, 1].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_sweep(templates: list[PsfTemplate], n_seeds: int, n_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    default_dropout = 0.045
    default_sep = 18.0
    default_clutter = 4
    conditions = []
    for value in [0.0, 0.05, 0.10, 0.20]:
        conditions.append(("dropout_prob", value, value, default_sep, default_clutter))
    for value in [8.0, 18.0, 32.0, 55.0]:
        conditions.append(("center_separation_px", value, default_dropout, value, default_clutter))
    for value in [0, 4, 8, 12]:
        conditions.append(("clutter_points_per_frame", value, default_dropout, default_sep, value))

    rows = []
    for sweep_name, sweep_value, dropout, sep, clutter in conditions:
        for seed in range(n_seeds):
            result = run_experiment(
                templates,
                seed=seed,
                n_frames=n_frames,
                dropout_prob=float(dropout),
                center_separation_px=float(sep),
                clutter_points_per_frame=int(clutter),
            )
            summary = result["summary"].copy()
            summary.insert(0, "sweep_name", sweep_name)
            summary.insert(1, "sweep_value", sweep_value)
            rows.append(summary)
            print(f"sweep={sweep_name}:{sweep_value} seed={seed + 1}/{n_seeds}", flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    return per_run, summarize_sweep(per_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dropout-prob", type=float, default=0.045)
    parser.add_argument("--center-separation-px", type=float, default=18.0)
    parser.add_argument("--clutter-points-per-frame", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = load_psf_templates()

    if args.sweep:
        per_run, sweep_summary = run_sweep(templates, n_seeds=args.n_seeds, n_frames=args.frames)
        per_run.to_csv(OUT_DIR / "sweep_per_run_metrics.csv", index=False)
        sweep_summary.to_csv(OUT_DIR / "sweep_summary_mean_sd.csv", index=False)
        plot_sweep(sweep_summary, OUT_DIR / "supp_fig_s6_algorithm_validation.png")
        print(sweep_summary.to_string(index=False))
        print(f"Saved sweep outputs to {OUT_DIR}")
        return

    result = run_experiment(
        templates,
        seed=args.seed,
        n_frames=args.frames,
        dropout_prob=args.dropout_prob,
        center_separation_px=args.center_separation_px,
        clutter_points_per_frame=args.clutter_points_per_frame,
    )
    z_cal = result["z_cal"]
    truth = result["truth"]
    detections = result["detections"]
    stack = result["stack"]
    groups = result["groups"]
    recovered = result["recovered"]
    summary = result["summary"].drop(columns=["seed", "dropout_prob", "center_separation_px", "clutter_points_per_frame"])

    io.imsave(str(OUT_DIR / "simulated_stack.tif"), stack, check_contrast=False)
    z_cal.to_csv(OUT_DIR / "psf_z_calibration.csv", index=False)
    truth.to_csv(OUT_DIR / "ground_truth_3d.csv", index=False)
    detections.to_csv(OUT_DIR / "subspot_tracks_truth.csv", index=False)
    groups["framewise_geometry"].to_csv(OUT_DIR / "framewise_groups.csv", index=False)
    groups["nearest_neighbor_framewise"].to_csv(OUT_DIR / "nearest_neighbor_framewise_groups.csv", index=False)
    groups["geometry_previous_frame"].to_csv(OUT_DIR / "geometry_previous_frame_groups.csv", index=False)
    groups["trajectory_level"].to_csv(OUT_DIR / "trajectory_level_groups.csv", index=False)
    groups["oracle_grouping"].to_csv(OUT_DIR / "oracle_groups.csv", index=False)
    recovered["framewise_geometry"].to_csv(OUT_DIR / "recovered_3d_framewise.csv", index=False)
    recovered["nearest_neighbor_framewise"].to_csv(OUT_DIR / "recovered_3d_nearest_neighbor_framewise.csv", index=False)
    recovered["geometry_previous_frame"].to_csv(OUT_DIR / "recovered_3d_geometry_previous_frame.csv", index=False)
    recovered["trajectory_level"].to_csv(OUT_DIR / "recovered_3d_trajectory_level.csv", index=False)
    recovered["oracle_grouping"].to_csv(OUT_DIR / "recovered_3d_oracle.csv", index=False)
    summary.to_csv(OUT_DIR / "association_accuracy_summary.csv", index=False)

    plot_grouping(groups["framewise_geometry"], groups["trajectory_level"], OUT_DIR / "grouping_comparison.png")
    plot_3d(truth, recovered["framewise_geometry"], recovered["trajectory_level"], OUT_DIR / "trajectory_3d_comparison.png")
    plot_failure_examples(detections, groups["framewise_geometry"], groups["trajectory_level"], OUT_DIR / "example_failure_frames.png")

    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
