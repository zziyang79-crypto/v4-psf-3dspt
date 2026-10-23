"""Localize and track the updated independent-validation ROIs.

20x: ROI/1 was replaced on 2026-10-24 (ROI/3 does not exist).
40x/60x: new ROI/3.
Calibration groups are unchanged: 20x ROI/2, 40x ROI/1, 60x ROI/1.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from v4psf.localization_methods import localize_four_spots
from v4psf.run_blur_angle_validation import track_unordered_diagonal_angles

root = Path(os.environ.get("V4PSF_FLUORESCENCE_ROOT", REPO_ROOT / "data" / "raw" / "fluorescence"))
out = Path(os.environ.get("V4PSF_OUT_DIR", REPO_ROOT / "outputs" / "fluorescence_calibration"))
out.mkdir(parents=True, exist_ok=True)

# objective, directory, new validation group
items = [
    ("20x_0.45NA_air", "20X0.45NA", "1"),
    ("40x_0.6NA_air", "40X0.6NA", "3"),
    ("60x_1.0NA_water", "60X1.0NA-W", "3"),
]
roi_radius = {"20x_0.45NA_air": 12, "40x_0.6NA_air": 16, "60x_1.0NA_water": 16}
SIGMA = 2.4


def line_angle(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


rows = []
for obj_key, obj_dir, grp in items:
    p = root / obj_dir / "ROI" / grp / f"{grp}-1.tif"
    stack = tifffile.imread(p)
    print(obj_key, grp, stack.shape)
    for i, img in enumerate(stack):
        im = ndimage.gaussian_filter(img.astype(np.float32), SIGMA)
        res = localize_four_spots(im, obj_key, "phasor", params={"roi_radius": roi_radius[obj_key]})
        base = {"objective": obj_key, "group": grp, "frame": i, "sigma": SIGMA,
                "success": res.success, "max_intensity": float(img.max())}
        if not res.success:
            base.update({"d1": np.nan, "d2": np.nan, "dmean": np.nan,
                         "theta1": np.nan, "theta2": np.nan, "cx": np.nan, "cy": np.nan,
                         "msg": res.message})
            rows.append(base)
            continue
        pts = np.array([[s.x, s.y] for s in res.spots])
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        g = pts[np.argsort(ang)]
        d1 = float(np.linalg.norm(g[0] - g[2]))
        d2 = float(np.linalg.norm(g[1] - g[3]))
        a1 = line_angle(g[0], g[2])
        a2 = line_angle(g[1], g[3])
        base.update({"d1": d1, "d2": d2, "dmean": (d1 + d2) / 2,
                     "theta1": a1, "theta2": a2, "cx": c[0], "cy": c[1], "msg": ""})
        rows.append(base)

df = pd.DataFrame(rows)
df.to_csv(out / "fl_roi_new_validation_localization.csv", index=False)

tracked = []
for (obj, grp), sub in df.groupby(["objective", "group"]):
    sub = sub.sort_values("frame").copy()
    t0, t1 = track_unordered_diagonal_angles(sub["theta1"].to_numpy(float),
                                             sub["theta2"].to_numpy(float))
    sub["t0"] = t0
    sub["t1"] = t1
    sub["theta_joint"] = (t0 + t1) / 2.0
    tracked.append(sub)
tr = pd.concat(tracked, ignore_index=True)
tr.to_csv(out / "fl_roi_new_validation_tracked.csv", index=False)

for (obj, grp), sub in tr.groupby(["objective", "group"]):
    sub = sub.sort_values("frame")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(sub.frame, sub.dmean, ".-", ms=4)
    axes[0].set_ylabel("d_mean (px)"); axes[0].grid(alpha=.3)
    axes[1].plot(sub.frame, sub.theta_joint, ".-", ms=4)
    axes[1].set_ylabel("theta_joint (deg)"); axes[1].set_xlabel("frame (1 um step)")
    axes[1].grid(alpha=.3)
    fig.suptitle(f"{obj} new validation ROI/{grp}")
    fig.tight_layout()
    fig.savefig(out / f"{obj}_new_validation_ROI{grp}_curves.png", dpi=170)
    plt.close(fig)

print("rows", len(df), "success", df.success.mean())
print(tr[["objective", "group", "frame", "dmean", "theta_joint"]].head(10).to_string(index=False))
print("saved", out / "fl_roi_new_validation_tracked.csv")
