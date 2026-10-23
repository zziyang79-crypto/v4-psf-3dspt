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

root = Path(os.environ.get("V4PSF_FLUORESCENCE_ROOT", REPO_ROOT / "data" / "raw" / "fluorescence"))
out = Path(os.environ.get("V4PSF_OUT_DIR", REPO_ROOT / "outputs" / "fluorescence_calibration"))
out.mkdir(parents=True, exist_ok=True)

objs = {
    "20x_0.45NA_air": "20X0.45NA",
    "40x_0.6NA_air": "40X0.6NA",
    "60x_1.0NA_water": "60X1.0NA-W",
}
sigmas = [1.8, 2.4, 3.0]
roi_radius = {"20x_0.45NA_air": 12, "40x_0.6NA_air": 16, "60x_1.0NA_water": 16}

def line_angle(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)

rows = []
for obj_key, obj_dir in objs.items():
    for grp in ["1", "2"]:
        p = root / obj_dir / "ROI" / grp / f"{grp}-1.tif"
        stack = tifffile.imread(p)
        for sigma in sigmas:
            for i, img in enumerate(stack):
                im = img.astype(np.float32)
                if sigma > 0:
                    im = ndimage.gaussian_filter(im, sigma)
                res = localize_four_spots(im, obj_key, "phasor", params={"roi_radius": roi_radius[obj_key]})
                base = {"objective": obj_key, "group": grp, "frame": i, "sigma": sigma,
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
                d1 = float(np.linalg.norm(g[0] - g[2])); d2 = float(np.linalg.norm(g[1] - g[3]))
                a1 = line_angle(g[0], g[2]); a2 = line_angle(g[1], g[3])
                base.update({"d1": d1, "d2": d2, "dmean": (d1 + d2) / 2,
                             "theta1": a1, "theta2": a2, "cx": c[0], "cy": c[1], "msg": ""})
                rows.append(base)

df = pd.DataFrame(rows)
df.to_csv(out / "fl_roi_localization.csv", index=False)
print("rows", len(df), "success rate", df.success.mean())

# plots per objective/group
for obj_key in objs:
    for grp in ["1", "2"]:
        sub = df[(df.objective == obj_key) & (df.group == grp)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        for sigma in sigmas:
            s = sub[sub.sigma == sigma].sort_values("frame")
            axes[0].plot(s.frame, s.dmean, ".-", label=f"sigma={sigma}")
            axes[1].plot(s.frame, s.theta1, ".-", label=f"sigma={sigma} θ1")
            axes[1].plot(s.frame, s.theta2, ".--", label=f"sigma={sigma} θ2", alpha=0.6)
        axes[0].set_ylabel("d_mean (px)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
        axes[1].set_ylabel("diagonal angle (deg)"); axes[1].set_xlabel("frame (1 um step)")
        axes[1].legend(fontsize=6, ncol=2); axes[1].grid(alpha=.3)
        fig.suptitle(f"{obj_key} ROI/{grp}")
        fig.tight_layout()
        fig.savefig(out / f"{obj_key}_ROI{grp}_curves.png", dpi=170)
        plt.close(fig)
print("saved", out)
