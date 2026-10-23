import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from v4psf.run_blur_angle_validation import track_unordered_diagonal_angles

base = Path(os.environ.get("V4PSF_OUT_DIR", REPO_ROOT / "outputs" / "fluorescence_calibration"))
df = pd.read_csv(base / "fl_roi_localization.csv")

rows = []
for (obj, grp, sigma), sub in df.groupby(["objective", "group", "sigma"]):
    sub = sub.sort_values("frame").copy()
    t0, t1 = track_unordered_diagonal_angles(sub["theta1"].to_numpy(float), sub["theta2"].to_numpy(float))
    sub["t0"] = t0
    sub["t1"] = t1
    sub["theta_joint"] = (t0 + t1) / 2.0
    rows.append(sub)
out = pd.concat(rows, ignore_index=True)
out.to_csv(base / "fl_roi_tracked.csv", index=False)

for obj in out.objective.unique():
    for grp in ["1", "2"]:
        sub = out[(out.objective == obj) & (out.group == grp)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
        for sigma, s in sub.groupby("sigma"):
            s = s.sort_values("frame")
            axes[0].plot(s.frame, s.dmean, ".-", label=f"sigma={sigma}")
            axes[1].plot(s.frame, s.theta_joint, ".-", label=f"sigma={sigma}")
            axes[2].plot(s.frame, s.t0, ".-", label=f"s{sigma} t0")
            axes[2].plot(s.frame, s.t1, ".--", label=f"s{sigma} t1", alpha=.5)
        axes[0].set_ylabel("d_mean (px)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
        axes[1].set_ylabel("theta_joint (deg)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
        axes[2].set_ylabel("tracked diag (deg)"); axes[2].legend(fontsize=5, ncol=2); axes[2].grid(alpha=.3)
        axes[2].set_xlabel("frame (1 um step)")
        fig.suptitle(f"{obj} ROI/{grp} tracked")
        fig.tight_layout()
        fig.savefig(base / f"{obj}_ROI{grp}_tracked.png", dpi=170)
        plt.close(fig)
print("saved tracked csv and plots")
