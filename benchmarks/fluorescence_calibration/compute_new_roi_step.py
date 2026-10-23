"""Step recovery on the updated independent-validation ROIs.

New validation data:
  20x: ROI/1 (updated; ROI/3 is absent)
  40x: ROI/3
  60x: ROI/3
Calibration models are unchanged (20x ROI/2, 40x ROI/1, 60x ROI/1).
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
TRACK = Path(os.environ.get("V4PSF_NEW_TRACKED_CSV", BASE / "outputs" / "fluorescence_calibration" / "fl_roi_new_validation_tracked.csv"))
MODEL = Path(os.environ.get("V4PSF_FLUORESCENCE_MODEL", BASE / "models" / "phasor_angle_tracked_models_fluorescence.json"))
OUT = Path(os.environ.get("V4PSF_RESULTS_DIR", BASE / "results"))
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(TRACK)
model_json = json.loads(MODEL.read_text(encoding="utf-8"))

# objective: (group, window, branch shift, label)
CFG = {
    "20x_0.45NA_air": ("1", (10, 32), 0.0, "20x"),
    "40x_0.6NA_air": ("3", (13, 23), -90.0, "40x"),
    "60x_1.0NA_water": ("3", (8, 16), -90.0, "60x"),
}

summary = []
pred_rows = []
for obj, (grp, window, shift, label) in CFG.items():
    m = model_json["models"][obj]["angle_spline"]
    kx = np.asarray(m["knots_x_theta_deg"], float)
    ky = np.asarray(m["knots_y_z_corrected_relative_um"], float)
    pchip = PchipInterpolator(kx, ky, extrapolate=True)
    tmin, tmax = float(kx.min()), float(kx.max())

    s = d[(d.objective == obj) & (d.group.astype(str) == grp)].sort_values("frame").copy()
    s = s[(s.frame >= window[0]) & (s.frame <= window[1]) & s.success.astype(bool)].copy()
    s["theta_cal"] = s.theta_joint.to_numpy(float) + shift
    s["z_pred_um"] = pchip(s.theta_cal.to_numpy(float))
    s["in_domain"] = (s.theta_cal >= tmin) & (s.theta_cal <= tmax)
    s["objective_label"] = label
    pred_rows.append(s[["objective_label", "frame", "theta_joint", "theta_cal",
                        "z_pred_um", "in_domain"]])

    errs = []
    for i in range(len(s) - 1):
        dframe = int(s.frame.iloc[i + 1] - s.frame.iloc[i])
        if dframe != 1:
            continue
        dz = float(s.z_pred_um.iloc[i + 1] - s.z_pred_um.iloc[i])
        errs.append(dz - 1.0)
    errs = np.asarray(errs, float)
    summary.append({
        "objective": obj, "group": grp, "window": f"{window[0]}-{window[1]}",
        "branch_shift_deg": shift, "n_points": len(s), "n_steps": len(errs),
        "domain_fraction": float(s.in_domain.mean()),
        "step_rmse_nm": float(np.sqrt(np.mean(errs ** 2)) * 1000) if len(errs) else np.nan,
        "step_mae_nm": float(np.mean(np.abs(errs)) * 1000) if len(errs) else np.nan,
    })

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=170)
    axes[0].plot(s.frame, s.theta_cal, "o-", ms=3)
    axes[0].axhline(tmin, color="0.6", lw=.7, ls="--")
    axes[0].axhline(tmax, color="0.6", lw=.7, ls="--")
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("theta + branch shift (deg)")
    axes[0].set_title(f"{label} new validation: angle", fontsize=8)
    axes[0].grid(alpha=.25)
    axes[1].plot(s.frame, s.z_pred_um, "o-", ms=3, label="predicted z")
    axes[1].plot(s.frame, s.frame - window[0], "k--", lw=.8, label="expected (1 um/frame)")
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("z (um)")
    axes[1].set_title(f"{label}: step RMSE {summary[-1]['step_rmse_nm']:.1f} nm", fontsize=8)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / f"new_validation_{label}_step.png", dpi=180)
    plt.close(fig)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "new_validation_step_recovery_summary.csv", index=False, encoding="utf-8-sig")
pred_df = pd.concat(pred_rows, ignore_index=True)
pred_df.to_csv(OUT / "new_validation_step_recovery_predictions.csv", index=False, encoding="utf-8-sig")
print(summary_df.round(2).to_string(index=False))
