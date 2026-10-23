import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

base = Path(__file__).resolve().parents[2]
src = Path(os.environ.get("V4PSF_TRACKED_CSV", base / "outputs" / "fluorescence_calibration" / "fl_roi_tracked.csv"))
out = Path(os.environ.get("V4PSF_MODEL_OUT", base / "results" / "fluorescence_calibration"))
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
df = df[df["success"].astype(bool)].copy()
SIGMA = 2.4

# direction: cal group, chk group, windows, zero frame, excluded duplicate-z frames (chk numbering).
# 20x ROI/1: 8->9 and 15->16 are ~2-um steps; 9/10 and 16/17 are duplicated-z pairs
# (stage double-step then skipped step). The excluded set is kept for the conservative
# clean-1-um-step subset; it is not evidence of optical saturation.
CFG = {
    "20x_0.45NA_air": {
        "cal_group": "2", "chk_group": "1", "cal": (9, 32), "chk": (9, 32),
        "alias": "20x_NA0.45_n1.0", "exclude": [9, 10, 16, 17],
    },
    "40x_0.6NA_air": {
        "cal_group": "1", "chk_group": "2", "cal": (12, 23), "chk": (13, 24),
        "alias": "40x_NA0.6_n1.0", "exclude": [],
    },
    "60x_1.0NA_water": {
        "cal_group": "1", "chk_group": "2", "cal": (7, 16), "chk": (7, 16),
        "alias": "60x_NA1.0_n1.333", "exclude": [],
    },
}


def fit_pchip(theta, z):
    theta = np.asarray(theta, float); z = np.asarray(z, float)
    m = np.isfinite(theta) & np.isfinite(z)
    theta, z = theta[m], z[m]
    if len(theta) < 4:
        return None, "too_few", np.array([]), np.array([])
    order = np.argsort(theta)
    g = pd.DataFrame({"t": theta[order], "z": z[order]}).groupby("t", as_index=False).mean()
    if len(g) < 4:
        return None, "too_few_unique", g["t"].to_numpy(), g["z"].to_numpy()
    return PchipInterpolator(g["t"], g["z"], extrapolate=True), "ok", g["t"].to_numpy(float), g["z"].to_numpy(float)


def loocv(theta, z):
    out = np.full(len(theta), np.nan)
    for i in range(len(theta)):
        tr = np.delete(np.arange(len(theta)), i)
        m, st, _, _ = fit_pchip(theta[tr], z[tr])
        if m is not None and st == "ok":
            out[i] = float(m(theta[i]))
    e = (out - z) * 1000
    e = e[np.isfinite(e)]
    return {"rmse_nm": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
            "mae_nm": float(np.mean(np.abs(e))) if len(e) else np.nan,
            "max_abs_nm": float(np.max(np.abs(e))) if len(e) else np.nan,
            "n": int(len(e))}


def step_metrics(frame, zp, ok, exclude=()):
    frame = np.asarray(frame, int); zp = np.asarray(zp, float); ok = np.asarray(ok, bool)
    errs_all, errs_f = [], []
    for i in range(len(frame) - 1):
        if frame[i+1] - frame[i] <= 0 or not (ok[i] and ok[i+1]):
            continue
        e = (zp[i+1] - zp[i]) - (frame[i+1] - frame[i])
        errs_all.append(e)
        if not ({int(frame[i]), int(frame[i+1])} & set(exclude)):
            errs_f.append(e)
    def stat(a):
        return (float(np.sqrt(np.mean(np.array(a)**2))) * 1000 if a else np.nan,
                float(np.mean(np.abs(np.array(a)))) * 1000 if a else np.nan,
                len(a))
    r_all, m_all, n_all = stat(errs_all)
    r_f, m_f, n_f = stat(errs_f)
    dirs = [zp[i+1]-zp[i] for i in range(len(frame)-1)
            if frame[i+1]-frame[i] > 0 and ok[i] and ok[i+1]]
    return {"rmse_all_nm": r_all, "mae_all_nm": m_all, "n_all": n_all,
            "rmse_filtered_nm": r_f, "mae_filtered_nm": m_f, "n_filtered": n_f,
            "direction": float(np.mean(np.array(dirs) > 0)) if dirs else np.nan}


def monotonic_fraction(y):
    d = np.diff(np.asarray(y, float))
    return float(max(np.mean(d >= 0), np.mean(d <= 0))) if len(d) else np.nan


def smoothness_rms(y):
    y = np.asarray(y, float)
    return float(np.sqrt(np.mean((y[2:] - 2*y[1:-1] + y[:-2])**2))) if len(y) >= 3 else np.nan


summary, models_json = [], {}
for obj, cfg in CFG.items():
    sub = df[(df["objective"] == obj) & (df["sigma"] == SIGMA)]
    cal = sub[(sub["group"].astype(str) == cfg["cal_group"]) & (sub["frame"] >= cfg["cal"][0]) & (sub["frame"] <= cfg["cal"][1])].sort_values("frame")
    chk = sub[(sub["group"].astype(str) == cfg["chk_group"]) & (sub["frame"] >= cfg["chk"][0]) & (sub["frame"] <= cfg["chk"][1])].sort_values("frame")
    theta = cal["theta_joint"].to_numpy(float)
    z = (cal["frame"].to_numpy(float) - cfg["cal"][0])
    model, status, kx, ky = fit_pchip(theta, z)
    loo = loocv(theta, z)
    tmin, tmax = float(kx.min()), float(kx.max())

    # choose branch shift maximizing in-domain coverage and minimizing step rmse
    best = None
    for k in [-2, -1, 0, 1, 2]:
        tc = chk["theta_joint"].to_numpy(float) + 90.0 * k
        ok = (tc >= tmin) & (tc <= tmax) & np.isfinite(tc)
        zp = model(tc)
        sm = step_metrics(chk["frame"].to_numpy(int), zp, ok, cfg["exclude"])
        frac = float(np.mean(ok))
        score = (1 - frac) * 100 + (sm["rmse_filtered_nm"] if np.isfinite(sm["rmse_filtered_nm"]) else 999)
        if best is None or score < best["score"]:
            best = {"k": k, "frac": frac, "score": score, "zp": zp, "ok": ok, **sm}

    row = {
        "objective": obj, "sigma_px": SIGMA, "cal_group": cfg["cal_group"], "chk_group": cfg["chk_group"],
        "cal_frames": f"{cfg['cal'][0]}-{cfg['cal'][1]}", "chk_frames": f"{cfg['chk'][0]}-{cfg['chk'][1]}",
        "zero_frame": cfg["cal"][0], "theta_min": tmin, "theta_max": tmax,
        "monotonic": monotonic_fraction(theta), "smooth_rms": smoothness_rms(theta),
        "loocv_rmse_nm": loo["rmse_nm"], "loocv_mae_nm": loo["mae_nm"],
        "branch_shift_deg": 90.0 * best["k"], "domain_frac": best["frac"],
        "indep_rmse_all_nm": best["rmse_all_nm"], "indep_n_all": best["n_all"],
        "indep_rmse_filtered_nm": best["rmse_filtered_nm"], "indep_n_filtered": best["n_filtered"],
        "indep_mae_filtered_nm": best["mae_filtered_nm"], "direction": best["direction"],
        "excluded_platform_frames": cfg["exclude"],
    }
    summary.append(row)

    models_json[obj] = {
        "display": obj,
        "objective_config": cfg["alias"],
        "localization_method": "phasor",
        "z_localization_method": "phasor_blur",
        "blur_sigma_px": SIGMA,
        "angle_signal": "tracked_simple_average",
        "channel": "fluorescence",
        "sample": "1 um fluorescent polystyrene microsphere",
        "angle_spline": {
            "primary_model": "joint_tracked_simple_average_monotonic_spline_fluorescence",
            "spline_kind": "pchip",
            "theta_input_column": "theta_tracked_simple_average_unwrapped_deg",
            "knots_x_theta_deg": [float(v) for v in kx],
            "knots_y_z_corrected_relative_um": [float(v) for v in ky],
            "domain_theta_deg": [tmin, tmax],
            "relative_zero_frame": int(cfg["cal"][0]),
            "allow_extrapolation_default": True,
        },
        "calibration_work_range_frames": {"start_frame": int(cfg["cal"][0]), "end_frame": int(cfg["cal"][1])},
        "independent_check_work_range_frames": {"start_frame": int(cfg["chk"][0]), "end_frame": int(cfg["chk"][1])},
        "validation_metrics": {
            "blur_sigma_px": SIGMA,
            "candidate": "tracked_simple_average",
            "calibration_start_frame": int(cfg["cal"][0]),
            "calibration_end_frame": int(cfg["cal"][1]),
            "independent_start_frame": int(cfg["chk"][0]),
            "independent_end_frame": int(cfg["chk"][1]),
            "real_model_status": status,
            "real_n_fit_points": int(len(kx)),
            "theta_monotonic_fraction": monotonic_fraction(theta),
            "theta_smoothness_rms_deg": smoothness_rms(theta),
            "calibration_loocv_rmse_nm": loo["rmse_nm"],
            "calibration_loocv_mae_nm": loo["mae_nm"],
            "independent_branch_shift_deg": 90.0 * best["k"],
            "independent_theta_domain_fraction": best["frac"],
            "independent_step_rmse_nm": best["rmse_filtered_nm"],
            "independent_step_mae_nm": best["mae_filtered_nm"],
            "independent_n_steps": int(best["n_filtered"]),
            "independent_step_rmse_all_nm": best["rmse_all_nm"],
            "independent_n_steps_all": int(best["n_all"]),
            "platform_frames_excluded": cfg["exclude"],
            "independent_direction_consistency": best["direction"],
        },
    }

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(cal["theta_joint"], cal["frame"] - cfg["cal"][0], "o", ms=4, label=f"ROI/{cfg['cal_group']} calibration")
    xx = np.linspace(tmin, tmax, 300)
    axes[0].plot(xx, model(xx), "-", label="PCHIP")
    axes[0].set_xlabel("theta_joint (deg)"); axes[0].set_ylabel("relative z (um)")
    axes[0].set_title(f"{obj}: calibration (sigma={SIGMA})"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].plot(chk["frame"], best["zp"], ".-", label="ROI/independent predicted z")
    axes[1].plot(chk["frame"], chk["frame"] - cfg["chk"][0], "k--", label="expected (1 um/frame)")
    for f in cfg["exclude"]:
        axes[1].axvline(f, color="gray", alpha=.3, lw=0.8)
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("z (um)")
    axes[1].set_title(f"independent: {best['rmse_filtered_nm']:.1f} nm (filtered)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / f"calibration_{obj}.png", dpi=170)
    plt.close(fig)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(out / "fluorescence_calibration_summary.csv", index=False)
print(summary_df.round(2).to_string(index=False))

payload = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "model_name": "phasor_joint_tracked_average_real_spline_fluorescence",
    "localization_method": "phasor",
    "z_metric": "joint_tracked_simple_average_angle_primary",
    "description": "Fluorescence-channel V4-PSF axial calibration using 1 um fluorescent polystyrene microspheres. theta_joint=(tracked_diag0+tracked_diag1)/2 after Gaussian blur sigma=2.4 px; independent ROI for step recovery. Not interchangeable with the scattering-channel model.",
    "source_files": {
        "raw_root": "../data/raw/fluorescence",
        "intermediate": str(src),
        "summary": str(out / "fluorescence_calibration_summary.csv"),
    },
    "objective_aliases": {
        "20x": "20x_0.45NA_air", "20x_0.45NA_air": "20x_0.45NA_air", "20x_NA0.45_n1.0": "20x_0.45NA_air",
        "40x": "40x_0.6NA_air", "40x_0.6NA_air": "40x_0.6NA_air", "40x_NA0.6_n1.0": "40x_0.6NA_air",
        "60x": "60x_1.0NA_water", "60x_1.0NA_water": "60x_1.0NA_water", "60x_water": "60x_1.0NA_water",
        "60x_NA1.0_n1.333": "60x_1.0NA_water",
    },
    "models": models_json,
}
with open(out / "phasor_angle_tracked_models_fluorescence.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print("saved", out)
