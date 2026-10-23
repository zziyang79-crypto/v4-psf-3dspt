"""Minimal four-spot -> theta -> z quickstart.

Example:
    python examples/quickstart_decode.py \
        --stack data/example_roi/20x_ROI2_2-1.tif \
        --objective 20x_0.45NA_air \
        --out outputs/quickstart_z.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from v4psf.localization_methods import localize_four_spots
from v4psf.run_blur_angle_validation import track_unordered_diagonal_angles

ROI_RADIUS = {"20x_0.45NA_air": 12, "40x_0.6NA_air": 16, "60x_1.0NA_water": 16}
ROOT = Path(__file__).resolve().parents[1]


def line_angle(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def decode(stack: np.ndarray, objective: str, model_path: Path, sigma: float = 2.4):
    rows = []
    for i, img in enumerate(stack):
        im = ndimage.gaussian_filter(img.astype(np.float32), sigma)
        res = localize_four_spots(im, objective, "phasor",
                                  params={"roi_radius": ROI_RADIUS[objective]})
        if not res.success:
            rows.append({"frame": i, "success": False, "dmean": np.nan,
                         "theta1": np.nan, "theta2": np.nan})
            continue
        pts = np.array([[s.x, s.y] for s in res.spots])
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        g = pts[np.argsort(ang)]
        d1 = float(np.linalg.norm(g[0] - g[2]))
        d2 = float(np.linalg.norm(g[1] - g[3]))
        rows.append({"frame": i, "success": True, "dmean": (d1 + d2) / 2,
                     "theta1": line_angle(g[0], g[2]),
                     "theta2": line_angle(g[1], g[3])})
    df = pd.DataFrame(rows)
    ok = df.success.to_numpy(bool)
    t0 = np.full(len(df), np.nan)
    t1 = np.full(len(df), np.nan)
    if ok.any():
        t0[ok], t1[ok] = track_unordered_diagonal_angles(
            df.loc[ok, "theta1"].to_numpy(float),
            df.loc[ok, "theta2"].to_numpy(float),
        )
    df["theta_joint"] = (t0 + t1) / 2.0

    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    spline = model["models"][objective]["angle_spline"]
    pchip = PchipInterpolator(spline["knots_x_theta_deg"],
                              spline["knots_y_z_corrected_relative_um"],
                              extrapolate=True)
    df["z_um"] = np.where(df.theta_joint.notna(), pchip(df.theta_joint.fillna(0)), np.nan)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--objective", default="20x_0.45NA_air",
                    choices=sorted(ROI_RADIUS))
    ap.add_argument("--model", default=str(ROOT / "models" / "phasor_angle_tracked_models_fluorescence.json"))
    ap.add_argument("--sigma", type=float, default=2.4)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "quickstart_z.csv"))
    args = ap.parse_args()

    stack = tifffile.imread(args.stack)
    df = decode(stack, args.objective, Path(args.model), args.sigma)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} frames)")


if __name__ == "__main__":
    main()
