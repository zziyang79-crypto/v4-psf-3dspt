"""Four-subimage descriptor analysis for V4 PSF stacks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, label


def find_four_peaks(img: np.ndarray):
    for size, frac in [(11, 0.08), (7, 0.05), (5, 0.03)]:
        mx = maximum_filter(img, size=size)
        mask = (img == mx) & (img > img.max() * frac)
        lab, num = label(mask)
        pts = []
        for i in range(1, num + 1):
            yy, xx = np.where(lab == i)
            w = img[yy, xx]
            if w.sum() <= 0:
                continue
            pts.append((float((yy * w).sum() / w.sum()),
                        float((xx * w).sum() / w.sum()), float(w.sum())))
        pts.sort(key=lambda t: -t[2])
        if len(pts) >= 4:
            pts = pts[:4]
            return np.array([[y, x] for y, x, _ in pts]), True
    return np.zeros((0, 2)), False


def descriptors(img: np.ndarray):
    pts, ok = find_four_peaks(img)
    if not ok:
        return {"n_peaks": len(pts), "d_mean_px": np.nan, "theta_deg": np.nan}
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
    order = np.argsort(ang)
    p = pts[order]
    # Opposite pairs after circular ordering: (0,2) and (1,3).
    d0 = p[2] - p[0]
    d1 = p[3] - p[1]
    t0 = math.degrees(math.atan2(d0[0], d0[1]))
    t1 = math.degrees(math.atan2(d1[0], d1[1]))
    # Align the two diagonal estimates modulo 180 and average.
    t1 = t1 + 180.0 * round((t0 - (t1 - 90.0)) / 180.0)
    theta = 0.5 * (t0 + (t1 - 90.0))
    return {
        "n_peaks": 4,
        "d_mean_px": 0.5 * (float(np.hypot(*d0)) + float(np.hypot(*d1))),
        "theta_deg": theta,
        "x0": float(center[1]), "y0": float(center[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--crlb_csv", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = np.load(args.stack)
    stack = data["stack"].astype(np.float64)
    depths = data["depths_um"].astype(float)
    stack = np.divide(stack, stack.sum(axis=(1, 2), keepdims=True),
                      out=np.zeros_like(stack), where=stack.sum(axis=(1, 2), keepdims=True) > 0)
    rows = []
    for z, img in zip(depths, stack):
        d = descriptors(img)
        d["z_um"] = float(z)
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("z_um")
    # Unwrap theta over valid points.
    valid = df["n_peaks"].to_numpy() >= 4
    th = df["theta_deg"].to_numpy(dtype=float)
    unwrapped = np.full_like(th, np.nan)
    if valid.any():
        idx = np.where(valid)[0]
        vals = np.rad2deg(np.unwrap(np.deg2rad(th[idx] * 2.0))) / 2.0
        unwrapped[idx] = vals
    df["theta_unwrapped_deg"] = unwrapped
    if args.crlb_csv:
        c = pd.read_csv(args.crlb_csv)
        df = df.merge(c[["z_um", "CRLB_Z_nm"]], on="z_um", how="left")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.stack).stem
    df.to_csv(out / f"{stem}_descriptors.csv", index=False)
    print(df.to_string(index=False))
    print(json.dumps({"n": len(df), "valid4": int(valid.sum()),
                      "d_mean_median_px": float(df["d_mean_px"].median()),
                      "theta_range_deg": float(np.nanmax(unwrapped) - np.nanmin(unwrapped))}, indent=2))


if __name__ == "__main__":
    main()
