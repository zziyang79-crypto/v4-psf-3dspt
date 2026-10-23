"""Poisson-Fisher CRLB for simulated PSF stacks.

Usage:
  python crlb.py --stack <npz> --pixel_um 0.1125 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def crlb_table(stack: np.ndarray, depths_um: np.ndarray, pixel_um: float,
               n_photons: float = 12000.0, background: float = 10.0) -> pd.DataFrame:
    stack = np.asarray(stack, dtype=np.float64)
    depths_um = np.asarray(depths_um, dtype=np.float64)
    sums = stack.sum(axis=(1, 2), keepdims=True)
    stack = np.divide(stack, sums, out=np.zeros_like(stack), where=sums > 0)
    dz_um = float(np.median(np.diff(depths_um)))
    rows = []
    for i in range(1, len(stack) - 1):
        h = stack[i]
        dz = (stack[i + 1] - stack[i - 1]) / 2.0
        dy, dx = np.gradient(h)
        nu = n_photons * h + background
        mask = nu > 1e-12
        grad = np.stack([
            n_photons * dx[mask],
            n_photons * dy[mask],
            n_photons * dz[mask],
            h[mask],
            np.ones(int(mask.sum())),
        ], axis=0)
        fim = (grad / nu[mask]) @ grad.T
        fim += np.eye(5) * 1e-15
        try:
            inv = np.linalg.inv(fim)
            sd = np.sqrt(np.maximum(np.diag(inv)[:3], 0.0))
            cx = sd[0] * pixel_um * 1000.0
            cy = sd[1] * pixel_um * 1000.0
            cz = sd[2] * dz_um * 1000.0
            rows.append({
                "z_um": float(depths_um[i]),
                "CRLB_X_nm": cx, "CRLB_Y_nm": cy, "CRLB_Z_nm": cz,
                "CRLB_3D_nm": math.sqrt(cx * cx + cy * cy + cz * cz),
            })
        except np.linalg.LinAlgError:
            rows.append({"z_um": float(depths_um[i]),
                         "CRLB_X_nm": np.nan, "CRLB_Y_nm": np.nan,
                         "CRLB_Z_nm": np.nan, "CRLB_3D_nm": np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--pixel_um", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=float, default=12000.0)
    ap.add_argument("--b", type=float, default=10.0)
    ap.add_argument("--threshold_nm", type=float, default=100.0)
    args = ap.parse_args()

    data = np.load(args.stack)
    stack = data["stack"]
    depths = data["depths_um"]
    meta = {}
    if "metadata" in data:
        try:
            meta = json.loads(str(data["metadata"].item()))
        except Exception:
            pass
    df = crlb_table(stack, depths, args.pixel_um, args.n, args.b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.stack).stem
    df.to_csv(out / f"{stem}_crlb.csv", index=False)
    valid = df.dropna(subset=["CRLB_Z_nm"])
    usable = df[df["CRLB_Z_nm"] <= args.threshold_nm]
    summary = {
        "stack": str(args.stack),
        "pixel_um": args.pixel_um,
        "n_photons": args.n,
        "background": args.b,
        "n_slices": int(len(df)),
        "crlb_z_min_nm": float(valid["CRLB_Z_nm"].min()),
        "crlb_z_median_nm": float(valid["CRLB_Z_nm"].median()),
        "crlb_z_max_nm": float(valid["CRLB_Z_nm"].max()),
        "usable_threshold_nm": args.threshold_nm,
        "usable_z_min_um": float(usable["z_um"].min()) if len(usable) else None,
        "usable_z_max_um": float(usable["z_um"].max()) if len(usable) else None,
        "metadata": meta,
    }
    (out / f"{stem}_crlb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
