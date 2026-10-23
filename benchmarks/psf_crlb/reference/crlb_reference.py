from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.special import j0


WORKSPACE = Path(__file__).resolve().parents[1]
GENERATED = WORKSPACE / "generated_psf_50nm"
RESULTS = WORKSPACE / "results"
EXCEL_PATH = RESULTS / "CRLB_20x40x60_full_50nm.xlsx"
LOG_PATH = RESULTS / "crlb_full_50nm_log.json"

N_PHOTONS = 12000.0
B_BACKGROUND = 10.0
CAMERA_PIXEL_PITCH_NM = 4500.0
WAVELENGTH_NM = 550.0
Z_STEP_UM = 0.05


@dataclass(frozen=True)
class ObjectiveConfig:
    key: str
    sheet_prefix: str
    label: str
    magnification: float
    na: float
    immersion_n: float
    zmin_um: float
    zmax_um: float

    @property
    def pixel_size_xy_nm(self) -> float:
        return CAMERA_PIXEL_PITCH_NM / self.magnification

    @property
    def depths_um(self) -> np.ndarray:
        return np.round(np.arange(self.zmin_um, self.zmax_um + Z_STEP_UM / 2.0, Z_STEP_UM), 6)


CONFIGS = [
    ObjectiveConfig("20x_0.45NA_air", "20x", "20X 0.45NA air", 20.0, 0.45, 1.0, -10.0, 10.0),
    ObjectiveConfig("40x_0.6NA_air", "40x", "40X 0.6NA air", 40.0, 0.60, 1.0, -5.0, 5.0),
    ObjectiveConfig("60x_1.0NA_water", "60x_water", "60X 1.0NA water", 60.0, 1.0, 1.333, -2.5, 2.5),
]


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    stack = stack.astype(np.float64, copy=False)
    sums = stack.sum(axis=(1, 2), keepdims=True)
    out = np.zeros_like(stack, dtype=np.float64)
    np.divide(stack, sums, out=out, where=sums > 0)
    return out


def scalar_widefield_psf_stack(cfg: ObjectiveConfig, depths_um: np.ndarray) -> np.ndarray:
    pixel_nm = cfg.pixel_size_xy_nm
    zmax_nm = float(np.max(np.abs(depths_um)) * 1000.0)
    geometric_half_nm = zmax_nm * cfg.na / max(cfg.immersion_n, 1e-9) * 1.5 + 3000.0
    half_width_nm = max(12000.0, geometric_half_nm)
    half_pixels = int(math.ceil(half_width_nm / pixel_nm))
    size = 2 * half_pixels + 1

    coords = (np.arange(size, dtype=np.float64) - half_pixels) * pixel_nm
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)

    radial_step_nm = max(pixel_nm / 3.0, 10.0)
    radial_grid = np.arange(0.0, float(radius.max()) + radial_step_nm, radial_step_nm)

    rho = np.linspace(0.0, 1.0, 768, dtype=np.float64)
    sin_theta = np.clip((cfg.na / cfg.immersion_n) * rho, 0.0, 1.0 - 1e-12)
    cos_theta = np.sqrt(1.0 - sin_theta * sin_theta)
    pupil_weight = rho
    kr_rho = (2.0 * np.pi / WAVELENGTH_NM) * cfg.na * np.outer(radial_grid, rho)
    bessel = j0(kr_rho)

    stack = np.empty((len(depths_um), size, size), dtype=np.float32)
    for zi, z_um in enumerate(depths_um):
        z_nm = z_um * 1000.0
        phase = np.exp(1j * (2.0 * np.pi / WAVELENGTH_NM) * cfg.immersion_n * z_nm * (cos_theta - 1.0))
        integrand = bessel * (phase * pupil_weight)[None, :]
        amp = np.trapz(integrand, rho, axis=1)
        radial_intensity = np.abs(amp) ** 2
        img = np.interp(radius.ravel(), radial_grid, radial_intensity).reshape(radius.shape)
        img = np.maximum(img, 0.0)
        s = img.sum()
        stack[zi] = (img / s).astype(np.float32) if s > 0 else img.astype(np.float32)
    return stack


def list_chunk_files(cfg: ObjectiveConfig) -> list[Path]:
    chunk_dir = GENERATED / cfg.key / "lfpsf_chunks"
    files = sorted(chunk_dir.glob("chunk_*.mat"))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    return files


def _mat_ref_at(dataset, idx: int):
    shape = dataset.shape
    if dataset.ndim == 3:
        if shape[0] > 1 and shape[1] == 1 and shape[2] == 1:
            return dataset[idx, 0, 0]
        if shape[2] > 1 and shape[0] == 1 and shape[1] == 1:
            return dataset[0, 0, idx]
    if dataset.ndim == 2:
        if shape[0] == 1 and shape[1] == 1:
            return dataset[0, 0]
        if shape[0] > 1 and shape[1] == 1:
            return dataset[idx, 0]
        if shape[1] > 1 and shape[0] == 1:
            return dataset[0, idx]
    raise ValueError(f"Unsupported MATLAB cell shape: {shape}")


def _mat_cell_length(dataset) -> int:
    return max(dataset.shape)


def _load_depths(mat_path: Path) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        depths = np.array(f["depths_um"]).astype(np.float64).ravel()
    return depths


def _aligned_depths_for_chunk(path: Path, H, depths: np.ndarray) -> np.ndarray:
    n_slices = _mat_cell_length(H)
    if n_slices == len(depths):
        return depths
    if n_slices < len(depths):
        return depths[:n_slices]
    raise ValueError(f"{path} has more H slices ({n_slices}) than saved depths ({len(depths)})")


def compute_union_bbox(cfg: ObjectiveConfig, chunk_files: list[Path]) -> tuple[int, int, int, int]:
    y0 = None
    y1 = None
    x0 = None
    x1 = None
    for path in chunk_files:
        with h5py.File(path, "r") as f:
            H = f["H"]
            for idx in range(_mat_cell_length(H)):
                ref = _mat_ref_at(H, idx)
                grp = f[ref]
                ir = np.array(grp["ir"]).ravel().astype(int)
                jc = np.array(grp["jc"]).ravel().astype(int)
                if ir.size == 0:
                    continue
                col_counts = np.diff(jc)
                nz_cols = np.flatnonzero(col_counts > 0)
                if nz_cols.size == 0:
                    continue
                row_min = int(ir.min())
                row_max = int(ir.max())
                col_min = int(nz_cols.min())
                col_max = int(nz_cols.max())
                y0 = row_min if y0 is None else min(y0, row_min)
                y1 = row_max if y1 is None else max(y1, row_max)
                x0 = col_min if x0 is None else min(x0, col_min)
                x1 = col_max if x1 is None else max(x1, col_max)
    if y0 is None:
        raise RuntimeError(f"No nonzero LFPSF support found for {cfg.label}")
    return y0, y1, x0, x1


def load_lfpsf_stack(cfg: ObjectiveConfig, chunk_files: list[Path], bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = bbox
    out_h = y1 - y0 + 1
    out_w = x1 - x0 + 1
    slices = []
    depths_all = []
    for path in chunk_files:
        depths = _load_depths(path)
        with h5py.File(path, "r") as f:
            H = f["H"]
            depths = _aligned_depths_for_chunk(path, H, depths)
            for idx in range(_mat_cell_length(H)):
                ref = _mat_ref_at(H, idx)
                grp = f[ref]
                data = np.array(grp["data"]).ravel()
                ir = np.array(grp["ir"]).ravel().astype(int)
                jc = np.array(grp["jc"]).ravel().astype(int)
                nr = int(ir.max()) + 1 if ir.size > 0 else 0
                nc = len(jc) - 1
                if nr == 0 or nc == 0:
                    cropped = np.zeros((out_h, out_w), dtype=np.float32)
                else:
                    M = csc_matrix((data, ir, jc), shape=(nr, nc))
                    cropped = np.zeros((out_h, out_w), dtype=np.float32)
                    yy0 = max(y0, 0)
                    yy1 = min(y1, nr - 1)
                    xx0 = max(x0, 0)
                    xx1 = min(x1, nc - 1)
                    if yy0 <= yy1 and xx0 <= xx1:
                        sub = M[yy0 : yy1 + 1, xx0 : xx1 + 1].toarray().astype(np.float32)
                        cropped[(yy0 - y0) : (yy1 - y0 + 1), (xx0 - x0) : (xx1 - x0 + 1)] = sub
                slices.append(cropped)
                depths_all.append(float(depths[idx]))
    depths_arr = np.array(depths_all, dtype=np.float64)
    order = np.argsort(depths_arr)
    stack = np.stack([slices[i] for i in order], axis=0)
    depths_sorted = depths_arr[order]
    return normalize_stack(stack), depths_sorted


def calc_crlb_dataframe(stack: np.ndarray, depths_um: np.ndarray, cfg: ObjectiveConfig) -> tuple[pd.DataFrame, int]:
    z_step_nm = float(np.median(np.diff(depths_um)) * 1000.0)
    rows = []
    failures = 0
    for i in range(1, stack.shape[0] - 1):
        h_prev = stack[i - 1]
        h_curr = stack[i]
        h_next = stack[i + 1]
        if float(h_curr.max()) < 1e-14:
            failures += 1
            rows.append(_nan_row(depths_um[i]))
            continue

        dz = (h_next - h_prev) / 2.0
        dy, dx = np.gradient(h_curr)
        nu = N_PHOTONS * h_curr + B_BACKGROUND
        mask = nu > 1e-12
        inv_nu = 1.0 / nu[mask]
        grad = np.stack(
            [
                N_PHOTONS * dx[mask],
                N_PHOTONS * dy[mask],
                N_PHOTONS * dz[mask],
                h_curr[mask],
                np.ones(int(mask.sum()), dtype=np.float64),
            ],
            axis=0,
        )
        fim = (grad * inv_nu) @ grad.T
        fim += np.eye(5) * 1e-15
        try:
            fim_inv = np.linalg.inv(fim)
            raw_std = np.sqrt(np.maximum(np.diag(fim_inv)[:3], 0.0))
            crlb_x_nm = raw_std[0] * cfg.pixel_size_xy_nm
            crlb_y_nm = raw_std[1] * cfg.pixel_size_xy_nm
            crlb_z_nm = raw_std[2] * z_step_nm
            var_x = crlb_x_nm * crlb_x_nm
            var_y = crlb_y_nm * crlb_y_nm
            var_z = crlb_z_nm * crlb_z_nm
            var_3d = var_x + var_y + var_z
            rows.append(
                {
                    "Sampling": "full_50nm",
                    "Z_um": float(depths_um[i]),
                    "CRLB_X_nm": crlb_x_nm,
                    "CRLB_Y_nm": crlb_y_nm,
                    "CRLB_Z_nm": crlb_z_nm,
                    "Var_X_nm2": var_x,
                    "Var_Y_nm2": var_y,
                    "Var_Z_nm2": var_z,
                    "Var_3D_nm2": var_3d,
                    "CRLB_3D_nm": math.sqrt(var_3d),
                }
            )
        except np.linalg.LinAlgError:
            failures += 1
            rows.append(_nan_row(depths_um[i]))
    return pd.DataFrame(rows), failures


def _nan_row(z_um: float) -> dict:
    return {
        "Sampling": "full_50nm",
        "Z_um": float(z_um),
        "CRLB_X_nm": np.nan,
        "CRLB_Y_nm": np.nan,
        "CRLB_Z_nm": np.nan,
        "Var_X_nm2": np.nan,
        "Var_Y_nm2": np.nan,
        "Var_Z_nm2": np.nan,
        "Var_3D_nm2": np.nan,
        "CRLB_3D_nm": np.nan,
    }


def summarize(df: pd.DataFrame, cfg: ObjectiveConfig, psf_type: str, failures: int, method: str) -> dict:
    row = {
        "Objective": cfg.label,
        "PSF_Type": psf_type,
        "Sampling": "full_50nm",
        "M": cfg.magnification,
        "NA": cfg.na,
        "Immersion_n": cfg.immersion_n,
        "Pixel_Size_XY_nm": cfg.pixel_size_xy_nm,
        "Photon_Count": N_PHOTONS,
        "Background_photons_per_pixel": B_BACKGROUND,
        "Z_Step_nm": 50.0,
        "Z_points": int(len(df)),
        "Failed_FIM_points": int(failures),
        "Method": method,
    }
    for col in ["CRLB_X_nm", "CRLB_Y_nm", "CRLB_Z_nm", "CRLB_3D_nm"]:
        values = df[col].to_numpy(dtype=np.float64)
        if np.all(np.isnan(values)):
            row[f"Min_{col}"] = np.nan
            row[f"Z_at_Min_{col}_um"] = np.nan
        else:
            idx = int(np.nanargmin(values))
            row[f"Min_{col}"] = float(values[idx])
            row[f"Z_at_Min_{col}_um"] = float(df.iloc[idx]["Z_um"])
    return row


def save_stack_npz(path: Path, stack: np.ndarray, depths_um: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stack=stack.astype(np.float32), depths_um=depths_um, metadata=json.dumps(metadata, ensure_ascii=False))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pd.DataFrame] = {}
    summary_rows = []
    log: dict[str, object] = {
        "workspace": str(WORKSPACE),
        "excel_path": str(EXCEL_PATH),
        "z_step_nm": 50.0,
        "constants": {
            "N_PHOTONS": N_PHOTONS,
            "B_BACKGROUND": B_BACKGROUND,
            "CAMERA_PIXEL_PITCH_NM": CAMERA_PIXEL_PITCH_NM,
            "WAVELENGTH_NM": WAVELENGTH_NM,
        },
        "configs": {},
    }

    for cfg in CONFIGS:
        print(f"\n=== {cfg.label} ===")
        chunk_files = list_chunk_files(cfg)
        bbox = compute_union_bbox(cfg, chunk_files)
        print("Bounding box:", bbox)
        lfpsf_stack, depths = load_lfpsf_stack(cfg, chunk_files, bbox)
        if len(depths) != len(cfg.depths_um):
            raise ValueError(f"{cfg.label}: expected {len(cfg.depths_um)} depths, got {len(depths)}")

        std_stack = scalar_widefield_psf_stack(cfg, cfg.depths_um)

        save_stack_npz(
            GENERATED / cfg.key / "lfpsf_full_50nm_cropped.npz",
            lfpsf_stack,
            depths,
            {"bbox_y0_y1_x0_x1": list(bbox), "source_chunks": [str(p) for p in chunk_files]},
        )
        save_stack_npz(
            GENERATED / cfg.key / "standard_full_50nm.npz",
            std_stack,
            cfg.depths_um,
            {"method": "scalar diffraction widefield PSF"},
        )

        lf_df, lf_fail = calc_crlb_dataframe(lfpsf_stack, depths, cfg)
        std_df, std_fail = calc_crlb_dataframe(std_stack, cfg.depths_um, cfg)

        lf_sheet = f"{cfg.sheet_prefix}_lfpsf"
        std_sheet = f"{cfg.sheet_prefix}_std"
        sheets[lf_sheet] = lf_df
        sheets[std_sheet] = std_df
        lf_df.to_csv(RESULTS / f"{lf_sheet}_full_50nm.csv", index=False)
        std_df.to_csv(RESULTS / f"{std_sheet}_full_50nm.csv", index=False)

        summary_rows.append(summarize(lf_df, cfg, "LFPSF", lf_fail, "true LFPSF recomputed in 50 nm steps from MATLAB chunks"))
        summary_rows.append(summarize(std_df, cfg, "Standard widefield", std_fail, "scalar diffraction generated in 50 nm steps"))

        log["configs"][cfg.key] = {
            "bbox_y0_y1_x0_x1": list(bbox),
            "chunk_count": len(chunk_files),
            "depth_count": len(depths),
            "lfpsf_shape": list(lfpsf_stack.shape),
            "std_shape": list(std_stack.shape),
        }

        del lfpsf_stack, std_stack, lf_df, std_df

    summary_df = pd.DataFrame(summary_rows)
    ordered_sheets = [
        "20x_std",
        "20x_lfpsf",
        "40x_std",
        "40x_lfpsf",
        "60x_water_std",
        "60x_water_lfpsf",
    ]
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for sheet in ordered_sheets:
            sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)

    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {EXCEL_PATH}")


if __name__ == "__main__":
    main()
