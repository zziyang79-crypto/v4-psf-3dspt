from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qc_v2_credibility_shrinkage as qc_v2


V4 = BASE / "qc_v4_spot_quality_trigger"
DATA = V4 / "data"
REPORTS = V4 / "reports"


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "pure_angle": "z_pred_pure_angle",
        "current_hard_qc": "z_pred_hybrid_angle_length_qc",
        "qc_v2_credibility_shrinkage": "z_pred_hybrid_qc_v2",
        "qc_v4_continuous_spot_shrinkage": "z_pred_hybrid_qc_v4",
        "qc_v4_gate_v2_correction": "z_pred_hybrid_qc_v4_gate_v2",
        "pure_length": "z_pred_pure_length",
    }
    return qc_v2.add_stack_offset_errors(df, methods)


def metric_row(df: pd.DataFrame, method: str, split: str) -> dict[str, Any]:
    err = pd.to_numeric(df[f"error_{method}_offset_aligned_um"], errors="coerce").to_numpy(dtype=float)
    err = err[np.isfinite(err)]
    abs_err = np.abs(err)
    return {
        "split": split,
        "method": method,
        "n_frames": int(len(err)),
        "rmse_um": float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "mae_um": float(np.mean(abs_err)) if len(err) else np.nan,
        "p95_abs_error_um": float(np.percentile(abs_err, 95)) if len(err) else np.nan,
        "p99_abs_error_um": float(np.percentile(abs_err, 99)) if len(err) else np.nan,
    }


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "pure_angle",
        "current_hard_qc",
        "qc_v2_credibility_shrinkage",
        "qc_v4_continuous_spot_shrinkage",
        "qc_v4_gate_v2_correction",
        "pure_length",
    ]
    rows = []
    for split, part in [
        ("train_seed_0_49", df[df["seed"].astype(int) < 50]),
        ("test_seed_50_99", df[df["seed"].astype(int) >= 50]),
        ("all", df),
    ]:
        for method in methods:
            rows.append(metric_row(part, method, split))
    return pd.DataFrame(rows)


def event_summary(df: pd.DataFrame) -> pd.DataFrame:
    q = df[df["hybrid_source"].astype(str).eq("angle_length_trend_interpolated")].copy()
    accepted = bool_series(q["qc_v4_trigger_accepted"])
    geom = bool_series(q["geometry_disturbance_applied"])
    for method in [
        "current_hard_qc",
        "qc_v2_credibility_shrinkage",
        "qc_v4_continuous_spot_shrinkage",
        "qc_v4_gate_v2_correction",
    ]:
        q[f"{method}_gain_um"] = q["abs_error_pure_angle_offset_aligned_um"] - q[
            f"abs_error_{method}_offset_aligned_um"
        ]
    subsets = [
        ("all_qc_candidates", q),
        ("v4_accepted_candidates", q[accepted]),
        ("v4_rejected_candidates", q[~accepted]),
        ("candidates_with_geometry", q[geom]),
        ("candidates_without_geometry", q[~geom]),
    ]
    rows = []
    for label, part in subsets:
        row: dict[str, Any] = {
            "subset": label,
            "n": int(len(part)),
            "v4_accepted": int(bool_series(part["qc_v4_trigger_accepted"]).sum()) if len(part) else 0,
            "geometry_fraction": float(bool_series(part["geometry_disturbance_applied"]).mean()) if len(part) else np.nan,
        }
        for method in [
            "current_hard_qc",
            "qc_v2_credibility_shrinkage",
            "qc_v4_continuous_spot_shrinkage",
            "qc_v4_gate_v2_correction",
        ]:
            gain = pd.to_numeric(part[f"{method}_gain_um"], errors="coerce")
            row[f"{method}_positive"] = int((gain > 0).sum())
            row[f"{method}_negative"] = int((gain <= 0).sum())
            row[f"{method}_mean_gain_um"] = float(gain.mean()) if len(part) else np.nan
            row[f"{method}_sum_gain_um"] = float(gain.sum()) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, events: pd.DataFrame) -> None:
    lines = [
        "# QC v4 gate + V2 correction",
        "",
        "本结果使用 V4 的采纳/拒绝规则，但对采纳帧直接使用 V2 的动态 shrinkage 修正；未采纳帧回到 pure angle。也就是说，V4 只作为长度信息是否可信的 gate，修正幅度仍由 V2 决定。",
        "",
        "```text",
        "if v4_accept:",
        "    z_final = z_pred_hybrid_qc_v2",
        "else:",
        "    z_final = z_pred_pure_angle",
        "```",
        "",
        "## Metrics",
        "",
        "| split | method | RMSE (um) | MAE (um) | P95 (um) | P99 (um) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    show = metrics[metrics["split"].isin(["train_seed_0_49", "test_seed_50_99", "all"])].copy()
    show = show.sort_values(["split", "method"])
    for _, row in show.iterrows():
        lines.append(
            f"| {row['split']} | {row['method']} | {row['rmse_um']:.6f} | {row['mae_um']:.6f} | {row['p95_abs_error_um']:.6f} | {row['p99_abs_error_um']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## QC Candidate Event Summary",
            "",
            "| subset | n | accepted | method | positive/negative | mean gain (um) | sum gain (um) |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    methods = [
        "current_hard_qc",
        "qc_v2_credibility_shrinkage",
        "qc_v4_continuous_spot_shrinkage",
        "qc_v4_gate_v2_correction",
    ]
    for _, row in events.iterrows():
        for method in methods:
            lines.append(
                f"| {row['subset']} | {int(row['n'])} | {int(row['v4_accepted'])} | {method} | "
                f"{int(row[f'{method}_positive'])}/{int(row[f'{method}_negative'])} | "
                f"{row[f'{method}_mean_gain_um']:.6f} | {row[f'{method}_sum_gain_um']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-frame predictions: `{DATA / 'per_frame_qc_v4_gate_v2_predictions.csv'}`",
            f"- Metrics: `{DATA / 'qc_v4_gate_v2_metrics.csv'}`",
            f"- Event summary: `{DATA / 'qc_v4_gate_v2_event_summary.csv'}`",
        ]
    )
    (REPORTS / "qc_v4_gate_v2_correction_report_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    source = DATA / "per_frame_qc_v4_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, low_memory=False)
    accepted = bool_series(df["qc_v4_trigger_accepted"])
    df["z_pred_hybrid_qc_v4_gate_v2"] = np.where(
        accepted.to_numpy(dtype=bool),
        pd.to_numeric(df["z_pred_hybrid_qc_v2"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["z_pred_pure_angle"], errors="coerce").to_numpy(dtype=float),
    )
    df = add_errors(df)
    metrics = compute_metrics(df)
    events = event_summary(df)
    df.to_csv(DATA / "per_frame_qc_v4_gate_v2_predictions.csv", index=False)
    metrics.to_csv(DATA / "qc_v4_gate_v2_metrics.csv", index=False)
    events.to_csv(DATA / "qc_v4_gate_v2_event_summary.csv", index=False)
    write_report(metrics, events)
    print(metrics[metrics["split"].eq("test_seed_50_99")].to_string(index=False))
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
