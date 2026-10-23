# Claim-to-code map

This repository contains the analysis code accompanying the V4-PSF 3D single-particle tracking manuscript. The map below is the intended public surface; data-generation and benchmark scripts are included so the reported simulation metrics can be regenerated from fixed seeds.

## Core method

| Claim | Code | Notes |
|---|---|---|
| Four-subimage detection and subpixel localization | `src/v4psf/spot_detection.py`, `src/v4psf/localization_methods.py` | LoG detection, phasor four-spot localization |
| Four-spot geometry and XY/intensity extraction | `src/v4psf/position_xy.py`, `src/v4psf/refine.py` | θ1/θ2, diagonal distance, subspot repair |
| Angle-primary axial calibration | `src/v4psf/Z-nihehanshu.py` | PCHIP θjoint→z, dmean consistency/QC |
| Three-spot fallback | `src/v4psf/position_xy.py`, `src/v4psf/trajectory_subsignal_core.py` | Parallelogram completion and confidence flag |
| Parent/trajectory stitching | `src/v4psf/parent_trajectory_compat.py`, `scripts/positions_3d.py` | Gap interpolation and 3D merge |
| Trajectory-level sub-signal association | `src/v4psf/trajectory_subsignal_core.py`, `src/v4psf/trajectory_association_adapter.py`, `src/v4psf/track_classification.py` | SciPy `linear_sum_assignment` (BSD-3); no LapTrack source copied |
| Automatic detection workflow | `src/v4psf/auto_detect.py` | CLI/benchmark-friendly entry point |

## PSF and CRLB

| Claim | Code | Notes |
|---|---|---|
| V4/MLA/DH vector PSF simulation | `benchmarks/psf_crlb/psf_sim.py` | 20×/40×/60× configs, raw-BFP oversampling for MLA/DH |
| Poisson-Fisher CRLB | `benchmarks/psf_crlb/crlb.py` | CRLBx/y/z/3D |
| Reference CRLB implementation | `benchmarks/psf_crlb/reference/crlb_reference.py` | Reference/validation implementation |

## QC performance

| Claim | Code | Notes |
|---|---|---|
| QC feature extraction/gating | `src/v4psf/Z-nihehanshu.py` | Production QC |
| Seed-based QC stack generation | `benchmarks/qc_performance/run_benchmark.py` | PSF-template rendering, noise/aberration/motion, Brownian z |
| QC evaluation | `benchmarks/qc_performance/evaluate_qc_v4_gate_v2_correction.py`, `evaluate_qc_v4_spot_quality_trigger.py` | Event/metric summaries |

## Trajectory-linking performance

| Claim | Code | Notes |
|---|---|---|
| Seed-based synthetic tracks/stacks | `benchmarks/linking_performance/simulate_and_compare.py` | PSF-template rendering, three particles, dropout/clutter/crossing |
| Seed sweeps | `benchmarks/linking_performance/run_trajectory_matching_advantage_sweep.py`, `run_center_separation_linked_sweep.py` | Fixed seeds and condition sweeps |
| Strict accuracy/recall | `benchmarks/linking_performance/compute_strict_overlap_tolerant_metrics.py` | Group-level strict accuracy and real-source recall |

## Fluorescence calibration/validation

| Claim | Code | Notes |
|---|---|---|
| ROI localization and tracking | `benchmarks/fluorescence_calibration/analyze_all_rois.py`, `analyze_new_roi.py`, `track_all_rois.py` | σ=2.4 px phasor pipeline |
| Calibration model | `benchmarks/fluorescence_calibration/finalize_fluorescence_models.py`, `models/phasor_angle_tracked_models_fluorescence.json` | PCHIP model |
| LOOCV / independent step recovery | `benchmarks/fluorescence_calibration/compute_new_roi_step.py`, `compute_statistics_ci.py` | Small expected CSVs in `results/` |

## Data included

- `data/psf_templates_20x/`: 21 measured 20× PSF templates (`10015`–`10035`).
- `data/psf_templates_20x_roicheck/`: 21 independent-check templates used by the QC stress mode.
- `data/example_roi/`: one small 20× fluorescence ROI for the decoder quickstart.
- `benchmarks/linking_performance/data/psf_20x_roi16bit/`: 21 templates required by the linking benchmark.
- `models/`: final scattering and fluorescence calibration models.
- `results/`: small summary tables only; no raw stacks.

## Explicitly excluded

- GUI and acquisition/orchestration code.
- Orientation/MSAD/viscosity analysis.
- Full raw stacks and full generated benchmark stacks.
- Paper-figure scripts and diagnostic/legacy files.
- Old ROI/plateau diagnostics and failed intermediate versions.
