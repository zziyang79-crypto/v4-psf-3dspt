# Inventory and refactor status

## Included

### Core
- `src/v4psf/spot_detection.py`
- `src/v4psf/refine.py`
- `src/v4psf/position_xy.py`
- `src/v4psf/cross_parent_overlap_repair.py`
- `src/v4psf/parent_trajectory_compat.py`
- `src/v4psf/z_fit.py` (renamed from `Z-nihehanshu.py`)
- `scripts/positions_3d.py` (renamed/moved from `3D positions.py`; still needs CLI/plot split)
- `src/v4psf/trajectory_subsignal_core.py`
- `src/v4psf/trajectory_association_adapter.py`
- `src/v4psf/track_classification.py`
- `src/v4psf/auto_detect.py`
- `src/v4psf/csv_io.py`
- `src/v4psf/localization_methods.py`
- `src/v4psf/run_blur_angle_validation.py`

### Models and results
- `models/phasor_angle_tracked_models.json`
- `models/phasor_length_backup_models.json`
- `models/phasor_angle_tracked_models_fluorescence.json`
- `results/fluorescence_calibration_summary.csv`
- `results/new_validation_step_recovery_summary_ci.csv`

### Benchmarks
- `benchmarks/psf_crlb/psf_sim.py`
- `benchmarks/psf_crlb/crlb.py`
- `benchmarks/psf_crlb/reference/crlb_reference.py`
- `benchmarks/psf_crlb/analyze_v4.py`
- `benchmarks/qc_performance/run_benchmark.py`
- `benchmarks/qc_performance/evaluate_qc_v4_gate_v2_correction.py`
- `benchmarks/qc_performance/evaluate_qc_v4_spot_quality_trigger.py`
- `benchmarks/linking_performance/simulate_and_compare.py`
- `benchmarks/linking_performance/run_trajectory_matching_advantage_sweep.py`
- `benchmarks/linking_performance/run_center_separation_linked_sweep.py`
- `benchmarks/linking_performance/compute_strict_overlap_tolerant_metrics.py`
- `benchmarks/fluorescence_calibration/analyze_all_rois.py`
- `benchmarks/fluorescence_calibration/analyze_new_roi.py`
- `benchmarks/fluorescence_calibration/track_all_rois.py`
- `benchmarks/fluorescence_calibration/finalize_fluorescence_models.py`
- `benchmarks/fluorescence_calibration/compute_new_roi_step.py`

### Data
- `data/psf_templates_20x/` (21 files)
- `data/psf_templates_20x_roicheck/` (21 files)
- `data/example_roi/20x_ROI2_2-1.tif`
- `benchmarks/linking_performance/data/psf_20x_roi16bit/` (21 files)

## Excluded
- GUI, pipeline runner, batch runner.
- Orientation/MSAD/viscosity: `anglesolver(shiyan).py`, `compute_ABCDEF.py`, `compute_acf.py`, `compute_msad.py`, `multi_particle_viscosity_batch.py`.
- Paper figure/plot scripts (`make_*`, `draw_*`, `plot_*`).
- Legacy/duplicate files and all generated/full stacks.
- `evaluate_qc_v3_trigger_logic.py`.
- Internal workbench, diagnostics, and old versions.

## License / LAP
- Trajectory association uses `scipy.optimize.linear_sum_assignment` (SciPy, BSD-3). No LapTrack source is copied.
- No GPL/AGPL dependency found in the copied set.

## Pending refactor
1. Split `scripts/positions_3d.py` into a pure 3D-merge function plus a separate animation/plot script.
2. Expand the unit-test suite beyond import/model contracts (θ unwrapping, step RMSE, benchmark smoke).
3. Add `CITATION.cff`, Zenodo metadata, author list, and final DOI.
4. Replace generic MIT copyright line with the approved author/institution wording if required.
5. Check benchmark defaults and expected CSVs against the final manuscript/SI versions.

