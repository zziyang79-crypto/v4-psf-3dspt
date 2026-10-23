# QC performance benchmark

Seed-based simulation benchmark for angle-primary Z with distance-based QC.

## Inputs

- `config.json`: condition grids and fixed seeds.
- `../../data/psf_templates_20x/`: 21 measured 20× PSF templates.
- `../../data/psf_templates_20x_roicheck/`: independent-check templates for the QC stress mode.
- `../../models/phasor_angle_tracked_models.json`
- `../../models/phasor_length_backup_models.json`

## Run

```bash
python benchmarks/qc_performance/run_benchmark.py --mode smoke
python benchmarks/qc_performance/run_benchmark.py --mode full
python benchmarks/qc_performance/run_benchmark.py --mode qc_stress_v2
```

Outputs are written under `benchmarks/qc_performance/<mode>/`; generated stacks are ignored by git. The smoke mode is the quick interface check; the full/QC-stress modes use the fixed seed lists in `config.json`.

## Evaluation

- `evaluate_qc_v4_gate_v2_correction.py`: soft/hard QC correction comparison.
- `evaluate_qc_v4_spot_quality_trigger.py`: spot-quality trigger evaluation.
- `expected_qc_v4_gate_v2_metrics.csv` and `expected_qc_v4_gate_v2_event_summary.csv`: small reference summaries.

The benchmark reports axial RMSE/MAE and event counts; the manuscript/SI contain the complete definitions and condition values.
