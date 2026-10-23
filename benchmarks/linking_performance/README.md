# Trajectory-linking performance benchmark

Self-contained simulation comparing frame-wise geometry grouping with trajectory-level sub-signal association.

## Inputs

- `data/psf_20x_roi16bit/`: 21 measured 20× PSF templates (`10015`–`10035`), treated as 0–20 µm.
- The simulator places three particles, renders four-subimage PSFs, and adds dropout, clutter, blinking, and a crossing region.

## Run

```bash
python benchmarks/linking_performance/simulate_and_compare.py --seed 7
python benchmarks/linking_performance/simulate_and_compare.py --sweep --n-seeds 20 --frames 40
```

Seed sweeps:

```bash
python benchmarks/linking_performance/run_trajectory_matching_advantage_sweep.py
python benchmarks/linking_performance/run_center_separation_linked_sweep.py
```

## Metrics

`compute_strict_overlap_tolerant_metrics.py` defines the strict association accuracy and real-source recall:

- a group is correct only when all supported visible subimages belong to the same real particle;
- groups mixing sources or unsupported clutter are incorrect;
- recall is evaluated only on source-frames with at least three visible subimages.

Small expected summaries are in `expected_association_accuracy_summary.csv` and `expected_sweep_summary_mean_sd.csv`. Full generated stacks are not committed; they are reproducible from the fixed seeds.
