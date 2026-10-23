# Fluorescence calibration and step recovery

Scripts for the fluorescence-channel θjoint–z calibration and independent step recovery.

## Expected data layout

Set `V4PSF_FLUORESCENCE_ROOT` to a directory containing the three objective folders with `ROI/<group>/<group>-1.tif` stacks (same layout as the acquisition data). The repository includes only one small 20× ROI example.

## Run

```bash
# localize/track calibration and validation ROIs
python benchmarks/fluorescence_calibration/analyze_all_rois.py
python benchmarks/fluorescence_calibration/track_all_rois.py
python benchmarks/fluorescence_calibration/finalize_fluorescence_models.py

# new independent validation (20x ROI/1, 40x/60x ROI/3)
python benchmarks/fluorescence_calibration/analyze_new_roi.py
python benchmarks/fluorescence_calibration/compute_new_roi_step.py
```

Intermediate files go to `outputs/fluorescence_calibration/` by default; model/result outputs go to `results/fluorescence_calibration/` unless overridden by environment variables. The final models used in the manuscript are in `models/`.

The σ=2.4 px, PCHIP, LOOCV, and step-RMSE definitions follow the manuscript/SI; this directory contains computation code, not the paper figure scripts.
