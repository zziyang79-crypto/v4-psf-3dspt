# V4-PSF analysis code

Code accompanying the manuscript on a prism-array V4-PSF for 3D single-particle tracking.

Repository: https://github.com/zziyang79-crypto/v4-psf-3dspt

This is a work-in-progress public package. The repository currently contains:

- the core detection, localization, calibration, QC, trajectory-association, and 3D-output modules;
- PSF and CRLB simulation code;
- seed-based QC and trajectory-linking benchmark code;
- small reference PSF templates and one example ROI;
- final calibration models and summary result tables.

The claim-to-code map is in `docs/claims_to_code.md`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick start

From the repository root:

```bash
python -m pip install -e .
python examples/quickstart_decode.py \
  --stack data/example_roi/20x_ROI2_2-1.tif \
  --objective 20x_0.45NA_air \
  --out outputs/quickstart_z.csv
```

If not installing the package, set `PYTHONPATH=src` first (PowerShell: `$env:PYTHONPATH="src"`). The example stack is one small 20× calibration ROI and is intended as an interface demonstration.

Benchmark smoke tests:

```bash
python benchmarks/qc_performance/run_benchmark.py --mode smoke
python benchmarks/linking_performance/simulate_and_compare.py --seed 7
```

PSF/CRLB entry points are `benchmarks/psf_crlb/psf_sim.py` and `benchmarks/psf_crlb/crlb.py`.

## Data availability

Only small reference data are included in this repository. Full raw image stacks are available from the authors upon reasonable request.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite:

> Zhang, Ziyang; Wang, Junjie; Che, Bingchen; Ren, Qian; Xie, Hao; He, Yan.
> *Configurable Prism-Array Four-View PSF Microscopy for Three-Dimensional Single-Particle Tracking.*
> Code version 0.1.0 (2026).

Ziyang Zhang and Junjie Wang contributed equally. Hao Xie and Yan He are corresponding authors. `CITATION.cff` contains the machine-readable metadata; a Zenodo DOI will be added after the first archived release.
