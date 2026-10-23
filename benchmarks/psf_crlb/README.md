# PSF and CRLB simulations

- `psf_sim.py`: vector Richards-Wolf PSF simulator for V4, three-hexagon MLA, and the Wang six-zone spiral-phase DH-PSF.
- `crlb.py`: Poisson-Fisher CRLB from a normalized PSF stack.
- `reference/crlb_reference.py`: reference implementation used during development.
- `analyze_v4.py`: four-subimage descriptor analysis for V4 stacks.

Example:

```bash
python benchmarks/psf_crlb/psf_sim.py \
  --config 20x --kind v4 --zmin -10 --zmax 10 --dz 1 \
  --crop 512 --out outputs/psf_v4_20x

python benchmarks/psf_crlb/crlb.py \
  --stack outputs/psf_v4_20x/20x_0.45NA_air_v4.npz \
  --pixel_um 0.225 --out outputs/crlb_v4_20x
```

For MLA/DH, use `--bfp_oversample 6` at 20× to avoid raw-BFP phase aliasing. Generated stacks are not committed.
