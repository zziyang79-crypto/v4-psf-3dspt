# Quickstart

From the repository root:

```bash
pip install -r requirements.txt
set PYTHONPATH=src        # Windows PowerShell: $env:PYTHONPATH="src"
python examples/quickstart_decode.py \
  --stack data/example_roi/20x_ROI2_2-1.tif \
  --objective 20x_0.45NA_air \
  --out outputs/quickstart_z.csv
```

The script localizes four subimages per frame, computes `theta_joint`/`dmean`, applies the final 20× fluorescence PCHIP model, and writes per-frame z values. The example stack is one small calibration ROI; it is intended as an interface demonstration, not as a replacement for the full validation datasets.
