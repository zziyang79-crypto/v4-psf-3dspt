from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

import v4psf  # noqa: F401
from v4psf.run_blur_angle_validation import track_unordered_diagonal_angles


class ContractTests(unittest.TestCase):
    def test_core_modules_import(self):
        modules = [
            "spot_detection", "csv_io", "refine", "position_xy",
            "parent_trajectory_compat", "z_fit", "trajectory_subsignal_core",
            "trajectory_association_adapter", "auto_detect", "localization_methods",
        ]
        for name in modules:
            __import__(f"v4psf.{name}")

    def test_calibration_models_are_finite(self):
        model = json.loads((ROOT / "models/phasor_angle_tracked_models.json").read_text())
        for obj in ("20x_0.45NA_air", "40x_0.6NA_air", "60x_1.0NA_water"):
            spline = model["models"][obj]["angle_spline"]
            x = np.asarray(spline["knots_x_theta_deg"], float)
            y = np.asarray(spline["knots_y_z_corrected_relative_um"], float)
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertGreaterEqual(len(x), 4)
            p = PchipInterpolator(x, y, extrapolate=True)
            self.assertTrue(np.isfinite(float(p(float(x[len(x) // 2])))))

    def test_tracking_returns_finite_angles(self):
        t0 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        t1 = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        a, b = track_unordered_diagonal_angles(t0, t1)
        self.assertEqual(a.shape, t0.shape)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(b)))


if __name__ == "__main__":
    unittest.main()
