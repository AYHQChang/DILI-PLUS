"""Contracts for the post-hoc P1 calibration, process, and subgroup audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.evaluation.p1_supplementary import (  # noqa: E402
    PROCESS_FEATURES,
    _subgroup_assignments,
    calibration_bin_intervals,
)
from diliplus.reporting.formal_assets import MODEL_COLORS  # noqa: E402
from diliplus.reporting.p1_contract import (  # noqa: E402
    AUDIT_COLORS,
    AUDIT_MODELS,
    PROCESS_MODEL,
)


class P1SupplementaryTests(unittest.TestCase):
    def test_process_baseline_uses_only_measurement_process_features(self):
        self.assertEqual(len(PROCESS_FEATURES), 9)
        joined = " ".join(PROCESS_FEATURES).lower()
        for forbidden in ("token", "drug", "diagnosis_name", "lab_value", "gender", "age"):
            self.assertNotIn(forbidden, joined)

    def test_formal_model_colours_remain_immutable_in_subgroup_figure(self):
        for model in AUDIT_MODELS:
            if model != PROCESS_MODEL:
                self.assertEqual(AUDIT_COLORS[model], MODEL_COLORS[model])
        self.assertNotIn(PROCESS_MODEL, MODEL_COLORS)

    def test_subgroup_assignment_separates_fairness_and_process_dimensions(self):
        frame = pd.DataFrame(
            {
                "dataset_index": np.arange(9),
                "gender_male": [0, 1, np.nan, 0, 1, 0, 1, 0, 1],
                "observation_hours": np.arange(1, 10, dtype=float),
            }
        )
        assignments, definitions = _subgroup_assignments(frame)
        self.assertEqual(set(assignments["audit_dimension"]), {"Sex", "Observation window"})
        self.assertEqual(definitions["sex"]["missing_encounters"], 1)
        self.assertIn("not a fairness attribute", definitions["observation_window_hours"]["interpretation"])

    def test_clustered_calibration_intervals_are_finite_and_reproducible(self):
        y = np.tile([0, 0, 0, 1], 25)
        p = np.linspace(0.001, 0.2, len(y))
        groups = np.array([f"p{i}" for i in range(len(y))])
        left = calibration_bin_intervals(
            y, p, groups, n_replicates=50, seed=20260816, n_bins=5
        )
        right = calibration_bin_intervals(
            y, p, groups, n_replicates=50, seed=20260816, n_bins=5
        )
        pd.testing.assert_frame_equal(left, right)
        self.assertTrue(np.isfinite(left[["ci_lower", "ci_upper"]]).all().all())
        self.assertTrue((left["ci_lower"] <= left["observed_fraction"]).all())
        self.assertTrue((left["observed_fraction"] <= left["ci_upper"]).all())

    def test_pipeline_isolates_analysis_from_rendering(self):
        source = (PROJECT_ROOT / "pipelines" / "06_build_p1_supplementary.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("subprocess.run", source)
        self.assertIn("check=True", source)
        self.assertIn("_default_render_python", source)
        self.assertIn('name == "analysis"', source)
        self.assertNotIn("from diliplus.evaluation.p1_supplementary import run_p1_analysis", source)


if __name__ == "__main__":
    unittest.main()
