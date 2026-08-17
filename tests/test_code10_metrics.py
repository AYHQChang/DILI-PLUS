from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.evaluation.metrics import (
    add_holm_adjustment,
    cluster_bootstrap_intervals,
    compute_binary_metrics,
    paired_cluster_bootstrap,
)


class Code10MetricTests(unittest.TestCase):
    def setUp(self):
        self.y = np.asarray([0, 0, 1, 0, 1, 0, 0, 1], dtype=int)
        self.p = np.asarray([0.01, 0.03, 0.80, 0.10, 0.70, 0.20, 0.04, 0.60])
        self.groups = np.asarray(["a", "b", "c", "d", "e", "f", "f", "g"])

    def test_metric_contract_contains_discrimination_calibration_and_alert_metrics(self):
        metrics = compute_binary_metrics(
            self.y,
            self.p,
            reference_prevalence=0.25,
            dca_thresholds=np.asarray([0.01, 0.02, 0.05]),
        )
        for key in (
            "AUROC",
            "AUPRC",
            "Brier",
            "Brier_Skill",
            "NLL",
            "Calibration_Intercept",
            "Calibration_Slope",
            "Normalized_pAUC_FPR_0p05",
            "Threshold_0p01_PPV",
            "Top_0p01_Recall",
            "DCA_AUDC",
        ):
            self.assertIn(key, metrics)

    def test_cluster_bootstrap_is_deterministic(self):
        first = cluster_bootstrap_intervals(
            self.y, self.p, self.groups, n_replicates=100, seed=17
        )
        second = cluster_bootstrap_intervals(
            self.y, self.p, self.groups, n_replicates=100, seed=17
        )
        self.assertTrue(first.equals(second))

    def test_paired_comparison_and_holm_contract(self):
        comparison = paired_cluster_bootstrap(
            self.y,
            self.p,
            np.full(len(self.y), 0.3),
            self.groups,
            n_replicates=100,
            seed=21,
        )
        adjusted = add_holm_adjustment(comparison)
        self.assertIn("p_value_holm", adjusted)
        self.assertTrue((adjusted["p_value_holm"] >= adjusted["p_value_two_sided"]).all())


if __name__ == "__main__":
    unittest.main()
