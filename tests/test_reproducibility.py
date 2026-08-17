"""Deterministic seed and aggregate split-manifest contract tests."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.reproducibility import derive_seed, seed_everything
from diliplus.reproducibility_audit import build_split_summary


class ReproducibilityTests(unittest.TestCase):
    def test_derived_seed_is_stable_and_namespaced(self):
        first = derive_seed(20260816, "model", 1)
        self.assertEqual(first, derive_seed(20260816, "model", 1))
        self.assertNotEqual(first, derive_seed(20260816, "model", 2))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31 - 1)

    def test_seed_everything_repeats_python_numpy_and_torch(self):
        seed_everything(12345)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        seed_everything(12345)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_group_split_summary_is_order_invariant_and_has_no_overlap(self):
        cohort = pd.DataFrame(
            {
                "encounter_id": [
                    "p1_a",
                    "p1_b",
                    "p2_a",
                    "p3_a",
                    "p4_a",
                    "p5_a",
                ],
                "label_ahi_proxy": [1, 0, 0, 1, 0, 0],
            }
        )
        first = build_split_summary(cohort, n_splits=2)
        second = build_split_summary(
            cohort.sample(frac=1.0, random_state=7), n_splits=2
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["total_group_overlap"], 0)
        self.assertEqual(first["unique_groups"], 5)


if __name__ == "__main__":
    unittest.main()
